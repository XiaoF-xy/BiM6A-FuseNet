from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer


TOKENIZER_HINT = (
    "Tokenizer files were not found. If the tokenizer is stored separately, "
    "rerun with an explicit path such as: --tokenizer_dir ./pretrained/birna-tokenizer"
)


def load_birna_tokenizer(tokenizer_dir: Path, max_length: int):
    tokenizer_dir = Path(tokenizer_dir)
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_dir}. {TOKENIZER_HINT}")
    if not any((tokenizer_dir / name).exists() for name in ("tokenizer.json", "vocab.txt")):
        raise FileNotFoundError(f"{TOKENIZER_HINT}. Checked directory: {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = max_length
    return tokenizer


def load_birna_backbone(model_dir: Path) -> nn.Module:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing model config.json under: {model_dir}")
    if not ((model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()):
        raise FileNotFoundError(f"Missing model weights under: {model_dir}")

    config = transformers.BertConfig.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    birna_model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    )
    birna_model.cls = torch.nn.Identity()
    return birna_model


def apply_lora_to_birna(
    birna_model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for LoRA. Install it with: pip install peft "
            "or recreate the environment with env/create_conda_env_cuda121.sh."
        ) from exc

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(birna_model, lora_config)


class BiRNANucClassifier(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        center_index: int = 20,
        use_center_pooling: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        self.birna_model = load_birna_backbone(model_dir)
        self.use_lora = use_lora
        self.center_index = center_index
        self.use_center_pooling = use_center_pooling
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))
        if use_lora:
            self.birna_model = apply_lora_to_birna(
                birna_model=self.birna_model,
                target_modules=lora_target_modules or ["Wqkv"],
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        classifier_input_size = hidden_size * (2 if use_center_pooling else 1)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        if freeze_backbone and not use_lora:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.birna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb = outputs.logits
        if emb.ndim != 3:
            raise ValueError(f"Expected BiRNA-BERT token embeddings with shape [B, L, H], got: {tuple(emb.shape)}")

        token_emb = emb[:, 1:-1, :]
        if self.use_center_pooling and token_emb.size(1) <= self.center_index:
            raise ValueError(
                "BiRNA-BERT output is too short for 41bp center pooling: "
                f"token_count={token_emb.size(1)}, center_index={self.center_index}. "
                "Check NUC tokenization and max_length."
            )
        mean_feat = token_emb.mean(dim=1)
        if self.use_center_pooling:
            center_feat = token_emb[:, self.center_index, :]
            feat = torch.cat([mean_feat, center_feat], dim=1)
        else:
            feat = mean_feat
        return self.classifier(feat)
