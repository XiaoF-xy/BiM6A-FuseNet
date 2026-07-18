from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_name = str(name).lower()
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}. Supported optimizers: adam, adamw")


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    patience: int | None,
    factor: float,
):
    if patience is None:
        return None
    if patience < 0:
        raise ValueError(f"scheduler patience must be non-negative, got: {patience}")
    if not 0.0 < factor < 1.0:
        raise ValueError(f"scheduler factor must be between 0 and 1, got: {factor}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
    )


def build_constant_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float | None,
):
    """Linearly warm up to the configured optimizer LR, then keep it constant."""
    if warmup_ratio is None:
        return None
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got: {total_steps}")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got: {warmup_ratio}")

    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
    )


class EarlyStopping:
    def __init__(self, patience: int):
        if patience <= 0:
            raise ValueError(f"early-stopping patience must be positive, got: {patience}")
        self.patience = int(patience)
        self.bad_epochs = 0

    def update(self, *, improved: bool) -> bool:
        if improved:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
