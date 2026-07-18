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
