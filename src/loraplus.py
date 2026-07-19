from __future__ import annotations

from collections.abc import Iterable
from typing import Any


LORA_A_MARKER = ".lora_A."
LORA_B_MARKER = ".lora_B."


def partition_loraplus_named_parameters(
    named_parameters: Iterable[tuple[str, Any]],
) -> dict[str, list[Any]]:
    """Partition all trainable parameters into exclusive LoRA A/B/head groups."""
    groups: dict[str, list[Any]] = {
        "lora_A": [],
        "lora_B": [],
        "classifier": [],
    }
    seen_parameter_ids: set[int] = set()

    for parameter_name, parameter in named_parameters:
        if not getattr(parameter, "requires_grad", False):
            continue
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            raise ValueError(
                "LoRA+ received a duplicate trainable parameter object: "
                f"{parameter_name}"
            )
        seen_parameter_ids.add(parameter_id)

        if LORA_A_MARKER in parameter_name:
            group_name = "lora_A"
        elif LORA_B_MARKER in parameter_name:
            group_name = "lora_B"
        else:
            group_name = "classifier"
        groups[group_name].append(parameter)

    missing = [name for name, parameters in groups.items() if not parameters]
    if missing:
        raise ValueError("LoRA+ parameter groups missing: " + ", ".join(missing))
    return groups
