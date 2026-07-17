"""Minimal hardware profile for reproducible AMD ROCm experiments."""

from __future__ import annotations

import argparse
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


HARDWARE_PROFILE_ID_PATTERN = re.compile(
    r"^hw-[a-z0-9]+(?:-[a-z0-9]+)*$"
)


class GpuInterconnectType(StrEnum):
    """Primary communication path between GPUs."""

    PCIE = "pcie"
    XGMI = "xgmi"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class HardwareProfile(BaseModel):
    """Stable hardware facts required to interpret an experiment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    hardware_profile_id: str

    provider: str = Field(min_length=2)
    node_type: str = Field(min_length=2)

    gpu_model: str = Field(min_length=2)
    gpu_architecture: str = Field(min_length=2)

    physical_gpu_count: int = Field(ge=1)
    logical_gpu_count: int = Field(ge=1)

    hbm_gib_per_gpu: float = Field(gt=0)

    compute_partition_mode: str = Field(min_length=2)
    memory_partition_mode: str = Field(min_length=2)

    gpu_interconnect: GpuInterconnectType

    cpu_model: str = Field(min_length=2)
    system_memory_gib: float = Field(gt=0)
    numa_node_count: int = Field(ge=1)

    @field_validator("hardware_profile_id")
    @classmethod
    def validate_hardware_profile_id(cls, value: str) -> str:
        """Require the agreed lowercase hardware-profile format."""
        if not HARDWARE_PROFILE_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "hardware_profile_id must start with 'hw-' "
                "and use lowercase kebab case"
            )
        return value

    @field_validator(
        "compute_partition_mode",
        "memory_partition_mode",
    )
    @classmethod
    def normalize_partition_mode(cls, value: str) -> str:
        """Store partition modes consistently in lowercase."""
        return value.lower()


def load_hardware_profile(
    profile_path: str | Path,
) -> HardwareProfile:
    """Load and validate a hardware profile from YAML."""
    resolved_path = Path(profile_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Hardware profile does not exist: {resolved_path}"
        )

    try:
        raw_data = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as error:
        raise ValueError(
            f"Hardware profile contains invalid YAML: {resolved_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Hardware profile root must be a YAML mapping"
        )

    return HardwareProfile.model_validate(raw_data)


def write_hardware_profile_json_schema(
    output_path: str | Path,
) -> Path:
    """Generate JSON Schema from the Pydantic model."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_path.write_text(
        json.dumps(
            HardwareProfile.model_json_schema(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return resolved_path


def main() -> None:
    """Validate a profile or generate its JSON Schema."""
    parser = argparse.ArgumentParser(
        description="Validate an AMD hardware profile."
    )

    parser.add_argument(
        "profile_path",
        nargs="?",
        help="Path to a YAML hardware profile.",
    )

    parser.add_argument(
        "--write-schema",
        dest="schema_path",
        help="Write the generated JSON Schema to this path.",
    )

    arguments = parser.parse_args()

    if arguments.schema_path:
        output_path = write_hardware_profile_json_schema(
            arguments.schema_path
        )
        print(f"Wrote hardware profile schema: {output_path}")

    if arguments.profile_path:
        profile = load_hardware_profile(
            arguments.profile_path
        )
        print(
            "Validated hardware profile: "
            f"{profile.hardware_profile_id}"
        )

    if not arguments.profile_path and not arguments.schema_path:
        parser.error(
            "Provide a profile path, --write-schema, or both."
        )


if __name__ == "__main__":
    main()
