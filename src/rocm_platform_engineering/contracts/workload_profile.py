"""Minimal workload profile for AMD ROCm experiments."""

from __future__ import annotations

import argparse
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


WORKLOAD_PROFILE_ID_PATTERN = re.compile(
    r"^wl-[a-z0-9]+(?:-[a-z0-9]+)*$"
)


class WorkloadType(StrEnum):
    """Supported high-level workload categories."""

    INFERENCE = "inference"
    TRAINING = "training"


class ArrivalPattern(StrEnum):
    """How inference requests enter the serving system."""

    SINGLE = "single"
    STEADY = "steady"
    BURST = "burst"
    CLOSED_LOOP = "closed_loop"


class WorkloadProfile(BaseModel):
    """Reusable workload inputs for an experiment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    workload_profile_id: str
    workload_type: WorkloadType

    model_name: str = Field(min_length=2)
    model_revision: str = Field(min_length=1)

    input_token_count: int = Field(ge=1)
    output_token_count: int = Field(ge=1)

    request_count: int = Field(ge=1)
    concurrency: int = Field(ge=1)

    arrival_pattern: ArrivalPattern

    random_seed: int = Field(ge=0)

    tensor_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    expert_parallel_size: int = Field(default=1, ge=1)

    @field_validator("workload_profile_id")
    @classmethod
    def validate_workload_profile_id(cls, value: str) -> str:
        """Require the agreed workload-profile identifier format."""
        if not WORKLOAD_PROFILE_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "workload_profile_id must start with 'wl-' "
                "and use lowercase kebab case"
            )
        return value


def load_workload_profile(
    profile_path: str | Path,
) -> WorkloadProfile:
    """Load and validate a workload profile from YAML."""
    resolved_path = Path(profile_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Workload profile does not exist: {resolved_path}"
        )

    try:
        raw_data = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as error:
        raise ValueError(
            "Workload profile contains invalid YAML: "
            f"{resolved_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Workload profile root must be a YAML mapping"
        )

    return WorkloadProfile.model_validate(raw_data)


def write_workload_profile_json_schema(
    output_path: str | Path,
) -> Path:
    """Generate JSON Schema from the Pydantic model."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_path.write_text(
        json.dumps(
            WorkloadProfile.model_json_schema(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return resolved_path


def main() -> None:
    """Validate a profile or generate its JSON Schema."""
    parser = argparse.ArgumentParser(
        description="Validate an AMD ROCm workload profile."
    )

    parser.add_argument(
        "profile_path",
        nargs="?",
        help="Path to a YAML workload profile.",
    )

    parser.add_argument(
        "--write-schema",
        dest="schema_path",
        help="Write the generated JSON Schema to this path.",
    )

    arguments = parser.parse_args()

    if arguments.schema_path:
        output_path = write_workload_profile_json_schema(
            arguments.schema_path
        )
        print(f"Wrote workload profile schema: {output_path}")

    if arguments.profile_path:
        profile = load_workload_profile(arguments.profile_path)
        print(
            "Validated workload profile: "
            f"{profile.workload_profile_id}"
        )

    if not arguments.profile_path and not arguments.schema_path:
        parser.error(
            "Provide a profile path, --write-schema, or both."
        )


if __name__ == "__main__":
    main()
