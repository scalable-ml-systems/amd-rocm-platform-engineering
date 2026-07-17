"""Minimal software-stack profile for AMD ROCm experiments."""

from __future__ import annotations

import argparse
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


SOFTWARE_STACK_ID_PATTERN = re.compile(
    r"^sw-[a-z0-9]+(?:-[a-z0-9]+)*$"
)

CONTAINER_DIGEST_PATTERN = re.compile(
    r"^sha256:[a-f0-9]{64}$"
)


class ExecutionEngine(StrEnum):
    """Primary framework or runtime being evaluated."""

    HIP = "hip"
    PYTORCH = "pytorch"
    RCCL = "rccl"
    VLLM = "vllm"
    SGLANG = "sglang"


class SoftwareStack(BaseModel):
    """Software versions needed to reproduce an experiment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    software_stack_id: str

    operating_system: str = Field(min_length=2)
    linux_kernel: str = Field(min_length=2)

    amdgpu_driver: str = Field(min_length=1)
    rocm_version: str = Field(min_length=1)

    python_version: str | None = None
    pytorch_version: str | None = None

    container_image: str | None = None
    container_image_digest: str | None = None

    execution_engine: ExecutionEngine
    execution_engine_version: str | None = None

    rccl_version: str | None = None
    triton_version: str | None = None
    aiter_version: str | None = None
    atom_version: str | None = None

    @field_validator("software_stack_id")
    @classmethod
    def validate_software_stack_id(cls, value: str) -> str:
        """Require the agreed lowercase software-stack identifier."""
        if not SOFTWARE_STACK_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "software_stack_id must start with 'sw-' "
                "and use lowercase kebab case"
            )
        return value

    @field_validator("container_image_digest")
    @classmethod
    def validate_container_digest(
        cls,
        value: str | None,
    ) -> str | None:
        """Require an immutable SHA-256 digest when supplied."""
        if value is None:
            return value

        if not CONTAINER_DIGEST_PATTERN.fullmatch(value):
            raise ValueError(
                "container_image_digest must use "
                "'sha256:<64 lowercase hexadecimal characters>'"
            )

        return value


def load_software_stack(
    profile_path: str | Path,
) -> SoftwareStack:
    """Load and validate a software-stack profile from YAML."""
    resolved_path = Path(profile_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Software-stack profile does not exist: {resolved_path}"
        )

    try:
        raw_data = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as error:
        raise ValueError(
            "Software-stack profile contains invalid YAML: "
            f"{resolved_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Software-stack profile root must be a YAML mapping"
        )

    return SoftwareStack.model_validate(raw_data)


def write_software_stack_json_schema(
    output_path: str | Path,
) -> Path:
    """Generate JSON Schema from the Pydantic model."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_path.write_text(
        json.dumps(
            SoftwareStack.model_json_schema(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return resolved_path


def main() -> None:
    """Validate a profile or generate its JSON Schema."""
    parser = argparse.ArgumentParser(
        description="Validate an AMD ROCm software-stack profile."
    )

    parser.add_argument(
        "profile_path",
        nargs="?",
        help="Path to a YAML software-stack profile.",
    )

    parser.add_argument(
        "--write-schema",
        dest="schema_path",
        help="Write the generated JSON Schema to this path.",
    )

    arguments = parser.parse_args()

    if arguments.schema_path:
        output_path = write_software_stack_json_schema(
            arguments.schema_path
        )
        print(f"Wrote software-stack schema: {output_path}")

    if arguments.profile_path:
        profile = load_software_stack(arguments.profile_path)
        print(
            "Validated software stack: "
            f"{profile.software_stack_id}"
        )

    if not arguments.profile_path and not arguments.schema_path:
        parser.error(
            "Provide a profile path, --write-schema, or both."
        )


if __name__ == "__main__":
    main()
