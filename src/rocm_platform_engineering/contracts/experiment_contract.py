"""Experiment contract for reproducible AMD ROCm platform experiments."""

from __future__ import annotations

import argparse
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


EXPERIMENT_ID_PATTERN = re.compile(
    r"^exp-p(?P<phase>0[0-8])-[0-9]{3}-"
    r"[a-z0-9]+(?:-[a-z0-9]+)*$"
)

PROFILE_ID_PATTERNS = {
    "hardware_profile_id": re.compile(
        r"^hw-[a-z0-9]+(?:-[a-z0-9]+)*$"
    ),
    "software_stack_id": re.compile(
        r"^sw-[a-z0-9]+(?:-[a-z0-9]+)*$"
    ),
    "workload_profile_id": re.compile(
        r"^wl-[a-z0-9]+(?:-[a-z0-9]+)*$"
    ),
}


class ExecutionType(StrEnum):
    """Supported experiment execution categories."""

    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    PROFILING = "profiling"
    FAILURE_INJECTION = "failure_injection"


class ExperimentStatus(StrEnum):
    """Lifecycle states for a declared experiment."""

    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class MetricDefinition(BaseModel):
    """A metric that the experiment intends to collect."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(
        min_length=3,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
        description="Metric name in lowercase snake case.",
    )
    unit: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
        description="Explicit measurement unit.",
    )
    description: str = Field(
        min_length=10,
        description="What the metric measures and why it matters.",
    )


class ExperimentContract(BaseModel):
    """Declared intent and acceptance boundary for one experiment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    experiment_id: str = Field(
        description=(
            "Stable identifier using "
            "exp-p<phase>-<sequence>-<descriptive-name>."
        )
    )
    title: str = Field(min_length=10, max_length=160)
    phase: int = Field(ge=0, le=8)
    workstream: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )

    execution_type: ExecutionType
    status: ExperimentStatus = ExperimentStatus.PLANNED

    objective: str = Field(min_length=20)
    question: str = Field(min_length=20)
    hypothesis: str = Field(min_length=20)

    prerequisites: list[str] = Field(default_factory=list)

    hardware_profile_id: str | None = None
    software_stack_id: str | None = None
    workload_profile_id: str | None = None

    independent_variable: str = Field(
        min_length=3,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    controlled_variables: list[str] = Field(min_length=1)

    metrics: list[MetricDefinition] = Field(min_length=1)

    acceptance_criteria: list[str] = Field(min_length=1)
    stop_conditions: list[str] = Field(min_length=1)
    evidence_required: list[str] = Field(min_length=1)

    notes: list[str] = Field(default_factory=list)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_identifier(cls, value: str) -> str:
        """Require the agreed experiment identifier format."""

        if not EXPERIMENT_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "experiment_id must match "
                "'exp-p<00-08>-<three-digit-sequence>-<descriptive-name>'"
            )
        return value

    @field_validator(
        "hardware_profile_id",
        "software_stack_id",
        "workload_profile_id",
    )
    @classmethod
    def validate_profile_identifier(
        cls,
        value: str | None,
        validation_info: Any,
    ) -> str | None:
        """Validate profile prefixes without defining future schemas here."""

        if value is None:
            return value

        field_name = validation_info.field_name
        pattern = PROFILE_ID_PATTERNS[field_name]

        if not pattern.fullmatch(value):
            expected_prefix = {
                "hardware_profile_id": "hw-",
                "software_stack_id": "sw-",
                "workload_profile_id": "wl-",
            }[field_name]

            raise ValueError(
                f"{field_name} must use the '{expected_prefix}' prefix "
                "and lowercase kebab case"
            )

        return value

    @field_validator(
        "prerequisites",
        "controlled_variables",
        "acceptance_criteria",
        "stop_conditions",
        "evidence_required",
        "notes",
    )
    @classmethod
    def require_unique_list_values(
        cls,
        values: list[str],
    ) -> list[str]:
        """Reject empty or duplicate list entries."""

        normalized_values = [value.strip() for value in values]

        if any(not value for value in normalized_values):
            raise ValueError("List entries must not be empty")

        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("List entries must be unique")

        return normalized_values

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "ExperimentContract":
        """Validate relationships between contract fields."""

        identifier_match = EXPERIMENT_ID_PATTERN.fullmatch(
            self.experiment_id
        )

        if identifier_match is None:
            return self

        identifier_phase = int(identifier_match.group("phase"))

        if identifier_phase != self.phase:
            raise ValueError(
                "The phase encoded in experiment_id must match "
                f"the phase field: {identifier_phase} != {self.phase}"
            )

        if self.execution_type in {
            ExecutionType.BENCHMARK,
            ExecutionType.PROFILING,
            ExecutionType.FAILURE_INJECTION,
        }:
            if self.hardware_profile_id is None:
                raise ValueError(
                    "hardware_profile_id is required for benchmark, "
                    "profiling, and failure-injection experiments"
                )

            if self.software_stack_id is None:
                raise ValueError(
                    "software_stack_id is required for benchmark, "
                    "profiling, and failure-injection experiments"
                )

        metric_names = [metric.name for metric in self.metrics]

        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Metric names must be unique")

        return self


def load_experiment_contract(
    contract_path: str | Path,
) -> ExperimentContract:
    """Load and validate an experiment contract from YAML."""

    resolved_path = Path(contract_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Experiment contract does not exist: {resolved_path}"
        )

    try:
        raw_data = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as error:
        raise ValueError(
            f"Experiment contract contains invalid YAML: {resolved_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Experiment contract root must be a YAML mapping"
        )

    return ExperimentContract.model_validate(raw_data)


def write_experiment_json_schema(
    output_path: str | Path,
) -> Path:
    """Generate the JSON Schema for external tooling and editors."""

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    schema = ExperimentContract.model_json_schema()

    resolved_path.write_text(
        json.dumps(schema, indent=2) + "\n",
        encoding="utf-8",
    )

    return resolved_path


def main() -> None:
    """Validate a contract or generate the JSON Schema."""

    parser = argparse.ArgumentParser(
        description="Validate an AMD ROCm experiment contract."
    )
    parser.add_argument(
        "contract_path",
        nargs="?",
        help="Path to the YAML experiment contract.",
    )
    parser.add_argument(
        "--write-schema",
        dest="schema_path",
        help="Write the generated JSON Schema to this path.",
    )

    arguments = parser.parse_args()

    if arguments.schema_path:
        output_path = write_experiment_json_schema(
            arguments.schema_path
        )
        print(f"Wrote experiment schema: {output_path}")

    if arguments.contract_path:
        contract = load_experiment_contract(
            arguments.contract_path
        )
        print(
            "Validated experiment contract: "
            f"{contract.experiment_id}"
        )

    if not arguments.schema_path and not arguments.contract_path:
        parser.error(
            "Provide a contract path, --write-schema, or both."
        )


if __name__ == "__main__":
    main()
