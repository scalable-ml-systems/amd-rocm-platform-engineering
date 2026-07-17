"""Minimal result manifest for one AMD ROCm experiment run."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


RUN_ID_PATTERN = re.compile(
    r"^run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)

EXPERIMENT_ID_PATTERN = re.compile(
    r"^exp-p0[0-8]-[0-9]{3}-"
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

GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


class RunStatus(StrEnum):
    """Final state of one experiment run."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    STOPPED = "stopped"


class SummaryMetric(BaseModel):
    """One summarized measurement from the experiment run."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(
        min_length=3,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    value: float | int
    unit: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )


class ExperimentResult(BaseModel):
    """Traceable outcome of one experiment execution."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    run_id: str
    experiment_id: str

    hardware_profile_id: str
    software_stack_id: str
    workload_profile_id: str | None = None

    git_revision: str

    started_at_utc: datetime
    finished_at_utc: datetime

    status: RunStatus
    exit_code: int

    summary_metrics: list[SummaryMetric] = Field(
        default_factory=list
    )
    evidence_paths: list[str] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """Require timestamp and Git revision in the run identifier."""
        if not RUN_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "run_id must match "
                "'run-<YYYYMMDDTHHMMSSZ>-<eight-character-git-revision>'"
            )
        return value

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        """Require the agreed experiment identifier format."""
        if not EXPERIMENT_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "experiment_id must use the agreed experiment format"
            )
        return value

    @field_validator(
        "hardware_profile_id",
        "software_stack_id",
        "workload_profile_id",
    )
    @classmethod
    def validate_profile_id(
        cls,
        value: str | None,
        validation_info: Any,
    ) -> str | None:
        """Validate profile prefixes."""
        if value is None:
            return value

        pattern = PROFILE_ID_PATTERNS[
            validation_info.field_name
        ]
        if not pattern.fullmatch(value):
            raise ValueError(
                f"{validation_info.field_name} has an invalid format"
            )

        return value

    @field_validator("git_revision")
    @classmethod
    def validate_git_revision(cls, value: str) -> str:
        """Require a hexadecimal Git revision."""
        if not GIT_REVISION_PATTERN.fullmatch(value):
            raise ValueError(
                "git_revision must contain 7 to 40 "
                "lowercase hexadecimal characters"
            )
        return value

    @field_validator("evidence_paths", "notes")
    @classmethod
    def validate_unique_list_entries(
        cls,
        values: list[str],
    ) -> list[str]:
        """Reject empty or duplicate path and note entries."""
        normalized_values = [value.strip() for value in values]

        if any(not value for value in normalized_values):
            raise ValueError("List entries must not be empty")

        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("List entries must be unique")

        return normalized_values

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "ExperimentResult":
        """Validate timestamps, exit code, and metric uniqueness."""
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError(
                "finished_at_utc must not be earlier than "
                "started_at_utc"
            )

        if self.status == RunStatus.PASS and self.exit_code != 0:
            raise ValueError(
                "A passing run must have exit_code 0"
            )

        metric_names = [metric.name for metric in self.summary_metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                "Summary metric names must be unique"
            )

        return self


def load_experiment_result(
    result_path: str | Path,
) -> ExperimentResult:
    """Load and validate an experiment result from JSON."""
    resolved_path = Path(result_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Experiment result does not exist: {resolved_path}"
        )

    try:
        raw_data = json.loads(
            resolved_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Experiment result contains invalid JSON: "
            f"{resolved_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Experiment result root must be a JSON object"
        )

    return ExperimentResult.model_validate(raw_data)


def write_experiment_result_json_schema(
    output_path: str | Path,
) -> Path:
    """Generate JSON Schema from the Pydantic model."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_path.write_text(
        json.dumps(
            ExperimentResult.model_json_schema(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return resolved_path


def main() -> None:
    """Validate a result or generate its JSON Schema."""
    parser = argparse.ArgumentParser(
        description="Validate an AMD ROCm experiment result."
    )

    parser.add_argument(
        "result_path",
        nargs="?",
        help="Path to an experiment-result JSON file.",
    )

    parser.add_argument(
        "--write-schema",
        dest="schema_path",
        help="Write the generated JSON Schema to this path.",
    )

    arguments = parser.parse_args()

    if arguments.schema_path:
        output_path = write_experiment_result_json_schema(
            arguments.schema_path
        )
        print(f"Wrote experiment-result schema: {output_path}")

    if arguments.result_path:
        result = load_experiment_result(arguments.result_path)
        print(f"Validated experiment result: {result.run_id}")

    if not arguments.result_path and not arguments.schema_path:
        parser.error(
            "Provide a result path, --write-schema, or both."
        )


if __name__ == "__main__":
    main()
