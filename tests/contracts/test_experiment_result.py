"""Tests for the minimal experiment-result contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rocm_platform_engineering.contracts.experiment_result import (
    ExperimentResult,
    load_experiment_result,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_RESULT_PATH = (
    REPOSITORY_ROOT
    / "benchmark-results"
    / "summaries"
    / "examples"
    / "run-manifest.example.json"
)


def valid_result_data() -> dict:
    """Return a minimal valid experiment result."""
    return {
        "schema_version": "1.0",
        "run_id": "run-20260716T120000Z-a1b2c3d4",
        "experiment_id": "exp-p02-003-pinned-memory-transfer",
        "hardware_profile_id": "hw-example-mi300x-8x-spx-nps1",
        "software_stack_id": "sw-example-vllm-aiter-bf16-r01",
        "workload_profile_id": None,
        "git_revision": "a1b2c3d4",
        "started_at_utc": "2026-07-16T12:00:00Z",
        "finished_at_utc": "2026-07-16T12:05:00Z",
        "status": "pass",
        "exit_code": 0,
        "summary_metrics": [
            {
                "name": "transfer_latency_milliseconds",
                "value": 3.8,
                "unit": "milliseconds",
            }
        ],
        "evidence_paths": [
            "stdout.log",
            "stderr.log",
        ],
        "notes": [],
    }


def test_example_result_loads() -> None:
    result = load_experiment_result(EXAMPLE_RESULT_PATH)

    assert result.status == "pass"
    assert result.exit_code == 0
    assert len(result.summary_metrics) == 2


def test_pass_result_requires_zero_exit_code() -> None:
    result_data = valid_result_data()
    result_data["exit_code"] = 1

    with pytest.raises(
        ValidationError,
        match="passing run must have exit_code 0",
    ):
        ExperimentResult.model_validate(result_data)


def test_finish_time_cannot_precede_start_time() -> None:
    result_data = valid_result_data()
    result_data["finished_at_utc"] = "2026-07-16T11:59:00Z"

    with pytest.raises(
        ValidationError,
        match="finished_at_utc",
    ):
        ExperimentResult.model_validate(result_data)
