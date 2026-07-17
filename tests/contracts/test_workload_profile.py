"""Tests for the minimal workload profile contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rocm_platform_engineering.contracts.workload_profile import (
    WorkloadProfile,
    load_workload_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "workloads"
    / "wl-example-dense-long-short-steady-c08.yaml"
)


def valid_profile_data() -> dict:
    """Return a minimal valid inference workload."""
    return {
        "schema_version": "1.0",
        "workload_profile_id": "wl-example-dense-long-short-steady-c08",
        "workload_type": "inference",
        "model_name": "example-dense-model",
        "model_revision": "example-model-revision",
        "input_token_count": 8192,
        "output_token_count": 256,
        "request_count": 100,
        "concurrency": 8,
        "arrival_pattern": "steady",
        "random_seed": 42,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
    }


def test_example_workload_profile_loads() -> None:
    profile = load_workload_profile(EXAMPLE_PROFILE_PATH)

    assert profile.workload_type == "inference"
    assert profile.input_token_count == 8192
    assert profile.output_token_count == 256
    assert profile.concurrency == 8


def test_unknown_fields_are_rejected() -> None:
    profile_data = valid_profile_data()
    profile_data["average_latency"] = 125

    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        WorkloadProfile.model_validate(profile_data)


def test_token_counts_and_concurrency_must_be_positive() -> None:
    profile_data = valid_profile_data()
    profile_data["concurrency"] = 0

    with pytest.raises(ValidationError):
        WorkloadProfile.model_validate(profile_data)
