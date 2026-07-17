"""Tests for the minimal hardware profile contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rocm_platform_engineering.contracts.hardware_profile import (
    HardwareProfile,
    load_hardware_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "hardware-profiles"
    / "hw-example-mi300x-8x-spx-nps1.yaml"
)


def valid_profile_data() -> dict:
    """Return a minimal valid hardware profile."""
    return {
        "schema_version": "1.0",
        "hardware_profile_id": "hw-example-mi300x-8x-spx-nps1",
        "provider": "example-provider",
        "node_type": "example-node-type",
        "gpu_model": "AMD Instinct MI300X",
        "gpu_architecture": "gfx942",
        "physical_gpu_count": 8,
        "logical_gpu_count": 8,
        "hbm_gib_per_gpu": 192,
        "compute_partition_mode": "spx",
        "memory_partition_mode": "nps1",
        "gpu_interconnect": "xgmi",
        "cpu_model": "AMD EPYC",
        "system_memory_gib": 1536,
        "numa_node_count": 8,
    }


def test_example_hardware_profile_loads() -> None:
    profile = load_hardware_profile(EXAMPLE_PROFILE_PATH)

    assert profile.gpu_model == "AMD Instinct MI300X"
    assert profile.physical_gpu_count == 8
    assert profile.logical_gpu_count == 8
    assert profile.compute_partition_mode == "spx"
    assert profile.memory_partition_mode == "nps1"


def test_unknown_fields_are_rejected() -> None:
    profile_data = valid_profile_data()
    profile_data["hourly_cost_usd"] = 1.25

    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        HardwareProfile.model_validate(profile_data)


def test_gpu_counts_must_be_positive() -> None:
    profile_data = valid_profile_data()
    profile_data["physical_gpu_count"] = 0

    with pytest.raises(ValidationError):
        HardwareProfile.model_validate(profile_data)
