"""Tests for the experiment contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rocm_platform_engineering.contracts.experiment_contract import (
    ExperimentContract,
    ExecutionType,
    load_experiment_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "examples"
    / "exp-p02-003-pinned-memory-transfer"
    / "experiment.yaml"
)


def valid_contract_data() -> dict:
    """Return a minimal valid benchmark experiment."""
    return {
        "schema_version": "1.0",
        "experiment_id": "exp-p02-003-pinned-memory-transfer",
        "title": "Pageable versus pinned host-memory transfer",
        "phase": 2,
        "workstream": "memory-and-streams",
        "execution_type": "benchmark",
        "status": "planned",
        "objective": "Measure host-memory transfer behavior on AMD Instinct.",
        "question": (
            "Does pinned memory improve transfer performance "
            "compared with pageable memory?"
        ),
        "hypothesis": (
            "Pinned memory will provide higher transfer bandwidth "
            "than pageable memory."
        ),
        "prerequisites": [
            "Host qualification passed.",
        ],
        "hardware_profile_id": "hw-mi300x-1x-spx-nps1-pcie",
        "software_stack_id": "sw-hip-native-bf16-r01",
        "workload_profile_id": None,
        "independent_variable": "host_memory_type",
        "controlled_variables": [
            "GPU device",
            "buffer sizes",
        ],
        "metrics": [
            {
                "name": "host_to_device_bandwidth_gib_per_second",
                "unit": "gib_per_second",
                "description": "Effective host-to-device transfer bandwidth.",
            }
        ],
        "acceptance_criteria": [
            "Every transferred buffer passes validation.",
        ],
        "stop_conditions": [
            "A transferred buffer fails validation.",
        ],
        "evidence_required": [
            "run-manifest.json",
            "transfer-metrics.json",
        ],
        "notes": [],
    }


def test_example_contract_loads() -> None:
    contract = load_experiment_contract(EXAMPLE_CONTRACT_PATH)

    assert contract.experiment_id == "exp-p02-003-pinned-memory-transfer"
    assert contract.phase == 2
    assert contract.execution_type == ExecutionType.BENCHMARK
    assert len(contract.metrics) == 3


def test_phase_must_match_experiment_identifier() -> None:
    contract_data = valid_contract_data()
    contract_data["phase"] = 3

    with pytest.raises(ValidationError, match="phase encoded in experiment_id"):
        ExperimentContract.model_validate(contract_data)


def test_unknown_fields_are_rejected() -> None:
    contract_data = valid_contract_data()
    contract_data["acceptence_criteria"] = [
        "This field name contains a typo."
    ]

    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentContract.model_validate(contract_data)


def test_benchmark_requires_hardware_profile() -> None:
    contract_data = valid_contract_data()
    contract_data["hardware_profile_id"] = None

    with pytest.raises(
        ValidationError,
        match="hardware_profile_id is required",
    ):
        ExperimentContract.model_validate(contract_data)


def test_benchmark_requires_software_stack() -> None:
    contract_data = valid_contract_data()
    contract_data["software_stack_id"] = None

    with pytest.raises(
        ValidationError,
        match="software_stack_id is required",
    ):
        ExperimentContract.model_validate(contract_data)


def test_duplicate_metric_names_are_rejected() -> None:
    contract_data = valid_contract_data()
    contract_data["metrics"].append(contract_data["metrics"][0].copy())

    with pytest.raises(
        ValidationError,
        match="Metric names must be unique",
    ):
        ExperimentContract.model_validate(contract_data)


def test_duplicate_controlled_variables_are_rejected() -> None:
    contract_data = valid_contract_data()
    contract_data["controlled_variables"] = [
        "GPU device",
        "GPU device",
    ]

    with pytest.raises(
        ValidationError,
        match="List entries must be unique",
    ):
        ExperimentContract.model_validate(contract_data)


def test_metric_names_must_use_snake_case() -> None:
    contract_data = valid_contract_data()
    contract_data["metrics"][0]["name"] = "TransferBandwidth"

    with pytest.raises(ValidationError):
        ExperimentContract.model_validate(contract_data)


def test_profile_identifiers_require_correct_prefixes() -> None:
    contract_data = valid_contract_data()
    contract_data["hardware_profile_id"] = "mi300x-node"

    with pytest.raises(
        ValidationError,
        match="hardware_profile_id must use the 'hw-' prefix",
    ):
        ExperimentContract.model_validate(contract_data)
