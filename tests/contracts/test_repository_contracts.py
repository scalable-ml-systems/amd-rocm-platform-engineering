"""Minimal repository-level validation for Phase 0 contracts."""

from pathlib import Path

from rocm_platform_engineering.contracts.experiment_contract import (
    load_experiment_contract,
)
from rocm_platform_engineering.contracts.experiment_result import (
    load_experiment_result,
)
from rocm_platform_engineering.contracts.hardware_profile import (
    load_hardware_profile,
)
from rocm_platform_engineering.contracts.software_stack import (
    load_software_stack,
)
from rocm_platform_engineering.contracts.workload_profile import (
    load_workload_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "examples"
    / "exp-p02-003-pinned-memory-transfer"
    / "experiment.yaml"
)

HARDWARE_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "hardware-profiles"
    / "hw-example-mi300x-8x-spx-nps1.yaml"
)

HIP_SOFTWARE_STACK_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "software-stacks"
    / "sw-hip-native-bf16-r01.yaml"
)

VLLM_SOFTWARE_STACK_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "software-stacks"
    / "sw-example-vllm-aiter-bf16-r01.yaml"
)

WORKLOAD_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "workloads"
    / "wl-example-dense-long-short-steady-c08.yaml"
)

EXPERIMENT_RESULT_PATH = (
    REPOSITORY_ROOT
    / "benchmark-results"
    / "summaries"
    / "examples"
    / "run-manifest.example.json"
)

REQUIRED_PHASE_ZERO_FILES = [
    REPOSITORY_ROOT / "NAMING-CONVENTIONS.md",
    REPOSITORY_ROOT / "EVIDENCE-CONTRACT.md",
    REPOSITORY_ROOT / "benchmark-contracts" / "EXPERIMENT-CONTRACT.md",
    REPOSITORY_ROOT / "benchmark-contracts" / "HARDWARE-PROFILE-CONTRACT.md",
    REPOSITORY_ROOT / "benchmark-contracts" / "SOFTWARE-STACK-CONTRACT.md",
    REPOSITORY_ROOT / "benchmark-contracts" / "WORKLOAD-PROFILE-CONTRACT.md",
    REPOSITORY_ROOT / "benchmark-contracts" / "EXPERIMENT-RESULT-CONTRACT.md",
    REPOSITORY_ROOT / "failure-atlas" / "TEMPLATE.md",
    REPOSITORY_ROOT / "operator-playbooks" / "RULE-TEMPLATE.md",
]


def profile_ids_by_type() -> dict[str, set[str]]:
    """Load the example profiles and return their identifiers."""
    hardware_profile = load_hardware_profile(HARDWARE_PROFILE_PATH)
    hip_software_stack = load_software_stack(
        HIP_SOFTWARE_STACK_PATH
    )
    vllm_software_stack = load_software_stack(
         VLLM_SOFTWARE_STACK_PATH
    )
    workload_profile = load_workload_profile(WORKLOAD_PROFILE_PATH)

    return {
        "hardware": {hardware_profile.hardware_profile_id},
        "software": { hip_software_stack.software_stack_id, 
                      vllm_software_stack.software_stack_id, 
        },
        "workload": {workload_profile.workload_profile_id},
    }


def test_required_phase_zero_files_exist() -> None:
    """Ensure important Phase 0 files were not deleted or renamed."""
    missing_files = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in REQUIRED_PHASE_ZERO_FILES
        if not path.is_file()
    ]

    assert not missing_files, (
        "Missing required Phase 0 files: "
        + ", ".join(missing_files)
    )


def test_example_contracts_validate() -> None:
    """Load every example through its existing contract model."""
    load_experiment_contract(EXPERIMENT_PATH)
    load_hardware_profile(HARDWARE_PROFILE_PATH)
    load_software_stack(HIP_SOFTWARE_STACK_PATH) 
    load_software_stack(VLLM_SOFTWARE_STACK_PATH)
    load_workload_profile(WORKLOAD_PROFILE_PATH)
    load_experiment_result(EXPERIMENT_RESULT_PATH)


def test_example_profile_references_resolve() -> None:
    """Ensure example contracts reference existing profiles."""
    available_profile_ids = profile_ids_by_type()

    experiment = load_experiment_contract(EXPERIMENT_PATH)
    result = load_experiment_result(EXPERIMENT_RESULT_PATH)

    assert (
        experiment.hardware_profile_id
        in available_profile_ids["hardware"]
    )
    assert (
        experiment.software_stack_id
        in available_profile_ids["software"]
    )
    if experiment.workload_profile_id is not None:
        assert (
            experiment.workload_profile_id
            in available_profile_ids["workload"]
        )

    assert (
        result.hardware_profile_id
        in available_profile_ids["hardware"]
    )
    assert (
        result.software_stack_id
        in available_profile_ids["software"]
    )
    if result.workload_profile_id is not None:
        assert (
            result.workload_profile_id
            in available_profile_ids["workload"]
        )
