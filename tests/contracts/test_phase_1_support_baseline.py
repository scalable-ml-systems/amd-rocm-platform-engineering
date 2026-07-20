"""Focused validation for the Phase 1 AMD support baseline."""

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SUPPORT_BASELINE_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "support-baselines"
    / "support-mi300x-rocm-host-r01.yaml"
)


def load_support_baseline() -> dict:
    """Load the Phase 1 support baseline."""
    raw_data = yaml.safe_load(
        SUPPORT_BASELINE_PATH.read_text(encoding="utf-8")
    )

    assert isinstance(raw_data, dict)
    return raw_data


def test_support_baseline_has_required_sections() -> None:
    """Ensure the baseline preserves its essential boundaries."""
    baseline = load_support_baseline()

    assert baseline["support_baseline_id"] == (
        "support-mi300x-rocm-host-r01"
    )

    assert baseline["target_hardware"]["gpu_model"] == (
        "AMD Instinct MI300X"
    )
    assert baseline["target_hardware"]["gpu_architecture"] == (
        "gfx942"
    )

    assert "target_host_stack" in baseline
    assert "target_framework_stack" in baseline
    assert "required_host_tools" in baseline
    assert "support_authority" in baseline
    assert "support_decision" in baseline


def test_selected_baseline_has_no_unresolved_required_versions() -> None:
    """Selected baselines must identify their target software tuple."""
    baseline = load_support_baseline()

    if baseline["baseline_status"] == (
        "selected_pending_host_verification"
    ):
        required_values = {
            "operating_system": baseline[
                "target_host_stack"
            ]["operating_system"],
            "operating_system_version": baseline[
                "target_host_stack"
            ]["operating_system_version"],
            "kernel_version": baseline[
                "target_host_stack"
            ]["kernel_version"],
            "amdgpu_driver_version": baseline[
                "target_host_stack"
            ]["amdgpu_driver_version"],
            "rocm_version": baseline[
                "target_host_stack"
            ]["rocm_version"],
            "python_version": baseline[
                "target_framework_stack"
            ]["python_version"],
            "pytorch_version": baseline[
                "target_framework_stack"
            ]["pytorch_version"],
        }

        unresolved = [
            name
            for name, value in required_values.items()
            if value is None
        ]

        assert not unresolved, (
            "Selected support baseline has unresolved values: "
            + ", ".join(unresolved)
        )
