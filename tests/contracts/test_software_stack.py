"""Tests for the minimal software-stack contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rocm_platform_engineering.contracts.software_stack import (
    SoftwareStack,
    load_software_stack,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "benchmark-contracts"
    / "software-stacks"
    / "sw-example-vllm-aiter-bf16-r01.yaml"
)


def valid_stack_data() -> dict:
    """Return a minimal valid software stack."""
    return {
        "schema_version": "1.0",
        "software_stack_id": "sw-example-vllm-aiter-bf16-r01",
        "operating_system": "Ubuntu 24.04",
        "linux_kernel": "example-kernel-version",
        "amdgpu_driver": "example-amdgpu-version",
        "rocm_version": "example-rocm-version",
        "python_version": "example-python-version",
        "pytorch_version": "example-pytorch-version",
        "container_image": (
            "example-registry/example-vllm-rocm:example-tag"
        ),
        "container_image_digest": None,
        "execution_engine": "vllm",
        "execution_engine_version": "example-vllm-version",
        "rccl_version": "example-rccl-version",
        "triton_version": "example-triton-version",
        "aiter_version": "example-aiter-version",
        "atom_version": None,
    }


def test_example_software_stack_loads() -> None:
    stack = load_software_stack(EXAMPLE_PROFILE_PATH)

    assert stack.execution_engine == "vllm"
    assert stack.aiter_version == "example-aiter-version"
    assert stack.atom_version is None


def test_unknown_fields_are_rejected() -> None:
    stack_data = valid_stack_data()
    stack_data["all_installed_packages"] = []

    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        SoftwareStack.model_validate(stack_data)


def test_invalid_container_digest_is_rejected() -> None:
    stack_data = valid_stack_data()
    stack_data["container_image_digest"] = "latest"

    with pytest.raises(
        ValidationError,
        match="container_image_digest",
    ):
        SoftwareStack.model_validate(stack_data)
