"""Tests for provider-visible node identity capture."""

from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from rocm_platform_engineering.host_qualification.provider_node_identity import (
    AdministrativeAccess,
    DeploymentType,
    DriverManagement,
    GpuAccessMode,
    build_provider_node_identity,
    load_provider_node_identity,
    write_provider_node_identity,
)


def valid_identity() -> object:
    """Return a valid provider-node identity."""
    return build_provider_node_identity(
        provider="example-provider",
        provider_node_id="example-node-001",
        provider_node_type="example-mi300x-node",
        provider_image_id="example-image",
        deployment_type=DeploymentType.BARE_METAL,
        gpu_access_mode=GpuAccessMode.DIRECT,
        advertised_gpu_model="AMD Instinct MI300X",
        advertised_physical_gpu_count=1,
        advertised_hbm_gib_per_gpu=1.0,
        allocated_cpu_count=1,
        advertised_system_memory_gib=1.0,
        advertised_storage_gib=1.0,
        advertised_network="not_disclosed",
        administrative_access=AdministrativeAccess.ROOT,
        driver_management=DriverManagement.PROVIDER_MANAGED,
        claim_source="provider console",
        notes=["Structural test values only."],
        hostname="example-host",
    )


def test_provider_node_identity_round_trip(tmp_path: Path) -> None:
    """A valid identity should write and reload."""
    output_path = tmp_path / "provider-node.json"

    write_provider_node_identity(
        identity=valid_identity(),
        output_path=output_path,
    )

    loaded_identity = load_provider_node_identity(output_path)

    assert loaded_identity.provider == "example-provider"
    assert loaded_identity.advertised_gpu_model == "AMD Instinct MI300X"
    assert loaded_identity.advertised_physical_gpu_count == 1


def test_gpu_count_must_be_positive() -> None:
    """A provider cannot advertise zero physical GPUs."""
    with pytest.raises(ValidationError):
        build_provider_node_identity(
            provider="example-provider",
            provider_node_id="example-node-001",
            provider_node_type="example-mi300x-node",
            provider_image_id="example-image",
            deployment_type=DeploymentType.BARE_METAL,
            gpu_access_mode=GpuAccessMode.DIRECT,
            advertised_gpu_model="AMD Instinct MI300X",
            advertised_physical_gpu_count=0,
            advertised_hbm_gib_per_gpu=1.0,
            allocated_cpu_count=1,
            advertised_system_memory_gib=1.0,
            advertised_storage_gib=1.0,
            advertised_network="not_disclosed",
            administrative_access=AdministrativeAccess.ROOT,
            driver_management=DriverManagement.PROVIDER_MANAGED,
            claim_source="provider console",
            hostname="example-host",
        )


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    """Unplanned or sensitive fields must not enter the record."""
    output_path = tmp_path / "provider-node.json"

    record_data = valid_identity().model_dump(mode="json")
    record_data["public_ip_address"] = "192.0.2.1"

    output_path.write_text(
        json.dumps(record_data),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs"):
        load_provider_node_identity(output_path)
