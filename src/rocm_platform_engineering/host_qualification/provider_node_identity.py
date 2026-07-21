"""Capture the provider-visible identity of an AMD GPU node."""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DeploymentType(StrEnum):
    """How the node itself is provisioned."""

    BARE_METAL = "bare_metal"
    VIRTUAL_MACHINE = "virtual_machine"


class GpuAccessMode(StrEnum):
    """How the operating system receives access to the GPU."""

    DIRECT = "direct"
    PCI_PASSTHROUGH = "pci_passthrough"
    SR_IOV = "sriov"
    UNKNOWN = "unknown"


class AdministrativeAccess(StrEnum):
    """Highest access level supplied by the provider."""

    ROOT = "root"
    SUDO = "sudo"
    USER = "user"


class DriverManagement(StrEnum):
    """Who controls the host AMD GPU driver."""

    PROVIDER_MANAGED = "provider_managed"
    USER_MANAGED = "user_managed"
    SHARED_RESPONSIBILITY = "shared_responsibility"
    UNKNOWN = "unknown"


class ProviderNodeIdentity(BaseModel):
    """Provider-side claim describing one provisioned node."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    captured_at_utc: datetime

    provider: str = Field(min_length=2)
    provider_node_id: str = Field(min_length=1)
    provider_node_type: str = Field(min_length=1)
    provider_image_id: str = Field(min_length=1)

    hostname: str = Field(min_length=1)

    deployment_type: DeploymentType
    gpu_access_mode: GpuAccessMode

    advertised_gpu_model: str = Field(min_length=2)
    advertised_physical_gpu_count: int = Field(ge=1)
    advertised_hbm_gib_per_gpu: float = Field(gt=0)

    allocated_cpu_count: int = Field(ge=1)
    advertised_system_memory_gib: float = Field(gt=0)
    advertised_storage_gib: float = Field(gt=0)
    advertised_network: str = Field(min_length=1)

    administrative_access: AdministrativeAccess
    driver_management: DriverManagement

    claim_source: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


def build_provider_node_identity(
    *,
    provider: str,
    provider_node_id: str,
    provider_node_type: str,
    provider_image_id: str,
    deployment_type: DeploymentType,
    gpu_access_mode: GpuAccessMode,
    advertised_gpu_model: str,
    advertised_physical_gpu_count: int,
    advertised_hbm_gib_per_gpu: float,
    allocated_cpu_count: int,
    advertised_system_memory_gib: float,
    advertised_storage_gib: float,
    advertised_network: str,
    administrative_access: AdministrativeAccess,
    driver_management: DriverManagement,
    claim_source: str,
    notes: list[str] | None = None,
    hostname: str | None = None,
) -> ProviderNodeIdentity:
    """Build a provider-node record with UTC capture time."""
    return ProviderNodeIdentity(
        captured_at_utc=datetime.now(timezone.utc),
        provider=provider,
        provider_node_id=provider_node_id,
        provider_node_type=provider_node_type,
        provider_image_id=provider_image_id,
        hostname=hostname or socket.gethostname(),
        deployment_type=deployment_type,
        gpu_access_mode=gpu_access_mode,
        advertised_gpu_model=advertised_gpu_model,
        advertised_physical_gpu_count=advertised_physical_gpu_count,
        advertised_hbm_gib_per_gpu=advertised_hbm_gib_per_gpu,
        allocated_cpu_count=allocated_cpu_count,
        advertised_system_memory_gib=advertised_system_memory_gib,
        advertised_storage_gib=advertised_storage_gib,
        advertised_network=advertised_network,
        administrative_access=administrative_access,
        driver_management=driver_management,
        claim_source=claim_source,
        notes=notes or [],
    )


def write_provider_node_identity(
    identity: ProviderNodeIdentity,
    output_path: Path,
) -> Path:
    """Write the provider-node record as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        identity.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_provider_node_identity(
    input_path: Path,
) -> ProviderNodeIdentity:
    """Load and validate an existing provider-node record."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Provider-node record does not exist: {input_path}"
        )

    try:
        raw_data = json.loads(
            input_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Provider-node record contains invalid JSON: "
            f"{input_path}"
        ) from error

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Provider-node record root must be a JSON object"
        )

    return ProviderNodeIdentity.model_validate(raw_data)


def main() -> None:
    """Capture a provider-visible node identity."""
    parser = argparse.ArgumentParser(
        description=(
            "Capture the provider-visible identity of an "
            "AMD GPU node."
        ),
    )

    parser.add_argument("--provider", required=True)
    parser.add_argument("--provider-node-id", required=True)
    parser.add_argument("--provider-node-type", required=True)
    parser.add_argument("--provider-image-id", required=True)

    parser.add_argument(
        "--deployment-type",
        required=True,
        choices=[dt.value for dt in DeploymentType],
    )

    parser.add_argument(
        "--gpu-access-mode",
        required=True,
        choices=[gm.value for gm in GpuAccessMode],
    )

    parser.add_argument(
        "--advertised-gpu-model",
        required=True,
    )
    parser.add_argument(
        "--advertised-physical-gpu-count",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--advertised-hbm-gib-per-gpu",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--allocated-cpu-count",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--advertised-system-memory-gib",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--advertised-storage-gib",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--advertised-network",
        required=True,
    )

    parser.add_argument(
        "--administrative-access",
        required=True,
        choices=[aa.value for aa in AdministrativeAccess],
    )
    parser.add_argument(
        "--driver-management",
        required=True,
        choices=[dm.value for dm in DriverManagement],
    )

    parser.add_argument(
        "--claim-source",
        required=True,
        help=(
            "Where the provider claim came from, such as "
            "'provider console' or 'provider API'."
        ),
    )

    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional note. May be supplied multiple times.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    arguments = parser.parse_args()

    identity = build_provider_node_identity(
        provider=arguments.provider,
        provider_node_id=arguments.provider_node_id,
        provider_node_type=arguments.provider_node_type,
        provider_image_id=arguments.provider_image_id,
        deployment_type=DeploymentType(arguments.deployment_type),
        gpu_access_mode=GpuAccessMode(arguments.gpu_access_mode),
        advertised_gpu_model=arguments.advertised_gpu_model,
        advertised_physical_gpu_count=arguments.advertised_physical_gpu_count,
        advertised_hbm_gib_per_gpu=arguments.advertised_hbm_gib_per_gpu,
        allocated_cpu_count=arguments.allocated_cpu_count,
        advertised_system_memory_gib=arguments.advertised_system_memory_gib,
        advertised_storage_gib=arguments.advertised_storage_gib,
        advertised_network=arguments.advertised_network,
        administrative_access=AdministrativeAccess(
            arguments.administrative_access
        ),
        driver_management=DriverManagement(
            arguments.driver_management
        ),
        claim_source=arguments.claim_source,
        notes=arguments.note,
    )

    output_path = write_provider_node_identity(
        identity=identity,
        output_path=arguments.output,
    )

    print(f"Wrote provider-node identity: {output_path}")


if __name__ == "__main__":
    main()
