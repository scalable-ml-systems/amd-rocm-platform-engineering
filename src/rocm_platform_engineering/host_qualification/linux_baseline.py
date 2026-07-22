"""Capture the Linux, kernel, and virtualization baseline."""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"
RELEVANT_MODULE_PREFIXES = (
    "amdgpu",
    "amd_iommu",
    "kvm",
    "vfio",
    "mdev",
)
IOMMU_PARAMETER_PREFIXES = (
    "iommu=",
    "amd_iommu=",
    "intel_iommu=",
)


def read_optional_text(path: Path) -> str | None:
    """Read a text file when it exists and is accessible."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def parse_os_release(raw_text: str) -> dict[str, str]:
    """Parse the standard /etc/os-release key-value format."""
    values: dict[str, str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def run_optional_command(
    command: Sequence[str],
    *,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Run a command when its executable is available."""
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "available": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "command not available",
        }
    try:
        completed_process = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "command": list(command),
            "available": True,
            "exit_code": None,
            "stdout": "",
            "stderr": f"command timed out after {timeout_seconds} seconds",
        }
    return {
        "command": list(command),
        "available": True,
        "exit_code": completed_process.returncode,
        "stdout": completed_process.stdout.strip(),
        "stderr": completed_process.stderr.strip(),
    }


def extract_iommu_parameters(
    kernel_command_line: str,
) -> list[str]:
    """Return IOMMU-related kernel command-line parameters."""
    try:
        parameters = shlex.split(kernel_command_line)
    except ValueError:
        parameters = kernel_command_line.split()
    return [
        parameter for parameter in parameters if parameter.startswith(IOMMU_PARAMETER_PREFIXES)
    ]


def filter_relevant_module_lines(
    raw_modules: str,
) -> list[str]:
    """Return AMD GPU and virtualization-related module lines."""
    relevant_lines: list[str] = []
    for raw_line in raw_modules.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        module_name = line.split(maxsplit=1)[0]
        if module_name.startswith(RELEVANT_MODULE_PREFIXES):
            relevant_lines.append(line)
    return relevant_lines


def count_loaded_modules(raw_modules: str) -> int:
    """Count non-empty entries in /proc/modules."""
    return sum(1 for line in raw_modules.splitlines() if line.strip())


def count_iommu_groups(
    iommu_group_directory: Path = Path("/sys/kernel/iommu_groups"),
) -> int:
    """Count visible kernel IOMMU groups."""
    try:
        return sum(1 for path in iommu_group_directory.iterdir() if path.is_dir())
    except (FileNotFoundError, PermissionError, OSError):
        return 0


def parse_uptime_seconds(raw_uptime: str | None) -> float | None:
    """Parse the first value from /proc/uptime."""
    if not raw_uptime:
        return None
    try:
        return float(raw_uptime.split()[0])
    except (ValueError, IndexError):
        return None


def extract_virtualization_lscpu_lines(
    raw_lscpu: str,
) -> list[str]:
    """Keep only virtualization and hypervisor fields."""
    prefixes = (
        "Virtualization:",
        "Virtualization type:",
        "Hypervisor vendor:",
    )
    return [line.strip() for line in raw_lscpu.splitlines() if line.strip().startswith(prefixes)]


def classify_virtualization(
    *,
    generic_detection: dict[str, Any],
    vm_detection: dict[str, Any],
    container_detection: dict[str, Any],
    lscpu_virtualization_lines: list[str],
) -> str:
    """Classify the observable execution environment."""
    if container_detection["available"] and container_detection["exit_code"] == 0:
        return "container"
    if vm_detection["available"] and vm_detection["exit_code"] == 0:
        return "virtual_machine"
    if any(line.startswith("Hypervisor vendor:") for line in lscpu_virtualization_lines):
        return "virtual_machine"
    if generic_detection["available"] and generic_detection["exit_code"] == 1:
        return "not_detected"
    return "unknown"


def build_virtualization_summary(
    *,
    environment_type: str,
    generic_detection: dict[str, Any],
    vm_detection: dict[str, Any],
    container_detection: dict[str, Any],
    lscpu_virtualization_lines: list[str],
    iommu_parameters: list[str],
    iommu_group_count: int,
) -> str:
    """Create a concise human-readable virtualization record."""
    lines = [
        f"environment_type={environment_type}",
        f"systemd_detect_virt={generic_detection.get('stdout') or 'none'}",
        f"systemd_detect_vm={vm_detection.get('stdout') or 'none'}",
        f"systemd_detect_container={container_detection.get('stdout') or 'none'}",
        f"iommu_kernel_parameters={' '.join(iommu_parameters) or 'none'}",
        f"iommu_group_count={iommu_group_count}",
    ]
    if lscpu_virtualization_lines:
        lines.append("")
        lines.append("lscpu virtualization fields:")
        lines.extend(lscpu_virtualization_lines)
    return "\n".join(lines) + "\n"


def collect_linux_baseline() -> tuple[dict[str, Any], dict[str, str]]:
    """Collect the Linux baseline and raw evidence files."""
    captured_at_utc = datetime.now(timezone.utc)
    notes: list[str] = []

    os_release_text = read_optional_text(Path("/etc/os-release"))
    kernel_command_line = read_optional_text(Path("/proc/cmdline")) or ""
    raw_modules = read_optional_text(Path("/proc/modules")) or ""
    raw_uptime = read_optional_text(Path("/proc/uptime"))
    boot_id = read_optional_text(Path("/proc/sys/kernel/random/boot_id"))

    if os_release_text is None:
        notes.append("/etc/os-release was not readable.")
    if not kernel_command_line:
        notes.append("/proc/cmdline was not readable.")
    if not raw_modules:
        notes.append("/proc/modules was not readable.")

    os_release = parse_os_release(os_release_text or "")
    uname = platform.uname()
    kernel_release = uname.release
    libc_name, libc_version = platform.libc_ver()
    getconf_glibc = run_optional_command(["getconf", "GNU_LIBC_VERSION"])
    uptime_seconds = parse_uptime_seconds(raw_uptime)
    booted_at_utc: str | None = None
    if uptime_seconds is not None:
        booted_at_utc = (captured_at_utc - timedelta(seconds=uptime_seconds)).isoformat()

    kernel_build_path = Path("/lib/modules") / kernel_release / "build"
    relevant_module_lines = filter_relevant_module_lines(raw_modules)
    relevant_module_names = [line.split(maxsplit=1)[0] for line in relevant_module_lines]

    generic_detection = run_optional_command(["systemd-detect-virt"])
    vm_detection = run_optional_command(["systemd-detect-virt", "--vm"])
    container_detection = run_optional_command(["systemd-detect-virt", "--container"])
    lscpu_result = run_optional_command(["lscpu"])
    lscpu_virtualization_lines = extract_virtualization_lscpu_lines(lscpu_result["stdout"])

    environment_type = classify_virtualization(
        generic_detection=generic_detection,
        vm_detection=vm_detection,
        container_detection=container_detection,
        lscpu_virtualization_lines=lscpu_virtualization_lines,
    )
    iommu_parameters = extract_iommu_parameters(kernel_command_line)
    iommu_group_count = count_iommu_groups()

    baseline = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": captured_at_utc.isoformat(),
        "hostname": socket.gethostname(),
        "operating_system": {
            "id": os_release.get("ID"),
            "name": os_release.get("NAME"),
            "pretty_name": os_release.get("PRETTY_NAME"),
            "version_id": os_release.get("VERSION_ID"),
            "version": os_release.get("VERSION"),
            "version_codename": os_release.get("VERSION_CODENAME"),
        },
        "kernel": {
            "release": kernel_release,
            "version": uname.version,
            "machine": uname.machine,
            "command_line": kernel_command_line,
            "build_path": str(kernel_build_path),
            "build_path_exists": kernel_build_path.exists(),
        },
        "glibc": {
            "platform_libc_name": libc_name or None,
            "platform_libc_version": libc_version or None,
            "getconf_result": getconf_glibc,
        },
        "boot": {
            "boot_id": boot_id,
            "uptime_seconds": uptime_seconds,
            "booted_at_utc": booted_at_utc,
        },
        "loaded_modules": {
            "total_count": count_loaded_modules(raw_modules),
            "relevant_module_names": relevant_module_names,
        },
        "virtualization": {
            "environment_type": environment_type,
            "systemd_detect_virt": generic_detection,
            "systemd_detect_vm": vm_detection,
            "systemd_detect_container": container_detection,
            "lscpu_virtualization_fields": lscpu_virtualization_lines,
        },
        "iommu": {
            "kernel_parameters": iommu_parameters,
            "group_count": iommu_group_count,
            "groups_observed": iommu_group_count > 0,
        },
        "notes": notes,
    }

    raw_evidence = {
        "kernel-command-line.txt": kernel_command_line + "\n",
        "loaded-modules.txt": "\n".join(relevant_module_lines) + ("\n" if relevant_module_lines else ""),
        "virtualization-baseline.txt": build_virtualization_summary(
            environment_type=environment_type,
            generic_detection=generic_detection,
            vm_detection=vm_detection,
            container_detection=container_detection,
            lscpu_virtualization_lines=lscpu_virtualization_lines,
            iommu_parameters=iommu_parameters,
            iommu_group_count=iommu_group_count,
        ),
    }
    return baseline, raw_evidence


def write_linux_baseline(
    *,
    output_directory: Path,
    overwrite: bool,
) -> list[Path]:
    """Write the structured and raw baseline evidence."""
    baseline, raw_evidence = collect_linux_baseline()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_directory / "linux-baseline.json",
        *[output_directory / filename for filename in raw_evidence],
    ]
    if not overwrite:
        existing_paths = [path for path in output_paths if path.exists()]
        if existing_paths:
            existing_list = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError("Refusing to overwrite existing evidence: " + existing_list)
    json_path = output_directory / "linux-baseline.json"
    json_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    for filename, content in raw_evidence.items():
        (output_directory / filename).write_text(content, encoding="utf-8")
    return output_paths


def main() -> None:
    """Capture the host Linux baseline."""
    parser = argparse.ArgumentParser(
        description=(
            "Capture Linux, kernel, boot, module, virtualization, and IOMMU baseline evidence."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("01-host-qualification/evidence"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Step 1.3 evidence files.",
    )
    arguments = parser.parse_args()
    try:
        output_paths = write_linux_baseline(
            output_directory=arguments.output_directory,
            overwrite=arguments.overwrite,
        )
    except FileExistsError as error:
        parser.error(str(error))
    print("Linux baseline evidence:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
