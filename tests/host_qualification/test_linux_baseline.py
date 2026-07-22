"""Tests for Linux and virtualization baseline parsing."""

from rocm_platform_engineering.host_qualification.linux_baseline import (
    extract_iommu_parameters,
    filter_relevant_module_lines,
    parse_os_release,
)


def test_parse_os_release() -> None:
    raw_text = """
    NAME="Ubuntu"
    VERSION_ID="24.04"
    ID=ubuntu
    PRETTY_NAME="Ubuntu 24.04.4 LTS"
    """
    parsed = parse_os_release(raw_text)
    assert parsed["ID"] == "ubuntu"
    assert parsed["VERSION_ID"] == "24.04"
    assert parsed["PRETTY_NAME"] == "Ubuntu 24.04.4 LTS"


def test_extract_iommu_parameters() -> None:
    command_line = (
        "BOOT_IMAGE=/vmlinuz "
        "root=/dev/sda1 "
        "amd_iommu=on "
        "iommu=pt "
        "quiet"
    )
    parameters = extract_iommu_parameters(command_line)
    assert parameters == [
        "amd_iommu=on",
        "iommu=pt",
    ]


def test_filter_relevant_module_lines() -> None:
    raw_modules = """
    amdgpu 12345 0 - Live 0x0
    kvm_amd 9876 0 - Live 0x0
    vfio_pci 4567 0 - Live 0x0
    snd_hda_intel 3210 0 - Live 0x0
    """
    relevant_lines = filter_relevant_module_lines(raw_modules)
    assert len(relevant_lines) == 3
    assert relevant_lines[0].startswith("amdgpu ")
    assert relevant_lines[1].startswith("kvm_amd ")
    assert relevant_lines[2].startswith("vfio_pci ")
