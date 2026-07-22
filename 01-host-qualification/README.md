# Phase 01 — AMD Host Qualification and ROCm Baseline

Phase 1 determines whether an AMD Instinct node provides a supported and internally consistent execution environment across:

- Linux and kernel
- amdgpu and firmware
- KFD
- ROCm user space
- partition state
- CPU and GPU topology
- ROCm Validation Suite
- container GPU access
- PyTorch ROCm

## Qualification experiment

`exp-p01-001-amd-host-qualification`

## Support baseline

`support-mi300x-rocm-host-r01`

## Phase boundary

This phase qualifies the node.

It does not perform HIP optimization, RCCL benchmarking,
inference serving, or Kubernetes deployment.

## Step 1.3 — Linux, Kernel, and Virtualization Baseline

The host operating-system and kernel environment is captured
before driver-health evaluation.

Evidence:

- `evidence/linux-baseline.json`
- `evidence/kernel-command-line.txt`
- `evidence/loaded-modules.txt`
- `evidence/virtualization-baseline.txt`

This step records virtualization and IOMMU signals but does not
independently declare the node supported or unsupported.
