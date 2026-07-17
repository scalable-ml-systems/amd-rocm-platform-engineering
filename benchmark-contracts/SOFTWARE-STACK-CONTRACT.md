# Software Stack Contract

A software-stack profile records the software combination required to reproduce an AMD ROCm experiment.

It is intentionally limited to software components that can materially affect the result.

## Required fields

- software-stack identifier
- operating system
- Linux kernel
- amdgpu driver
- ROCm version
- primary execution engine

## Optional fields

Populate these only when the experiment uses them:

- Python
- PyTorch
- container image
- container image digest
- execution-engine version
- RCCL
- Triton
- AITER
- ATOM

## Not included

The profile does not contain:

- every installed operating-system package
- every installed Python package
- hardware configuration
- workload configuration
- benchmark results
- environment-variable dumps
- runtime health status

Detailed package and environment captures may be stored as raw evidence for a benchmark run.

## Identifier format

```text
sw-<engine>-<backend>-<precision>-r<revision>

Examples:

sw-hip-native-bf16-r01
sw-vllm-upstream-bf16-r01
sw-vllm-aiter-fp8-r01
sw-vllm-atom-fp8-r01
sw-sglang-aiter-fp8-r01

The revision identifies the project-owned stack definition.

Exact component versions belong inside the profile.

When to create a new profile

Create a new software-stack profile when a result-relevant component changes, including:

amdgpu driver
ROCm
PyTorch
RCCL
Triton
AITER
ATOM
vLLM
SGLang
container image digest

Do not create a new software profile for a hardware-only change.
