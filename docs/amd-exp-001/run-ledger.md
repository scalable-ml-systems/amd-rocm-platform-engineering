
# AMD-EXP-001 — Run Ledger

## Frozen environment

- Hardware: AMD Instinct MI300X VF
- GPU count: 1
- Architecture: gfx942
- Compute units: 304
- Partition: SPX / NPS1
- HBM: ~192 GiB
- OS: Ubuntu 24.04.4
- Kernel: 6.8.0-138-generic
- ROCm: 10.0.0
- Profiler package: amdrocm-profiler10.0 (10.0.0-4)
- Profiler path: /opt/rocm/core-10.0/bin/rocprof-compute

---

## SETUP-001 — Profiler installation

Date: 2026-09-22
Purpose: Prepare the TP1 kernel investigation.

Preflight:
- Package candidate: 10.0.0-4
- Upgrades: 0
- New packages: 1
- Removals: 0

Result: PASS

ROCm Compute Profiler installed and located at:
  /opt/rocm/core-10.0/bin/rocprof-compute

Next: Validate profiler execution and establish the
empirical MI300X roofline.

---
### AMD001-S1-ROOFLINE-R1

Result: PASS

ROCm Compute Profiler: 3.8.0
Environment override: ROCM_VER=10.0.0

Measured:
- MFMA BF16: 1.029 PFLOP/s
- MFMA FP8: 1.867 PFLOP/s
- LDS bandwidth: 76.17 TB/s
- HBM bandwidth: pending CSV inspection

Issue: Initial ROCm version detection failed.
Resolved with ROCM_VER=10.0.0.

Evidence:
roofline/workloads/MI300X-SPX-NPS1-BASELINE/
MI300X_A1/roofline.csv
