#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ROCM_MAJOR="10.0"
PROFILER_PACKAGE="amdrocm-profiler10.0"

echo "===== AMD-EXP-001 PROFILER INSTALL ====="

echo
echo "===== CURRENT ROCM ====="
amd-smi version

echo
echo "===== PACKAGE CANDIDATE ====="
apt-cache policy "${PROFILER_PACKAGE}"

candidate="$(
  apt-cache policy "${PROFILER_PACKAGE}" |
  awk '/Candidate:/ {print $2}'
)"

if [[ -z "${candidate}" || "${candidate}" == "(none)" ]]; then
    echo "ERROR: No candidate found for ${PROFILER_PACKAGE}"
    exit 1
fi

echo
echo "Candidate: ${candidate}"

if [[ "${candidate}" != ${EXPECTED_ROCM_MAJOR}* ]]; then
    echo "ERROR: Profiler candidate ${candidate} does not match ROCm ${EXPECTED_ROCM_MAJOR}"
    exit 1
fi

echo
echo "===== DRY RUN ====="
apt-get -s install "${PROFILER_PACKAGE}"

echo
echo "Dry run completed."
echo "No packages were installed."
