# Evidence Contract

Every experiment run must produce enough evidence to verify what ran, what happened, and why the result was accepted.

## Minimum Required Evidence

Each run directory should contain:

```text
run-manifest.json
stdout.log
stderr.log
environment.json

Additional files depend on the experiment.

Examples:

HIP experiment:
  transfer-metrics.json
  rocprofv3-summary.csv

RCCL experiment:
  rccl-results.csv
  rccl-debug.log
  topology.txt

Inference experiment:
  latency-summary.json
  throughput-summary.json
  server.log
  output-correctness.json

Kubernetes experiment:
  pod-description.txt
  events.txt
  resource-state.yaml

Two-node experiment:
  rdma-bandwidth.txt
  nic-topology.txt
  network-counters.txt
Run Directory Structure
benchmark-results/raw/
└── <experiment-id>/
    └── <run-id>/
        ├── run-manifest.json
        ├── environment.json
        ├── stdout.log
        ├── stderr.log
        └── experiment-specific evidence
Required Evidence Principles
1. Capture commands

The evidence must show which command or script was executed.

2. Capture versions

The run must reference the hardware and software profiles used.

3. Preserve raw measurements

Summary metrics must remain traceable to raw output.

4. Capture failures completely

Do not remove warnings, failed commands, or error output from the evidence bundle.

5. Separate observations from conclusions

Raw output belongs in evidence files.

Interpretation belongs in experiment summaries and failure-atlas entries.

6. Use UTC timestamps

All recorded timestamps must use UTC.

7. Do not commit secrets

Evidence must not contain:

API keys
access tokens
cloud credentials
private registry credentials
private SSH keys
8. Avoid committing unnecessarily large files

Large profiler traces, packet captures, and model artifacts may be stored externally.

The run manifest should record where they are stored.

Minimum Evidence by Experiment Type
Validation
run-manifest.json
environment.json
stdout.log
stderr.log
validation-summary.json
Benchmark
run-manifest.json
environment.json
stdout.log
stderr.log
raw metrics
summary metrics
Profiling
run-manifest.json
environment.json
stdout.log
stderr.log
profiler output
profiler interpretation
Failure Injection
run-manifest.json
environment.json
baseline evidence
injected condition
failure evidence
recovery evidence
Evidence Acceptance Rule

An experiment result should not be accepted when:

the command is unknown

the software stack is unknown

the hardware profile is unknown

summary metrics cannot be traced to raw output

correctness evidence is missing

the relevant error logs were discarded

