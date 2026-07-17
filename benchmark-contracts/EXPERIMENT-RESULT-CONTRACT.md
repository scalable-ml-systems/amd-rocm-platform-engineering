# Experiment Result Contract

An experiment result records what happened during one execution of an experiment contract.

It references the hardware, software, and workload profiles used for the run.

## Required fields

- run identifier
- experiment identifier
- hardware profile identifier
- software-stack identifier
- optional workload profile identifier
- Git revision
- UTC start time
- UTC finish time
- final run status
- process exit code
- summary metrics
- raw evidence paths

## Run statuses

```text
pass:
    The run met its acceptance criteria.

warn:
    The run completed but requires review.

fail:
    The run failed an acceptance criterion.

stopped:
    The run was halted after reaching a stop condition.
Result format

Generated result manifests use JSON.

Human-authored experiment and profile contracts use YAML.

Result directory
benchmark-results/raw/
└── <experiment-id>/
    └── <run-id>/
        ├── run-manifest.json
        ├── stdout.log
        ├── stderr.log
        └── additional evidence
Summary metrics

Only important final measurements belong in summary_metrics.

Examples:

transfer_bandwidth_gib_per_second
ttft_p95_milliseconds
throughput_tokens_per_second
rccl_bus_bandwidth_gib_per_second

Large metric series belong in separate evidence files.

Evidence paths

Evidence paths should be relative to the repository root or run directory.

Examples:

stdout.log
stderr.log
transfer-metrics.json
profiler-traces/rocprofv3-trace.csv

