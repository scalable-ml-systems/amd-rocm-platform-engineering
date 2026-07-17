# Failure XXX — Descriptive Failure Name

## Summary

One or two sentences describing what failed and why it matters.

## Symptom

What was observed? Include:

```text
error message
unexpected behavior
performance regression
timeout
incorrect output
resource mismatch
```

## Affected Layer

Choose the primary failure boundary:

```text
hardware
driver
kfd
rocm-runtime
container
hip
memory partitioning
xgmi
rccl
inference-engine
kernel-backend
model
kubernetes
scheduler
observability
network
rdma
```

## Environment Reference

Record identifiers that tie this failure to contracts:

```text
hardware_profile_id
software_stack_id
workload_profile_id
experiment_id
run_id
```

## Expected Behavior

What should have happened?

## Raw Evidence

List the evidence files and the most relevant observations. Example:

```text
stderr.log
rccl-debug.log
topology.txt
run-manifest.json
```

Do not paste excessively long raw logs into this document.

## Initial Hypotheses

List the plausible causes considered before the root cause was known.

## Diagnostic Probes

Document the checks used to eliminate or confirm each hypothesis. For each probe, record:

- command  
- reason for running it  
- important output  
- conclusion  

## Root Cause

State the confirmed failure cause. If the cause remains uncertain, say so clearly. Do not label a hypothesis as a confirmed root cause without evidence.

## Corrective Action

Describe exactly what changed. Examples:

```text
permission update
container device mapping
partition reset
process-placement change
RCCL configuration change
backend change
software rollback
network correction
```

## Validation

Show how the fix was proven. Include before-and-after evidence where possible.

## Operator Rule

State the reusable operational lesson. Example:

```text
Do not begin RCCL diagnosis until xGMI topology and pairwise bandwidth have passed qualification.
```

## Prevention or Regression Test

Describe the check that should prevent the same failure from returning.

## Limitations

Record anything not fully validated or any conditions under which the conclusion may not apply.
