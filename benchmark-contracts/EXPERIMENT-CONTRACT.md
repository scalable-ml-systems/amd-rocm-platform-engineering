# Experiment Contract

An experiment contract defines what an experiment intends to test before execution begins.

It is not a benchmark-result document.

## Contract Responsibilities

Each contract declares:

- experiment identity
- objective
- technical question
- hypothesis
- prerequisites
- referenced hardware, software, and workload profiles
- independent variable
- controlled variables
- metrics
- acceptance criteria
- stop conditions
- required evidence

## Separation of Responsibilities

```text
Experiment contract:
    What are we testing?

Hardware profile:
    Where are we testing it?

Software-stack profile:
    Which software combination is under test?

Workload profile:
    What work is applied?

Result manifest:
    What happened during one execution?
Required Naming

Experiment identifiers use:

exp-p<phase>-<three-digit-sequence>-<descriptive-name>

Example:

exp-p02-003-pinned-memory-transfer
Experiment Status Values
planned
ready
running
completed
blocked
cancelled

The source-controlled experiment contract normally remains planned or ready.

Generated execution results record the actual runtime outcome.

Execution Types
validation
benchmark
profiling
failure_injection
Contract Design Rule

The contract should describe the technical question without embedding every runtime detail in the identifier.

Exact versions and hardware properties belong in the referenced profile manifests.
