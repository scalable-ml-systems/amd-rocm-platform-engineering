# Rule XXX — Descriptive Operator Rule

## Rule

State the production rule in one direct sentence.

Example:

> Do not schedule distributed inference on a node that has not passed RCCL qualification.

## Why

Explain the failure or operational risk this rule prevents.

## Applies When

Describe the systems and conditions where the rule applies.

## Evidence

Reference the experiment runs or failure-atlas entries that support the rule.

Examples:

```text
experiment_id
run_id
failure-atlas entry
benchmark summary
qualification report
Validation Procedure

List the checks an operator should run before proceeding.

For each check, include:

command or procedure
expected evidence
pass condition
Pass Condition

Define what must be true before the operation or workload may proceed.

Failure Action

Define what the operator should do when validation fails.

Examples:

stop the experiment
taint the node
quarantine the node
restore the approved software stack
rerun node qualification
collect additional evidence
Exceptions

Document legitimate cases where the rule does not apply.

When there are no known exceptions, write:

No known exceptions.
Related Failures

Reference relevant failure-atlas entries.

Related Playbooks

Reference any operational procedures that implement this rule.
