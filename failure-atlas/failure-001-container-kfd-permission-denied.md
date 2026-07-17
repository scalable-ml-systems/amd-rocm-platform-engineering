# Failure 001 — Container KFD Permission Denied

## Summary

The ROCm container could see `/dev/kfd`, but ROCm initialization failed because the container process did not have permission to access the device.

## Symptom

The container reported a permission error while running `rocminfo`.

The device existed inside the container, but the ROCm runtime could not initialize.

## Affected Layer

```text
container
Environment
hardware_profile_id: placeholder
software_stack_id: placeholder
workload_profile_id: null
experiment_id: placeholder
run_id: placeholder
Expected Behavior

rocminfo should enumerate the same logical GPU devices inside the container that are visible on the host.

Raw Evidence

Expected evidence files:

stdout.log
stderr.log
host-device-permissions.txt
container-device-permissions.txt
run-manifest.json
Initial Hypotheses
/dev/kfd was not passed into the container

the render devices were missing

the container user lacked the required group membership

the security context blocked device access
Diagnostic Probes
Compare host and container device visibility

Command:

ls -l /dev/kfd /dev/dri/renderD*

Purpose:

Confirm that the required devices exist inside the container.

Check process identity and groups

Command:

id

Purpose:

Confirm whether the container process belongs to the group that owns the GPU devices.

Root Cause

Placeholder example:

The container process did not have the group permissions required to access /dev/kfd.

This root cause must be replaced with real evidence when the failure is reproduced.

Corrective Action

Add the required supplemental group or explicit device-access configuration without granting unnecessary privileged access.

Validation

Rerun:

rocminfo

Then compare host and container logical GPU counts.

Operator Rule

Device presence inside a container does not prove device access. Validate both device mapping and process permissions.

Prevention or Regression Test

Add a container qualification test that verifies:

/dev/kfd exists

required render devices exist

the process can open the devices

rocminfo succeeds

host and container GPU counts match
Limitations

This is a structural example. The final entry must contain actual commands, logs, environment identifiers, and before-and-after evidence.
