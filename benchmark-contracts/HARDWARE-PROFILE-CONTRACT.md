# Hardware Profile Contract

A hardware profile records the stable machine configuration required to interpret AMD ROCm experiment results. It is intentionally small.

## Required fields

- hardware profile identifier  
- provider  
- node type  
- GPU model  
- GPU architecture  
- physical GPU count  
- logical GPU count  
- HBM capacity per GPU  
- compute partition mode  
- memory partition mode  
- GPU interconnect  
- CPU model  
- system memory  
- NUMA node count  

## Not included

The hardware profile does **not** contain:

- software versions  
- driver health  
- firmware inventory  
- benchmark results  
- GPU telemetry  
- RCCL measurements  
- network tuning  
- hourly cost  
- qualification status  

Detailed topology and inventory are captured as raw evidence during the relevant experiment phases.

## Identifier format

```text
hw-<descriptive-hardware-name>
```

Examples:

- `hw-mi300x-1x-spx-nps1`  
- `hw-mi300x-8x-spx-nps1`  
- `hw-mi300x-8x-cpx-nps4`  

## When to create a new profile

Create a new hardware profile when any of these materially change:

- GPU model or architecture  
- physical GPU count  
- logical GPU count  
- compute partition mode  
- memory partition mode  
- interconnect type  
- node type  
- CPU or NUMA configuration  

Software-only changes do **not** require a new hardware profile.
