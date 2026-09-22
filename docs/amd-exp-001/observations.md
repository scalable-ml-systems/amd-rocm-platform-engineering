## Hardware Discovery — 2026-09-14

### Accelerator

AMD Developer Cloud exposes one AMD Instinct MI300X VF:

- Architecture: gfx942
- Compute units: 304
- HBM: 196288 MB
- Reported max HBM bandwidth: 5325 GB/s
- Compute partition: SPX
- Memory partition: NPS1
- XCC resources: 8

SPX presents all eight XCCs as one logical GPU.
NPS1 exposes the HBM stacks as one unified memory domain.

This configuration is frozen for AMD-EXP-001 Slice 1.

### CPU

The guest exposes:

- 20 vCPU
- one NUMA node
- ~240 GB system RAM

Physical host NUMA locality cannot be inferred reliably
from the guest VM topology.

### Network

Two Virtio network devices are visible.

No RDMA or InfiniBand devices are exposed:

- `/sys/class/infiniband` is empty
- `rdma link` returns no links

Therefore this instance will not be used for RDMA,
RoCE, or multi-node fabric measurements.

### Experimental implication

This VM is suitable for:

- MI300X TP1 execution
- empirical roofline measurement
- kernel profiling
- HBM/cache investigation
- TP1 vLLM serving baseline

Multi-GPU XGMI/RCCL and RDMA experiments require
a separate multi-GPU environment.

```

AMD Developer Cloud VM
│
├── CPU
│   └── 20 vCPU
│       └── NUMA node 0 only
│
├── MI300X VF — gfx942
│   ├── 304 CUs
│   ├── 8 XCCs
│   ├── SPX
│   │   └── all 8 XCCs presented as ONE logical GPU
│   ├── NPS1
│   │   └── all 8 HBM stacks form ONE unified memory domain
│   ├── ~192 GiB HBM
│   └── reported theoretical HBM max: 5.325 TB/s
│
└── Network
    ├── Virtio NIC #1
    ├── Virtio NIC #2
    └── NO exposed RDMA / InfiniBand device

```
