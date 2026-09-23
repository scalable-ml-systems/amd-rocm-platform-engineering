
# AMD ROCm Platform Engineering

## From GPU Hardware to Real Inference Performance

**An experiment on AMD MI300X hardware to understand how GPU
performance affects real AI applications.**

This project follows an AI model through several layers of the
system: from the small programs running on the GPU, through
memory and communication between GPUs, all the way to the
speed experienced by someone using the model.

The goal is to develop a deeper understanding of AMD hardware
by building, measuring and explaining how the system behaves.

**Current experiment:** AMD-EXP-001  
**Hardware:** AMD Instinct MI300X  
**Software:** AMD ROCm 10.0 and vLLM  
**Model:** Qwen3-30B-A3B  
**Status:** Active — single-GPU investigation

---

## 1. What are we trying to understand?

A GPU may be capable of enormous computing power and memory
bandwidth. But an AI model running on it rarely uses all of
those resources perfectly.

Why?

The limitation might be:

- The speed of the GPU's calculations.
- The speed at which data moves between memory and the GPU.
- Communication between multiple GPUs.
- The way the inference software schedules incoming requests.

We want to measure these limitations individually and then
understand how they affect the performance of the complete
system.

Our central question is:

**Where does performance get limited as an AI workload moves
from one GPU to multiple GPUs, and how does that affect
response times?**

---

## 2. Four experiments

We divide the investigation into four manageable pieces.

| Experiment | What we want to understand |
|---|---|
| 1. Kernel | Is the GPU spending more time calculating or waiting for data from memory? |
| 2. Fabric | What changes when multiple GPUs need to exchange data? |
| 3. Serving | How do these limitations affect the speed of real inference requests? |
| 4. Cross-layer analysis | How are the measurements from all three layers connected? |

A kernel is a small program executed by the GPU. An AI model
runs many different kernels to perform calculations, process
attention and manage memory.

The important part of this investigation is connecting
what happens inside the GPU to what happens when someone
sends a request to the model.

---

## 3. Our hardware

We started with one AMD Instinct MI300X on AMD Developer Cloud.

- GPU memory: approximately 192 GiB.
- GPU architecture: AMD CDNA3.
- Software: ROCm 10.0.
- Operating system: Ubuntu 24.04.
- GPU configuration: SPX/NPS1, exposing one logical GPU
  with a unified memory domain.

This machine lets us study GPU calculations, memory access
and single-GPU inference.

The multi-GPU investigation will use separate hardware
with two or more connected AMD GPUs.

---

## 4. What have we measured so far?

### Hardware performance

Our first experiment measured the performance limits of
the MI300X using ROCm Compute Profiler.

| Measurement | Result |
|---|---:|
| GPU memory bandwidth | 4.204 TB/s |
| BF16 matrix computation | 1.029 PFLOP/s |
| FP8 matrix computation | 1.867 PFLOP/s |

**What do these numbers mean?**

Memory bandwidth tells us how quickly the GPU can move
data to and from its high-bandwidth memory (HBM).

BF16 and FP8 are numerical formats commonly used for
AI calculations. They use 16 bits and 8 bits,
respectively.

MFMA stands for **Matrix Fused Multiply-Add**. It refers
to specialized GPU instructions that perform matrix
calculations efficiently.

The BF16 and FP8 numbers above measure the throughput
of these matrix instructions.

A PFLOP/s is one quadrillion floating-point operations
per second.

These results establish the measured capabilities of
our GPU. They do not tell us how efficiently our AI
model uses the hardware. That is what we investigate next.

### Running our first AI model

We successfully ran Qwen3-30B-A3B using vLLM on one MI300X.

We prepared two workloads:

| Workload | Input | Output | Purpose |
|---|---:|---:|---|
| Long prompt | 4,700 tokens | 64 tokens | Study prompt processing |
| Long generation | 35 tokens | 512 tokens | Study token generation |

The first workload emphasizes processing a large input.
The second emphasizes generating many output tokens.

### Our first kernel investigation

We recorded approximately 54,000 GPU kernel dispatches
during the long-prompt request, excluding model
initialization and warmup.

One kernel, `fused_moe_kernel`, accounted for 29.53%
of the recorded GPU kernel execution time.

This kernel performs part of the model's
Mixture-of-Experts (MoE) computation.

The request also generated 64 output tokens, so the
trace contains both prompt processing and generation.

We are now separating those activities before
deciding which kernels are limited by computation
and which are limited by memory.

---

## 5. Where the project goes next

The next steps follow our four experiments:

1. Finish measuring the important kernels on one GPU.
2. Add multiple GPUs and measure the cost of communication.
3. Measure how those costs affect inference response times.
4. Connect the results into one technical report.

We will use the measured hardware limits as reference
points, rather than relying only on published GPU
specifications.

---

## 6. Repository structure

- `scripts/amd-exp-001/` — scripts used to run the experiments.
- `benchmark-contracts/` — definitions of what we measure
  and how we keep experiments consistent.
- `benchmark-results/amd-exp-001/` — measurements and
  supporting evidence.
- `profiler-traces/` — detailed records of GPU execution.
- `docs/` — technical notes, explanations and findings.

The original ROCm learning and engineering directories
remain available, but AMD-EXP-001 is the active project.

---

## 7. Key terminology

| Term | Meaning |
|---|---|
| ROCm | AMD's software platform for programming and running workloads on its GPUs |
| CDNA3 | AMD's GPU architecture used by the MI300X |
| HBM | High Bandwidth Memory — the GPU's high-speed memory |
| MFMA | Matrix Fused Multiply-Add — specialized instructions for matrix calculations |
| MoE | Mixture of Experts — a model architecture that activates selected groups of neural-network parameters for each token |
| TP | Tensor Parallelism — splitting model calculations across multiple GPUs |
| XGMI | AMD's high-speed connection for communication between GPUs |
| RCCL | ROCm Communication Collectives Library — software for coordinating data exchange between GPUs |
| TTFT | Time to First Token — how long a request waits before its first generated token |
| TPOT | Time per Output Token — how long generating each output token takes |
| SLO | Service Level Objective — a measurable performance target, such as a latency limit |
| p99 | 99th-percentile latency — 99% of measured requests finish within this time |

---

## Final objective

Understand how the AMD GPU behaves under real AI workloads,
identify what limits performance at each layer, and explain
how those limitations affect the complete serving system.

The final deliverable is a reproducible technical report
supported by our own hardware measurements and experiments.

