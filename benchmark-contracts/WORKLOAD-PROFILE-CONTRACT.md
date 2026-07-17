# Workload Profile Contract

A workload profile records the reusable input conditions applied during an AMD ROCm experiment. It is intentionally small.

## Required fields

- workload profile identifier  
- workload type  
- model name  
- model revision  
- input token count  
- output token count  
- request count  
- concurrency  
- arrival pattern  
- random seed  
- tensor-parallel size  
- data-parallel size  
- expert-parallel size  

## Not included

The workload profile does not contain:

- hardware configuration  
- software versions  
- benchmark results  
- raw private prompts  
- GPU measurements  
- framework-specific environment variables  
- every serving-engine option  

Those details belong in hardware profiles, software-stack profiles, experiment contracts, or generated evidence.

## Identifier format

```text
wl-<descriptive-workload-name>
```

Examples:

```text
wl-dense-short-short-steady-c08
wl-dense-long-short-burst-c32
wl-deepseek-mla-long-long-steady-c64
wl-deepseek-moe-decode-heavy-c128
```

## Arrival patterns

```text
single: One request used for correctness or smoke testing.
steady: Requests enter at a controlled steady rate.
burst: Requests are submitted in a concentrated burst.
closed_loop: A fixed number of clients submit a new request after the previous request completes.
```

## When to create a new workload profile

Create a new profile when a result-relevant workload property changes, including:

- model or model revision  
- input token count  
- output token count  
- request count  
- concurrency  
- arrival pattern  
- random seed  
- tensor, data, or expert parallel size  

Hardware-only and software-only changes do not require a new workload profile.
