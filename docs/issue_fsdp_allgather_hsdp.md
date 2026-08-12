# Multi-node FSDP2: NCCL AllGather dominates step time (little compute overlap)

## Summary

On multi-node training with FSDP2 (`with_fsdp: True`) and a **1D world mesh**, Nsight Systems shows that **NCCL AllGather / ReduceScatter dominate GPU time** and are largely **not overlapped with compute**. The bottleneck is communication topology and FSDP shard granularity, not FlashAttention or GEMM kernels.

We propose enabling optional **HSDP (2D device mesh)**: shard within node, replicate across nodes, controlled by a config flag `hsdp_shard_size`.

---

## Environment

| Item | Value |
|------|--------|
| Code | WeatherGenerator + FSDP2 (`fully_shard`) |
| Config | `config/config_operan_georing_avhrr_forecasting_lowres.yml` |
| Hardware | Jupiter · **8 nodes × 4 GPUs** = **32 ranks** |
| Run ID | `xas6tr73` |
| Parallelism | `with_fsdp: True`, **no HSDP** (full-world FSDP) |
| Profiler | Nsight Systems (`--nsys-profiling`) |
| Trace (rank 0) | `nsys_profile_0_xas6tr73_1310493.nsys-rep` |

### Repro command

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --time 15 --nodes=8 \
  --base-config ./config/config_operan_georing_avhrr_forecasting_lowres.yml \
  --nsys-profiling
```

---

## Observed behavior

Rank-0 profile window ≈ **203 s**.

### GPU kernel time (top contributors)

| Kernel / class | Share of GPU kernel time | Notes |
|----------------|--------------------------|--------|
| `ncclDevKernel_AllGather_RING_LL` | **42.5%** | 2233 calls · avg **~20.3 ms** |
| `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | **21.9%** | 703 calls · avg **~33.2 ms** |
| `ncclDevKernel_AllReduce_Sum_f32_TREE_LL` | **3.6%** | logging / norms / throughput syncs |
| LayerNorm / FlashAttention / GEMM | remainder | compute is secondary |

**NCCL ≈ 68%** of summed GPU kernel time.

### Compute–communication overlap

Sweep over CUPTI kernel intervals (rank 0):

| Metric | Value |
|--------|--------|
| NCCL wall time overlapping any compute kernel | **~16%** |
| NCCL wall time with **no** concurrent compute | **~84%** |
| AllGathers per `Model.forward` | **~203** |
| NCCL stream vs compute stream | AllGather on stream **24**, compute mostly stream **7** |

Overlap is *possible* (different streams) but almost unused: the step is largely **serialize-on-AllGather**.

Host-side, `cudaStreamSynchronize` is the top CUDA API (~22% of API time), consistent with frequent waits / syncs around collectives and step accounting.

---

## Root cause

```text
[forward block i] ──AllGather (32 ranks)──► [compute] ──reshard──► [block i+1] ──AllGather──► …
                      ▲
                      └── small RING_LL messages, often cross-node, little prefetch
```

1. **Full-world FSDP (1D mesh)**  
   Every parameter AllGather / grad ReduceScatter uses the **world process group (32)**, so traffic crosses nodes on every shard.

2. **Fine-grained `fully_shard`**  
   Many Attention/MLP modules are wrapped individually in `model_interface.py` → **hundreds of small AllGathers** per forward. NCCL selects **`RING_LL`** (latency protocol for small payloads).

3. **Activation checkpointing**  
   Recomputation re-triggers AllGather, multiplying (2).

4. **No HSDP / no forward prefetch**  
   Default path has no 2D mesh and no `set_modules_to_forward_prefetch`, so communication rarely hides behind compute.

This is not primarily a “slow FlashAttention kernel” problem; it is a **sharding + multi-node collective** problem.

---

## Proposed solution: optional HSDP 2D mesh

Use FSDP2 **Hybrid Shard (HSDP)**:

| Mesh dim | Role | Example (8×4) |
|----------|------|----------------|
| **shard** | AllGather / ReduceScatter group | **4** (intra-node) |
| **replicate** | Replica across nodes (DDP-like) | **8** |

Effect: parameter AllGather group size **32 → 4**; cross-node traffic shifts to replica gradient sync instead of every FSDP AllGather.

### Config API

```yaml
with_fsdp: True

# null / false / 0  → classic full-world FSDP (current default behavior)
# true              → auto: world_size / SLURM_JOB_NUM_NODES
# <int>             → explicit shard size (must divide world_size), e.g. 4 on Jupiter
hsdp_shard_size: 4
```

### Implementation sketch

In `src/weathergen/model/model_interface.py`:

```python
mesh = init_device_mesh(
    "cuda",
    (replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)
fully_shard(module, mesh=mesh, ...)
```

Default remains full-world FSDP when `hsdp_shard_size` is unset/`null` (no behavior change for existing configs).

---

## Expected impact

| Metric | Baseline (this issue) | With `hsdp_shard_size: 4` (expected) |
|--------|----------------------|--------------------------------------|
| AllGather process group | 32 | **4** |
| Cross-node AllGather | yes (every shard) | mostly avoided for params |
| NCCL share of step | ~68% | should drop |
| Compute–comm overlap | ~16% | should improve (esp. with prefetch later) |

Follow-ups (not required for the first fix): coarser FSDP wraps, forward prefetch, selective `reshard_after_forward=False`, gate per-step `cuda.synchronize` in throughput tracking.

---

## Validation plan

- [ ] Re-run same config with `hsdp_shard_size: 4` and `--nsys-profiling`
- [ ] Confirm log line: `HSDP DeviceMesh: replicate=8 × shard=4`
- [ ] Compare rank-0 `.nsys-rep`: AllGather count, avg duration, NCCL % of GPU time, overlap %
- [ ] Sanity-check loss / convergence vs baseline for a short run
- [ ] Confirm checkpoint load/save still works under HSDP DTensor placements

---

## References

- PyTorch FSDP2 `fully_shard`: 2D mesh = HSDP (`Replicate` on dim 0, `Shard` on dim 1)
- Local write-up: `docs/hsdp_first_experiment.md`
- Profile artifacts: `/p/scratch/oneprotgpt/kasravi/xas6tr73/` (cluster scratch; not in git)
