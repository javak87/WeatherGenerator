# FSDP vs HSDP experiments (AllGather → AllReduce)

Nsight Systems comparison of full-world FSDP2 vs hybrid sharding (HSDP) on Jupiter, plus an ongoing **1→8 node** throughput sweep **without** nsys.

## Runs

| | Baseline (no 2D mesh) | HSDP (`hsdp_shard_size: 4`) |
|--|----------------------|-----------------------------|
| Run ID | `xas6tr73` | `rzl66dge` |
| Config | `config/config_operan_georing_avhrr_forecasting_lowres.yml` | same + `hsdp_shard_size: 4` |
| Cluster | Jupiter, 8 nodes × 4 GPUs (32 ranks) | same |
| Parallelism | FSDP2, 1D world mesh | FSDP2 HSDP: **replicate=8 × shard=4** (logged) |
| Rank-0 profile | `/p/scratch/oneprotgpt/kasravi/xas6tr73/nsys_profile_0_xas6tr73_1310493.nsys-rep` | `/e/scratch/weatherai/slurm/slurm_weathergen_rzl66dge_dir/WeatherGenerator/profiling/rzl66dge/nsys_profile_0_rzl66dge_1311181.nsys-rep` |

Launch:

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --time 15 --nodes=8 \
  --base-config ./config/config_operan_georing_avhrr_forecasting_lowres.yml \
  --nsys-profiling
```

---

## Experiment 1 — Baseline (`xas6tr73`, no HSDP)

### What we saw (rank 0, ~203 s window)

| Metric | Value |
|--------|--------|
| GPU time in NCCL AllGather | **42.5%** |
| GPU time in NCCL ReduceScatter | **21.9%** |
| Combined NCCL share of kernel time | **~68%** |
| AllGathers per `Model.forward` | **~203** |
| Avg AllGather duration | **~20 ms** (`RING_LL`) |
| NCCL overlapping compute | **~16%** |
| NCCL with GPU compute idle | **~84%** |
| `Model.forward` avg (NVTX) | **5.46 s** |
| Log `s/sec` @ step 10 | **0.040** |

### Why it was slow

1. **Full-world FSDP** — AllGather / ReduceScatter over **32** ranks (cross-node every shard).
2. **Fine-grained `fully_shard`** — many Attention/MLP wraps → hundreds of small AllGathers → NCCL **`RING_LL`**.
3. **Activation checkpointing** — recomputation re-triggers AllGather.
4. **Little compute–comm overlap** — no prefetch; timeline mostly NCCL-only.

---

## Solution tried — HSDP 2D mesh

Shard within the node, replicate across nodes:

```yaml
with_fsdp: True
# null/false: full-world FSDP
# true: auto (= world_size / num_nodes)
# 4: shard within 4 GPUs/node (Jupiter)
hsdp_shard_size: 4
```

Implemented in `src/weathergen/model/model_interface.py`:

```text
init_device_mesh("cuda", (replicate, shard), mesh_dim_names=("replicate", "shard"))
fully_shard(..., mesh=mesh)
```

On 8×4 with `hsdp_shard_size: 4` → AllGather group **4**, replica AllReduce group **8**.

---

## Experiment 2 — HSDP (`rzl66dge`)

Log confirms: `HSDP DeviceMesh: replicate=8 × shard=4 (AllGather group size 4)`.

### Side-by-side (rank 0)

| Metric | Baseline `xas6tr73` | HSDP `rzl66dge` | Change |
|--------|---------------------|-----------------|--------|
| AllGather total GPU time | 45.3 s | **5.7 s** | **−87%** |
| AllGather avg | 20.3 ms | **2.9 ms** | **−86%** |
| ReduceScatter total | 23.3 s | **3.5 s** | **−85%** |
| AllReduce total | 3.9 s | **49.1 s** | **+12.6×** |
| NCCL share of GPU kernels | 68% | 64% | slight ↓ |
| NCCL∩compute overlap | 16% | 22% | +6 pp |
| `Model.forward` avg | 5.46 s | **4.77 s** | **~13% faster** |
| Log `s/sec` @ step 10 | 0.040 | 0.043 | ~+8% |

### Per-`Model.forward` NCCL (normalized)

| Collective | Baseline | HSDP |
|------------|----------|------|
| AllGather | ~4117 ms | **~571 ms** |
| ReduceScatter | ~2121 ms | **~353 ms** |
| AllReduce | ~354 ms | **~4906 ms** |
| AG + RS (FSDP shard path) | ~6238 ms | **~924 ms** |
| All NCCL | ~6591 ms | ~5830 ms |

### Interpretation

- **AllGather goal: achieved** — intra-node shard group of 4 makes AG/RS much cheaper.
- **New bottleneck: AllReduce** on the **replicate** dim (8 nodes). Fine-grained shards still issue many small cross-node AllReduces (`RING_LL`, avg ~76 ms) → **53%** of GPU kernel time in `rzl66dge`.
- **Net end-to-end:** modest win (~10–13% faster forward), not a large speedup, until AllReduce is fixed.

### Memory footprint

| Source | Estimate |
|--------|----------|
| Theory | Params/grads/optim held at **1/4** of ranks instead of **1/32** → up to **~8×** more of those tensors **per GPU** |
| nsys CUDA alloc tracking (rank 0; frees not recorded → rough) | ~36.1 → ~40.7 GiB ≈ **+4.6 GiB (~+13%)** |

HSDP **increases** memory; it does not save it. Prefer `torch.cuda.max_memory_allocated` / `nvidia-smi` for a hard peak.

---

## Experiment 3 — Node scaling 1→8 (no nsys) — *in progress*

Throughput / scaling study **without** Nsight Systems (avoid profiler overhead). Same lowres config and HSDP settings as Experiment 2.

### Goal

Measure how training throughput scales with node count under HSDP, and how the **replicate** AllReduce group grows.

### Setup

| Item | Value |
|------|--------|
| Config | `config/config_operan_georing_avhrr_forecasting_lowres.yml` |
| HSDP | `hsdp_shard_size: 4` (shard = 4 GPUs/node) |
| Nodes swept | **1, 2, 4, 8** (4 GPUs/node → 4 / 8 / 16 / 32 ranks) |
| Profiling | **none** (no `--nsys-profiling`) |
| Primary metric | log `s/sec` after warmup (e.g. step ≥ 20) |

```bash
# Example for N nodes (repeat N = 1 .. 8); do NOT pass --nsys-profiling
../WeatherGenerator-private/hpc/launch-slurm.py --time 15 --nodes=N \
  --base-config ./config/config_operan_georing_avhrr_forecasting_lowres.yml
```

### Expected HSDP mesh vs nodes

With `hsdp_shard_size: 4`:

| Nodes | World size | Mesh `(replicate × shard)` | AllGather group | AllReduce (replicate) group |
|------:|-----------:|----------------------------|----------------:|----------------------------:|
| 1 | 4 | 1 × 4 | 4 | **1** (no cross-node AR) |
| 2 | 8 | 2 × 4 | 4 | 2 |
| 4 | 16 | 4 × 4 | 4 | 4 |
| 8 | 32 | 8 × 4 | 4 | **8** |

AllGather cost should stay similar across node counts (always shard=4). Cross-node **AllReduce** should grow with `replicate = num_nodes` — this is the scaling cliff to quantify in logs.

### Results table (fill as runs finish)

| Nodes | Run ID | Slurm job | `s/sec` (post-warmup) | Notes |
|------:|--------|-----------|----------------------:|-------|
| 1 | `wwxs6vv6` | 1311829 | | |
| 2 | `bm8eogar` | 1311833 | | |
| 4 | `zpg63djd` | 1311837 | | |
| 8 | `lpfwaiy9` | 1311846 | | compare to nsys run `rzl66dge` |

All launched with `--time 60`, no `--nsys-profiling` (`WEATHERGEN_NSYS_PROFILING=0`). Confirm each log contains `HSDP DeviceMesh: replicate=<N> × shard=4`.

### How to read `s/sec` from logs

```bash
rg "s/sec=" /e/scratch/weatherai/slurm/slurm_weathergen_<RUN>_dir/WeatherGenerator/logs/<RUN>/log.txt
```

Use a stable post-warmup step (e.g. 20 or 40), not step 10 alone.

---

## Next steps (AllReduce bottleneck)

1. **Finish Experiment 3** — fill the 1→8 node `s/sec` table (no nsys).
2. **Coarser `fully_shard`** — wrap whole blocks, not every Attention/MLP (fewer, larger AllReduces).
3. **Tune `hsdp_shard_size`** — e.g. `8` shrinks replicate group 8→4 (trades some AG cost for less AR).
4. **Backward prefetch** — overlap replica AllReduce with backward compute once messages are larger.
5. **Cheaper reduce dtype** — try `bf16` instead of fp32 `reduce_dtype` if numerically acceptable.
6. Optional: disable EMA for short scaling runs (EMA update was slower under HSDP).

After code changes, optionally re-enable `--nsys-profiling` on 8 nodes and compare **AllReduce total ms** / **avg ms**.
