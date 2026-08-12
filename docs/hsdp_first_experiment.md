# FSDP vs HSDP experiments (AllGather → AllReduce)

Nsight Systems comparison of full-world FSDP2 vs hybrid sharding (HSDP) on Jupiter, plus a completed **1/2/4/8 node** throughput sweep (**ingested samples**, no nsys) on `develop` vs `javad/hsdp-2d-mesh`.

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

## Experiment 3 — Node scaling (no nsys) — completed

Throughput / scaling study **without** Nsight Systems. Metric: **ingested samples** over the job (higher is better). Same lowres config; HSDP branch uses `hsdp_shard_size: 4`.

### Setup

| Item | Value |
|------|--------|
| Config | `config/config_operan_georing_avhrr_forecasting_lowres.yml` |
| Branches | `develop` (full-world FSDP) vs `javad/hsdp-2d-mesh` (`hsdp_shard_size: 4`) |
| Nodes swept | **1, 2, 4, 8** (4 GPUs/node) |
| Wall time | `--time 60` |
| Profiling | **none** (`WEATHERGEN_NSYS_PROFILING=0`) |
| Primary metric | **ingested samples** |

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --time 60 --nodes=N \
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

### Results — ingested samples

| Nodes | `develop` run | Ingested samples | `javad/hsdp-2d-mesh` run | Ingested samples | Δ (HSDP − develop) |
|------:|---------------|-----------------:|--------------------------|-----------------:|-------------------:|
| 1 | `x7n38kzc` | **880** | `wwxs6vv6` | **870** | −10 (−1.1%) |
| 2 | — | — | `bm8eogar` | **682** | — |
| 4 | `q4ol6600` | **622** | `zpg63djd` | **652** | **+30 (+4.8%)** |
| 8 | `ay98z59e` | **376** | `lpfwaiy9` | **394** | **+18 (+4.8%)** |

### Interpretation

- **1 node:** HSDP ≈ develop (within noise). Replicate group size is 1, so no cross-node AllReduce; no expected win.
- **4 / 8 nodes:** HSDP ingests **~5% more** samples than develop in the same wall time — modest but consistent with the nsys finding (cheaper AllGather, AllReduce still limits scaling).
- **Scaling efficiency is poor on both branches:** ingested samples **drop** as nodes increase (880 → 376 on develop; 870 → 394 on HSDP). Communication grows faster than useful compute — matches Experiment 2 (replica AllReduce dominates at 8 nodes).
- **2-node HSDP** (`bm8eogar`: 682) sits between 1- and 4-node points; no develop baseline was recorded for 2 nodes.

---

## Next steps (AllReduce bottleneck)

1. **Coarser `fully_shard`** — wrap whole blocks, not every Attention/MLP (fewer, larger AllReduces).
2. **Tune `hsdp_shard_size`** — e.g. `8` shrinks replicate group 8→4 (trades some AG cost for less AR).
3. **Backward prefetch** — overlap replica AllReduce with backward compute once messages are larger.
4. **Cheaper reduce dtype** — try `bf16` instead of fp32 `reduce_dtype` if numerically acceptable.
5. Optional: disable EMA for short scaling runs (EMA update was slower under HSDP).

After code changes, re-run the 1/4/8 node ingested-samples sweep and optionally `--nsys-profiling` on 8 nodes.
