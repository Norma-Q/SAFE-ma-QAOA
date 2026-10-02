# SAFE ma-QAOA: Reproducibility Repository

Code, problem instances, and pipelines to reproduce every experimental figure and
table of

> **SAFE ma-QAOA: Surrogate-Assisted and Fine-Tuning Enhanced Multi-Angle QAOA
> with Parameter Distillation**, Hyunwoo Kim and Youngseok Lee (NORMA Inc.).
> arXiv:2605.23377, <https://arxiv.org/abs/2605.23377>.

SAFE trains multi-angle QAOA (ma-QAOA) in three stages: **(1) surrogate
pre-training** with Low-Weight Pauli Propagation (LWPP) as a cheap classical
surrogate, **(2) parameter distillation**, which removes angles whose magnitude
remains at or below a threshold, and **(3) exact fine-tuning** of the remaining
active angles on the exact energy objective. This repository regenerates the
full data grid behind the paper: three problem families (Sherrington-Kirkpatrick,
two-dimensional square-lattice spin glass, Max-Cut), 12 to 20 qubits, depths
$p \in \{2, 3, 4\}$, LWPP truncation weights $w_{\max} \in \{3, 4\}$, and distillation
thresholds $\{0, 0.01, 0.3\}$, with eleven initialization settings per instance.

## Repository layout

```
safe_maqaoa/      experiment package
  problem_factory.py                 SK / 2D-grid / Max-Cut instance builders
  make_problem.py                    CLI instance generator
  run_ma_qaoa_lwpp_exact_compare.py  SAFE pipeline: LWPP pre-train -> exact
                                     fine-tune + exact-only baseline
  run_distill_finetune.py            parameter distillation + exact fine-tune
  aggregate.py                       metrics (alpha, tau_0.99, C_ballpark)
  aggregate_cost.py                  LWPP term counts, wall time, peak VRAM
  plot_problem_family_schematic.py   Fig. 2 (instance schematic)
data/problems/    all problem instances used by the sweeps (JSON, seed-exact)
sweeps/           sweep launchers (5-way parallel, resume-safe)
figures/          Fig. 4/5/6/7 and Table I/II generators
smoke/            fast verification: engine-vs-reference, instance
                  regeneration, reduced end-to-end pipeline, device benchmark
results/          sweep outputs (created at run time)
environment.yml   conda environment spec (name: SAFE)
```

The LWPP surrogate is evaluated by [padopauli](https://github.com/Norma-Q/PADO-Pauli),
a GPU-accelerated differentiable Pauli-propagation engine distributed as a prebuilt
wheel; it is installed as a dependency, not vendored here.

## Setup

Target platform: Linux x86-64 with an AMD Instinct GPU (validated on MI300X, ROCm 7.2
host, torch 2.11.0+rocm7.2). Everything also runs on CPU-only hosts via PennyLane's
`lightning.qubit` (slower; see the device notes below).

```bash
conda env create -f environment.yml      # torch (ROCm), padopauli wheel, PennyLane, matplotlib, pandas
conda activate SAFE
python -c "import padopauli; print(padopauli.__version__)"
```

`environment.yml` installs everything, including the padopauli engine as a prebuilt
wheel from the [PADO-Pauli GitHub Releases](https://github.com/Norma-Q/PADO-Pauli/releases)
(Python 3.11; the release page also carries a cp312 wheel, so swap the URL if you change
the Python version). CUDA hosts: replace the torch line with a CUDA build of torch
2.10 to 2.12 and `pennylane-lightning-amdgpu` with `pennylane-lightning-gpu`, then pass
`EXACT_DEVICE=lightning.gpu` to the sweeps.

GPU note: the ROCm build of PyTorch exposes the AMD GPU through the standard
`torch.cuda` API (HIP). Exact fine-tuning uses the PennyLane state-vector device
`lightning.amdgpu`. If `torch.cuda.is_available()` is `False` on a ROCm host,
check that your user is in the `render`/`video` groups (`/dev/kfd`,
`/dev/dri/renderD*`).

### Quick verification

```bash
bash smoke/run_smoke_pipeline.sh                # CPU checks + reduced end-to-end pipeline (GPU)
# CPU-only host:
#   EXACT_DEVICE=lightning.qubit LWPP_PRESET=cpu bash smoke/run_smoke_pipeline.sh
```

The CPU checks (`smoke/verify_problem_regeneration.py`: one shipped instance per
family regenerates from its seed; `smoke/verify_engine_vs_pennylane.py`: untruncated
surrogate vs PennyLane state vector) can also be run on their own. Smoke outputs go
to `results_smoke/`.

### Choosing the exact-fine-tuning device

```bash
python smoke/benchmark_exact_device.py          # lightning.amdgpu vs lightning.qubit
```

The benchmark times the batched energy+gradient step used by exact fine-tuning
at $q = 12, 16, 20$ and prints an `EXACT_DEVICE=...` recommendation; pass it to the
sweeps below to override their default. By default (`EXACT_DEVICE=auto`) 12-qubit
cases run on `lightning.qubit` and larger cases on `lightning.amdgpu`. The paper's
runs used `lightning.amdgpu`.

## Reproducing the paper

### 1. Problem instances

All instances are shipped under `data/problems/` and are exactly regenerable
from their seeds (`smoke/verify_problem_regeneration.py` checks this):

- **SK**: fully connected, $J_{ij} \sim \mathcal{N}(0, 1/N)$, no fields.
- **2D grid**: $3 \times 4$, $4 \times 4$, $4 \times 5$ lattice; nearest neighbours $J \sim \mathcal{N}(0, 1)$,
  next-nearest $J \sim \mathcal{N}(0, 0.3^2)$, fields $h \sim \mathcal{N}(0, 0.1^2)$.
- **Max-Cut**: Erdős-Rényi $G(n, 0.3)$, unit weights, no fields.

Seed scheme: `700000 + q*1000 + p*100 + instance` for SK/Max-Cut (five
instances 0 to 4 per $(q, p)$), `120490 + instance` for the 2D grids (shared across
depths).

### 2. Sweeps

```bash
bash sweeps/run_all_paper_sweeps.sh
```

runs, in order: $w_{\max} = 4$ then $w_{\max} = 3$, each for all families over
$q \in \{12, 16, 20\}$ (grids $3 \times 4$, $4 \times 4$, $4 \times 5$) $\times$ $p \in \{2, 3, 4\}$ $\times$ 5 instances, 270
pipeline runs total (the paper reports $w_{\max} = 3$ only at 20 qubits; run
`QUBITS_LIST=20 MW=3 FAMILY=... bash sweeps/run_family_sweep.sh` to skip the rest).
Each run performs LWPP pre-training
(500 steps, float16 surrogate), exact fine-tuning and the exact-only baseline
(100 steps each, float64 parameters, eleven initializations as one batch),
then distillation fine-tuning at thresholds 0.01 and 0.3 (100 steps).
Every optimization stage uses Adam with lr 0.02 and default betas (paper
footnote 1). Initialization settings (paper Sec. III-B): 5 random, 5 constant
(0.01, 0.05, 0.1, 0.2, 0.4 with $\beta = -\text{const}$), 1 QAOA-Relax (TQA warm start,
$\Delta t = 0.5$, 200 tied-parameter steps).

Sweeps are 5-way parallel (one worker per instance), resume-safe (relaunching
skips completed runs and resumes interrupted ones from checkpoints), and
controlled by environment knobs, most usefully `EXACT_DEVICE`, `FORCE_RERUN=1`,
and `CONDA_ENV`. A single family/tier can be run directly, e.g.:

```bash
FAMILY=maxcut MW=4 bash sweeps/run_family_sweep.sh
```

The full grid is a multi-day job on a single GPU; the heaviest cases are $q = 20$,
$p = 4$. `smoke/benchmark_exact_device.py` measures the per-step cost on your host.

### 3. Aggregation, figures, and tables

```bash
python safe_maqaoa/aggregate.py                 # -> figures/output/runs_long.csv
python safe_maqaoa/aggregate_cost.py            # -> figures/output/cost_metrics.csv
                                                # LWPP zero-filtering term counts (pre/post),
                                                # per-phase wall time and peak VRAM per run
python figures/make_summary_tables.py           # Tables I & II (+ best-case CSVs)
python figures/make_fig4.py                     # Fig. 4  threshold overview
python figures/make_fig5.py                     # Fig. 5  first-hit steps
python figures/make_fig6.py --run-dir results/mw4/sk/sk_q20_p4_inst0_seed720400
                                                # Fig. 6  fine-tuning trajectories
python figures/make_fig7.py                     # Fig. 7  cost-angle structure
python safe_maqaoa/plot_problem_family_schematic.py --output figures/output/fig2.png
                                                # Fig. 2  instance schematic
```

| Paper artifact | Source data | Generator |
|---|---|---|
| Fig. 2 | instance builders | `plot_problem_family_schematic.py` |
| Fig. 4 | `runs_long.csv` (20-qubit tier, $w_{\max} = 3, 4$) | `figures/make_fig4.py` |
| Fig. 5 | `runs_long.csv` ($w_{\max} = 4$, threshold 0.3) | `figures/make_fig5.py` |
| Fig. 6 | per-run trace CSVs | `figures/make_fig6.py` |
| Fig. 7 | per-run thetas + traces | `figures/make_fig7.py` |
| Table I | best-case summary CSV | `figures/make_summary_tables.py` |
| Table II | `runs_long.csv` | `figures/make_summary_tables.py` |

(Figs. 1, 3, and 8 are conceptual illustrations, not generated from data.)

### Metrics (paper Sec. III-D)

- **Approximation ratio** $\alpha(\theta) = (E_{\max} - E(\theta)) / (E_{\max} - E_{\min})$, with
  $E_{\min}$ and $E_{\max}$ from exhaustive enumeration of all $2^n$ spin configurations.
- **First-hit step** $\tau_{0.99}$: the first fine-tuning step whose $\alpha$ reaches 99% of the
  trajectory's own best value. Exact-stage traces record the pre-update energy,
  so step 0 is the starting point for every method (a distilled start already
  in the near-best regime gives $\tau_{0.99} = 0$).
- **Cost ballpark** $C = N_{\text{active}} \cdot \tau_{0.99}$, averaged over instances and
  initializations; Table II reports SAFE-vs-exact-only reduction factors.

## Reproducibility conventions and caveats

- **Determinism.** Problem instances are bit-exact functions of their recorded
  seeds. Training runs use fixed seeds for initialization pools (grid2d passes
  the case seed, 120490+instance, to the trainer; SK/Max-Cut use the trainer
  default seed 0), but GPU floating-point reductions and the float16 LWPP
  surrogate are not bit-exact across hardware or library versions, so trained
  energies reproduce statistically rather than bit-for-bit.
- **Distillation implementation.** Distilled rotation gates are not built: for each initialization the ma-QAOA circuit is reconstructed from only
  the active rotations (parameter indices remapped), a fresh exact qnode is
  compiled for that pruned circuit, and the initializations are fine-tuned
  sequentially, each on its own smaller circuit.
- **Circular cosine similarity** (Fig. 7) between cost-angle vectors $\theta$, $\phi$ is
  computed as $\mathrm{mean}_i \cos(\theta_i - \phi_i)$ over all cost angles (edge and field
  rotations, all layers).
- The `padopauli` engine evaluates the LWPP surrogate; its full-weight
  (untruncated) mode is exact and is checked against a PennyLane state-vector
  reference in `smoke/verify_engine_vs_pennylane.py`.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
