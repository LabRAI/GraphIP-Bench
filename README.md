<div align="center">

# GraphIP-Bench

### How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?

**A unified benchmark and library for evaluating model-extraction attacks and ownership defenses on graph neural networks under a single reproducible black-box protocol.**

<p>
  <a href="#why-graphip-bench"><img alt="Project" src="https://img.shields.io/badge/Benchmark-GNN%20IP-4F46E5?style=for-the-badge"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="DGL" src="https://img.shields.io/badge/DGL-2.1.0-7C3AED?style=for-the-badge">
  <img alt="PyG" src="https://img.shields.io/badge/PyG-2.7.0-14B8A6?style=for-the-badge">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-0F766E?style=for-the-badge"></a>
</p>

<p>
  <a href="#why-graphip-bench">Why GraphIP-Bench</a> ·
  <a href="#highlights">Highlights</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#research-questions">Research Questions</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#citation">Citation</a>
</p>

<p>
  <img alt="Attacks" src="https://img.shields.io/badge/Attacks-12-EF4444?style=flat-square">
  <img alt="Defenses" src="https://img.shields.io/badge/Defenses-12-0EA5E9?style=flat-square">
  <img alt="Datasets" src="https://img.shields.io/badge/Datasets-10-22C55E?style=flat-square">
  <img alt="Backbones" src="https://img.shields.io/badge/Backbones-3-7C3AED?style=flat-square">
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-3-F59E0B?style=flat-square">
  <img alt="Joint Track" src="https://img.shields.io/badge/Joint%20Track-Watermark%20Survival-DB2777?style=flat-square">
</p>

</div>

---

## Why GraphIP-Bench

Graph neural networks deployed as cloud services can be **stolen** through model-extraction attacks: an adversary submits carefully chosen queries, records the labels or confidence scores that the endpoint returns, and trains a local surrogate which reproduces the target's behaviour. A successful theft leaks the owner's intellectual property, undermines pay-per-query revenue, and lets competitors recreate proprietary functionality at low cost.

A growing line of ownership defenses — watermarking, fingerprinting, output perturbation, query-pattern detection — tries to prevent or trace such theft. **But experimental practice in this area is fragmented:** studies use private splits, incompatible budgets, and inconsistent metrics; the few existing testbeds focus on robustness or privacy and exclude model extraction together with ownership verification.

GraphIP-Bench addresses this gap. It standardises the evaluation of two complementary tracks — the **extraction track** (an adversary trains a surrogate that imitates a deployed GNN) and the **ownership track** (the model owner verifies a watermark or fingerprint after extraction) — under a single black-box protocol with shared splits, queries, and budgets. We then add a **joint attack-and-defense track** that runs every extraction attack on every defended target and measures *watermark survival* on the extracted surrogate — the setting that actually determines whether a defense is useful.

| Component | What it gives you |
|---|---|
| **12 attacks** | Six MEA-style baselines (MEA0–MEA5), adversarial (AdvMEA), centrality-driven (CEGA), structure-aware (Realistic), and three data-free variants (DFEA_I/II/III) |
| **12 defenses** | 5 ownership-tracing (BackdoorWM, RandomWM, SurviveWM, ImperceptibleWM, Integrity) + 7 information-limiting (OP_low, OP_high, PR_2bit, PR_top1, PRADA, AdaptMisinfo, GradRedir) |
| **10 datasets** | Homophilic citation (Cora, CiteSeer, PubMed), coauthor/product (CoauthorCS, CoauthorPhysics, Computers, Photo), large-scale (OGBN-Arxiv), heterophilic (RomanEmpire, AmazonRatings) |
| **3 backbones × 3 tasks** | GCN, GAT, GraphSAGE on node classification, link prediction, graph classification |
| **Joint track** | Surrogate fidelity *and* watermark survival on the extracted surrogate, all under one protocol |
| **Reproducibility** | Public splits, shared query sets, fixed seeds, JSON-Lines outputs, ready-made aggregation scripts |

---

## Highlights

<table>
  <tr>
    <td width="50%">
      <h3>One protocol, two sides</h3>
      <p>Every attack is evaluated against every defense under the same splits, budgets, and endpoint assumptions. Fidelity, ownership verification, utility, and compute cost are reported on a shared metric suite.</p>
    </td>
    <td width="50%">
      <h3>Watermark survival, not just embedding</h3>
      <p>The joint track measures whether a watermark still verifies after the model is actually extracted — exposing the gap that single-model evaluations miss.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>Heterophilic & large-scale graphs</h3>
      <p>Beyond the seven homophilic graphs of the core protocol, we add OGBN-Arxiv (169K nodes, 40 classes) and the heterophilic RomanEmpire / AmazonRatings to stress test attack and defense behaviour.</p>
    </td>
    <td width="50%">
      <h3>Faithful, paper-aligned attacks</h3>
      <p>MEA2 follows the Wu 2022 structure-only protocol with one-hot node-ID features; DFEA_I/II/III ship with both the data-free synthetic-query and the paper-faithful real-graph proxy variants used in the Zhuang 2024 paper.</p>
    </td>
  </tr>
</table>

---

## At A Glance

| Signal | Scale in the benchmark |
|---|---:|
| Extraction attacks | **12** |
| Ownership / information-limiting defenses | **12** |
| Graph datasets | **10** |
| GNN backbones | **3** (GCN, GAT, GraphSAGE) |
| Graph-learning tasks | **3** (node / link / graph) |
| Data-availability regimes | **4** (both, features-only, structure-only, data-free) |
| Standardized query budgets | **5** (0.05× — 1.00× test size) |
| Seeds per cell | **3** |
| Evaluation tracks | **3** (extraction, ownership, joint) |

---

## Quick Start

```bash
git clone https://github.com/LabRAI/GraphIP-Bench.git
cd GraphIP-Bench
conda create -n graphip python=3.11 -y
conda activate graphip
pip install -r requirements.txt
```

Run one cell of the extraction track:

```bash
python scripts/run_rq1_single.py \
  --dataset Cora \
  --attack MEA0 \
  --regime both \
  --budget 0.25 \
  --seed 0 \
  --gpu \
  --output-dir outputs/RQ1
```

Run the joint attack-and-defense track on a single dataset:

```bash
python examples/run_joint_evaluation.py \
  --dataset Cora \
  --seed 0 \
  --gpu
```

Each run writes one JSON-Lines record per (attack, defense, regime, budget, seed) tuple to `outputs/`. Records can be merged and analysed with the helpers in `scripts/` (e.g. `merge_rq1_rq5_faithful_results.py`).

---

## Installation

GraphIP-Bench supports **Linux** (recommended) with **Python 3.10+**. We pin PyTorch / DGL / PyG versions because DGL 2.1.0 graphbolt kernels only ship pre-built shared libraries for PyTorch 2.0–2.2 on CUDA 12.1.

### Option A: Conda (recommended)

```bash
conda create -n graphip python=3.11 -y
conda activate graphip
pip install -r requirements.txt
```

### Option B: pip + venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary><b>📦 Pinned stack</b></summary>

| Library | Version | Why pinned |
|:---|:---|:---|
| PyTorch | 2.2.1 + cu121 | DGL 2.1.0 graphbolt wheels only support PT 2.0–2.2 / CUDA 12.1 |
| DGL | 2.1.0 + cu121 | Graphbolt kernels used by the loaders |
| PyTorch Geometric | 2.7.0 | Link-prediction and graph-classification adapters |
| OGB | 1.3.6 | OGBN-Arxiv loader |
| NumPy | < 2.0 | DGL kernel ABI |

</details>

---

## Research Questions

GraphIP-Bench is organised around five research questions; each maps to a top-level reproduction script.

| RQ | Question | Entry point |
|:---:|:---|:---|
| **RQ1** | How does extraction effectiveness change with query budget, and does the trend hold on heterophilic and large-scale graphs? | [`scripts/run_rq1_single.py`](scripts/run_rq1_single.py) |
| **RQ2** | How effective are existing defenses on the protected model? | [`scripts/run_defense_single.py`](scripts/run_defense_single.py) |
| **RQ3** | How well do defenses balance protection and utility? | [`scripts/run_defense_single.py`](scripts/run_defense_single.py) + [`scripts/run_defense_hp_ablation.py`](scripts/run_defense_hp_ablation.py) |
| **RQ4** | What are the computational complexity and practical efficiency of attacks and defenses? | wall-clock and peak-mem fields recorded in every JSONL record |
| **RQ5** | How effective are defenses in the joint adversarial setting, and does the watermark signal survive on the extracted surrogate? | [`examples/run_joint_evaluation.py`](examples/run_joint_evaluation.py) |

Supplementary tracks:

| Track | Entry point |
|:---|:---|
| Cross-architecture extraction (undefended) | [`scripts/run_cross_arch.py`](scripts/run_cross_arch.py) |
| Cross-architecture on defended targets | [`scripts/run_cross_arch_defended.py`](scripts/run_cross_arch_defended.py) |
| Link prediction (Cora) | [`scripts/run_link_prediction.py`](scripts/run_link_prediction.py), [`examples/run_link_pred_experiments.py`](examples/run_link_pred_experiments.py) |
| Graph classification (ENZYMES, PROTEINS) | [`scripts/run_graph_class.py`](scripts/run_graph_class.py) |
| Endpoint ablation (hard-label vs. confidence scores) | [`examples/run_endpoint_ablation.py`](examples/run_endpoint_ablation.py) |
| Budget grid ablation | [`scripts/run_budget_ablation.py`](scripts/run_budget_ablation.py) |
| Query-split / structure analysis | [`scripts/run_query_split_ablation.py`](scripts/run_query_split_ablation.py), [`scripts/run_structure_analysis.py`](scripts/run_structure_analysis.py) |
| Baseline utility across backbones | [`scripts/run_baseline_utility.py`](scripts/run_baseline_utility.py) |

---

## Attacks

<table>
<tr><th align="left">Category</th><th align="left">Name</th><th align="left">Key idea</th></tr>
<tr><td rowspan="7"><b>Data-driven</b></td>
<td><code>MEA0</code>–<code>MEA1</code></td><td>Random / shuffled-order subgraph queries (Wu 2022 baselines)</td></tr>
<tr><td><code>MEA2</code></td><td>Structure-only extraction with one-hot node-ID features (paper-faithful Wu 2022)</td></tr>
<tr><td><code>MEA3</code>–<code>MEA5</code></td><td>Shadow-graph and feature-shuffled variants of MEA</td></tr>
<tr><td><code>AdvMEA</code></td><td>Adversarial-query attack with policy search</td></tr>
<tr><td><code>CEGA</code></td><td>Centrality- and entropy-driven node selection</td></tr>
<tr><td><code>Realistic</code></td><td>Structure-aware pipeline with auxiliary edge model</td></tr>
<tr><td rowspan="3"><b>Data-free</b></td>
<td><code>DFEA_I</code></td><td>KL-divergence soft-label distillation; ships with synthetic-graph + real-graph proxy variants</td></tr>
<tr><td><code>DFEA_II</code></td><td>Hard-label supervision; ships with synthetic-graph + real-graph proxy variants</td></tr>
<tr><td><code>DFEA_III</code></td><td>Label-only + consistency loss between two surrogates</td></tr>
</table>

Each attack supports four **data-availability regimes**: `both`, `x_only`, `a_only`, and `data_free`, controlled by `--regime`.

---

## Defenses

<table>
<tr><th align="left">Family</th><th align="left">Name</th><th align="left">Mechanism</th></tr>
<tr><td rowspan="5"><b>Ownership-tracing</b></td>
<td><code>BackdoorWM</code></td><td>Trigger-based backdoor watermark</td></tr>
<tr><td><code>RandomWM</code></td><td>Random-graph trigger watermark</td></tr>
<tr><td><code>SurviveWM</code></td><td>SNNL-based watermark designed to survive extraction</td></tr>
<tr><td><code>ImperceptibleWM</code></td><td>Representation-level imperceptible watermark</td></tr>
<tr><td><code>Integrity</code></td><td>Query-time fingerprint verifier (no model-side trigger)</td></tr>
<tr><td rowspan="4"><b>Output perturbation</b></td>
<td><code>OP_low</code> / <code>OP_high</code></td><td>Gaussian noise on returned logits at two scales</td></tr>
<tr><td><code>PR_2bit</code></td><td>Two-bit quantization of returned probabilities</td></tr>
<tr><td><code>PR_top1</code></td><td>Top-1 label-only output</td></tr>
<tr><td>—</td></tr>
<tr><td rowspan="3"><b>Query detection</b></td>
<td><code>PRADA</code></td><td>Distance-based query-stream detector</td></tr>
<tr><td><code>AdaptMisinfo</code></td><td>Adaptive misinformation on flagged queries</td></tr>
<tr><td><code>GradRedir</code></td><td>Gradient-redirection on flagged queries</td></tr>
</table>

---

## Datasets

All datasets are downloaded automatically on first use and cached under `data/`.

| Dataset | Nodes | Edges | Classes | Edge homophily | Source |
|:---|---:|---:|---:|---:|:---|
| Cora | 2,708 | 5,278 | 7 | 0.81 | DGL Planetoid |
| CiteSeer | 3,327 | 4,614 | 6 | 0.74 | DGL Planetoid |
| PubMed | 19,717 | 44,325 | 3 | 0.80 | DGL Planetoid |
| Computers | 13,752 | 252,737 | 10 | 0.78 | DGL Amazon |
| Photo | 7,650 | 122,906 | 8 | 0.83 | DGL Amazon |
| CoauthorCS | 18,333 | 81,894 | 15 | 0.81 | DGL Coauthor |
| CoauthorPhysics | 34,493 | 247,962 | 5 | 0.93 | DGL Coauthor |
| OGBN-Arxiv | 169,343 | 667,793 | 40 | 0.70 | OGB |
| RomanEmpire | 22,662 | 44,258 | 18 | 0.29 | Heterophilic |
| AmazonRatings | 24,492 | 105,296 | 5 | 0.45 | Heterophilic |

ENZYMES and PROTEINS (graph classification) are loaded through TUDataset in PyG.

---

## Reproducing the Paper

Every paper figure or table is generated from JSON-Lines records produced by the entry-point scripts above. A typical workflow:

```bash
# 1. Sweep RQ1 over (dataset × attack × regime × budget × seed)
for ds in Cora CiteSeer PubMed Computers Photo CoauthorCS CoauthorPhysics OGBNArxiv RomanEmpire AmazonRatings; do
  for atk in MEA0 MEA1 MEA2_Wu2022 MEA3 MEA4 MEA5 AdvMEA CEGA Realistic \
             DFEA_I_RealGraph DFEA_II_E DFEA_III_E; do
    for regime in both x_only a_only data_free; do
      for budget in 0.05 0.10 0.25 0.50 1.00; do
        for seed in 0 1 2; do
          python scripts/run_rq1_single.py \
            --dataset $ds --attack $atk \
            --regime $regime --budget $budget \
            --seed $seed --gpu \
            --output-dir outputs/RQ1
        done
      done
    done
  done
done
```

```bash
# 2. Sweep RQ5 (joint track) at the medium budget 0.25×
for ds in Cora CiteSeer PubMed Computers Photo CoauthorCS CoauthorPhysics OGBNArxiv RomanEmpire AmazonRatings; do
  for seed in 0 1 2; do
    python examples/run_joint_evaluation.py --dataset $ds --seed $seed --gpu
  done
done
```

```bash
# 3. Merge & summarise into the unified table format
python scripts/merge_rq1_rq5_faithful_results.py
```

All runs use fixed seeds (`0`, `1`, `2`) and the shared query sets defined in `pygip/datasets/`, so a single-seed run is directly comparable to the corresponding cell in the paper.

---

## Repository Layout

```text
GraphIP-Bench/
├── pygip/                              # Core library
│   ├── datasets/                       # 10 graph datasets + link-pred & graph-class loaders
│   ├── models/
│   │   ├── attack/                     # 12 extraction attacks
│   │   │   ├── mea/MEA.py              #   MEA0–MEA5 (incl. paper-faithful MEA2 Wu2022)
│   │   │   ├── AdvMEA.py               #   adversarial-query attack
│   │   │   ├── CEGA.py                 #   centrality + entropy
│   │   │   ├── Realistic.py            #   structure-aware pipeline
│   │   │   ├── DataFreeMEA.py          #   DFEA_I/II/III + real-graph proxies
│   │   │   └── linkpred_attacks.py     #   link-prediction adapters
│   │   ├── defense/                    # 12 defenses
│   │   │   ├── BackdoorWM.py · SurviveWM.py · RandomWM.py
│   │   │   ├── ImperceptibleWM.py · Integrity.py
│   │   │   ├── OutputPerturbation.py · PredictionRounding.py
│   │   │   ├── NonWatermarkDefenses.py (PRADA / AdaptMisinfo / GradRedir)
│   │   │   └── linkpred_defenses.py
│   │   └── nn/                         # GCN, GAT, GraphSAGE backbones (DGL + PyG)
│   ├── evaluation/                     # Watermark-survival evaluator
│   └── utils/                          # Metrics, hardware probing
├── examples/                           # Reproduction entry points
│   ├── run_joint_evaluation.py         # RQ5 joint attack × defense × watermark survival
│   ├── run_cross_arch_attacks.py
│   ├── run_cross_arch_node_class.py
│   ├── run_endpoint_ablation.py
│   └── run_link_pred_experiments.py
├── scripts/                            # Per-cell single-run scripts & post-processing
│   ├── run_rq1_single.py               # one (dataset, attack, regime, budget, seed) cell
│   ├── run_defense_single.py
│   ├── run_baseline_utility.py
│   ├── run_cross_arch.py · run_cross_arch_defended.py
│   ├── run_budget_ablation.py · run_query_split_ablation.py
│   ├── run_defense_hp_ablation.py · run_structure_analysis.py
│   ├── run_link_prediction.py · run_graph_class.py
│   ├── run_joint_new_defense.py · run_new_defense.py
│   └── merge_rq1_rq5_faithful_results.py
├── data/                               # Auto-populated dataset cache
├── outputs/                            # JSON-Lines run records
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Output Format

Every script writes one **JSON-Lines** record per run to `outputs/`. The schema is consistent across all tracks so records can be filtered and joined trivially:

```json
{
  "track": "RQ1",
  "dataset": "Cora",
  "attack": "MEA0",
  "regime": "both",
  "budget": 0.25,
  "seed": 0,
  "fidelity": 87.55,
  "accuracy": 79.7,
  "f1": 77.92,
  "train_target_time": 1.21,
  "query_target_time": 0.0019,
  "train_surrogate_time": 0.78,
  "total_time": 2.00,
  "peak_gpu_mem(GB)": 0.094,
  "status": "ok"
}
```

For the joint track, the record additionally carries `defense`, `defense_arch`, `surrogate_fidelity_to_defended`, and `wm_acc_on_surrogate` (watermark survival).

---

## Hardware

All reported results use a single **NVIDIA A100 80 GB** GPU with CUDA 12.1. The lightweight attacks and defenses run comfortably on a single GPU with at most 16 GB; only the `Realistic` pipeline and `ImperceptibleWM` representation-level optimization require the full 80 GB allocation. OGBN-Arxiv runs request 192 GB system memory.

---

## License

This project is released under the [MIT License](./LICENSE).

---

## Citation

If you use GraphIP-Bench, please cite:

```bibtex
@inproceedings{graphipbench2026,
  title={GraphIP-Bench: How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?},
  author={Anonymous Authors},
  booktitle={NeurIPS},
  year={2026}
}
```

---

<div align="center">

**GraphIP-Bench turns model extraction and ownership defense into a single reproducible benchmark — and exposes the gap that single-model evaluations miss.**

</div>
