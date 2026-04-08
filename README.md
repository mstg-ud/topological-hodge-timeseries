# Topological and Geometric Analysis of Time-Series Complexity Dynamics via Discrete Hodge Decomposition
[![DOI](https://zenodo.org/badge/1203992682.svg)](https://doi.org/10.5281/zenodo.19456477)

This repository contains the implementation and experimental scripts accompanying the paper:

> **Topological and Geometric Analysis of Time-Series Complexity Dynamics via Discrete Hodge Decomposition**  
> Masatsugu Ueda, Independent Researcher, 2026  
> Preprint: https://www.preprints.org/manuscript/202604.0554/v1

---

## Overview

We propose a framework that encodes the local complexity dynamics of a time series as a simplicial complex and applies discrete Hodge decomposition to a KL-antisymmetric flow on that complex. The pipeline produces three scalar summary statistics — **G%** (gradient), **C%** (curl), and **H%** (harmonic) — that characterize the global topological and geometric structure of local complexity transitions.

```
x(t)  →  sliding window  →  Hann + FFT  →  local PSD {P_i}
      →  Wasserstein distance matrix D
      →  Vietoris-Rips complex K
      →  KL-antisymmetric edge flow f(i,j)
      →  Discrete Hodge decomposition  →  G%, C%, H%
```

---

## Repository Structure

```
.
├── pipeline_psd_wass_rips.py      # Core pipeline (Steps 1–6)
├── exp_4signal_pipelineB.py       # Synthetic signal experiments (Section 4)
├── exp_ppg_characterization.py    # PPG characterization: W1 dist + PCA-2D (Section 5.1)
├── batch_pleth_analysis.py        # PPG batch processing, 53 recordings (Section 5)
├── baseline_features.py           # Classical HRV/PPG feature extraction (Section 5.4)
├── exp_statistical_validation.py  # Bootstrap CI + multiple regression + LOO-CV (Appendix A)
├── requirements.txt
└── README.md
```

---

## Requirements

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list of dependencies.

---

## Data

The PPG experiments (Section 5) use the **BIDMC PPG and Respiration Dataset**:

> Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220, 2000.  
> Available at: https://physionet.org/content/bidmc/1.0.0/

Download the dataset from PhysioNet and place the `bidmc_*_Signals.csv` files in a local directory. Pass the directory path to the relevant scripts via `--ppg_dir`.

---

## Usage

### 1. Synthetic signal experiments (Section 4)

```bash
python3 exp_4signal_pipelineB.py
```

Generates `exp_4signal_comparison.png` and prints G%/C%/H% results for four synthetic signals (pure_sin, comm, incomm, noisy) across Rips radius quantiles q = 0.01 … 0.20.

Distance matrices are cached as `_cache_{name}_D.npy` / `_cache_{name}_P.npy` on first run.

### 2. PPG characterization: W₁ distributions and PCA-2D trajectories (Section 5.1)

```bash
python3 exp_ppg_characterization.py \
    --ppg_dir /path/to/bidmc_data \
    --out_dir ./figures \
    --cache_dir ./cache
```

Generates:
- `fig2_wass_dist_comparison.png` — W₁ distance distributions (CDF + histogram)
- `fig3_trajectory_pca2d.png` — {P_i} trajectories in PCA-2D space

### 3. PPG batch processing (Section 5.2–5.4)

```bash
python3 batch_pleth_analysis.py \
    --ppg_dir /path/to/bidmc_data \
    --out_dir ./results
```

Processes all 53 BIDMC recordings (8 epochs × 60 s each) and saves per-patient Hodge decomposition results.

### 4. Classical HRV/PPG feature extraction (Section 5.4)

```bash
python3 baseline_features.py \
    --ppg_dir /path/to/bidmc_data \
    --out_dir ./results
```

Computes 10 classical HRV and PPG features per epoch for comparison with Hodge components.

### 5. Statistical validation: Bootstrap CI, multiple regression, LOO-CV (Appendix A)

```bash
python3 exp_statistical_validation.py \
    --comparison_csv ./results/comparison.csv \
    --out_dir ./figures
```

Generates:
- `fig_bootstrap_ci.png` — Bootstrap distributions (Figure S1)
- `statistical_validation.json` — Summary of all results

---

## Parameter Notes

Key pipeline parameters (see Section 3.7 of the paper):

| Parameter | Synthetic | PPG |
|---|---|---|
| Sampling frequency $f_s$ | 50 Hz | 125 Hz |
| Window length $L$ | 64 samples | 250 samples |
| Overlap rate $r$ | 0.875 | 0.875 |
| Rips radius quantile $q$ | 0.05, 0.10 | 0.01 |
| Triangle limit $|T|_{\max}$ | 8,000 | 8,000 |

The Rips radius quantile $q = 0.01$ for PPG (versus $q \in \{0.05, 0.10\}$ for synthetic signals) is required to keep the triangle count below 8,000; PPG signals produce a denser $W_1$ distance matrix that causes $|T|$ to grow much more rapidly with $\varepsilon$.

---

## Key Design Decisions

- **Trivial zero-flow regime**: When $\|\mathbf{f}\|^2 < 10^{-15}$, G%, C%, H% are reported as **undefined** (not 0%). This occurs for pure sine waves where all local complexity states are identical.
- **Wasserstein distance** is used as the ground metric for the Rips complex (preserves physical frequency units).
- **KL-antisymmetric flow** $f(i,j) = D_\text{KL}(P_i \| P_j) - D_\text{KL}(P_j \| P_i)$ is antisymmetric by construction.
- All three core components (Rips complex, Wasserstein distance, Hodge decomposition) are implemented from scratch using NumPy and SciPy only — no specialized TDA libraries required.

---

## Citation

If you use this code, please cite:

```bibtex
@article{ueda2026hodge,
  title   = {Topological and Geometric Analysis of Time-Series Complexity
             Dynamics via Discrete Hodge Decomposition},
  author  = {Ueda, Masatsugu},
  year    = {2026},
  note    = {Preprint. https://www.preprints.org/manuscript/202604.0554/v1}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

Pipeline implementation, manuscript drafting, and iterative revision were assisted by Claude (Anthropic). Critical review of successive drafts was conducted with Gemini (Google DeepMind). All experimental designs, interpretations, and conclusions are the sole responsibility of the author.
