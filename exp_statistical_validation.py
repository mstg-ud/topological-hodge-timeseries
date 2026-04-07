"""
exp_statistical_validation.py
==============================
Statistical validation experiments for the manuscript.
Performs:
  1. Bootstrap 95% CI for Hodge vs HR correlations (N=53 patient averages)
  2. Multiple regression: HR ~ baseline HRV features + H%
  3. LOO-CV (Leave-One-Out Cross-Validation) for overfitting assessment

Usage:
  python3 exp_statistical_validation.py --comparison_csv /path/to/comparison.csv

Requires:
  comparison.csv  — output of baseline_features.py merged with batch_pleth_analysis.py summary

Output files:
  fig_bootstrap_ci.png        — Bootstrap distribution plot (Figure S1)
  statistical_validation.json — Summary of all results
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

# ── Settings ──────────────────────────────────────────────────────────────────
N_BOOT        = 10000
SEED          = 42
CI_LEVEL      = 95
BASELINE_COLS = ['SDNN', 'RMSSD', 'pNN50', 'sample_entropy', 'permutation_entropy']


# ── 1. Paired bootstrap CI ────────────────────────────────────────────────────

def bootstrap_r(x, y, n_boot=N_BOOT, seed=SEED):
    """
    Paired bootstrap confidence interval for Pearson r.
    Resamples subject index jointly to preserve pairing.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Patient-averaged feature values.

    Returns
    -------
    np.ndarray, shape (n_boot,)
        Bootstrap distribution of Pearson r.
    """
    rng = np.random.default_rng(seed)
    n   = len(x)
    rs  = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = pearsonr(x[idx], y[idx])
        rs.append(r)
    return np.array(rs)


# ── 2. Multiple regression ────────────────────────────────────────────────────

def multiple_regression(pat, baseline_cols, h_col='H', target='HR'):
    """
    Compare two regression models:
      Model A: HR ~ baseline features (simultaneously)
      Model B: HR ~ baseline features + H%

    Returns dict with keys:
        r2_baseline, r2_full, delta_r2, F, p_F, coefs, features
    """
    available = [c for c in baseline_cols if c in pat.columns]
    y   = pat[target].values
    H   = pat[h_col].values

    X_b    = pat[available].values
    sc_b   = StandardScaler()
    X_b_s  = sc_b.fit_transform(X_b)
    lr_b   = LinearRegression().fit(X_b_s, y)
    r2_b   = r2_score(y, lr_b.predict(X_b_s))

    X_f    = np.column_stack([X_b, H])
    sc_f   = StandardScaler()
    X_f_s  = sc_f.fit_transform(X_f)
    lr_f   = LinearRegression().fit(X_f_s, y)
    r2_f   = r2_score(y, lr_f.predict(X_f_s))

    n, k_f = len(y), len(available) + 1
    ss_b   = np.sum((y - lr_b.predict(X_b_s))**2)
    ss_f   = np.sum((y - lr_f.predict(X_f_s))**2)
    F      = ((ss_b - ss_f) / 1) / (ss_f / (n - k_f - 1))
    p_F    = float(f_dist.sf(F, 1, n - k_f - 1))

    coefs  = dict(zip(available + ['H%'], lr_f.coef_))

    return dict(r2_baseline=float(r2_b), r2_full=float(r2_f),
                delta_r2=float(r2_f - r2_b),
                F=float(F), p_F=p_F,
                coefs=coefs, features=available)


# ── 3. LOO-CV ─────────────────────────────────────────────────────────────────

def loocv_regression(pat, baseline_cols, h_col='H', target='HR'):
    """
    Leave-One-Out Cross-Validation for Model A (baseline only) and
    Model B (baseline + H%).

    For each left-out subject, the scaler and regression are fitted on
    the remaining N-1 subjects only, then applied to the held-out subject.
    This prevents data leakage in the scaling step.

    Returns dict with keys:
        insample_r2_A, insample_r2_B,
        loo_r2_A, loo_r2_B,
        delta_insample, delta_loo,
        shrinkage_A, shrinkage_B,
        n_features_A, n_features_B, epv_B
    """
    available = [c for c in baseline_cols if c in pat.columns]
    y   = pat[target].values
    H   = pat[h_col].values
    X_b = pat[available].values
    X_f = np.column_stack([X_b, H])

    loo    = LeaveOneOut()
    pred_A = np.zeros(len(y))
    pred_B = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_b):
        # Model A — fit scaler on training fold only
        sc_A = StandardScaler().fit(X_b[train_idx])
        lr_A = LinearRegression().fit(
            sc_A.transform(X_b[train_idx]), y[train_idx])
        pred_A[test_idx] = lr_A.predict(sc_A.transform(X_b[test_idx]))

        # Model B — fit scaler on training fold only
        sc_B = StandardScaler().fit(X_f[train_idx])
        lr_B = LinearRegression().fit(
            sc_B.transform(X_f[train_idx]), y[train_idx])
        pred_B[test_idx] = lr_B.predict(sc_B.transform(X_f[test_idx]))

    ss_tot   = np.sum((y - y.mean())**2)
    loo_r2_A = float(1 - np.sum((y - pred_A)**2) / ss_tot)
    loo_r2_B = float(1 - np.sum((y - pred_B)**2) / ss_tot)

    # In-sample R² for shrinkage
    sc_A_full = StandardScaler().fit(X_b)
    lr_A_full = LinearRegression().fit(sc_A_full.transform(X_b), y)
    ins_r2_A  = float(r2_score(y, lr_A_full.predict(sc_A_full.transform(X_b))))

    sc_B_full = StandardScaler().fit(X_f)
    lr_B_full = LinearRegression().fit(sc_B_full.transform(X_f), y)
    ins_r2_B  = float(r2_score(y, lr_B_full.predict(sc_B_full.transform(X_f))))

    return dict(
        insample_r2_A=ins_r2_A,
        insample_r2_B=ins_r2_B,
        loo_r2_A=loo_r2_A,
        loo_r2_B=loo_r2_B,
        delta_insample=float(ins_r2_B - ins_r2_A),
        delta_loo=float(loo_r2_B - loo_r2_A),
        shrinkage_A=float(ins_r2_A - loo_r2_A),
        shrinkage_B=float(ins_r2_B - loo_r2_B),
        n_features_A=len(available),
        n_features_B=len(available) + 1,
        epv_B=float(len(y) / (len(available) + 1)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Statistical validation: Bootstrap CI, multiple regression, LOO-CV')
    parser.add_argument('--comparison_csv', required=True,
                        help='Path to comparison.csv (patient-epoch level)')
    parser.add_argument('--out_dir', default='.',
                        help='Output directory for figures and results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df  = pd.read_csv(args.comparison_csv)
    pat = df.groupby('pid').mean(numeric_only=True).reset_index()
    N   = len(pat)

    H  = pat['H'].values
    C  = pat['C'].values
    G  = pat['G'].values
    HR = pat['HR'].values

    print('=' * 65)
    print('Statistical Validation Experiments')
    print(f'  N = {N} patients  (Bootstrap N_boot = {N_BOOT}, seed = {SEED})')
    print('=' * 65)

    # ── 1. Bootstrap CI ────────────────────────────────────────────────────────
    print()
    print('1. Bootstrap 95% CI: Hodge components vs HR')
    print()
    results_boot = {}
    for name, vec in [('G%', G), ('C%', C), ('H%', H)]:
        r_obs, p_obs = pearsonr(vec, HR)
        boot         = bootstrap_r(vec, HR)
        lo, hi       = np.percentile(boot,
                           [(100-CI_LEVEL)/2, 100-(100-CI_LEVEL)/2])
        crosses      = bool(lo < 0 < hi)
        results_boot[name] = dict(r=float(r_obs), p=float(p_obs),
                                   ci_lo=float(lo), ci_hi=float(hi),
                                   boot=boot, crosses_zero=crosses)
        print(f'  {name}: r={r_obs:+.3f}  p={p_obs:.4f}  '
              f'95% CI=[{lo:.3f}, {hi:.3f}]  '
              f'{"CI crosses zero" if crosses else "CI does NOT cross zero"}')

    # ── 2. Multiple regression ─────────────────────────────────────────────────
    print()
    print('2. Multiple Regression: HR ~ baseline + H%')
    reg = multiple_regression(pat, BASELINE_COLS)
    print(f'  Model A (baseline only): R² = {reg["r2_baseline"]:.3f}')
    print(f'  Model B (+ H%):          R² = {reg["r2_full"]:.3f}  '
          f'ΔR² = {reg["delta_r2"]:+.3f}')
    print(f'  F-test for H% increment: F = {reg["F"]:.3f}  '
          f'p = {reg["p_F"]:.4f}  '
          f'{"significant" if reg["p_F"] < 0.05 else "not significant"}')
    print()
    print('  Standardized coefficients (Model B):')
    for feat, coef in reg['coefs'].items():
        print(f'    {feat:<30}: β = {coef:+.3f}')

    # ── 3. LOO-CV ──────────────────────────────────────────────────────────────
    print()
    print('3. Leave-One-Out Cross-Validation (LOO-CV)')
    loo = loocv_regression(pat, BASELINE_COLS)
    print(f'  EPV (events-per-variable, Model B): '
          f'{loo["epv_B"]:.1f}  [N={N} / k={loo["n_features_B"]}]')
    print()
    hdr = f'  {"Model":<25}  {"In-sample R²":>14}  {"LOO-CV R²":>12}  {"Shrinkage":>10}'
    print(hdr)
    print('  ' + '-' * 66)
    print(f'  {"A (baseline only)":<25}  '
          f'{loo["insample_r2_A"]:>14.3f}  '
          f'{loo["loo_r2_A"]:>12.3f}  '
          f'{loo["shrinkage_A"]:>10.3f}')
    print(f'  {"B (baseline + H%)":<25}  '
          f'{loo["insample_r2_B"]:>14.3f}  '
          f'{loo["loo_r2_B"]:>12.3f}  '
          f'{loo["shrinkage_B"]:>10.3f}')
    print(f'  {"ΔR²":<25}  '
          f'{loo["delta_insample"]:>+14.3f}  '
          f'{loo["delta_loo"]:>+12.3f}  '
          f'{"—":>10}')
    print()
    if loo['delta_loo'] > 0:
        verdict = 'H% adds genuine out-of-sample predictive value'
    else:
        verdict = 'H% increment does NOT generalize out-of-sample'
    print(f'  LOO-CV ΔR² = {loo["delta_loo"]:+.3f}  → {verdict}')
    if loo['delta_loo'] >= loo['delta_insample']:
        print('  Note: LOO-CV ΔR² ≥ in-sample ΔR² — no evidence of overfitting.')

    # ── 4. Bootstrap distribution figure ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor='#1a1a2e')
    fig.suptitle(
        f'Bootstrap distribution of Pearson r  '
        f'(N_boot={N_BOOT}, N={N} patients)',
        color='white', fontsize=10, y=0.99)

    colors = {'G%': '#4fc3f7', 'C%': '#ffb74d', 'H%': '#81c784'}
    for ax, (name, res) in zip(axes, results_boot.items()):
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, color='#222', lw=0.5)
        ax.hist(res['boot'], bins=50, color=colors[name], alpha=0.8,
                edgecolor='none', density=True)
        ax.axvline(res['r'], color='white', lw=2,
                   label=f'r = {res["r"]:+.3f}')
        ax.axvline(res['ci_lo'], color='#ff9999', lw=1.5, ls='--',
                   label=f'95% CI [{res["ci_lo"]:.3f}, {res["ci_hi"]:.3f}]')
        ax.axvline(res['ci_hi'], color='#ff9999', lw=1.5, ls='--')
        ax.axvline(0, color='#888', lw=0.8, ls=':')
        ax.set_xlabel('Pearson r', color='#aaa', fontsize=9)
        ax.set_ylabel('density', color='#aaa', fontsize=9)
        ax.set_title(f'{name} vs HR', color='white', fontsize=10, pad=4)
        ax.legend(framealpha=0.3, labelcolor='white', fontsize=7)

    plt.tight_layout()
    boot_png = os.path.join(args.out_dir, 'fig_bootstrap_ci.png')
    plt.savefig(boot_png, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    print()
    print(f'  Figure saved: {boot_png}')

    # ── 5. Save JSON summary ───────────────────────────────────────────────────
    summary = {
        'bootstrap': {
            name: {k: v for k, v in res.items() if k != 'boot'}
            for name, res in results_boot.items()
        },
        'multiple_regression': {
            k: v for k, v in reg.items() if k != 'coefs'
        },
        'multiple_regression_coefs': {
            k: float(v) for k, v in reg['coefs'].items()
        },
        'loocv': loo,
    }
    json_path = os.path.join(args.out_dir, 'statistical_validation.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Results saved: {json_path}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
