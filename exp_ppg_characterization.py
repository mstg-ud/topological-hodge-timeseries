"""
exp_ppg_characterization.py
============================
PPG signal characterization: commensurable vs incommensurable.
Generates:
  fig2_wass_dist_comparison.png  — W₁ distance distribution (CDF + histogram)
  fig3_trajectory_pca2d.png      — {P_i} trajectory in PCA-2D space

Usage:
  python3 exp_ppg_characterization.py \\
      --ppg_dir /path/to/bidmc_data \\
      --out_dir ./figures

Requires:
  pipeline_psd_wass_rips.py  (same directory)
  bidmc_*_Signals.csv  (BIDMC dataset from PhysioNet)

Synthetic reference signals are generated internally at fs=125Hz, L=250.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_psd_wass_rips import local_psd, wasserstein_matrix

# ── Settings ──────────────────────────────────────────────────────────────────
FS         = 125.0
L          = 250
R          = 0.875
EPOCH_SEC  = 60
F_HR       = 1.5          # reference heart rate frequency [Hz]
F_COMM     = 3.0          # commensurable second frequency [Hz]
F_INC      = F_HR * np.sqrt(2)  # incommensurable second frequency [Hz]

# Default PPG file patterns (bidmc_01 ~ 03 as representatives)
DEFAULT_PPG = [
    ('PPG_01', 'bidmc_01_Signals.csv'),
    ('PPG_02', 'bidmc_02_Signals.csv'),
    ('PPG_03', 'bidmc_03_Signals.csv'),
]

COLORS = {
    'comm':   '#4fc3f7',
    'incomm': '#ffb74d',
    'noisy':  '#ef9a9a',
    'PPG_01': '#81c784',
    'PPG_02': '#a5d6a7',
    'PPG_03': '#c8e6c9',
}
STYLES = {
    'comm': '-', 'incomm': '-', 'noisy': '-',
    'PPG_01': '--', 'PPG_02': '--', 'PPG_03': '--',
}


def find_ppg_file(ppg_dir, pattern):
    """Find a file matching *pattern* in ppg_dir."""
    import glob
    matches = glob.glob(os.path.join(ppg_dir, f'*{pattern}'))
    return matches[0] if matches else None


def load_ppg_epoch(path, epoch_sec=EPOCH_SEC, fs=FS):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df['PLETH'].values[:int(fs * epoch_sec)]


def compute_psd_and_wass(x, cache_prefix=None):
    """Compute {P_i} and W₁ distance matrix, with optional caching."""
    if cache_prefix:
        cd = f'{cache_prefix}_D.npy'
        cp = f'{cache_prefix}_P.npy'
        if os.path.exists(cd) and os.path.exists(cp):
            return np.load(cp), np.load(cd)
    freqs, P = local_psd(x, FS, L, R)
    D = wasserstein_matrix(P, freqs)
    if cache_prefix:
        np.save(cd, D)
        np.save(cp, P)
    return P, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppg_dir', required=True,
                        help='Directory containing bidmc_*_Signals.csv files')
    parser.add_argument('--out_dir', default='.',
                        help='Output directory for figures')
    parser.add_argument('--cache_dir', default=None,
                        help='Directory for W₁ matrix cache (default: out_dir)')
    args = parser.parse_args()

    out_dir   = args.out_dir
    cache_dir = args.cache_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print('=' * 65)
    print('PPG Characterization: comm vs incomm vs PPG')
    print(f'  fs={FS}Hz  L={L}  r={R}  epoch={EPOCH_SEC}s')
    print(f'  f_hr={F_HR}Hz  f_comm={F_COMM}Hz  f_incomm={F_INC:.3f}Hz')
    print('=' * 65)

    # ── Generate synthetic signals ─────────────────────────────────────────────
    t = np.arange(0, EPOCH_SEC, 1.0 / FS)
    rng = np.random.default_rng(42)
    signals = {
        'comm':   np.sin(2*np.pi*F_HR*t) + np.sin(2*np.pi*F_COMM*t),
        'incomm': np.sin(2*np.pi*F_HR*t) + np.sin(2*np.pi*F_INC*t),
        'noisy':  np.sin(2*np.pi*F_HR*t) + 0.3 * rng.standard_normal(len(t)),
    }

    # ── Load PPG epochs ────────────────────────────────────────────────────────
    for label, pattern in DEFAULT_PPG:
        path = find_ppg_file(args.ppg_dir, pattern)
        if path:
            signals[label] = load_ppg_epoch(path)
            print(f'  Loaded {label}: {path}')
        else:
            print(f'  WARNING: {pattern} not found in {args.ppg_dir}')

    # ── Compute {P_i} and W₁ for all signals ──────────────────────────────────
    print()
    print('Computing {P_i} and W₁ distance matrices...')
    data = {}
    for name, x in signals.items():
        cache_pfx = os.path.join(cache_dir, f'_ppg_char_{name}')
        P, D = compute_psd_and_wass(x, cache_prefix=cache_pfx)
        upper = D[np.triu_indices(len(P), k=1)]
        data[name] = {'P': P, 'D': D, 'upper': upper, 'N': len(P)}
        print(f'  {name:<12}: N={len(P)}  W₁_mean={upper.mean():.4f}  '
              f'W₁_max={upper.max():.4f}')

    # ── KS test ────────────────────────────────────────────────────────────────
    print()
    print('KS test: each signal vs comm / vs incomm')
    print(f'  {"signal":<12}  {"D vs comm":>10}  {"D vs incomm":>12}  closer to')
    print('  '+'-'*50)
    for name in signals:
        if name in ('comm', 'incomm'):
            continue
        D_comm,  _ = ks_2samp(data[name]['upper'], data['comm']['upper'])
        D_incomm,_ = ks_2samp(data[name]['upper'], data['incomm']['upper'])
        closer = 'incomm' if D_comm > D_incomm else 'comm'
        print(f'  {name:<12}  {D_comm:>10.4f}  {D_incomm:>12.4f}  {closer}')

    # ── Figure 2: W₁ distribution ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
    fig.suptitle(
        'Wasserstein distance distribution: comm vs incomm vs PPG\n'
        f'(fs={FS}Hz, L={L}, r={R}, Hann window, {EPOCH_SEC}s epoch)',
        color='white', fontsize=11, y=0.99)

    # CDF
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.tick_params(colors='#888', labelsize=8)
    ax.grid(True, color='#222', lw=0.5)
    for name, d in data.items():
        u = np.sort(d['upper'])
        cdf = np.arange(1, len(u)+1) / len(u)
        ax.plot(u, cdf, color=COLORS.get(name, '#aaa'),
                lw=2, ls=STYLES.get(name, '-'),
                label=f'{name} (μ={d["upper"].mean():.4f})')
    ax.set_xlabel('W₁ distance', color='#aaa', fontsize=9)
    ax.set_ylabel('CDF', color='#aaa', fontsize=9)
    ax.set_title('Cumulative distribution of pairwise W₁ distances',
                 color='white', fontsize=9, pad=4)
    ax.legend(framealpha=0.3, labelcolor='white', fontsize=7)
    ax.set_xlim(-0.01, 0.6)

    # Histogram
    ax = axes[1]
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.tick_params(colors='#888', labelsize=8)
    ax.grid(True, color='#222', lw=0.5)
    bins = np.linspace(0, 0.6, 60)
    for name, d in data.items():
        clipped = d['upper'][d['upper'] <= 0.6]
        ax.hist(clipped, bins=bins, density=True, histtype='step',
                color=COLORS.get(name, '#aaa'), lw=2,
                ls=STYLES.get(name, '-'), label=name)
    ax.set_xlabel('W₁ distance', color='#aaa', fontsize=9)
    ax.set_ylabel('density', color='#aaa', fontsize=9)
    ax.set_title('Distribution of pairwise W₁ distances (clipped at 0.6)',
                 color='white', fontsize=9, pad=4)
    ax.legend(framealpha=0.3, labelcolor='white', fontsize=7)

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, 'fig2_wass_dist_comparison.png')
    plt.savefig(fig2_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    print(f'\n  Figure 2 saved: {fig2_path}')

    # ── Figure 3: PCA-2D trajectory ────────────────────────────────────────────
    n_signals = len(signals)
    n_rows    = 2
    n_cols    = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(16, 11), facecolor='#1a1a2e')
    fig.suptitle(
        '{P_i} trajectory in PCA-2D space  (time-colored)\n'
        f'fs={FS}Hz, L={L}, r={R}, Hann window, {EPOCH_SEC}s epoch',
        color='white', fontsize=11, y=0.99)

    all_names = list(signals.keys())
    for idx, name in enumerate(all_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.tick_params(colors='#888', labelsize=7)
        ax.grid(True, color='#222', lw=0.5)

        P   = data[name]['P']
        N   = len(P)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(P)
        ev     = pca.explained_variance_ratio_
        spread = np.std(coords, axis=0).mean()

        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=np.arange(N), cmap='plasma',
                        s=8, alpha=0.8, edgecolors='none')
        ax.plot(coords[:, 0], coords[:, 1],
                color='white', lw=0.3, alpha=0.2)

        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', color='#aaa', fontsize=8)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', color='#aaa', fontsize=8)
        ax.set_title(f'{name}\nspread={spread:.4f}  N={N}',
                     color='white', fontsize=9, pad=4)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(colors='#888', labelsize=6)
        cbar.set_label('window index (time →)', color='#888', fontsize=7)

    # Hide unused subplots
    for idx in range(len(signals), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig3_path = os.path.join(out_dir, 'fig3_trajectory_pca2d.png')
    plt.savefig(fig3_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    print(f'  Figure 3 saved: {fig3_path}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
