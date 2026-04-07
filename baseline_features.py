"""
Baseline feature extraction for PLETH epochs
Computes classical PPG/HRV features per epoch for comparison with Hodge G/C/H%.

Features computed per epoch:
  Time-domain HRV : SDNN, RMSSD, pNN50
  Frequency-domain HRV : LF power, HF power, LF/HF ratio
  PPG waveform   : spectral entropy, template correlation
  Nonlinear      : sample entropy, permutation entropy

Usage:
  python3 baseline_features.py --data_dir /path/to/data --out_dir ./pleth_results

Output:
  pleth_results/baseline_features.csv  — one row per epoch
  pleth_results/comparison.csv         — merged with Hodge summary.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy
warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
FS        = 125.0
EPOCH_SEC = 60
# IBI resampling rate for HRV frequency analysis
FS_IBI    = 4.0   # Hz
LF_BAND   = (0.04, 0.15)
HF_BAND   = (0.15, 0.40)


# ── Peak detection → IBI ──────────────────────────────────────────────────────

def detect_peaks_ibi(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect PPG peaks and return IBI array (seconds).
    Uses scipy find_peaks with physiological constraints.
    """
    min_dist = int(0.4 * fs)   # minimum 150 ms between beats (400 bpm max)
    peaks, _ = find_peaks(seg, distance=min_dist,
                          prominence=0.01 * (seg.max() - seg.min()))
    if len(peaks) < 2:
        return np.array([])
    ibi = np.diff(peaks) / fs
    # physiological filter: 0.3s-2.0s (30-200 bpm)
    ibi = ibi[(ibi >= 0.3) & (ibi <= 2.0)]
    return ibi


# ── Time-domain HRV ───────────────────────────────────────────────────────────

def hrv_time_domain(ibi: np.ndarray) -> dict:
    if len(ibi) < 3:
        return dict(SDNN=np.nan, RMSSD=np.nan, pNN50=np.nan, mean_IBI=np.nan)
    diff_ibi = np.diff(ibi)
    pnn50 = 100.0 * np.sum(np.abs(diff_ibi) > 0.05) / len(diff_ibi)
    return dict(
        SDNN     = float(np.std(ibi, ddof=1)),
        RMSSD    = float(np.sqrt(np.mean(diff_ibi**2))),
        pNN50    = float(pnn50),
        mean_IBI = float(np.mean(ibi)),
    )


# ── Frequency-domain HRV ─────────────────────────────────────────────────────

def hrv_frequency_domain(ibi: np.ndarray) -> dict:
    if len(ibi) < 8:
        return dict(LF=np.nan, HF=np.nan, LF_HF=np.nan, total_HRV_power=np.nan)
    # Resample IBI to uniform grid
    t_ibi = np.cumsum(ibi) - ibi[0]
    t_grid = np.arange(0, t_ibi[-1], 1.0 / FS_IBI)
    if len(t_grid) < 8:
        return dict(LF=np.nan, HF=np.nan, LF_HF=np.nan, total_HRV_power=np.nan)
    ibi_resampled = np.interp(t_grid, t_ibi, ibi)
    ibi_resampled -= ibi_resampled.mean()

    # Welch PSD
    freqs, psd = scipy_signal.welch(ibi_resampled, fs=FS_IBI,
                                     nperseg=min(len(ibi_resampled), int(FS_IBI*30)))
    df_freq = freqs[1] - freqs[0]

    def band_power(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.sum(psd[mask]) * df_freq) if mask.any() else np.nan

    lf  = band_power(*LF_BAND)
    hf  = band_power(*HF_BAND)
    tot = band_power(0.003, 0.40)
    lf_hf = lf / hf if (hf is not None and hf > 0) else np.nan
    return dict(LF=lf, HF=hf, LF_HF=lf_hf, total_HRV_power=tot)


# ── PPG waveform features ─────────────────────────────────────────────────────

def ppg_spectral_entropy(seg: np.ndarray, fs: float) -> float:
    """Shannon entropy of the normalized PSD of the PPG waveform."""
    _, psd = scipy_signal.welch(seg, fs=fs, nperseg=min(len(seg), 256))
    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 0]
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def ppg_template_correlation(seg: np.ndarray, fs: float) -> float:
    """
    Mean pairwise correlation between individual beat templates.
    Measures morphological consistency of PPG beats within the epoch.
    """
    peaks, _ = find_peaks(seg, distance=int(0.4*fs),
                          prominence=0.01*(seg.max()-seg.min()))
    if len(peaks) < 3:
        return np.nan
    # extract fixed-length templates centered on each peak
    half = int(0.4 * fs)
    templates = []
    for pk in peaks:
        if pk - half >= 0 and pk + half < len(seg):
            t = seg[pk-half:pk+half]
            t = (t - t.mean()) / (t.std() + 1e-10)
            templates.append(t)
    if len(templates) < 2:
        return np.nan
    T = np.array(templates)
    # mean correlation (upper triangle)
    n = len(T)
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            corrs.append(float(np.corrcoef(T[i], T[j])[0,1]))
    return float(np.mean(corrs))


# ── Nonlinear features ────────────────────────────────────────────────────────

def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample entropy of a time series.
    r = r_factor * std(x)
    """
    x = np.asarray(x, dtype=float)
    r = r_factor * x.std()
    N = len(x)
    if N < 2*m+1 or r <= 0:
        return np.nan

    def count_matches(m_):
        count = 0
        for i in range(N - m_):
            template = x[i:i+m_]
            for j in range(i+1, N - m_):
                if np.max(np.abs(template - x[j:j+m_])) < r:
                    count += 1
        return count

    Bm = count_matches(m)
    Am = count_matches(m + 1)
    if Bm == 0:
        return np.nan
    return float(-np.log(Am / Bm)) if Am > 0 else np.inf


def permutation_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy of a time series.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < m:
        return np.nan
    # embed
    n_patterns = N - (m-1)*delay
    patterns = np.array([x[i:i+(m-1)*delay+1:delay] for i in range(n_patterns)])
    orders = np.argsort(patterns, axis=1)
    # convert to int keys
    keys = [tuple(o) for o in orders]
    from collections import Counter
    counts = Counter(keys)
    total  = sum(counts.values())
    probs  = np.array([c/total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs)))


# ── Single epoch processing ───────────────────────────────────────────────────

def compute_epoch_features(seg: np.ndarray, pid: str, ep: int, t0: int) -> dict:
    ibi    = detect_peaks_ibi(seg, FS)
    td     = hrv_time_domain(ibi)
    fd     = hrv_frequency_domain(ibi)
    sp_ent = ppg_spectral_entropy(seg, FS)
    tc     = ppg_template_correlation(seg, FS)
    # sample entropy on downsampled signal (speed)
    seg_ds = seg[::4]   # 125Hz → ~31Hz
    samp_e = sample_entropy(seg_ds[:200])
    perm_e = permutation_entropy(seg_ds, m=3)

    return dict(pid=pid, ep=ep, t0=t0,
                n_ibi=len(ibi),
                **td, **fd,
                spectral_entropy=sp_ent,
                template_corr=tc,
                sample_entropy=samp_e,
                permutation_entropy=perm_e)


# ── File discovery (same as batch script) ─────────────────────────────────────

def find_patient_files(data_dir: Path) -> list[dict]:
    sig_files = sorted(data_dir.glob('*_Signals.csv'))
    patients  = []
    for sig in sig_files:
        prefix = sig.name.replace('_Signals.csv', '')
        num    = sig.parent / (prefix + '_Numerics.csv')
        if num.exists():
            m   = re.search(r'(bidmc_\d+)', prefix)
            pid = m.group(1) if m else prefix
            patients.append(dict(pid=pid, signals_path=sig, numerics_path=num))
    return patients


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  required=True)
    parser.add_argument('--out_dir',   default='./pleth_results')
    parser.add_argument('--hodge_csv', default=None,
                        help='Path to summary.csv from batch_pleth_analysis.py')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / 'baseline_features.csv'

    if feat_path.exists() and not args.force:
        print(f'Loading existing features: {feat_path}')
        df_feat = pd.read_csv(feat_path)
    else:
        patients = find_patient_files(data_dir)
        print(f'Found {len(patients)} patients')
        epoch_len = int(FS * EPOCH_SEC)
        all_rows  = []

        for pat in patients:
            pid = pat['pid']
            try:
                df_sig = pd.read_csv(pat['signals_path'])
                df_sig.columns = df_sig.columns.str.strip()
                pleth = df_sig['PLETH'].values
            except Exception as e:
                print(f'  {pid}: ERROR — {e}')
                continue

            n_epochs = len(pleth) // epoch_len
            print(f'  {pid}: {n_epochs} epochs', flush=True)

            for ep in range(n_epochs):
                t0  = ep * EPOCH_SEC
                seg = pleth[ep * epoch_len:(ep+1) * epoch_len]
                row = compute_epoch_features(seg, pid, ep, t0)
                all_rows.append(row)

        df_feat = pd.DataFrame(all_rows)
        df_feat.to_csv(feat_path, index=False)
        print(f'Saved: {feat_path}')

    print()
    print(f'Baseline features: {df_feat.shape}')
    print(df_feat.describe().round(3).to_string())

    # ── Merge with Hodge summary ──────────────────────────────────────────────
    hodge_path = args.hodge_csv or str(out_dir / 'summary.csv')
    if not Path(hodge_path).exists():
        print(f'\nHodge summary not found: {hodge_path}')
        print('Run batch_pleth_analysis.py first, then rerun with --hodge_csv')
        return

    df_hodge = pd.read_csv(hodge_path)
    df_all   = pd.merge(df_hodge, df_feat, on=['pid','ep','t0'], how='inner')
    comp_path = out_dir / 'comparison.csv'
    df_all.to_csv(comp_path, index=False)
    print(f'\nMerged dataset: {df_all.shape}  → {comp_path}')

    # ── Correlation comparison ────────────────────────────────────────────────
    from scipy.stats import pearsonr

    hodge_cols   = ['G', 'C', 'H']
    baseline_cols = ['SDNN','RMSSD','pNN50','LF','HF','LF_HF',
                     'spectral_entropy','template_corr',
                     'sample_entropy','permutation_entropy']
    numerics_cols = ['HR','SpO2','RESP']

    print()
    print('=== Correlation with HR (epoch level, N=all epochs) ===')
    print(f'  {"feature":<25}  {"r":>7}  {"p":>8}  sig')
    print('  '+'-'*50)

    target = df_all['HR'].dropna()
    for col in hodge_cols + baseline_cols:
        if col not in df_all.columns:
            continue
        vals = df_all[col]
        mask = target.notna() & vals.notna()
        if mask.sum() < 10:
            continue
        r, p = pearsonr(vals[mask], target[mask])
        sig  = '***' if p<0.001 else ('** ' if p<0.01 else ('*  ' if p<0.05 else '   '))
        tag  = ' ← Hodge' if col in hodge_cols else ''
        print(f'  {col:<25}  {r:+7.3f}  {p:8.4f}  {sig}{tag}')

    print()
    print('=== Patient-average correlations (N=patients) ===')
    pat_avg = df_all.groupby('pid')[hodge_cols + baseline_cols + numerics_cols].mean()
    print(f'  {"feature":<25}  {"HR r":>7}  {"SpO2 r":>8}  {"RESP r":>8}')
    print('  '+'-'*58)
    for col in hodge_cols + baseline_cols:
        if col not in pat_avg.columns:
            continue
        row_str = f'  {col:<25}'
        for num in numerics_cols:
            mask = pat_avg[col].notna() & pat_avg[num].notna()
            if mask.sum() < 5:
                row_str += f'  {"nan":>7}'
                continue
            r, p = pearsonr(pat_avg[col][mask], pat_avg[num][mask])
            sig  = '***' if p<0.001 else ('**' if p<0.01 else ('* ' if p<0.05 else '  '))
            row_str += f'  {r:+6.3f}{sig}'
        tag = ' ← Hodge' if col in hodge_cols else ''
        print(row_str + tag)


if __name__ == '__main__':
    main()
