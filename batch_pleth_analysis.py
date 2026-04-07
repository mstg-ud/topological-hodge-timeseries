"""
Batch PLETH epoch analysis — Pipeline B
  PSD + Wasserstein + Rips + KL Hodge

Usage:
  python3 batch_pleth_analysis.py --data_dir /path/to/data --out_dir ./results

Input files expected (per patient):
  {prefix}_Signals.csv   — contains PLETH column, fs=125Hz
  {prefix}_Numerics.csv  — contains HR, SpO2, RESP columns, fs=1Hz
  {prefix}_Fix.txt       — (optional) metadata

Output per patient:
  results/{pid}_epochs.json   — epoch-level G/C/H/numerics
  results/{pid}_cache_D.npy   — Wasserstein distance matrix cache (per epoch)

Output summary:
  results/summary.csv         — all patients × all epochs in one table

Features:
  - Skips already processed patients (resume-friendly)
  - Caches Wasserstein matrices per epoch to avoid recomputation
  - Logs warnings when signal variation is near leakage floor
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Pipeline import ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_psd_wass_rips import (local_psd, wasserstein_matrix,
                                     build_rips, kl_flow, hodge_decompose,
                                     n_components)

# ── Constants ─────────────────────────────────────────────────────────────────
FS         = 125.0   # Hz
L          = 250     # window length (samples)
R          = 0.875   # overlap rate
S          = max(1, int(L * (1 - R)))   # stride = 31 samples
EPOCH_SEC  = 60      # seconds per epoch
FLOOR      = 0.0047  # leakage floor (pure sine 1.2Hz, this setting)
MAX_TRIS   = 8000    # triangle limit
PCT_LIST   = [1.0, 0.5, 0.3]   # epsilon percentile fallback sequence


# ── File discovery ────────────────────────────────────────────────────────────

def find_patient_files(data_dir: Path) -> list[dict]:
    """
    Scan data_dir for bidmc-style file pairs.
    Matches: *_Signals.csv + *_Numerics.csv with the same prefix.
    Returns list of dicts with keys: pid, signals_path, numerics_path
    """
    sig_files = sorted(data_dir.glob('*_Signals.csv'))
    patients  = []
    for sig in sig_files:
        prefix = sig.name.replace('_Signals.csv', '')
        num    = sig.parent / (prefix + '_Numerics.csv')
        if num.exists():
            # extract patient id from prefix (e.g. "bidmc_01")
            m = re.search(r'(bidmc_\d+)', prefix)
            pid = m.group(1) if m else prefix
            patients.append(dict(pid=pid, signals_path=sig, numerics_path=num))
    return patients


# ── Single epoch processing ───────────────────────────────────────────────────

def process_epoch(seg: np.ndarray, t0: int,
                  cache_dir: Path, pid: str, ep: int) -> dict:
    """
    Process one 60-second PLETH epoch.
    Wasserstein matrix is cached to disk.
    """
    epoch_len = int(FS * EPOCH_SEC)
    assert len(seg) == epoch_len, f'Expected {epoch_len} samples, got {len(seg)}'

    # {P_i}
    freqs, P_arr = local_psd(seg, FS, L, R)
    N = len(P_arr)

    # Wasserstein matrix (cached)
    cache_path = cache_dir / f'{pid}_ep{ep:02d}_D.npy'
    if cache_path.exists():
        D = np.load(cache_path)
    else:
        D = wasserstein_matrix(P_arr, freqs)
        np.save(cache_path, D)

    upper = D[np.triu_indices(N, k=1)]
    wmax  = float(upper.max())
    wmean = float(upper.mean())
    floor_ratio = wmax / FLOOR

    # epsilon selection (fallback to smaller pct if too many triangles)
    eps   = None
    edges = []
    tris  = []
    used_pct = None
    for pct in PCT_LIST:
        eps_try   = float(np.percentile(upper, pct))
        e_try, t_try = build_rips(D, eps_try)
        if len(t_try) <= MAX_TRIS:
            eps, edges, tris, used_pct = eps_try, e_try, t_try, pct
            break
    if eps is None:
        return dict(ep=ep, t0=t0, N=N, wmax=wmax, wmean=wmean,
                    floor_ratio=floor_ratio, eps=None, used_pct=None,
                    Ne=None, Nt=None, G=None, C=None, H=None,
                    E=None, b1=None, nc=None, trivial=None,
                    warn='ALL_LARGE')

    # Hodge decomposition
    f_flow              = kl_flow(P_arr, edges)
    G, C, H, E, b1, tv = hodge_decompose(N, edges, tris, f_flow)
    nc                  = n_components(N, edges)

    warn = ''
    if floor_ratio < 2:
        warn = 'NEAR_FLOOR'

    return dict(ep=ep, t0=t0, N=N, wmax=wmax, wmean=wmean,
                floor_ratio=floor_ratio, eps=eps, used_pct=used_pct,
                Ne=len(edges), Nt=len(tris),
                G=G, C=C, H=H, E=E, b1=b1, nc=nc, trivial=tv, warn=warn)


# ── Single patient processing ─────────────────────────────────────────────────

def process_patient(pid: str, signals_path: Path, numerics_path: Path,
                    out_dir: Path, cache_dir: Path,
                    force: bool = False) -> list[dict] | None:
    """
    Process all epochs for one patient.
    Skips if result file already exists (unless force=True).
    """
    result_path = out_dir / f'{pid}_epochs.json'
    if result_path.exists() and not force:
        print(f'  {pid}: already processed — skipping (use --force to rerun)')
        with open(result_path) as f:
            return json.load(f)

    # Load signals
    try:
        df_sig = pd.read_csv(signals_path)
        df_sig.columns = df_sig.columns.str.strip()
        pleth = df_sig['PLETH'].values
    except Exception as e:
        print(f'  {pid}: ERROR loading signals — {e}')
        return None

    # Load numerics
    try:
        df_num = pd.read_csv(numerics_path)
        df_num.columns = df_num.columns.str.strip()
    except Exception as e:
        print(f'  {pid}: ERROR loading numerics — {e}')
        return None

    epoch_len = int(FS * EPOCH_SEC)
    n_epochs  = len(pleth) // epoch_len

    if n_epochs == 0:
        print(f'  {pid}: too short ({len(pleth)} samples) — skipping')
        return None

    print(f'  {pid}: {len(pleth)/FS:.0f}s → {n_epochs} epochs', flush=True)

    results = []
    for ep in range(n_epochs):
        t0  = ep * EPOCH_SEC
        seg = pleth[ep * epoch_len:(ep + 1) * epoch_len]

        r = process_epoch(seg, t0, cache_dir, pid, ep)

        # attach numerics
        seg_num  = df_num[(df_num['Time [s]'] >= t0) &
                          (df_num['Time [s]'] <  t0 + EPOCH_SEC)]
        r['HR']   = float(seg_num['HR'].mean())   if 'HR'   in seg_num else None
        r['SpO2'] = float(seg_num['SpO2'].mean()) if 'SpO2' in seg_num else None
        r['RESP'] = float(seg_num['RESP'].mean()) if 'RESP' in seg_num else None

        results.append(r)

        # inline progress
        gch = (f'G={r["G"]:5.1f}% C={r["C"]:5.1f}% H={r["H"]:5.1f}%'
               if r['G'] is not None else f'WARN:{r["warn"]}')
        print(f'    ep{ep}: {gch}  '
              f'HR={r["HR"]:.1f} SpO2={r["SpO2"]:.2f} RESP={r["RESP"]:.1f}  '
              f'[floor×{r["floor_ratio"]:.1f}]', flush=True)

    # save
    def _serial(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, bool): return v
        return v

    with open(result_path, 'w') as f:
        json.dump([{k: _serial(v) for k, v in r.items()} for r in results],
                  f, indent=2)

    return results


# ── Summary CSV ───────────────────────────────────────────────────────────────

def build_summary(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(out_dir.glob('*_epochs.json')):
        pid = path.stem.replace('_epochs', '')
        with open(path) as f:
            epochs = json.load(f)
        for r in epochs:
            row = {'pid': pid}
            row.update(r)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'summary.csv', index=False)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch PLETH epoch analysis — Pipeline B')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing *_Signals.csv + *_Numerics.csv')
    parser.add_argument('--out_dir', default='./pleth_results',
                        help='Output directory (default: ./pleth_results)')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess even if result file exists')
    parser.add_argument('--pid', default=None,
                        help='Process only this patient ID (e.g. bidmc_01)')
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    cache_dir = out_dir / '_cache'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    print('=' * 65)
    print('Batch PLETH analysis — Pipeline B')
    print(f'  fs={FS}Hz  L={L}  r={R}  S={S}')
    print(f'  epoch={EPOCH_SEC}s  max_tris={MAX_TRIS}')
    print(f'  leakage floor = {FLOOR}')
    print(f'  data_dir : {data_dir}')
    print(f'  out_dir  : {out_dir}')
    print('=' * 65)

    patients = find_patient_files(data_dir)
    if not patients:
        print(f'No patient files found in {data_dir}')
        return

    if args.pid:
        patients = [p for p in patients if p['pid'] == args.pid]
        if not patients:
            print(f'Patient {args.pid} not found')
            return

    print(f'Found {len(patients)} patient(s):',
          ', '.join(p['pid'] for p in patients))
    print()

    for pat in patients:
        process_patient(pat['pid'], pat['signals_path'], pat['numerics_path'],
                        out_dir, cache_dir, force=args.force)
        print()

    # summary
    df = build_summary(out_dir)
    if not df.empty:
        n_pat = df['pid'].nunique()
        n_ep  = len(df)
        print(f'Summary saved: {out_dir}/summary.csv')
        print(f'  {n_pat} patients × {n_ep} epochs total')
        print()
        # quick stats
        for col in ['G', 'C', 'H']:
            vals = df[col].dropna()
            print(f'  {col}%: mean={vals.mean():.1f}  '
                  f'std={vals.std():.1f}  '
                  f'min={vals.min():.1f}  max={vals.max():.1f}')


if __name__ == '__main__':
    main()
