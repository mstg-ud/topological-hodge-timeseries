"""
Canonical pipeline: PSD + Wasserstein + Rips + KL Hodge
Version: 1.0
Steps:
  1. Sliding window
  2. Hann window + local PSD  (DC removed, total-power normalized)
  3. Wasserstein distance matrix  (freq bins as support)
  4. Vietoris-Rips complex at radius epsilon
  5. KL antisymmetric edge flow
  6. Hodge decomposition -> G%, C%, H%

Baseline contract:
  - pure sine -> trivial zero-flow regime (total_E ~ machine precision)
  - spectral leakage floor: W_1 max ~ 0.025  (L=64, fs=50, f=1Hz, Hann)
"""
from __future__ import annotations
import numpy as np
from itertools import combinations


# ── Step 1: sliding window ────────────────────────────────────────────────────

def sliding_windows(x: np.ndarray, L: int, r: float):
    """Yield window patches of length L with overlap rate r."""
    S = max(1, int(L * (1 - r)))
    for s in range(0, len(x) - L + 1, S):
        yield x[s:s + L]


# ── Step 2: Hann window + local PSD ──────────────────────────────────────────

def local_psd(x: np.ndarray, fs: float, L: int, r: float,
              eps_smooth: float = 1e-10):
    """
    Compute local PSD distributions {P_i} for each sliding window.

    Parameters
    ----------
    x          : input time series
    fs         : sampling frequency (Hz)
    L          : window length (samples)
    r          : overlap rate in [0, 1)
    eps_smooth : additive smoothing to prevent log(0) in KL

    Returns
    -------
    freqs : frequency axis (Hz), length = L//2  (DC removed, for even L)
    P_arr : array of shape (N, L//2), each row sums to 1
    """
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)[1:]   # DC bin removed
    hann  = np.hanning(L)
    P_list = []
    for w in sliding_windows(x, L, r):
        power = np.abs(np.fft.rfft(w * hann)) ** 2
        power = power[1:]                          # remove DC
        total = power.sum()
        if total < 1e-10:
            continue
        p = (power + eps_smooth) / (total + len(power) * eps_smooth)
        P_list.append(p)
    return freqs, np.array(P_list)


# ── Step 3: Wasserstein distance matrix ──────────────────────────────────────

def wasserstein_matrix(P_arr: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Pairwise 1D Wasserstein distances using actual Hz values as support.
    Computed via the CDF-difference formula: W_1 = sum |CDF_i - CDF_j| * dfreq
    """
    N    = len(P_arr)
    cdfs = np.cumsum(P_arr, axis=1)[:, :-1]   # (N, bins-1)
    dx   = np.diff(freqs)                       # (bins-1,)
    D    = np.zeros((N, N))
    for i in range(N):
        diff       = np.abs(cdfs[i] - cdfs[i + 1:])   # (N-i-1, bins-1)
        D[i, i+1:] = diff @ dx
        D[i+1:, i] = D[i, i+1:]
    return D


# ── Step 4: Vietoris-Rips complex ────────────────────────────────────────────

def build_rips(D: np.ndarray, eps: float):
    """
    Build edges and triangles of the Rips complex at radius eps.
    beta1 = |E| - |V| + n_components - |T|  (2-dimensional complex only)
    """
    N     = D.shape[0]
    edges = [(i, j) for i in range(N)
             for j in range(i + 1, N) if D[i, j] <= eps]
    es    = set(edges)
    tris  = [(i, j, k) for i, j, k in combinations(range(N), 3)
             if (i, j) in es and (i, k) in es and (j, k) in es]
    return edges, tris


def n_components(N: int, edges) -> int:
    parent = list(range(N))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj: parent[rj] = ri
    return len(set(find(v) for v in range(N)))


# ── Step 5: KL antisymmetric flow ────────────────────────────────────────────

def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p2 = np.clip(p, eps, None); q2 = np.clip(q, eps, None)
    return float(np.sum(p2 * np.log(p2 / q2)))

def kl_flow(P_arr: np.ndarray, edges) -> np.ndarray:
    """f(i,j) = KL(P_i||P_j) - KL(P_j||P_i)  [antisymmetric]"""
    return np.array([kl_div(P_arr[i], P_arr[j]) - kl_div(P_arr[j], P_arr[i])
                     for i, j in edges])


# ── Step 6: Hodge decomposition (sparse solver) ──────────────────────────────

def hodge_decompose(N: int, edges, tris, f: np.ndarray):
    """
    Decompose f into gradient + curl + harmonic components.
    Uses sparse least-squares (scipy.sparse.linalg.lsqr) for efficiency.

    Returns
    -------
    G, C, H   : energy shares (%) — meaningful only when total_E > 0
    total_E   : ||f||^2
    b1        : first Betti number (2-dim complex, unreliable when beta2 > 0)
    trivial   : True when total_E < 1e-18 (zero-flow regime)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr as sparse_lsqr

    m     = len(edges)
    total = float(f @ f)
    nc    = n_components(N, edges)
    b1    = m - N + nc - len(tris)

    if total < 1e-18:
        return None, None, None, total, b1, True   # trivial zero-flow

    ei = {e: k for k, e in enumerate(edges)}

    # sparse B1  (N x m)
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(edges):
        rows += [i, j]; cols += [k, k]; vals += [-1.0, 1.0]
    B1   = csr_matrix((vals, (rows, cols)), shape=(N, m))
    phi  = sparse_lsqr(B1 @ B1.T, B1 @ f)[0]
    grad = B1.T @ phi

    if tris:
        # sparse B2  (m x |T|)
        rows2, cols2, vals2 = [], [], []
        for t, (i, j, k) in enumerate(tris):
            for e, s in [((i, j), 1), ((j, k), 1), ((i, k), -1)]:
                idx = ei.get(e)
                if idx is None:
                    idx = ei.get((e[1], e[0])); s = -s
                if idx is not None:
                    rows2.append(idx); cols2.append(t); vals2.append(float(s))
        B2   = csr_matrix((vals2, (rows2, cols2)), shape=(m, len(tris)))
        psi  = sparse_lsqr(B2.T @ B2, B2.T @ f)[0]
        curl = B2 @ psi
    else:
        curl = np.zeros_like(f)

    harm = f - grad - curl
    G = 100.0 * float(grad @ grad) / total
    C = 100.0 * float(curl @ curl) / total
    H = 100.0 * float(harm @ harm) / total
    return G, C, H, total, b1, False

# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(x: np.ndarray, fs: float, L: int, r: float,
                 eps_quantile: float, max_tris: int = 5000):
    """
    Run the full pipeline on signal x.

    Returns a dict with keys:
      N, freqs, P_arr, D, eps, edges, tris, f,
      G, C, H, total_E, b1, trivial, n_comp
    """
    freqs, P_arr = local_psd(x, fs, L, r)
    N = len(P_arr)

    D     = wasserstein_matrix(P_arr, freqs)
    upper = D[np.triu_indices(N, k=1)]
    eps   = float(np.quantile(upper, eps_quantile))

    edges, tris = build_rips(D, eps)

    if len(tris) > max_tris:
        return {'error': f'|T|={len(tris)} exceeds max_tris={max_tris}',
                'N': N, 'eps': eps, 'N_e': len(edges), 'N_t': len(tris)}

    f = kl_flow(P_arr, edges)
    G, C, H, total_E, b1, trivial = hodge_decompose(N, edges, tris, f)
    nc = n_components(N, edges)

    return dict(N=N, freqs=freqs, P_arr=P_arr, D=D,
                upper=upper, eps=eps,
                edges=edges, tris=tris, f=f,
                G=G, C=C, H=H, total_E=total_E,
                b1=b1, trivial=trivial, n_comp=nc)


# ── Baseline: pure sine ───────────────────────────────────────────────────────

if __name__ == '__main__':
    fs       = 50.0
    f_signal = 1.0
    duration = 20.0
    L        = 64
    r        = 0.875

    freq_res   = fs / L
    signal_bin = f_signal / freq_res

    t = np.arange(0, duration, 1.0 / fs)
    x = np.sin(2 * np.pi * f_signal * t)

    print('=' * 65)
    print('Pure sine baseline  (PSD + Wasserstein + Rips + KL Hodge)')
    print('=' * 65)
    print(f'  signal      : sin(2π·{f_signal}Hz·t)')
    print(f'  fs          : {fs} Hz')
    print(f'  duration    : {duration} s')
    print(f'  L           : {L} samples')
    print(f'  r           : {r}')
    print(f'  S           : {max(1,int(L*(1-r)))} samples')
    print(f'  freq res    : {freq_res:.4f} Hz/bin')
    print(f'  signal bin  : {signal_bin:.2f}  (must be >= 1)')
    print(f'  window      : Hann')
    print()

    # P_i uniformity check
    freqs, P_arr = local_psd(x, fs, L, r)
    N = len(P_arr)
    print(f'  N windows   : {N}')
    print(f'  P_i dim     : {P_arr.shape[1]}  (= L/2 bins, DC removed, even L)')
    print(f'  p1 mean     : {P_arr[:,0].mean():.6f}')
    print(f'  p1 std      : {P_arr[:,0].std():.2e}  (leakage-induced variation)')
    print(f'  max |P_i-P_0|: {np.max(np.abs(P_arr-P_arr[0])):.2e}')
    print()

    # Wasserstein distance distribution
    D     = wasserstein_matrix(P_arr, freqs)
    upper = D[np.triu_indices(N, k=1)]
    print(f'  W_1 distance floor (leakage):')
    print(f'    min  : {upper.min():.2e}')
    print(f'    max  : {upper.max():.2e}  <- spectral leakage floor')
    print(f'    mean : {upper.mean():.2e}')
    print()

    # epsilon sweep
    print(f'  {"eps_q":>6}  {"ε":>10}  {"Ne":>5}  {"Nt":>5}  {"result":>40}')
    print('  ' + '-' * 70)
    for eps_q in [0.01, 0.05, 0.10, 0.25, 0.50]:
        res = run_pipeline(x, fs, L, r, eps_q)
        if 'error' in res:
            print(f'  {eps_q:.2f}  {res["eps"]:>10.3e}  '
                  f'{res["N_e"]:>5}  {res["N_t"]:>5}  {res["error"]}')
            continue
        if res['trivial']:
            result_str = 'trivial zero-flow  (G/C/H undefined)'
        else:
            result_str = (f'G={res["G"]:5.1f}%  C={res["C"]:5.1f}%  '
                          f'H={res["H"]:5.1f}%  E={res["total_E"]:.2e}')
        print(f'  {eps_q:.2f}  {res["eps"]:>10.3e}  '
              f'{len(res["edges"]):>5}  {len(res["tris"]):>5}  {result_str}')

    print()
    print('Baseline established:')
    print(f'  spectral leakage floor : W_1 max = {upper.max():.4f}')
    print('  pure sine result       : trivial zero-flow across all eps_q')
    print('  G/C/H%                 : undefined (not 0%) in trivial regime')
