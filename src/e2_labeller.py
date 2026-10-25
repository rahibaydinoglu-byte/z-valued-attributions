"""Simulated CIC-type schedule labeller with imperfections, applied to NetFlow-v3 ground truth.
y (published label) = clean; returns noisy y_tilde and the mechanism of each flip.
Mechanisms (parameters drawn per run from the seed):
  M1a missed attacker hosts : per attack class, each attacker source host is unknown to the labeller with prob 1-q
                              (>= 1 host kept per class); its attack flows become benign.
  M1b missed phases         : attack segments (per class/day, gaps <= 3 min) lose a contiguous fraction f of their
                              minutes with prob p_drop; attack flows of that class in the dropped minutes become benign.
  M2  over-wide windows     : each kept segment is extended by e minutes on both sides; benign flows from a known
                              attacker host of the class to a victim of that class inside the extension become attacks.
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import numpy as np, pandas as pd

def segments(minutes, gap=3):
    m = np.unique(minutes)
    if m.size == 0: return []
    cut = np.flatnonzero(np.diff(m) > gap)
    starts = np.r_[m[0], m[cut + 1]]; ends = np.r_[m[cut], m[-1]]
    return list(zip(starts, ends))

def label(df, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.6, 0.9); p_drop = rng.uniform(0.2, 0.4); e = int(rng.integers(2, 6))
    y = df["y"].values; yt = y.copy(); mech = np.zeros(len(df), np.int8)   # 1 = M1a, 2 = M1b, 3 = M2
    src, dst, minute, day = df["src"].values, df["dst"].values, df["minute"].values, df["day"].values
    cls = df["cls"].astype(str).values
    for c in np.unique(cls[y == 1]):
        ia = np.flatnonzero((cls == c) & (y == 1))
        hosts = np.unique(src[ia])
        keep = hosts[rng.random(hosts.size) < q]
        if keep.size == 0: keep = rng.choice(hosts, 1)
        miss = ia[~np.isin(src[ia], keep)]
        yt[miss] = 0; mech[miss] = 1
        victims = np.unique(dst[ia])
        for dday in np.unique(day[ia]):
            idd = ia[day[ia] == dday]
            for s, t in segments(minute[idd]):
                if rng.random() < p_drop:
                    L = t - s + 1; f = rng.uniform(0.2, 0.5); ln = max(1, int(round(f * L)))
                    a0 = s + int(rng.integers(0, max(1, L - ln + 1)))
                    dm = idd[(minute[idd] >= a0) & (minute[idd] < a0 + ln) & (yt[idd] == 1)]
                    yt[dm] = 0; mech[dm] = 2
                ext = np.flatnonzero((y == 0) & (day == dday) & np.isin(src, keep) & np.isin(dst, victims) &
                                     (((minute >= s - e) & (minute < s)) | ((minute > t) & (minute <= t + e))))
                yt[ext] = 1; mech[ext] = 3
    return yt.astype(np.int8), mech, dict(q=q, p_drop=p_drop, e=e)

def provenance(df, yt, pad=2, rw=0.2, rp=0.5):
    """Generic E1 rule family, computed from noisy labels only.
    cat 0: noisy-benign within +-pad min of a noisy-attack minute on the same day; cat 2: noisy-attack with no response
    (OUT_PKTS == 0); cat 1: other noisy-benign; cat 3: other noisy-attack."""
    am = set(zip(df["day"].values[yt == 1], df["minute"].values[yt == 1]))
    win = set((d, m + k) for d, m in am for k in range(-pad, pad + 1))
    inwin = np.fromiter(((d, m) in win for d, m in zip(df["day"].values, df["minute"].values)), bool, len(df))
    nores = df["OUT_PKTS"].values == 0
    cat = np.select([(yt == 0) & inwin, yt == 0, (yt == 1) & nores], [0, 1, 2], 3).astype(np.int8)
    rho = np.full(len(df), 1e-3); rho[cat == 0] = rw; rho[cat == 2] = rp
    return rho, cat, inwin
