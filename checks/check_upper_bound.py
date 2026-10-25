"""Numerical check of the certified upper bound (Theorem 2, second inequality):
P(|Delta| <= tau) <= min_k [ P(|Delta_h| <= tau + h k / 2) + P(K > k) ], by exhaustive enumeration (n <= 12)."""
import itertools, numpy as np
rng = np.random.default_rng(7); viol = 0; gaps = []
for inst in range(2000):
    n = rng.integers(2, 13); d = rng.normal(size=n) * rng.choice([0.1, 1, 5]); rho = rng.uniform(0, 0.5, n)
    tau = abs(rng.normal()) * np.abs(d).sum() / 3 + 1e-3; h = rng.choice([0.05, 0.2, 0.5]) * np.abs(d).mean()
    q = np.round(d / h); F = np.array(list(itertools.product([0, 1], repeat=n)))
    w = np.prod(np.where(F == 1, rho, 1 - rho), 1); D = F @ d; Dh = h * (F @ q); K = F.sum(1)
    p = w[np.abs(D) <= tau].sum()
    up = min(w[np.abs(Dh) <= tau + h * k / 2].sum() + w[K > k].sum() for k in range(n + 1))
    lo = max(w[np.abs(Dh) <= tau - h * k / 2].sum() - w[K > k].sum() for k in range(n + 1))
    viol += (p > up + 1e-12) or (p < lo - 1e-12); gaps.append(up - lo)
print(f"2000 instances: violations {viol}; median width of certified interval {np.median(gaps):.3f}")
