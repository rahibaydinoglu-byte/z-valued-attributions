import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def oof_proba(X, yn, seed, kind="gb"):
    p1 = np.zeros(len(yn))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yn):
        if kind == "gb":
            m = GradientBoostingClassifier(random_state=seed)
        elif kind == "gb_cal":
            m = CalibratedClassifierCV(GradientBoostingClassifier(random_state=seed), method="sigmoid", cv=3)
        else:
            m = LogisticRegression(C=1.0, max_iter=2000)
        m.fit(X[tr], yn[tr]); p1[te] = m.predict_proba(X[te])[:, 1]
    return np.clip(p1, 1e-4, 1 - 1e-4)

def rho_naive(p1, yn):
    return np.clip(np.where(yn == 1, 1 - p1, p1), 1e-3, 0.999)

def rho_cl(p1, yn):
    """Confident-learning-style: per-class threshold t_c = mean p(c) over samples labelled c;
    a sample is 'suspect' if p(ỹ) < t_ỹ and the other class is above its threshold; rho = naive on suspects, ~0 else."""
    p_obs = np.where(yn == 1, p1, 1 - p1)
    t1, t0 = p1[yn == 1].mean(), (1 - p1)[yn == 0].mean()
    p_other = 1 - p_obs
    t_other = np.where(yn == 1, t0, t1)
    suspect = p_other >= t_other
    return np.where(suspect, np.clip(1 - p_obs, 1e-3, 0.999), 1e-3)

def rho_bayes(p1, yn, q=0.02):
    """Class-conditional noise inversion (anchor estimates): eta0 = P(ỹ=1|y=0) ≈ min p̃, eta1 = P(ỹ=0|y=1) ≈ 1 - max p̃
    (robust quantiles q), clean posterior pi = (p̃-eta0)/(1-eta0-eta1); rho_j = posterior of y_j != ỹ_j."""
    eta0 = np.quantile(p1, q); eta1 = 1 - np.quantile(p1, 1 - q)
    pi = np.clip((p1 - eta0) / max(1 - eta0 - eta1, 1e-3), 0, 1)
    num1 = eta0 * (1 - pi); den1 = (1 - eta1) * pi + eta0 * (1 - pi)      # ỹ=1, y=0
    num0 = eta1 * pi;       den0 = eta1 * pi + (1 - eta0) * (1 - pi)      # ỹ=0, y=1
    r = np.where(yn == 1, num1 / np.maximum(den1, 1e-9), num0 / np.maximum(den0, 1e-9))
    return np.clip(r, 1e-3, 0.999)
