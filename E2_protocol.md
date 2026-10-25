# E2 (NetFlow-v3 mechanism transplant): PRE-REGISTERED PROTOCOL (frozen 25.10.2026, before any E2 outcome)
Allowed before freezing (done): pool construction and noise-rate / prior-coverage calibration on seeds 900–902 (no AUROC, no Brier, no model trained). Any later change → amendment at the bottom.

## Data
Fixed pools (e2_pools.py, seed 20261025):
- NF-UNSW-NB15-v3: all 2 365 424 flows;
- NF-ToN-IoT-v3: 2 000 644 flows (≈ 7.3% uniform subsample);
- NF-CICIDS2018-v3: 2 000 711 flows (≈ 9.9%).

NF-BoT-IoT-v3 is EXCLUDED: 99.7% of its flows are attacks, so benign→attack mechanisms are undefined.

Feature set (fixed a priori, analogue of E1 FS-A): L4_DST_PORT, FLOW_DURATION_MILLISECONDS, IN_PKTS, OUT_PKTS, IN_BYTES, SRC_TO_DST_AVG_THROUGHPUT, TCP_WIN_MAX_IN, TCP_FLAGS. Signed log1p + standardisation.

## Noise (e2_labeller.py)
Clean y = published label (binary). Noisy ỹ comes from a simulated CIC-type schedule labeller, with per-run imperfections drawn from the run seed:
- M1a, missed attacker hosts: keep probability q ~ U(0.6, 0.9), at least 1 host per class.
- M1b, missed phases: segments with gap ≤ 3 min; drop probability p_drop ~ U(0.2, 0.4); contiguous fraction ~ U(0.2, 0.5).
- M2, over-wide windows: extension e ∈ {2..5} min; benign flows from a known attacker to a victim of that class, inside the extension, become attacks.

Calibration facts (seeds 900–902):
- Noise rates: UNSW 1.6–2.4%, CICIDS2018 1.8–5.2%, ToN 17–20% (few attacker hosts carry most of the attack traffic).
- M2 contributes ≈ 0, except on ToN (≤ 0.4%).

## Provenance prior (noisy labels only)
- **PROV (primary)** = window rule only: noisy-benign flows within ±2 min of a noisy-attack minute of the same day → ρ = 0.2; all others 1e-3.
  - E1's rule (ii) (no-response attack flows, ρ = 0.5) is dropped, because E2 simulates no failed-attempt mechanism and calibration showed the rule makes the prior anti-informative. PROV+ii (with rule ii) is kept as a sensitivity mode.
- **BLK-FS (primary Z mode):** units = (category ∈ {benign-in-window, no-response attack}) × day × source host; unit flip probability = flow marginal; other flows independent. (Same definition as E1c.)

## Design
- 20 runs per dataset (seeds 200–219; labeller seed = run seed): 60 runs in total.
- Training: 6 000 flows uniform from the pool.
- Test: 120 flows, defined by noisy labels only:
  - 60 "near" (within ±30 min of a noisy-attack minute, same day);
  - 60 "far" (the rest).
- Model and SHAP as in E1 (zero-order TSK, R = 16, λ = 0.01, exact SHAP, 30 background points).
- Events: rel, sign (primary); abs (secondary).
- Baselines: S, bootstrap SD (8), |φ̃|, bootSD/|φ̃|.
- Z modes: CL, PROV, PROV+ii, BLK-FS, CONST, ORACLE. PROV-AUD is dropped because E1 showed no benefit.

## Hypotheses
- **H1-E2 (primary):** AUROC(Z[BLK-FS]) > best label-blind baseline per run, for rel and sign, tested PER DATASET (one-sided Wilcoxon, n = 20), Holm over 3 datasets × 2 events = 6 tests.
- **H2-E2:** AUROC(Z[BLK-FS]) > AUROC(Z[CONST]) per dataset (one-sided, Holm over 6).
- **H8 (dose-response, primary):** across the 60 runs, the Spearman correlation between [AUROC(Z[BLK-FS]) − best baseline] and the prior's ρ-AUC (vs true flips) is positive (one-sided, per event rel/sign, Holm over 2). Prediction: ToN (weak prior) shows small or no gain.
- **H5-E2 (calibration):** Brier(Z[BLK-FS]) < Brier(Z[PROV]), pooled over 60 runs (one-sided Wilcoxon), for rel and sign, Holm over 2.
- **Descriptive:** Friedman/Nemenyi over methods (60 runs as blocks); near/far strata; noise rate and mechanism mix per run.
- **Falsification of the generalisation claim:** H1-E2 fails on ≥ 2 of 3 datasets for both primary events.

**Caveat stated in advance:** E2 is semi-synthetic; the prior and the noise share the labelling-rule family, and BLK-FS units align with M1a by construction. E1/E1c remain the only fully real tests.
