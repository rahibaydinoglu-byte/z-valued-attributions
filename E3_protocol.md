# E3 evidence-quality sweep: PRE-REGISTERED PROTOCOL (frozen 25.10.2026, before any E3 outcome)

## Purpose
Turn the E2 failure into an operating condition: how informative must label-reliability evidence be for Z-attributions to beat label-blind baselines?

## Design
- **Runs.** The same 60 runs as E2 (3 NetFlow-v3 pools × seeds 200–219), with identical labeller noise, training sample, test points, background and model.
  - Reproduction is verified by matching φ̃ to the saved E2 npz; the run aborts on mismatch.
  - Baselines and events are taken from the saved E2 npz (no recomputation).
- **Synthetic evidence of controlled quality.** It uses the true flips of the training sample by construction, so this is a controlled experiment, not a deployable estimator.
  - Flagged set: each truly flipped training flow with probability recall r, plus uniformly drawn clean flows, so that the flagged set has precision π = min(0.95, L · noise_train).
  - ρ = π on flagged flows; ρ = residual rate (unflagged flips / unflagged flows, in expectation) on the rest.
  - The evidence is therefore calibrated in expectation, and its lift is L by construction.
  - RNG for flagging: seed + 50 000.
- **Lift levels.** L ∈ {1, 1.5, 2, 3, 5, 10, 20}; r = 0.9.
  - Levels with L · noise > 0.95 are capped at π = 0.95 and reported as such.
  - Sensitivity: r = 0.5 at L ∈ {3, 10}.
- **Propagation.** Independent flows (A3), windowed exact DP, and the same events (rel, sign primary; abs secondary) and AUROC definition as E2.

## Hypotheses
- **H9 (monotone dose-response).** Per dataset and event, the Spearman correlation between L and mean AUROC(Z[L]) over levels is positive. Additionally, a paired one-sided Wilcoxon test of AUROC(Z[L=20 or cap]) > AUROC(Z[L=1]).
- **Primary quantity: L\*.** The smallest lift at which AUROC(Z[L]) > best label-blind baseline per run (one-sided Wilcoxon, n = 20), per dataset × event (rel, sign).
  - Found with a fixed-sequence procedure from the highest level downwards at α = 0.05: stop at the first non-significant level; L\* = the last significant level.
  - If no level is significant, L\* = "none".
- **Prediction (stated in advance):**
  - L\* ≤ 3 for CICIDS2018-v3 (the only dataset where E2 succeeded);
  - L\* > the observed E2 lift (1.0–1.8) for all three datasets, which would explain the E2 failure quantitatively.
