# E1 full study: PRE-REGISTERED PROTOCOL (frozen 24.10.2026, before any E1 result)
Any change after the first E1 result must be logged at the bottom with date and reason.

## Data
Matched CIC-IDS2017 flows (1 601 326). ỹ = original label, y = corrected label, binary attack vs benign (Attempted = benign). Features from the original CSVs.

## Design
- 20 independent subsamples (seeds 100–119); 6 000 training flows each, drawn uniformly from all matched flows.
- Two feature sets fixed a priori (not selected on data):
  - FS-A (pilot): destination port, flow duration, total fwd packets, total backward packets, total length of fwd packets, flow bytes/s, init_win_bytes_forward, psh flag count.
  - FS-B (alternative, domain-chosen): flow iat mean, fwd packet length max, bwd packet length mean, total length of bwd packets, fin flag count, syn flag count, init_win_bytes_backward, flow packets/s.
- Model: zero-order TSK, R = 16, k-means antecedents on X only, ridge λ = 0.01, trained on noisy labels. Exact interventional SHAP, 30 background points.
- Test points per run: 30 from each of four strata:
  - S1: Wednesday (DoS→Attempted mechanism);
  - S2: Thursday afternoon (Infiltration-Portscan);
  - S3: Friday DDoS file;
  - S4: clean files (Monday, Tuesday, Thursday morning, Friday morning, Friday PortScan).
- Events: rel (primary), sign (primary), abs (secondary).
- Methods: Z with ρ modes
  - CL;
  - PROV (pad 2 min, ρ_win 0.2, ρ_payload 0.5);
  - PROV-AUD (ρ per provenance category estimated from a 1% audit of the corpus excluding the run's training flows, Laplace-smoothed);
  - CONST (global rate; structure-only ablation);
  - ORACLE (upper bound).
- Baselines: label sensitivity S, bootstrap SD (8 refits), |φ̃|, bootstrap SD/|φ̃|.
- Provenance sensitivity (seeds 100–109, FS-A): pad ∈ {0, 1, 5} (ρ 0.2/0.5); ρ_win ∈ {0.1, 0.3}; ρ_payload ∈ {0.3, 0.7}.
- Lattice: windowed exact DP (±12 sd; lost mass recorded), grid min(0.05·sd_i, sd)/40.

## Hypotheses and tests
- **H1 (primary):** AUROC(Z[PROV]) > AUROC(best label-blind baseline per run) for rel and sign. Unit = run (40 runs = 20 subsamples × 2 FS); paired one-sided Wilcoxon signed-rank test, Holm over the 2 primary events. Also reported vs each baseline separately.
- **H2:** AUROC(Z[PROV]) > AUROC(Z[CONST]) (the gain comes from the reliability evidence).
- **H3:** |mean P − observed rate| is smaller for PROV-AUD than for PROV (audit improves calibration); Brier scores reported.
- **H4 (descriptive):** per-stratum AUROC; S4 (clean files) expected to have low event rates.
- **Falsification:** H1 fails if p ≥ 0.05 after Holm or the mean difference is ≤ 0.
- **Additional:** Friedman test over methods (runs as blocks) + Nemenyi critical difference, per event.

---
## Amendment 1 (24.10.2026, after E1 results): block-noise confirmation study E1c
**Reason.** H3 was falsified. Real errors were shown to be block-constant: 99.7% of (file, minute, host-pair) blocks are all-or-none, and 97% of the flip variance lies between blocks. An exploratory analysis on E1 seeds 100–104 (10 runs) compared five block definitions; this selection is EXPLORATORY and is not used as evidence.

**Frozen for confirmation** on NEW seeds 120–139 (never run before); otherwise the same design as E1:
- New modes:
  - BLK-FS (primary): units = (provenance category ∈ {benign-in-window, attack-zero-payload}) × capture file × source IP. A unit flips all its training members, with the flow-level PROV marginal (0.2 / 0.5); other flows are independent (ρ = 1e-3).
  - BLK-FMH (secondary): units = category × file × minute × host pair.
- **H5 (primary, calibration):** Brier(Z[BLK-FS]) < Brier(Z[PROV]) for rel and sign; paired one-sided Wilcoxon over 40 runs, Holm over 2 events. |mean P − rate| is also reported.
- **H6 (ranking):** AUROC(Z[BLK-FS]) > AUROC(best label-blind baseline per run) for rel and sign (one-sided Wilcoxon, Holm).
- **H7 (secondary):** AUROC(Z[BLK-FS]) ≥ AUROC(Z[PROV]) (two-sided Wilcoxon, descriptive).
- H1/H2 are re-tested on the new seeds as replication.
