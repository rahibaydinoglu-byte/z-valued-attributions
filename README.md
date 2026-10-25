# Z-valued feature attributions under unreliable supervision

Code, pre-registered protocols and per-run results for:

> R. Imamguluyev, *Z-valued feature attributions under unreliable supervision: exact propagation of label-reliability evidence, with a case study on network intrusion detection* (manuscript submitted to Information Sciences).

Each feature attribution is represented as a Z-number `(A, B)`: `A` restricts the value of the attribution, and `B` restricts the probability that the attribution obtained under the *correct* training labels lies in `A`. For models that are linear in the training labels (zero-order TSK fuzzy systems with label-free antecedents, kernel smoothers, linear heads on frozen features), label noise propagates to Shapley attributions exactly. This repository contains the implementation, the experiments on CIC-IDS2017 (original vs corrected labels) and NetFlow-v3, and every per-run result reported in the paper.

## Layout

| Folder | Contents |
|---|---|
| `src/` | Core library and data preparation |
| `experiments/` | One driver per study (E1, E1c, E2, E3) and per revision analysis (e6, e7, e8) |
| `analysis/` | Scripts that turn per-run results into the numbers in the paper |
| `checks/` | Numerical checks of the theoretical results (Appendix B) |
| `protocols/` | Pre-registered protocols, with the date each was frozen |
| `results/` | Per-run outputs (`.json` and `.npz`) and the analysis printouts |
| `pilots/` | Pilot experiments that preceded the protocols (disclosed in Sec. 6.5) |
| `figures/` | Figure script |
| `data/` | Provenance files only; the datasets are not redistributed (see below) |

Contents of `src/`:
- TSK model, firing strengths, ridge hat matrix `H`, exact interventional SHAP of the normalised firings, lattice DP (`pilot_controlled.py`);
- windowed exact lattice DP with escaped mass (`lattice_window.py`);
- block units and aggregation (`block_noise.py`);
- confident-learning evidence (`rho_estimators.py`);
- provenance evidence (`pilot_cicids.py`);
- flow matching (`match2.py`);
- NetFlow-v3 pools and the simulated schedule labeller (`e2_pools.py`, `e2_labeller.py`).

All scripts find the repository root automatically. To use a different root (for example, with the data elsewhere), set `ZXAI_ROOT`.

## Environment

The paper's results were produced with Python 3.13.17, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, pyarrow 26.0.0 and matplotlib 3.11.2, on CPU only.

```
pip install -r requirements.txt
```

## Data (not redistributed)

1. **CIC-IDS2017, original flows.** Download `GeneratedLabelledFlows.zip` from the Canadian Institute for Cybersecurity (registration required). The expected MD5 is `5ca3f8f69e3514950681615824149973`. Extract it to `data/cic_original/extracted/` (this creates the folder `TrafficLabelling ` — note the trailing space in the original archive).
2. **CIC-IDS2017, corrected labels.** Download the redistribution of the corrected release from Zenodo, DOI [10.5281/zenodo.22016274](https://doi.org/10.5281/zenodo.22016274). Save it as `data/zenodo_huang2026/cicids2017_corrected_huang2026.csv`. The expected MD5 is `ea6046dd1a3399ecbc88d682c3b5ba3e`.
3. **NetFlow-v3.** Download NF-UNSW-NB15-v3, NF-ToN-IoT-v3 and NF-CICIDS2018-v3 from the University of Queensland research data repository into `data/nfv3/`. The sizes and SHA-256 sums are listed in `data/download_manifest.txt`.

Then build the matched set and the NetFlow pools:

```
python src/match2.py          # -> data/matched/matched.parquet, hosts.parquet, match_log.json (≈ 1.60 M matched flows)
python src/e2_pools.py        # -> data/nfv3/pool_<dataset>.parquet (seed 20261025)
```

`data/match_log.json` and `data/transition_matrix.csv` are the logs of our own run, for comparison: 1 601 326 matched flows, a +3 h shift, and the label transition counts.

## Reproducing the paper

The seeds are fixed. Each run writes `results/<study>/<name>_<seed>[_<FS>].{json,npz}` and skips runs that already exist, so delete a file to recompute it.

| Paper | Command(s) | Analysis |
|---|---|---|
| Table 2 (block structure) | — | `analysis/block_structure.py` |
| E1, Table 3 (Sec. 7.1) | `python experiments/e1_full.py 100 101 … 119` | `analysis/e1_analyze.py`, `analysis/subsample_tests.py` |
| E1c, Table 4 (Sec. 7.2) | `python experiments/e1_confirm.py 120 121 … 139` | `analysis/e1c_analyze.py`, `analysis/subsample_tests.py` |
| Block-unit exploration (disclosed, exploratory) | `python experiments/e1_block.py results/blk_explore 100 101 102 103 104` | `analysis/blk_analyze.py` |
| Gaussian vs lattice (Sec. 7.3) | `python experiments/e6_simple.py 120 … 139` | `analysis/e6_analyze.py` |
| Thresholds, label policy, interval B, Table 5 (Sec. 7.3) | `python experiments/e7_revision.py 120 … 139` | printout in `results/e7_analysis.txt` |
| 30 features, KernelSHAP (Sec. 7.3) | `python experiments/e8_kernelshap.py 141 … 150` (seed 140 = smoke test) | `analysis/e8_analyze.py` |
| E2, Table 6 (Sec. 7.4) | `python experiments/e2_run.py <dataset> 200 … 219` | `analysis/e2_analyze.py`, `analysis/e2_revision.py` |
| E3, Fig. 1 (Sec. 7.5) | `python experiments/e3_sweep.py <dataset> 200 … 219` | `analysis/e3_analyze.py`, `figures/fig_e3.py` |
| Matched-flow selection (Sec. 6.1) | — | `analysis/match_rates.py` |
| Appendix B checks | `cd checks && python zxai_checks.py && python zxai_checks2.py && python zxai_checks3.py && python check_upper_bound.py` | printed |

Datasets for E2/E3: `NF-CICIDS2018-v3`, `NF-ToN-IoT-v3`, `NF-UNSW-NB15-v3`.

Runtime per run on one CPU core: about 2.5–8 min for E1/E1c and about 3 min for e8.

**Reproducibility check.** E1c seed 120 was rerun from a clean copy of this repository. For both feature sets, every stored array (attributions, events, baselines and all Z-scores) was reproduced with a maximum absolute difference of 0.0 (python 3.13.17 and the package versions above). Other platforms or BLAS builds may differ in the last digits. The revision scripts (e6, e7, e3) recompute each run up to the attributions and abort unless they match the stored results to 1e-10.

## Pre-registration

The protocols in `protocols/` were frozen at the stated dates, before the outcomes of the corresponding studies were computed. The provenance rule, the primary events and feature set A were fixed after the pilot experiments in `pilots/`, whereas feature set B and the E1c subsamples were not used in any pilot. Analyses added during revision (e6, e7, e8, `subsample_tests.py`, `e2_revision.py`, `match_rates.py`) are post hoc, except e8, whose protocol was fixed before its seeds were run.

## Citation

Please cite the paper (details will be added on acceptance) and this archive: Imamguluyev, R. (2026). *Z-valued feature attributions under unreliable supervision: code, protocols and results* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23645353

## License

Code: MIT (see `LICENSE`). Per-run results: CC BY 4.0. The datasets remain under their original licences.

## Disclosure

Parts of the analysis code were written with the assistance of an AI system (Claude, Anthropic). All code was run, checked against independent computations where indicated, and is the responsibility of the author.
