# QuBridge — Reproducibility Artifact

Data, scripts and source code for the paper
_"QuBridge: Layer-wise Score Decomposition in Quantum Computation Pipeline"_.

Everything here runs on a cached calibration snapshot of IBM Marrakesh.
**No access to IBM Quantum hardware or to the IBM Quantum API is needed.**

| The paper says the artifact provides | Where it is                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| CSV data for all figures and tables  | `data/` (12 files; mapping below)                                                                     |
| the calibration snapshot             | `source/static/data/backend_errors.json` (IBM Marrakesh, fetched 2026-10-28)                          |
| the QuBridge source code             | `source/` (the simulation pipeline and the three experiments)                                         |
| reproduction of the results          | `scripts/reproduce_numbers.py` (seconds), or re-run the simulations from `source/` (minutes to hours) |

## Quick check (a few seconds, pandas only)

```bash
pip install pandas
python scripts/reproduce_numbers.py --check
```

The script recomputes every number the paper reports (the three tables, the values quoted in
the text and in the captions) from the CSV files in `data/`, prints them, and compares them with
the values printed in the paper. It ends with `N of N reported values match the CSV data.`
and exits with status 1 if any value differs.

## Layout

```
README.md  LICENSE  requirements.txt  environment.yml  MANIFEST.json
data/        CSV data behind every figure, table and in-text number
figures/     the result figures as they appear in the paper (PDF, for reference)
scripts/     reproduce_numbers.py — CSV -> the numbers in the paper
source/      the code that produced the data (a slice of the QuBridge repository)
```

`MANIFEST.json` lists the SHA-256 of every file.

## Data ↔ paper

Figure and table numbers are those of the submitted manuscript.

| In the paper                                                                             | File in `data/`                                                             | What it holds                                                                                                                                                   |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Table I, Fig. 2(b) — cumulative L2 filtering                                             | `l2_filter_cascade.csv`                                                     | per filter stage: the chosen chain (`position = best`) and the worst surviving configuration (`worst`), at the untuned L3/L4 setting                            |
| Table II, Fig. 3(b), Fig. 6, end-to-end trace, L3 and L4 sections — progressive ablation | `pipeline_ablation.csv`                                                     | every configuration of the four ablation stages: 384 chains × 4 L3 settings × 2 L4 settings (stage 0), then the chosen chain with L3 and L4 fixed one at a time |
| Table III, Fig. 8 — error-detected teleportation                                         | `error_detected_teleportation.csv`                                          | physical and error-detected circuits, `\|+>` and `\|1>`, noise scale 0.5–2.5                                                                                    |
| L1 section — three input states                                                          | `l1_input_states.csv`                                                       | `\|+>`, `\|0>`, `\|1>` at the four L3 settings on the chosen chain                                                                                              |
| L3 section and the L2–L3 remark in "Separability" — gain by chain                        | `l3_gain_by_chain.csv`                                                      | the four L3 settings on the default and on the chosen chain                                                                                                     |
| L4 section — readout error set to zero                                                   | `l4_correction_variants.csv`                                                | real-time correction, post-processed frame, post-selection, ideal readout (single seed)                                                                         |
| Fig. 7 and the shares quoted in "Separability"                                           | `layer_interaction_shares.csv`, `layer_interaction_noise_floor.csv`         | share of Var(ln Y) per layer and layer pair, and its shot-noise floor, for four applications                                                                    |
| raw cells behind Fig. 7                                                                  | `layer_interaction_grid_{teleportation,ghz,entanglement_swapping,chsh}.csv` | one row per cell of the factorial grid (16,384 shots each)                                                                                                      |

Common columns of the three experiment files: `fidelity` is the estimate from one simulator seed
(2^18 shots in each of the three fixed measurement bases), `sigma` its binomial standard error,
`acceptance` the fraction of shots kept (1 unless the syndrome is post-selected), `seed` the
simulator seed (0–4). The paper reports the mean over the five seeds.
In the grid files `sched` is the scheduling policy, `dd` the decoupling sequence, `corr` the
implementation of the correction, `ps` post-selection, `F` the score, `var` its shot-noise variance
and `acc` the acceptance.

## Re-running the simulations

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
cd source

# Table III / Fig. 8 — about 3 minutes
python -m experiments.e3d_noise_scaling.runner --output-dir ../rerun/e3d

# Table I / Fig. 2(b) — about 50 minutes
python -m experiments.e1g_qubit_quality_fidelity.runner_v6 --output-dir ../rerun/e1g

# Table II / Fig. 3(b) / Fig. 6 — about 3.5 hours (3,072 configurations × 5 seeds in stage 0)
python -m experiments.e1h_pipeline_convergence.runner --output-dir ../rerun/e1h
# stages 1–3 only, reusing the shipped stage 0 — about 1 minute
python -m experiments.e1h_pipeline_convergence.runner --output-dir ../rerun/e1h \
    --reuse-stage0 experiments/results/e1h_pipeline_convergence/e1h_pipeline_convergence.csv
```

Times are wall-clock times on the authors' workstation. The simulator seeds are fixed, so a re-run
with the pinned Qiskit and Qiskit Aer versions reproduces the shipped `fidelity` columns.
Add `--dry-run` to any runner for a reduced run that finishes in seconds.
Without `--output-dir` a runner writes to `source/experiments/results/<experiment>/` and replaces
the copy of the CSV kept there; `data/` is never written.

The four result figures of these experiments are drawn by the analysis modules, which read the
CSV copies under `source/experiments/results/`:

```bash
python -m experiments.e1g_qubit_quality_fidelity.analyze_v6 --paper
python -m experiments.e1h_pipeline_convergence.analyze --paper
python -m experiments.e3d_noise_scaling.analyze --paper
```

### What is and is not re-simulated from this artifact

- Re-simulated from `source/`: the three experiments above, i.e. Tables I–III and Figs. 2(b), 3(b), 6 and 8.
- Provided as data only: the three input states of the L1 section, the L4 variants, and the
  factorial grids behind Fig. 7. They were produced with the same pipeline (`source/services/pipeline`);
  the notebooks that drove those runs are not part of this artifact.

## Simulation setup

| Item                | Value                                                                                                                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Device model        | IBM Marrakesh (156 qubits, Heron r2), calibration fetched 2026-10-28                                                                                                                                                         |
| Qiskit / Qiskit Aer | 1.3.1 / 0.16.0                                                                                                                                                                                                               |
| Python              | 3.10 or newer (the data were produced with 3.12)                                                                                                                                                                             |
| Estimation          | shot-based simulation of the dynamic circuits; 2^18 shots per basis, three fixed bases, seeds 0–4                                                                                                                            |
| Transpilation       | optimisation level 3, seed 42                                                                                                                                                                                                |
| Noise model         | thermal relaxation + sparse Pauli-Lindblad residual per gate, calibrated readout error on every measurement, static ZZ of 3.4 kHz, T2\* = 0.8 T2 without decoupling, fast initial idle dephasing; 600 ns feedforward latency |

The constants of the noise model are in `source/core/config.py` (`PAPER_*`) and
`source/experiments/config.py`.

## About `source/`

`source/` is a slice of the QuBridge repository: the files keep their paths and contents, so the
commands above are the ones used to produce the data. Two things differ from the repository.
The calibration file is reduced to the one device the paper uses, and run metadata
(timestamps, local paths) is left out. Comments and docstrings in the source are in Japanese.

| Directory                            | Role                                                                                                                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `services/pipeline/`                 | the pipeline one application runs through: qubit selection, noise model, idle-time model, feedforward, measurement, shot-based estimation                                                       |
| `services/apps/`                     | the teleportation circuits (physical and error-detected) and their score                                                                                                                        |
| `services/mode1/`, `services/mode3/` | the service functions behind the QuBridge pages; the experiments call the same functions (`evaluate_step`, `simulate`, `compute_fidelity_*`), so the paper and the tool report the same numbers |
| `experiments/`                       | the three experiments (runner and analysis) and their shared configuration                                                                                                                      |
| `core/`                              | the constants of the noise model and of the paper's configuration (`config.py`)                                                                                                                 |

The web application itself (routes, pages, session handling) is not part of this artifact;
nothing here starts a server or makes a network request.

## Earlier version

The artifact of the earlier conference submission of this work (IBM Torino, density-matrix
simulation, three layers) is kept under the git tag `qce26`.

## License

MIT — see [LICENSE](LICENSE).
