"""
E1-G v6: Cumulative Filter → Fidelity Band

推定は論文の既定（ショット）で、L3 は手つかず（asap・DD なし・形は IBM の既定。CommonConfig.l3_untuned。2026-10-28 に
All Square から替えた）。E1-H stage 0 と同じ列挙・同じ L3 なので、Fig. 2(b) と Fig. 3(b) の Best F が整合する
（v6 の名は結果ディレクトリ results/e1g_v6 が論文の正本から指されるため残す）。

設定は experiments.config.E1GConfig を参照（CommonConfig 経由で backend,
test_states, n_repeats 等を共有）。

Usage:
    python -m experiments.e1g_qubit_quality_fidelity.runner_v6 --verbose
    python -m experiments.e1g_qubit_quality_fidelity.runner_v6 --dry-run   # results/dry_run/e1g_v6 に書く（本番の CSV は触らない）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.chains import baseline_chains
from experiments.common.cli import create_runner_parser
from experiments.common.logging_utils import ExperimentLogger, paper_estimation_meta
from experiments.common.output import default_output_dir
from experiments.common.rows import ascii_label, paper_shots_config, seed_runs, shot_columns
from experiments.config import RESULTS_DIR, E1GConfig
from services.static_backend import load_static_backend_data

EXPERIMENT_ID = "e1g_v6"
E1G_DIR = Path(RESULTS_DIR) / EXPERIMENT_ID   # 本番の置き場（analyze_v6 が図と E1-H の CSV の場所をここから引く）

E1G_COLUMNS = [
    "state_label", "theta", "phi",
    "filter_stage", "filter_label", "n_paths",
    "position", "path", "path_error",
    "repeat", "fidelity", "sigma", "acceptance", "n_accepted_min", "shots", "seed",
]

CFG = E1GConfig()
BACKEND = CFG.common.backend
# CommonConfig.test_states（|+⟩ と |1⟩）を両方回す。Fig 3b が読むのは |+⟩ で、段の全経路の F は E1-H stage 0 から取る（analyze_v6）
STATES = CFG.common.test_states
N_REPEATS = CFG.common.n_repeats


def get_filtered_paths(backend_name):
    """段階的フィルタリングで各段階のパスリストを生成。段 0 は Baseline の鎖（両向き。experiments/common/chains.baseline_chains）"""
    data = load_static_backend_data(backend_name)
    qubit_props = {qp["qubit"]: qp for qp in data.get("qubit_properties", [])}

    def qubit_ro(q):
        qp = qubit_props.get(q, {})
        return qp.get("readout_error", qp.get("prob_meas1_prep0", 1.0))

    # ---- Baseline（壊れた qubit・辺を除く。正本は services/pipeline/qubits.baseline_paths）。両向き（2026-10-20、E1-H と同じ）----
    baseline_paths = baseline_chains(backend_name, CFG.common.baseline_thresholds)
    stages = [("Baseline", list(baseline_paths))]

    # ---- Progressive filters applied on top of Baseline ----
    # Readout error filter (first visible stage after Baseline)
    def ro_ok(path):
        return all(qubit_ro(q) <= CFG.ro_threshold for q in path)
    ro_filtered = [(p, e) for p, e in baseline_paths if ro_ok(p)]
    ro_filtered.sort(key=lambda x: x[1])
    stages.append((f"+RO<{CFG.ro_threshold*100:.0f}%", list(ro_filtered)))

    # 2Q gate error cumulative tightening (below gate2q_max = 10%)
    for tau, label in CFG.gate2q_thresholds:
        further = [(p, e) for p, e in ro_filtered if e <= tau]
        further.sort(key=lambda x: x[1])
        if further:
            stages.append((label, list(further)))

    return stages


def _fidelity(theta: float, phi: float, physical_qubits, seed) -> dict:
    """L3 を手つかず（asap・DD なし・既定の形）に固定して L2 だけを掃く。推定法は論文の既定（shots）。run_app の dict を返す"""
    from services.mode1 import simulate
    return simulate(theta, phi, BACKEND, list(physical_qubits), method=CFG.common.estimation,
                    shots=paper_shots_config(CFG.common, seed), **CFG.common.l3_untuned.kwargs())


def _mean_f(cache: dict, state_label: str, path_tuple, seeds) -> float:
    """seed の平均の F（worst の選び方と、位置の見出しに使う）。受理 0 の seed は 0 と数える"""
    vals = [cache[(state_label, path_tuple, s)]["fidelity"] for s in seeds]
    return sum(0.0 if v is None else v for v in vals) / len(vals)


def run_experiment(dry_run: bool = False, verbose: bool = False, output_dir: str | None = None, seeds=None):
    """seeds を渡すと論文の seed（CommonConfig.paper_seeds）の代わりにそれで回す（別の seed での回し直し。metadata にも書く）"""
    paper_seeds = tuple(seeds) if seeds else CFG.common.paper_seeds
    if output_dir is None:
        output_dir = default_output_dir(EXPERIMENT_ID, dry_run)   # dry-run は results/dry_run/e1g_v6（本番の CSV を "w" で開かない）
    logger = ExperimentLogger(EXPERIMENT_ID, E1G_COLUMNS, output_dir=output_dir)

    stages = get_filtered_paths(BACKEND)

    if verbose:
        for label, paths in stages:
            if paths:
                print(f"  {label}: {len(paths)} paths, "
                      f"best={paths[0][1]:.5f} worst={paths[-1][1]:.5f}")
            else:
                print(f"  {label}: 0 paths")

    best_path_tuple = tuple(CFG.common.best_path)
    best_path_list = list(best_path_tuple)
    best_path_err = None
    for _, paths_at_stage in stages:
        for p, e in paths_at_stage:
            if list(p) == best_path_list:
                best_path_err = e
                break
        if best_path_err is not None:
            break

    # Pre-compute fidelity for every baseline path × state so that "worst"
    # can be selected by min-fidelity (matches E1-H methodology) instead of
    # max-2Q-error (which tied paths at 0.045405 and picked [116,117,118]
    # instead of the true worst [115,116,117] = 0.778).
    baseline_paths_all = stages[0][1] if stages else []
    unique_paths = {}
    for p, e in baseline_paths_all:
        unique_paths[tuple(p)] = e
    unique_paths.setdefault(best_path_tuple,
                            best_path_err if best_path_err is not None else 0.0)

    fidelity_cache = {}
    cache_total = len(STATES) * len(unique_paths)
    cache_count = 0
    # 2026-10-28 からショット推定: 論文の seed ごとに 1 回。dry-run は seed 無し（F=0 の stub）を n_repeats 回
    runs = seed_runs(dry_run, N_REPEATS, paper_seeds)
    seeds = [seed for _, seed in runs]
    for state_label, theta, phi in STATES:
        for path_tuple in unique_paths:
            cache_count += 1
            for seed in seeds:
                key = (state_label, path_tuple, seed)
                if dry_run:
                    fidelity_cache[key] = {"fidelity": 0.0, "sigma": 0.0, "acceptance": 1.0, "n_accepted_min": 0, "shots": 0}
                else:
                    fidelity_cache[key] = _fidelity(theta, phi, path_tuple, seed)
            if verbose:
                print(f"  [cache {cache_count}/{cache_total}] {ascii_label(state_label)} "
                      f"{list(path_tuple)} F={_mean_f(fidelity_cache, state_label, path_tuple, seeds):.4f}")

    total = len(STATES) * len(stages) * 2 * len(runs)
    count = 0

    for state_label, theta, phi in STATES:
        for stage_idx, (label, paths) in enumerate(stages):
            if not paths:
                continue

            path_fidelities = [
                (tuple(p), e, _mean_f(fidelity_cache, state_label, tuple(p), seeds))
                for p, e in paths
            ]
            path_fidelities.sort(key=lambda x: x[2])
            worst_tuple, worst_err, _ = path_fidelities[0]
            best_f_for_log = _mean_f(fidelity_cache, state_label, best_path_tuple, seeds)
            for position, (path_tuple, path_err, base_f) in [
                ("best", (best_path_tuple,
                          best_path_err if best_path_err is not None else 0.0,
                          best_f_for_log)),
                ("worst", (worst_tuple, worst_err,
                           _mean_f(fidelity_cache, state_label, worst_tuple, seeds))),
            ]:
                for rep, seed in runs:
                    count += 1
                    r = fidelity_cache[(state_label, path_tuple, seed)]
                    if verbose:
                        print(f"  [{count}/{total}] {ascii_label(state_label)} "
                              f"{label} {position} rep={rep} seed={seed} F={base_f:.4f}")
                    logger.log_row({
                        "state_label": state_label,
                        "theta": round(theta, 6),
                        "phi": round(phi, 6),
                        "filter_stage": stage_idx,
                        "filter_label": label,
                        "n_paths": len(paths),
                        "position": position,
                        "path": str(list(path_tuple)),
                        "path_error": round(path_err, 6),
                        "repeat": rep,
                        **shot_columns(r, seed),
                    })

    logger.save_metadata({
        "experiment": "E1-G v6: Cumulative Filter Band (L3 untuned: asap, no DD, default shapes)",
        "backend": BACKEND,
        "simulation_method": CFG.common.simulation_method,
        "l3_baseline": CFG.common.l3_untuned.name,
        "filters": {
            "RO": CFG.ro_threshold,
            "1Q": CFG.common.baseline_thresholds.gate1q_max,
            "T1_min": CFG.common.baseline_thresholds.t1_min,
            "T2_min": CFG.common.baseline_thresholds.t2_min,
            "2Q_thresholds": [t for t, _ in CFG.gate2q_thresholds],
        },
        "n_repeats": N_REPEATS,
        **paper_estimation_meta(CFG.common, paper_seeds),
    })
    print(f"Saved to {output_dir}")


def main():
    parser = create_runner_parser("E1-G v6: Cumulative Filter Band (shots, L3 untuned)")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run, verbose=args.verbose, output_dir=args.output_dir, seeds=args.seeds)


if __name__ == "__main__":
    main()
