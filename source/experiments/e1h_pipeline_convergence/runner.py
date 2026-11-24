"""
E1-H: Pipeline Convergence — Cumulative Fidelity Band Narrowing

Measures how the achievable fidelity band narrows as each pipeline stage
is cumulatively optimized.

Stages（状態は |+⟩ に固定。L1 は動かさない。L3 ＝ 時刻決め × DD、形は IBM の既定。L4 ＝ 補正の実装、
読み出し・遅れ・後選択は較正のモデルに固定。2026-10-28、L4 の段は 2026-11-04）:
  Stage 0 (baseline):    Baseline の全鎖（両向き）× L3 の 4 通り × L4 の 2 通り → 何も固定していないので全格子（384 × 8）。帯がいちばん広い
                         （2026-11-04 夕方まで L3・L4 を手つかずに固定した近似だった。ケイ「Best は平のほうがいい」で全格子に。S_0 ⊃ S_1 ⊃ S_2 ⊃ S_3 の入れ子が成り立つ）
  Stage 1 (+L2):         best の経路 × L3 の 4 通り（asap/alap × DD なし/あり）× L4 の 2 通り → まだ固定していない層を全部掃く
  Stage 2 (+L2+L3):      best の経路 × 整えた L3（alap＋DD）× L4 の 2 通り（実時間／後処理の補正）
  Stage 3 (+L2+L3+L4):   best の経路 × 整えた L3 × 整えた L4（後処理の補正）→ 1 点（F_opt）

推定は論文の既定（ショット。CommonConfig.estimation）で、E1-G と同じ列挙・同じ手つかずの L3 を使う。
1 セルの計算は画面の Results と同じ services/mode1/pipeline_eval.evaluate_step（L3・L4 の knob を同じ関数に通す）。

Usage:
    python -m experiments.e1h_pipeline_convergence.runner --dry-run --verbose   # results/dry_run/e1h_pipeline_convergence に書く
    python -m experiments.e1h_pipeline_convergence.runner --verbose
"""

import csv
import math

import numpy as np

from experiments.common.chains import baseline_chains
from experiments.common.cli import create_runner_parser
from experiments.common.logging_utils import ExperimentLogger, paper_estimation_meta
from experiments.common.output import default_output_dir
from experiments.common.rows import ascii_label, paper_shots_config, seed_runs, shot_columns
from experiments.config import E1HConfig

CFG = E1HConfig()
EXPERIMENT_ID = "e1h_pipeline_convergence"

E1H_COLUMNS = [
    "stage",           # 0-2
    "stage_label",     # "baseline", "+L2", "+L2+L3"
    "state_label",     # quantum state name
    "theta",
    "phi",
    "path_q1",
    "path_q2",
    "path_q3",
    "avg_2q_error",
    "l3_config",      # L3 の決め事の名前（L3Config.name）
    "l4_config",      # L4 の決め事の名前（L4Config.name。2026-11-04）
    "repeat",
    "fidelity",
    "sigma",            # ショット推定の統計誤差（dm・dry-run は 0）
    "acceptance",       # 後選択で残った割合（物理は 1）
    "n_accepted_min",   # 観測量ごとの受理 shot の最小
    "shots",            # 1 観測量あたりのショット数（dm・dry-run は 0）
    "seed",             # 使った seed（dry-run は -1）
]

STAGE_LABELS = ["baseline", "+L2", "+L2+L3", "+L2+L3+L4"]

# Fixed input state: CommonConfig.test_states[0] (paper では |+⟩ — 最大ノイズ感受)
_TS_LABEL, _TS_THETA, _TS_PHI = CFG.common.test_states[0]
FIXED_STATE = (_TS_THETA, _TS_PHI, f"|{_TS_LABEL}⟩")


def _baseline_paths(backend_name: str):
    """Baseline の鎖を両向きで（正本は experiments/common/chains.baseline_chains。E1-G の段 0 と同じ列挙）。
    名前は knob_inventory のノートとテストが import するので残す"""
    return baseline_chains(backend_name, CFG.common.baseline_thresholds)


def stage0_rows(csv_path) -> list[dict]:
    """段 0（全格子。3,072 構成 × 5 seed で約 3.6 時間）を前の CSV から文字のまま読む。段 0 は全経路を掃くので L2 の選択に依存せず、
    L2 を替えたときは段 1〜3 だけ回せばよい（2026-11-04）。段 0 が無ければ ValueError"""
    with open(csv_path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["stage"] == "0"]
    if not rows:
        raise ValueError(f"{csv_path} に段 0 の行が無い")
    return rows


def run_experiment(dry_run: bool = False, verbose: bool = False, reuse_stage0=None, output_dir: str | None = None,
                   seeds=None):
    """seeds を渡すと論文の seed（CommonConfig.paper_seeds）の代わりにそれで回す（別の seed での回し直し。metadata にも書く）"""
    paper_seeds = tuple(seeds) if seeds else CFG.common.paper_seeds
    # 読むのは logger の前（logger が同じ CSV を新しく作り直すので、後だと中身が消える）
    reused = stage0_rows(reuse_stage0) if reuse_stage0 else None
    if output_dir is None:
        output_dir = default_output_dir(EXPERIMENT_ID, dry_run)   # dry-run は results/dry_run/<id>（本番の CSV を "w" で開かない）
    logger = ExperimentLogger(
        experiment_id=EXPERIMENT_ID,
        csv_columns=E1H_COLUMNS,
        output_dir=output_dir,
    )

    n_repeats = CFG.dry_run_repeats if dry_run else CFG.common.n_repeats
    best_path = list(CFG.common.best_path)
    baseline_paths = _baseline_paths(CFG.common.backend)

    if verbose:
        print(f"E1-H Pipeline Convergence ({'dry-run' if dry_run else 'real'})")
        print(f"  Baseline paths: {len(baseline_paths)} (after broken-qubit/edge exclusion)")
        print(f"  L3 configs (+L2 stage): {[c.name for c in CFG.l3_configs]}")
        print(f"  Repeats: {n_repeats}")

    for stage in range(4):
        stage_label = STAGE_LABELS[stage]
        if verbose:
            print(f"\n--- Stage {stage}: {stage_label} ---")

        if stage == 0 and reused is not None:
            for row in reused:
                logger.log_row(row)
            if verbose:
                print(f"  {len(reused)} rows reused from {reuse_stage0}")
            continue

        # Stage 0 (baseline):  all adjacent L2 paths, both orientations × L3 の 4 通り × L4 の 2 通り（全格子。E1-G の列挙と同じ鎖）
        # Stage 1 (+L2):       best path (= 段 0 で F が最高の鎖、CommonConfig.best_path) × L3 の 4 通り × L4 の 2 通り（段 k はまだ固定して
        #                      いない層を全部掃く。Fig 4b は手つかずの L4 の 4 行だけ読む）
        # Stage 2 (+L2+L3):    best path × 整えた L3（alap＋DD）× L4 の 2 通り
        # Stage 3 (+L2+L3+L4): best path × 整えた L3 × 整えた L4（後処理の補正）
        if stage == 0:
            paths = baseline_paths
            cells = [(l3, l4) for l3 in CFG.l3_configs for l4 in CFG.l4_configs]
        elif stage == 1:
            paths = [(best_path, 0.0)]
            cells = [(l3, l4) for l3 in CFG.l3_configs for l4 in CFG.l4_configs]
        elif stage == 2:
            paths = [(best_path, 0.0)]
            cells = [(CFG.common.l3_tuned, l4) for l4 in CFG.l4_configs]
        else:
            paths = [(best_path, 0.0)]
            cells = [(CFG.common.l3_tuned, CFG.common.l4_tuned)]

        theta, phi, state_label = FIXED_STATE
        for (path, avg_err) in paths:
            q1, q2, q3 = path[0], path[1], path[2]
            physical_qubits = [q1, q2, q3]

            for l3, l4 in cells:
                # 2026-10-28 からショット推定: 論文の seed（CommonConfig.paper_seeds）ごとに 1 行。dry-run は解析値を n_repeats 行
                runs = seed_runs(dry_run, n_repeats, paper_seeds)
                for rep, seed in runs:
                    if dry_run:
                        r = {"fidelity": _dry_run_fidelity(theta, avg_err, l3, l4),
                             "sigma": 0.0, "acceptance": 1.0, "n_accepted_min": 0, "shots": 0}
                    else:
                        r = _real_fidelity(theta, phi, physical_qubits, l3, l4, seed)
                    fidelity = r["fidelity"]
                    logger.log_row({
                        "stage": stage,
                        "stage_label": stage_label,
                        "state_label": state_label,
                        "theta": round(theta, 6),
                        "phi": round(phi, 6),
                        "path_q1": q1,
                        "path_q2": q2,
                        "path_q3": q3,
                        "avg_2q_error": round(avg_err, 6),
                        "l3_config": l3.name,
                        "l4_config": l4.name,
                        "repeat": rep,
                        **shot_columns(r, seed),
                    })
                    if verbose:
                        safe_label = ascii_label(state_label)
                        f_txt = "None" if fidelity is None else f"{fidelity:.4f}"
                        print(f"  {stage_label} | {safe_label} | "
                              f"q[{q1},{q2},{q3}] | {l3.name} | {l4.name} | "
                              f"rep={rep} seed={seed} | F={f_txt}")

    logger.save_metadata({
        "experiment": "E1-H Pipeline Convergence",
        "description": "Cumulative fidelity band narrowing across pipeline stages",
        "backend": CFG.common.backend,
        "n_baseline_paths": len(baseline_paths),
        "best_path": list(CFG.common.best_path),
        "stage0_reused_from": str(reuse_stage0) if reuse_stage0 else None,
        "l3_untuned": CFG.common.l3_untuned.name,
        "l3_tuned": CFG.common.l3_tuned.name,
        "l3_configs": {c.name: c.kwargs() for c in CFG.l3_configs},
        "l4_untuned": CFG.common.l4_untuned.name,
        "l4_tuned": CFG.common.l4_tuned.name,
        "l4_configs": {c.name: c.kwargs() for c in CFG.l4_configs},
        "dry_run": dry_run,
        "n_repeats": n_repeats,
        **paper_estimation_meta(CFG.common, paper_seeds),   # save_metadata が get_experiment_metadata() から始めるので、ここでは広げない（鍵の順は同じ）
    })

    if verbose:
        print(f"\nResults saved to: {output_dir}")


def _real_fidelity(theta, phi, physical_qubits, l3, l4, seed):
    """論文の推定法（core.config.PAPER_ESTIMATION_METHOD ＝ shots。CommonConfig.estimation と同じ）で 1 セルを回し、
    evaluate_step の dict（fidelity・fidelity_sigma・acceptance・n_accepted_min・shots・seed）を sigma の鍵にそろえて返す。
    画面の Results と同じ関数に L3・L4 の knob を通す（2026-11-04）。形は IBM の既定"""
    from services.mode1.pipeline_eval import L3Knobs, L4Knobs, evaluate_step
    assert CFG.common.estimation == "shots"
    r = evaluate_step(theta, phi, CFG.common.backend, physical_qubits, L3Knobs(**l3.knobs()),
                      L4Knobs(**l4.kwargs()), shots=paper_shots_config(CFG.common, seed))
    return {**r, "sigma": r["fidelity_sigma"]}


def _dry_run_fidelity(theta, avg_2q_err, l3, l4):
    """dry-run 用の解析値（形はもう knob でないので、時刻決め・DD・補正の実装の分を Table I の内訳（alap +1.2・DD +1.4・frame +2.1 pt）で足す）"""
    state_factor = 1.0 - 0.005 * abs(math.sin(theta))        # L1: 状態の雑音感受
    qubit_factor = (1.0 - avg_2q_err) ** 2                     # L2: 2Q 誤り
    l3_gain = (0.012 if l3.scheduling_method == "alap" else 0.0) + (0.014 if l3.dd else 0.0)   # L3: 時刻決めと DD
    l4_gain = 0.021 if l4.correction == "frame" else 0.0       # L4: 後処理の補正は Bob の待ちを無くす
    fidelity = 0.97 * state_factor * qubit_factor + l3_gain + l4_gain
    fidelity += np.random.normal(0, 0.002)
    return max(0.0, min(1.0, fidelity))


def main():
    parser = create_runner_parser("E1-H: Pipeline Convergence")
    parser.add_argument("--reuse-stage0", default=None, help="段 0 をこの CSV から読み、段 1〜3 だけ回す")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run, verbose=args.verbose, reuse_stage0=args.reuse_stage0, output_dir=args.output_dir,
                   seeds=args.seeds)


if __name__ == "__main__":
    main()
