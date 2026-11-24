"""
E3-D: ノイズスケーリング Physical vs Logical テレポーテーション交差点分析ランナー

ノイズスケール(0.0〜2.0)を変化させて、物理/論理テレポーテーションのフィデリティが
どこで交差するか（論理量子ビットが有利になる閾値）を探る。

測定プロトコル — 他 paper 図 (Fig 2b/3b/4b) と完全整合（2026-10-28 にショット推定へ統一）:
  - estimation        : CommonConfig.estimation = "shots"。services/mode3/fidelity.compute_fidelity_{physical,logical}
                        （run_app の包み）に paper_shots と seed を渡し、seed ごとに 1 行。dry-run は 4,096 ショット・seed 0 の 1 回
  - circuit           : 動的回路をそのまま回す（補正は if_test。論理は中間シンドロームの後選択で acceptance の列が 1 未満）
  - fidelity          : Bob を固定 3 基底で測って F ± σ（services/pipeline/shots.py。sigma の列）
  - test state        : CommonConfig.test_states（|+⟩ と |1⟩。Fig 5 は両方）
  - n_repeats         : metadata に書く値（行数は paper_seeds で決まる）
  - readout           : 較正の読み違いを雑音モデルの測定に付ける（noise_scale で割る）

レイアウト方針 (CommonConfig 参照。物理は全格子で F が最高の鎖、論理は transpile の選択 = services/pipeline/target.paper_chain。2026-11-04):
  - Physical: CommonConfig.best_path (Fig.2b/3b/4b と同一)
  - Logical : CommonConfig.best_logical_path（2026-10-20 ケイ「論理テレポも Qiskit の関数でレイアウト」。それまでは VF2 の完全部分グラフを雑音で採点）
"""
import sys
import os
import time

# プロジェクトルートをパスに追加（直接実行用）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiments.config import E3DConfig
from experiments.common.logging_utils import ExperimentLogger, paper_estimation_meta
from experiments.common.output import default_output_dir
from services.pipeline.noise import create_reduced_noise_model

EXPERIMENT_ID = "e3d_noise_scaling"

E3D_COLUMNS = [
    "noise_scale",      # ノイズスケール係数
    "circuit_type",     # "physical" / "logical"
    "state_label",
    "theta",
    "phi",
    "repeat",
    "backend",
    "physical_qubits",  # 実行に使った物理量子ビット（phys=best_path, logi=VF2 layout）
    "fidelity",
    "error_detection_rate",
    "shots",
    "seed",             # 使った seed（dry-run は 0 の 1 回）
    "sigma",            # ショット推定の統計誤差
    "acceptance",       # 中間シンドロームの受理率（物理は 1）
    "n_accepted_min",
]


def run_e3d(
    backend_name: str = None,
    dry_run: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
    seeds=None,
) -> str:
    """E3-D ノイズスケーリング テレポーテーション比較実験を実行する（ショット推定。CommonConfig.estimation）。CSV のパスを返す。
    seeds を渡すと論文の seed（CommonConfig.paper_seeds）の代わりにそれで回す（別の seed での回し直し。metadata にも書く）"""
    from services.mode3 import (
        compute_fidelity_physical, compute_fidelity_logical,
    )
    from services.pipeline.shots import ShotsConfig

    config = E3DConfig()
    paper_seeds = tuple(seeds) if seeds else config.common.paper_seeds
    if backend_name is None:
        backend_name = config.common.backend

    if output_dir is None:
        output_dir = default_output_dir(EXPERIMENT_ID, dry_run)   # dry-run は results/dry_run/<id>（本番の CSV を "w" で開かない）
    logger = ExperimentLogger(EXPERIMENT_ID, E3D_COLUMNS, output_dir=output_dir)

    noise_scales = config.dry_run_noise_scales if dry_run else config.noise_scales
    n_repeats = config.dry_run_repeats if dry_run else config.common.n_repeats
    # CommonConfig.test_states で全 paper 図と統一（E3-D は |+⟩ と |1⟩ の両方を回す）
    states = config.common.test_states

    # --- 配置（実験中は固定。手で写した凍結の値。物理は tests/test_experiments/test_e1h_fixed_best.py、論理は tests/test_scheduling_in_pipeline.py が導出と照合） ---
    physical_qubits = list(config.common.best_path)
    logical_qubits = list(config.common.best_logical_path)

    total = len(noise_scales) * len(states) * n_repeats * 2
    start_time = time.time()

    print("=== E3-D ノイズスケーリング テレポーテーション比較実験 (shots) ===")
    print(f"Backend: {backend_name}")
    print(f"Physical qubits (best_path): {physical_qubits}")
    print(f"Logical qubits  (transpile): {logical_qubits}")
    print(f"Noise scales: {noise_scales}")
    print(f"States: {[s[0] for s in states]}, Repeats: {n_repeats}")
    print(f"Simulation method: {config.common.simulation_method}")
    print(f"Total data rows: {total}")
    if dry_run:
        print("** DRY-RUN MODE **")
    print()

    # noise_scale x layout ごとに noise model を 1 回だけ構築（state/repeat で不変）
    for ns in noise_scales:
        if verbose:
            print(f"  noise_scale={ns}")

        try:
            phys_noise = create_reduced_noise_model(backend_name, physical_qubits, noise_scale=ns)
            logi_noise = create_reduced_noise_model(backend_name, logical_qubits, noise_scale=ns)
        except Exception as e:
            print(f"  [ERROR] noise model ns={ns}: {e}")
            continue

        for state_label, theta, phi in states:
            label_display = f"|{state_label}>"

            # 2026-10-28 からショット推定: 論文の seed ごとに 1 行。dry-run は 4,096 ショット・seed 0 の 1 回
            runs = [(0, 0)] if dry_run else list(enumerate(paper_seeds))
            for repeat, seed in runs:
                est = {"method": config.common.estimation,
                       "shots": ShotsConfig(shots=4096 if dry_run else config.common.paper_shots, seed=seed)}
                try:
                    phys = compute_fidelity_physical(
                        theta, phi, noise_model=phys_noise,
                        backend_name=backend_name, selected_qubits=physical_qubits,
                        noise_scale=ns, **est,
                    )
                except Exception as e:
                    if verbose:
                        print(f"    [ERROR] phys ns={ns} {label_display}: {e}")
                    phys = None
                try:
                    logi_result = compute_fidelity_logical(
                        theta, phi, noise_model=logi_noise,
                        backend_name=backend_name, selected_qubits=logical_qubits,
                        noise_scale=ns, **est,
                    )
                except Exception as e:
                    if verbose:
                        print(f"    [ERROR] logi ns={ns} {label_display}: {e}")
                    logi_result = None
                for circuit_type, res, qubits_used in (("physical", phys, physical_qubits), ("logical", logi_result, logical_qubits)):
                    if res is None:
                        continue
                    logger.log_row({
                        "noise_scale": ns,
                        "circuit_type": circuit_type,
                        "state_label": label_display,
                        "theta": theta,
                        "phi": phi,
                        "repeat": repeat,
                        "backend": backend_name,
                        "physical_qubits": str(qubits_used),
                        "fidelity": res["fidelity"],
                        "error_detection_rate": res.get("error_detection_rate", 0.0),   # 物理には符号空間なし
                        "shots": res["shots"] or 0,
                        "seed": seed,
                        "sigma": res["fidelity_sigma"],
                        "acceptance": res["acceptance"],
                        "n_accepted_min": res["n_accepted_min"] or 0,
                    })
                if verbose:
                    print(f"    ns={ns} {label_display} seed={seed}: phys={phys['fidelity'] if phys else None}, "
                          f"logi={logi_result['fidelity'] if logi_result else None}")

    elapsed = time.time() - start_time
    print(f"\nDone: {logger.row_count} rows in {elapsed:.1f}s")
    print(f"CSV: {logger.csv_path}")

    logger.save_metadata({
        "backend_name": backend_name,
        "dry_run": dry_run,
        "noise_scales": list(noise_scales),
        "n_states": len(states),
        "n_repeats": n_repeats,
        "simulation_method": config.common.simulation_method,
        **paper_estimation_meta(config.common, paper_seeds),
        "physical_qubits": physical_qubits,
        "logical_qubits": logical_qubits,
        "layout": "physical: highest F on the full E1-H grid (core.config.PAPER_BEST_PATH); logical: transpile (services/pipeline/target.paper_chain, seed 42)",
    })

    return logger.csv_path


if __name__ == "__main__":
    from experiments.common.cli import create_runner_parser
    args = create_runner_parser("E3-D: ノイズスケーリング テレポーテーション比較 (shots)").parse_args()
    run_e3d(backend_name=args.backend, dry_run=args.dry_run, verbose=args.verbose, output_dir=args.output_dir,
            seeds=args.seeds)
