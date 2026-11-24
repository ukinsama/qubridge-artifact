"""Mode1 physical qubit teleportation services.

論文 §III の物理テレポーテーションパイプラインの計算コア。サブモジュールに分割:

- ``simulation``           遅延測定回路 + DM シミュレーションによる fidelity / Bloch 角
- ``error_decomposition``  SX / CZ / Readout 損失の error budget + waterfall layer fidelity
- ``pipeline_eval``        画面の L1〜L4 の決め事を渡し値で受け、baseline → +L2 → +L3 → +L4 の累積の F（2026-10-20）
- ``pulse_matrix``         per-gate × per-shape のフィデリティマトリクス（L3 UI 用）
- ``strategies``           qubit 選択戦略の比較と VF2 layout 統計（L2 UI 用）
- ``endpoint_*``           ルートが呼ぶ、入力を受けて応答の dict を返す関数（endpoints.py が facade）

ここから再輸出するのは、パッケージ経由で引かれている名前だけ（routes・tests・experiments。2026-11-07 に 16 個を外した）。
それ以外はサブモジュールから直接 import する。

Web の配管を読むサブモジュール（``workflow_context``。utils と core.session_manager、その先の Flask を読む）は、ここから再輸出しない。
論文の実験（E1-G の simulate、E1-H の pipeline_eval）がこの package を読むので、再輸出すると実験と公開用の一式に Flask が付いてくる
（2026-11-24。歯止め tests/test_dependency_direction.py）。
"""
from services.mode1.simulation import (
    simulate,
    simulate_dm,
)
from services.mode1.error_decomposition import (
    clamp_error,
    get_baseline_qubits,
    decompose_errors,
)
from services.mode1.results import (
    build_ideal_vector_setting,
    compute_state_vector_entry,
)
from services.mode1.strategies import (
    run_strategy_comparison,
    compute_vf2_stats,
)

__all__ = [
    # simulation
    "simulate",
    "simulate_dm",
    # error_decomposition
    "clamp_error",
    "get_baseline_qubits",
    "decompose_errors",
    # results
    "build_ideal_vector_setting",
    "compute_state_vector_entry",
    # strategies
    "run_strategy_comparison",
    "compute_vf2_stats",
]
