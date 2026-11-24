"""Mode3 logical qubit teleportation services.

論文 §IV [[2,1,1]] 論理テレポーテーションの計算コア。サブモジュールに分割:

- 回路は services/apps/teleportation_circuits.py（2026-09-28 に移した。ここから再輸出している）
- ``fidelity``  density-matrix (true state) のフィデリティ（物理・論理）
- ``sweeps``    ノイズスケーリング掃引と Fig.5 形式 stress test、crossover 検出

ここから再輸出するのは、パッケージ経由で引かれている名前だけ（endpoints・tests・experiments/e3d。2026-11-07 に 9 個を外した。
basis_gates は services.static_backend.basis_gates_for へ移した）。それ以外はサブモジュールから直接 import する。

Web の配管を読むサブモジュール（``transpile``。回路図の表示用で、utils と core.session_manager、その先の Flask を読む）は、
ここから再輸出しない。論文の実験（E3-D の compute_fidelity_*）がこの package を読むので、再輸出すると実験と公開用の一式に
Flask が付いてくる（2026-11-24。歯止め tests/test_dependency_direction.py）。
"""
from services.apps.teleportation_circuits import (
    prepare_bloch_state,
    create_logical_teleportation_circuit,
)
from services.mode3.fidelity import (
    ideal_bloch_probs,
    compute_fidelity_logical,
    compute_fidelity_physical,
)
from services.mode3.sweeps import (
    find_crossover,
    run_noise_sweep,
    run_stress_test_sweep,
)

__all__ = [
    # circuits
    "prepare_bloch_state",
    "create_logical_teleportation_circuit",
    # fidelity
    "ideal_bloch_probs",
    "compute_fidelity_logical",
    "compute_fidelity_physical",
    # sweeps
    "find_crossover",
    "run_noise_sweep",
    "run_stress_test_sweep",
]
