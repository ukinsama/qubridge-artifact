"""Baseline の鎖の列挙（E1-H の段 0 と E1-G の段 0 が共有する）。

役割: services/pipeline/qubits.baseline_paths（閾値で絞った鎖）に both_orientations（両向き）を重ねた 1 つの式を置く。
境界: 閾値の意味と列挙は services 側。ここは論文の閾値（CommonConfig.baseline_thresholds）を既定にして呼ぶだけ。
存在理由: 2026-11-07 まで e1h runner._baseline_paths と e1g runner_v6.get_filtered_paths が同じ式を別々に持っていた。
     その 2 つの名前は knob_inventory とテストが import するので、包みとして残している。
契約: tests/test_experiments/test_chains_common.py
"""
from experiments.config import BaselineThresholds, CommonConfig
from services.pipeline.qubits import baseline_paths, both_orientations


def baseline_chains(backend_name: str, thresholds: BaselineThresholds | None = None):
    """Baseline の鎖を両向きで [(鎖, 最大 2Q 誤り), ...]。正向きと逆向きが隣り合う（both_orientations の並び）。
    2026-10-20 から向きも L2 の構成に入れる（transpile は向きを選ぶ。10/16 の境目の監査 4）"""
    if thresholds is None:
        thresholds = CommonConfig().baseline_thresholds
    return both_orientations(baseline_paths(backend_name, thresholds))
