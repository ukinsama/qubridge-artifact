"""アプリの定義の型: 回路・補正・待ち・残す qubit・理想・読み出し・指標を1つにまとめる。

役割: どのアプリも同じ共通の実行部品（services/pipeline/runner.py の run_app）で回せるよう、アプリ固有の部分だけを持つ。
境界: 計算はしない。関数を持つだけ。位置の範囲は __post_init__ で検査する。
契約: 読み出しの ops は Hermitian な Pauli（op† = op）。metric は MetricInput を1つ受け、score（0〜1）を必ず返す。
spec: docs/superpowers/specs/2026-09-27-multi-app-pipeline-design.md（第1部の続き）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class ReadoutMix:
    """フィードフォワードに使う測定の読み出し誤りを、補正の誤りとして残す qubit に混ぜる。

    op i は measured[i] の測定が決める補正。p_i = min(readout_error[measured[i]] * noise_scale, 0.5) を
    独立に混ぜる（部分集合すべて。runner.apply_readout_mix）。measured と ops は同じ数。
    """
    measured: tuple[int, ...]       # 選んだ qubit の中の位置（op ごとに 1 つ）
    ops: tuple[np.ndarray, ...]     # 残す qubit の空間の演算子


@dataclass(frozen=True)
class MetricInput:
    """指標の入力。run_app が組む。"""
    state: object                           # 残した qubit の DensityMatrix（並びは位置の昇順）
    ideal: object | None                    # 理想の DensityMatrix。定義が ideal を持たなければ None
    params: Mapping[str, float]
    acceptance: float                       # 受理率（postselect が無ければ 1）
    probabilities: dict[str, float] | None  # measure_final のときの測定の確率（Qiskit の並び）。ほかは None
    kept_qubits: list[int] | None           # 残した qubit の物理番号（state の並びと同じ）。qubits が無ければ None


RESERVED_METRIC_KEYS = frozenset({"state", "acceptance", "probabilities", "kept_qubits",
                                  "sigma", "bloch", "bloch_sigma", "acceptance_sigma", "detected_at_readout", "shots", "seed", "method",
                                  "n_accepted_min"})


@dataclass(frozen=True, eq=False)   # eq=False: ndarray を持つので値で比べず、同一性で hash する（キャッシュの鍵に使える）
class AppSpec:
    id: str
    n_qubits: int
    circuit_edges: tuple[tuple[int, int], ...]      # 回路が 2 qubit ゲートを掛ける辺
    core_circuit: Callable[[Mapping[str, float]], object]   # 補正の手前までの QuantumCircuit
    correction: Callable[[], object] | None         # 誤り無しで当てる補正。None ならフィードフォワード無し
    measured: tuple[int, ...]                       # 待ちの長さを決める測定の位置
    waiting: tuple[int, ...]                        # 待つ qubit（仮想の番号）
    kept: tuple[int, ...]                           # 最後に残す qubit
    ideal: Callable[[Mapping[str, float]], object] | None   # kept の空間の理想の DensityMatrix。無い応用（CHSH・QAOA）は None
    readout: ReadoutMix | None
    uses_layout: bool                               # 選んだ qubit の部分グラフでトランスパイルするか
    postselect: Callable[[np.ndarray, list[int], int], tuple[np.ndarray, float]] | None
    metric: Callable[[MetricInput], dict]           # 返す dict は score（0〜1）を必ず持つ。RESERVED_METRIC_KEYS は使えない
    measure_final: bool = False                     # True なら最後の測定の確率（混同行列つき）を MetricInput.probabilities に渡す
    observables: tuple = ()                         # ショット推定の固定の測り方（services.pipeline.shots.Observable の組）。空なら shots では推定できない
    accept_mid: Callable[[tuple[int, ...]], bool] | None = None   # ショット推定の後選択（中間測定の bit、measured の順）。None なら全部受理

    def __post_init__(self):
        n = self.n_qubits
        if n < 1:
            raise ValueError(f"{self.id}: n_qubits は 1 以上")
        if not self.kept:
            raise ValueError(f"{self.id}: kept が空")
        if len(set(self.kept)) != len(self.kept):
            raise ValueError(f"{self.id}: kept {self.kept} に同じ位置が重なっている")
        if self.readout is not None and not self.readout.measured:
            raise ValueError(f"{self.id}: readout.measured が空（読み出しを混ぜる測定が無いなら readout=None）")
        if self.readout is not None and len(self.readout.measured) != len(self.readout.ops):
            raise ValueError(f"{self.id}: readout.measured が {len(self.readout.measured)} 個、ops が {len(self.readout.ops)} 個で対応が取れない")
        for name, positions in (("measured", self.measured), ("waiting", self.waiting), ("kept", self.kept),
                                ("readout.measured", self.readout.measured if self.readout else ())):
            bad = [p for p in positions if not 0 <= p < n]
            if bad:
                raise ValueError(f"{self.id}: {name} の位置 {bad} が 0..{n - 1} の外")
        for a, b in self.circuit_edges:
            if not (0 <= a < n and 0 <= b < n) or a == b:
                raise ValueError(f"{self.id}: circuit_edges の辺 ({a}, {b}) が不正")
