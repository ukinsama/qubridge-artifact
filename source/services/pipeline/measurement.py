"""測定の確率: 残した密度行列の Z 基底の確率に、較正の読み出しの混同行列を掛ける。

役割: 最後に測定する応用（CHSH・QAOA）の指標の入力を作る。フィードフォワードに使う測定の誤りは
ReadoutMix（補正の誤りとして混ぜる）が担い、こちらは最後の測定だけを扱う。
境界: 較正は呼び出し側（run_app）が読んで qp_map で渡す。ここは I/O をしない。
契約: 文字列は Qiskit の並び（右端が縮めた空間の qubit 0）。混同行列は列＝用意した値、行＝読んだ値、
p = min(p × noise_scale, 0.5)。向きごとの値が無ければ readout_error を両向きに使う（noise.py と同じ）。
spec: docs/superpowers/specs/2026-09-27-multi-app-pipeline-design.md（第1部の続き）
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(qp: dict, noise_scale: float = 1.0) -> np.ndarray:
    """2×2 の混同行列 M[読んだ値][用意した値]。"""
    p10 = qp.get("prob_meas1_prep0")   # |0⟩ を用意して 1 と読む
    p01 = qp.get("prob_meas0_prep1")   # |1⟩ を用意して 0 と読む
    if p10 is None or p01 is None:
        p10 = p01 = qp.get("readout_error", 0.0)
    p10 = min(float(p10) * noise_scale, 0.5)
    p01 = min(float(p01) * noise_scale, 0.5)
    return np.array([[1 - p10, p01], [p10, 1 - p01]])


def measurement_probabilities(rho, qp_map: dict | None, kept_qubits: list[int] | None,
                              noise_scale: float = 1.0) -> dict[str, float]:
    """Z 基底で全部測ったときの確率。qp_map と kept_qubits があれば qubit ごとに混同行列を掛ける。"""
    probs = np.asarray(rho.probabilities(), dtype=float)
    k = int(round(np.log2(probs.size)))
    if qp_map is not None and kept_qubits is not None:
        # 添字のビット j が qubit j（Qiskit は little-endian）。reshape([2]*k) では軸 k-1-j が qubit j
        p = probs.reshape([2] * k)
        for j, q in enumerate(kept_qubits):
            m = confusion_matrix(qp_map.get(q, {}), noise_scale)
            axis = k - 1 - j
            p = np.moveaxis(np.tensordot(m, p, axes=([1], [axis])), 0, axis)
        probs = p.reshape(-1)
    return {format(i, f"0{k}b"): float(v) for i, v in enumerate(probs)}
