"""テレポーテーションのアプリの定義（物理 3 qubit と、[[2,1,1]] 符号の論理 6 qubit）。

役割: 回路・補正・待つ qubit・残す qubit・理想・読み出しの混ぜ方・指標を AppSpec にまとめる。
境界: 計算は services/pipeline/runner.py の run_app がする。回路の正本は services/apps/teleportation_circuits.py。
契約: Mode1 の simulate_dm と Mode3 の compute_dm_fidelity_physical/logical がこの定義を使う（2026-09-27 に畳んだ）。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from services.apps.teleportation_circuits import (
    bloch_state,
    create_teleportation_core_circuit,
    create_deferred_correction_circuit,
    create_logical_teleportation_core_circuit,
    create_deferred_logical_correction_circuit,
)
from services.pipeline.app_spec import AppSpec, ReadoutMix
from services.pipeline.shots import Observable

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# ショット推定の固定の測り方（入力によらず同じ組。2026-10-28 ケイ「回路の中身はインプットに対して不変」）
OBS_1Q = (Observable("X", ("X",), (0,), X), Observable("Y", ("Y",), (0,), Y), Observable("Z", ("Z",), (0,), Z))
# [[2,1,1]] の論理パウリ（数式の順は kept の位置の昇順 = Bob の 1 つ目・2 つ目）: Z_L = Z⊗I、X_L = X⊗X、Y_L = Y⊗X（= i X_L Z_L）。
# matrix は Qiskit の little-endian（np.kron の左が kept 空間の qubit 1）なので、数式と左右が逆になる。Bob の最終パリティは F の条件に入れない（決定③）
OBS_211 = (Observable("Z_L", ("Z", "Z"), (0,), np.kron(I2, Z)),
           Observable("X_L", ("X", "X"), (0, 1), np.kron(X, X)),
           Observable("Y_L", ("Y", "X"), (0, 1), np.kron(X, Y)))


def _physical_core(params):
    return create_teleportation_core_circuit(params["theta"], params["phi"])


def _physical_correction():
    return create_deferred_correction_circuit()


def _physical_ideal(params):
    return DensityMatrix(Statevector(bloch_state(params["theta"], params["phi"])))


def _fidelity_metric(m):
    f = float(state_fidelity(m.ideal, m.state))
    return {"score": f, "fidelity": f}


PHYSICAL = AppSpec(
    id="teleportation",
    n_qubits=3,
    circuit_edges=((0, 1), (1, 2)),
    core_circuit=_physical_core,
    correction=_physical_correction,
    measured=(0, 1),          # Alice の2つを並行して読む
    waiting=(2,),             # Bob が待つ
    kept=(2,),
    ideal=_physical_ideal,
    readout=ReadoutMix(measured=(0, 1), ops=(Z, X)),   # m0 → Z の補正、m1 → X の補正
    uses_layout=False,
    postselect=None,
    metric=_fidelity_metric,
    observables=OBS_1Q,
)


def postselect_alice00(flips: tuple[float, float]):
    """Alice の 2 qubit（仮想 0・1）を観測値 00 で後選択する関数（AppSpec.postselect の形: (arr, pos, n_qubits) → (射影後, 受理率)）。

    flips = (p0, p1) は読み違いの確率。真の結果 m の成分は、観測 00 と読み違える重み Π(1−p_i)（m_i = 0）・p_i（m_i = 1）で混ざる。
    読み出しが理想なら純粋な射影で、受理率は理想で 1/4。κ の格子（layer_coupling_matrix_l4.py、2026-10-16）から本番へ移した（2026-10-20）
    """
    p0, p1 = flips

    def ps(arr, pos, n_qubits):
        idx = np.arange(2 ** n_qubits)
        bit0, bit1 = (idx >> pos[0]) & 1, (idx >> pos[1]) & 1     # Qiskit は qubit 0 が最下位ビット
        out = np.zeros_like(arr)
        for m0 in (0, 1):
            for m1 in (0, 1):
                w = (p0 if m0 else 1 - p0) * (p1 if m1 else 1 - p1)
                if w == 0:
                    continue
                sel = (bit0 == m0) & (bit1 == m1)
                out[np.ix_(sel, sel)] += w * arr[np.ix_(sel, sel)]
        return out, float(np.real(np.trace(out)))
    return ps


def physical_pauli_frame() -> AppSpec:
    """後処理の補正（Pauli frame）の物理テレポ（L4 の「補正の実装」）: Bob は Alice の結果を待たずにすぐ測り、補正の X・Z は
    Bob の測定結果の読み替え（X は Z・Y 基底の結果を反転、Z は X・Y 基底の結果を反転）で済ませる。密度行列版では待ちの減衰を
    当てない（waiting=()）だけで、補正と、読み違いで補正を誤る混合はそのまま。Bob の状態をすぐ測る（F の推定）場合に使える。
    2026-10-28 ケイ「進めはそれがいいです」"""
    return replace(PHYSICAL, waiting=())


def physical_postselected(flips: tuple[float, float]) -> AppSpec:
    """後選択あり（L4）の物理テレポ: 補正が要らないので補正・待ち・読み違いの混合を外し、Alice 00 で射影する。
    flips は run_app に渡す readout_scale と同じ倍率で出す（runner.readout_flip_probabilities）"""
    return replace(PHYSICAL, correction=None, measured=(0, 1), waiting=(), readout=None, postselect=postselect_alice00(flips),
                   accept_mid=lambda bits: tuple(bits) == (0, 0))   # measured は中間レジスタのために残す（correction が None なので dm は素通り）


# 論理テレポーテーションの VF2 の辺（T 字、Ltelepo.ipynb 準拠）: Alice{0,1} - Med{2,3} - Bob{4,5}、3 で分岐
#   (0,1) Alice 内の符号化  (1,2) Alice → Mediator  (2,3) Mediator 内のベル対  (3,4)・(3,5) Mediator → Bob
LOGICAL_CIRCUIT_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5))


def logical_bloch_state(theta: float, phi: float) -> np.ndarray:
    """論理の |ψ_L⟩ = cos(θ/2)|00⟩ + e^{iφ}·sin(θ/2)|11⟩"""
    c, s = bloch_state(theta, phi)
    return np.array([c, 0, 0, s])


def project_onto_valid_syndrome(arr: np.ndarray, layout: list[int], n_qubits: int):
    """X の誤り（パリティ違反）のシンドロームを持つ成分を捨てる。条件は Alice の2つが同じ、かつ Mediator の2つが同じ。

    [[2,1,1]] は Z の誤りを検出できないので、捨てるのはビット反転由来の成分だけ（|+⟩ で論理が物理を下回る理由）。

    Returns:
        (rho_post, p_valid): 正規化前の射影した行列と、受理される確率の重み
    """
    phys_to_virt = [0] * n_qubits
    for virt, pos in enumerate(layout):
        if pos < n_qubits:
            phys_to_virt[pos] = virt
    dim = 1 << n_qubits
    valid = np.zeros(dim, dtype=bool)
    for idx in range(dim):
        bits = [0] * n_qubits
        for k in range(n_qubits):
            bits[phys_to_virt[k]] = (idx >> k) & 1
        # 仮想の番号: 0,1 = Alice、2,3 = Mediator、4,5 = Bob
        valid[idx] = bits[0] == bits[1] and bits[2] == bits[3]
    rho_post = np.zeros_like(arr)
    rho_post[np.ix_(valid, valid)] = arr[np.ix_(valid, valid)]
    return rho_post, float(np.real(np.trace(rho_post)))


def project_bob_onto_codespace(bob_arr: np.ndarray, ideal_dm):
    """Bob の 4×4 を符号空間 |00⟩,|11⟩ に射影してから理想と比べる。

    Returns:
        (fidelity, p_meas_0, p_meas_1, p_codespace)
    """
    p_00 = float(bob_arr[0, 0].real)
    p_11 = float(bob_arr[3, 3].real)
    p_codespace = p_00 + p_11
    if p_codespace <= 1e-10:
        return 0.0, 0.0, 0.0, p_codespace
    proj = np.zeros((4, 4), dtype=complex)
    proj[0, 0] = 1
    proj[3, 3] = 1
    bob_ps = (proj @ bob_arr @ proj) / p_codespace
    return float(state_fidelity(ideal_dm, DensityMatrix(bob_ps))), p_00 / p_codespace, p_11 / p_codespace, p_codespace


def _logical_core(params):
    return create_logical_teleportation_core_circuit(params["theta"], params["phi"])


def _logical_correction():
    return create_deferred_logical_correction_circuit()


def _logical_ideal(params):
    return DensityMatrix(Statevector(logical_bloch_state(params["theta"], params["phi"])))


def _logical_metric(m):
    fidelity, p0, p1, p_code = project_bob_onto_codespace(m.state.data, m.ideal)
    return {"score": fidelity, "fidelity": fidelity,
            "error_detection_rate": max(0.0, 1.0 - m.acceptance * p_code),
            "p_measured_0": p0, "p_measured_1": p1}


LOGICAL = AppSpec(
    id="teleportation_logical",
    n_qubits=6,
    circuit_edges=LOGICAL_CIRCUIT_EDGES,
    core_circuit=_logical_core,
    correction=_logical_correction,
    measured=(0, 1, 2, 3),    # Alice と Mediator の4つを並行して読む
    waiting=(4, 5),
    kept=(4, 5),
    ideal=_logical_ideal,
    readout=ReadoutMix(measured=(0, 2), ops=(np.kron(Z, np.eye(2)), np.kron(X, X))),   # Z_L = Z⊗I、X_L = X⊗X
    uses_layout=True,
    postselect=project_onto_valid_syndrome,
    metric=_logical_metric,
    observables=OBS_211,
    accept_mid=lambda bits: bits[0] == bits[1] and bits[2] == bits[3],   # Alice の 2 bit が同じ、Mediator の 2 bit が同じ（project_onto_valid_syndrome と同じ条件）
)
