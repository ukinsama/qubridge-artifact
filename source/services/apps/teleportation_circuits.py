"""テレポーテーションの回路（物理 3 qubit と、[[2,1,1]] 符号の論理 6 qubit）。

役割: 回路を作る関数だけを置く。物理は旧 utils/circuit.py、論理は旧 services/mode3/circuits.py から集めた（2026-09-28）。
境界: 定義（AppSpec）は services/apps/teleportation.py、計算は services/pipeline/runner.py。ここは I/O をしない。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


def bloch_state(theta: float, phi: float) -> np.ndarray:
    """Ry(θ)·Rz(φ)|0⟩ の状態ベクトル cos(θ/2)|0⟩ + e^{iφ}·sin(θ/2)|1⟩（大域位相を除く）"""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def prepare_bloch_state(theta_rad: float, phi_rad: float):
    """1量子ビットのBloch球状態を準備し、状態ベクトルを返す。

    L1で送信量子状態を設定する際に使用。

    Args:
        theta_rad: 極角（ラジアン）
        phi_rad: 方位角（ラジアン）

    Returns:
        dict: alpha, beta の複素数成分
    """
    alpha, beta = (complex(v) for v in bloch_state(theta_rad, phi_rad))

    return {
        'alpha': {'real': round(alpha.real, 4), 'imag': round(alpha.imag, 4)},
        'beta': {'real': round(beta.real, 4), 'imag': round(beta.imag, 4)},
        'state_notation': (
            f"|ψ⟩ = {alpha.real:.3f}|0⟩ + "
            f"({beta.real:.3f}{beta.imag:+.3f}i)|1⟩"
        ),
    }


def create_logical_teleportation_circuit(theta: float, phi: float):
    """[[2,1,1]] 論理テレポーテーション回路を作成する (CX_L + H_L 方式)。

    プロトコル:
      1. Alice: ry(θ)·rz(φ)·CX(0,1) で |ψ_L⟩ = α|00⟩+β|11⟩ を準備
      2. Mediator-Bob に論理 Bell pair |Φ+_L⟩ = (|0000⟩+|1111⟩)/√2
      3. 論理 Bell 測定 on (Alice, Mediator):
         - CX_L (transversal): CX(A_0,M_0); CX(A_1,M_1)
         - H_L (decode-encode): CX(A_0,A_1); H(A_0); CX(A_0,A_1)
         - Z 基底測定 on Alice, Mediator
      4. Bob 論理 Pauli 補正: X_L^{m_M} Z_L^{m_A} (X_L=X⊗X, Z_L=Z⊗I)

    H_L は [[2,1,1]] では Eastin-Knill より transversal 不可能。
    decode-encode (3 ゲート) で非 transversal unitary 実装。

    6量子ビット構成:
      Alice: qa0(0), qa1(1)  |  Mediator: qm0(2), qm1(3)  |  Bob: qb0(4), qb1(5)

    Args:
        theta: 極角（ラジアン）
        phi: 方位角（ラジアン）
    """
    qa = QuantumRegister(2, "alice")
    qm = QuantumRegister(2, "med")
    qb = QuantumRegister(2, "bob")

    cr_meas_a = ClassicalRegister(2, "meas_a")
    cr_meas_m = ClassicalRegister(2, "meas_m")
    cr_bob = ClassicalRegister(2, "c_bob")
    qc = QuantumCircuit(qa, qm, qb, cr_meas_a, cr_meas_m, cr_bob)

    # Step 1: Alice state preparation + encoding
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.cx(0, 1)
    qc.barrier(label="Alice encoded")

    # Step 2: Mediator-Bob logical Bell pair
    # (|0000⟩+|1111⟩)/√2 — Med内Bell pair → トランスバーサルCNOTでBobへ拡張
    # Ltelepo.ipynb 準拠の構成順: cx(2,3) → cx(2,4) → cx(3,5)
    qc.h(2)
    qc.cx(2, 3)
    qc.cx(2, 4)
    qc.cx(3, 5)
    qc.barrier(label="Bell pair")

    # Step 3: Logical Bell measurement (Method 2: CX_L + H_L decode-encode)
    # Eastin-Knill 定理より [[2,1,1]] ZZ code では H_L は transversal に作れない。
    # 非 transversal だが unitary な decode-encode 実装で H_L を構成する:
    #   H_L = CX(A_0, A_1) ; H(A_0) ; CX(A_0, A_1)
    # Codeword 入力 α|00⟩+β|11⟩ に対して H_L は正しく (α+β)|00⟩+(α-β)|11⟩ = H_L|ψ_L⟩ を返す。
    # CX_L (Alice → Mediator) は transversal: CX(A_0,M_0); CX(A_1,M_1)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.barrier(label="CX_L")
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier(label="H_L decode-encode")
    qc.measure(0, cr_meas_a[0])
    qc.measure(1, cr_meas_a[1])
    qc.measure(2, cr_meas_m[0])
    qc.measure(3, cr_meas_m[1])
    qc.barrier(label="Measure A, M")

    # Step 4: Bob logical Pauli correction
    # 測定結果 (m_A = cr_meas_a[0], m_M = cr_meas_m[0]) で Bob は X_L^{m_M} Z_L^{m_A} |ψ_L⟩.
    # 逆を掛けて復元: X_L^{m_M} 先 → Z_L^{m_A}.
    # X_L = X ⊗ X, Z_L = Z ⊗ I.
    # 符号空間内で a[0]==a[1], m[0]==m[1] が理想。エラー時は a[0]/m[0] を logical bit と採用。
    with qc.if_test((cr_meas_m[0], 1)):
        qc.x(4)
        qc.x(5)
    with qc.if_test((cr_meas_a[0], 1)):
        qc.z(4)
    qc.barrier(label="Corrected")

    # Step 5: Bob measurement
    qc.measure(4, cr_bob[0])
    qc.measure(5, cr_bob[1])

    return qc


def create_logical_teleportation_core_circuit(theta: float, phi: float):
    """論理テレポーテーションの、補正の手前までの回路（符号化・論理ベル対・論理ベル測定の手前の操作）。

    ノイズつきで回すのはここまで。補正は実際には測定と古典の制御なので、ゲートの誤りを載せない
    （services/mode3/fidelity.py が services/pipeline/feedforward.py で誤り無しで当てる）。

    Args:
        theta: 極角（ラジアン）
        phi: 方位角（ラジアン）

    Returns:
        QuantumCircuit (6量子ビット: alice 0–1, med 2–3, bob 4–5。古典レジスタなし)
    """
    qa = QuantumRegister(2, "alice")
    qm = QuantumRegister(2, "med")
    qb = QuantumRegister(2, "bob")
    qc = QuantumCircuit(qa, qm, qb)

    # State prep + encoding
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.cx(0, 1)

    # Mediator-Bob logical Bell pair
    qc.h(2)
    qc.cx(2, 3)
    qc.cx(2, 4)
    qc.cx(3, 5)

    # Logical Bell measurement: CX_L (transversal) + H_L (decode-encode)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def create_deferred_logical_correction_circuit():
    """論理の補正（X_L^{m_M} のあと Z_L^{m_A}）を、遅延測定の原理で量子制御ゲートに置き換えた部分。

    遅延測定置換:
      measure(qm0) + if(1): X(qb0); X(qb1)  ≡  CX(qm0, qb0); CX(qm0, qb1)
      measure(qa0) + if(1): Z(qb0)          ≡  CZ(qa0, qb0)
    """
    qc = QuantumCircuit(6)
    qc.cx(2, 4)
    qc.cx(2, 5)
    qc.cz(0, 4)
    return qc



def create_teleportation_circuit(theta_angle: float, phi_angle: float):
    """
    Create quantum teleportation circuit with rotation.

    Args:
        theta_angle: Theta rotation angle for RY gate (radians) - ブロッホ球の極角θ
        phi_angle: Phi rotation angle for RZ gate (radians) - ブロッホ球の方位角φ

    Returns:
        QuantumCircuit: The teleportation circuit
    """
    qr = QuantumRegister(3)
    a_0 = ClassicalRegister(1)
    a_1 = ClassicalRegister(1)
    b_0 = ClassicalRegister(1)
    qc = QuantumCircuit(qr, a_0, a_1, b_0)
    # θ（極角）はRy回転、φ（方位角）はRz回転
    qc.ry(theta_angle, 0)
    qc.rz(phi_angle, 0)
    qc.barrier()
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(1, a_1)
    qc.measure(0, a_0)
    qc.barrier()
    with qc.if_test((a_1, 1)):
        qc.x(2)
    with qc.if_test((a_0, 1)):
        qc.z(2)
    return qc


def create_teleportation_core_circuit(theta_angle: float, phi_angle: float):
    """テレポーテーションの、補正の手前までの回路（状態準備・ベル対・Alice の操作）。

    ノイズつきで回すのはここまで。補正は実際には測定と古典の制御なので、ゲートの誤りを載せない
    （services/pipeline/runner.py の run_app が誤り無しで当てる）。

    Args:
        theta_angle: Ry回転角 (radians) - ブロッホ球の極角θ
        phi_angle: Rz回転角 (radians) - ブロッホ球の方位角φ

    Returns:
        QuantumCircuit: 3量子ビット、古典レジスタなし
    """
    qc = QuantumCircuit(3)
    # 状態準備
    qc.ry(theta_angle, 0)
    qc.rz(phi_angle, 0)
    # ベル状態生成
    qc.h(1)
    qc.cx(1, 2)
    # テレポーテーション操作
    qc.cx(0, 1)
    qc.h(0)
    return qc


def create_deferred_correction_circuit():
    """Alice の測定と古典の補正を、遅延測定の原理で量子制御ゲートに置き換えた部分。

    理論的等価性:
        measure(q1) + if(1): X(q2)  ≡  CNOT(q1, q2)
        measure(q0) + if(1): Z(q2)  ≡  CZ(q0, q2)
    """
    qc = QuantumCircuit(3)
    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc


def create_deferred_teleportation_circuit(theta_angle: float, phi_angle: float):
    """Create teleportation circuit using deferred measurement principle.

    mid-circuit測定の代わりに量子制御ゲート(CNOT, CZ)を使用する等価回路。
    補正の手前（create_teleportation_core_circuit）と補正（create_deferred_correction_circuit）をつないだもの。
    回路の表示や transpile の指標に使う。fidelity の計算（simulate_dm）は、補正を誤り無しで別に当てる。

    Args:
        theta_angle: Ry回転角 (radians) - ブロッホ球の極角θ
        phi_angle: Rz回転角 (radians) - ブロッホ球の方位角φ

    Returns:
        QuantumCircuit: deferred measurement版テレポーテーション回路（3量子ビット、古典レジスタなし）
    """
    return create_teleportation_core_circuit(theta_angle, phi_angle).compose(create_deferred_correction_circuit())
