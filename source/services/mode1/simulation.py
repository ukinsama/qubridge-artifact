"""Mode1 の物理テレポーテーションの密度行列シミュレーション（Bob の F と Bloch 角）。

役割: 較正からノイズモデルを作り（gate_pulse_map つき）、共通の実行部品 services/pipeline/runner.py の run_app で
物理テレポーテーションの定義（services/apps/teleportation.py の PHYSICAL）を回す包み。
境界: 計算の順番（補正の手前まで → 待ち → 誤り無しの補正 → Alice を捨てる → 読み出しの混ぜ方）は run_app が持つ。
契約: 返す形 (F, θ_deg, φ_deg, 長さ) と値は 2026-09-27 以前と同じ（tests/test_paper_numbers.py）。
"""
import numpy as np

from core.config import PAPER_ESTIMATION_METHOD, PAPER_SCHEDULING_METHOD, PAPER_T2STAR_RATIO, PAPER_ZZ_HZ
from services.apps.teleportation import PHYSICAL
from services.pipeline.feedforward import DEFAULT_FEEDFORWARD_LATENCY_S
from services.pipeline import noise
from services.pipeline.idle import PAPER_IDLE_FAST
from services.pipeline.runner import run_app
from services.pipeline.shots import ShotsConfig


def _extract_bloch_angles(dm):
    """Extract Bloch sphere angles from a single-qubit density matrix.

    Returns:
        (theta_deg, phi_deg, bloch_length)
    """
    from qiskit.quantum_info import Pauli

    n_x = dm.expectation_value(Pauli('X')).real
    n_y = dm.expectation_value(Pauli('Y')).real
    n_z = dm.expectation_value(Pauli('Z')).real
    bloch_length = float(np.sqrt(n_x**2 + n_y**2 + n_z**2))
    theta_out = float(np.degrees(np.arccos(np.clip(n_z / max(bloch_length, 1e-10), -1, 1))))
    phi_out = float(np.degrees(np.arctan2(n_y, n_x)))
    return round(theta_out, 2), round(phi_out, 2), round(bloch_length, 4)



def simulate_dm(theta: float, phi: float, backend_name: str, qubits: list,
                gate_pulse_map: dict | None = None,
                feedforward_latency_s: float = DEFAULT_FEEDFORWARD_LATENCY_S,
                scheduling_method: str | None = PAPER_SCHEDULING_METHOD, dd: bool = False,
                zz_hz: float | None = PAPER_ZZ_HZ, t2star_ratio: float | None = PAPER_T2STAR_RATIO,
                idle_fast: dict | None = PAPER_IDLE_FAST):
    """密度行列で物理テレポーテーションを回し、Bob の F と Bloch 角を返す。

    Args:
        feedforward_latency_s: Alice の測定のあと、補正までの古典の遅れ（秒）。較正に無いので、
            既定は IBM の公式の値 約 600 ns（services/pipeline/feedforward.py）。
        scheduling_method: 時刻決め（asap／alap）。既定は run_app と同じ論文の値（core.config.PAPER_SCHEDULING_METHOD）。None で delay 無し。
        dd / zz_hz / t2star_ratio: L3 の DD と、機体の ZZ・DD 無しの空きの T2*（既定は run_app と同じ論文の代表値。None で 10/20 のモデル）。
        idle_fast: 空きの速い位相緩和（鍵は DD の有無。既定は論文の代表値 idle.PAPER_IDLE_FAST。None で 10/21 のモデル。2026-10-28）。

    Returns:
        tuple: (fidelity, theta_deg, phi_deg, bloch_length)
    """
    r = simulate(theta, phi, backend_name, qubits, gate_pulse_map=gate_pulse_map, method="dm",
                 feedforward_latency_s=feedforward_latency_s, scheduling_method=scheduling_method,
                 dd=dd, zz_hz=zz_hz, t2star_ratio=t2star_ratio, idle_fast=idle_fast)
    return r["fidelity"], r["theta_deg"], r["phi_deg"], r["bloch_length"]


def simulate(theta: float, phi: float, backend_name: str, qubits: list, *,
             gate_pulse_map: dict | None = None,
             method: str = PAPER_ESTIMATION_METHOD, shots: ShotsConfig | None = None,
             noise_scale: float = 1.0, readout_scale: float | None = None,
             feedforward_latency_s: float = DEFAULT_FEEDFORWARD_LATENCY_S,
             scheduling_method: str | None = PAPER_SCHEDULING_METHOD, dd: bool = False,
             zz_hz: float | None = PAPER_ZZ_HZ, t2star_ratio: float | None = PAPER_T2STAR_RATIO,
             idle_fast: dict | None = PAPER_IDLE_FAST) -> dict:
    """物理テレポーテーションを 1 回回し、run_app の dict に Bob の Bloch 角（theta_deg・phi_deg・bloch_length）を足して返す。
    method: "dm"（密度行列）か "shots"（動的回路をショットで。shots は ShotsConfig）。受理 0 で state が None なら Bloch 角も None。
    2026-10-28 のショット統一で足した正本。simulate_dm は method="dm" の薄い包み（4 つ組を返す）"""
    # モジュール越しに呼ぶ: テスト（tests/test_feedforward.py）が noise.create_reduced_noise_model を差し替える
    noise_model = noise.create_reduced_noise_model(backend_name, qubits, gate_pulse_map=gate_pulse_map, noise_scale=noise_scale)
    r = run_app(PHYSICAL, {"theta": float(theta), "phi": float(phi)}, backend_name, list(qubits),
                noise_model=noise_model, noise_scale=noise_scale, readout_scale=readout_scale,
                feedforward_latency_s=feedforward_latency_s, scheduling_method=scheduling_method,
                dd=dd, zz_hz=zz_hz, t2star_ratio=t2star_ratio, idle_fast=idle_fast, method=method, shots=shots)
    if r["state"] is not None:
        r["theta_deg"], r["phi_deg"], r["bloch_length"] = _extract_bloch_angles(r["state"])
    else:
        r["theta_deg"], r["phi_deg"], r["bloch_length"] = None, None, None
    return r
