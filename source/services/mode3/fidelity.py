"""Mode3 の密度行列の忠実度（物理・論理）。services/apps/teleportation.py の PHYSICAL・LOGICAL を
services/pipeline/runner.py の run_app で回す包みで、順番は run_app の docstring を読む。

境界: 計算は run_app。ここは引数の形（theta_rad・phi_rad・noise_model・backend_name・selected_qubits・noise_scale）を
     Mode3 の UI と E3-D に合わせ、返す dict の丸めを決めるだけ。
契約: 返す値は 2026-09-27 以前と同じ（tests/test_paper_numbers.py の E3-D の 4 点）。counts ベースの Bhattacharyya の
     2 関数は呼び手が experiments/archive（削除済み）だけだったので 2026-10-20 に消した。
"""
import numpy as np

from core.config import PAPER_ESTIMATION_METHOD, PAPER_SCHEDULING_METHOD, PAPER_T2STAR_RATIO, PAPER_ZZ_HZ
from services.apps.teleportation import LOGICAL, PHYSICAL
from services.pipeline.feedforward import DEFAULT_FEEDFORWARD_LATENCY_S
from services.pipeline.idle import PAPER_IDLE_FAST
from services.pipeline.runner import run_app


# =============================================================================
# Ideal probabilities (Bloch projection)
# =============================================================================

def ideal_bloch_probs(theta_rad: float) -> tuple:
    """理想的な Bloch 球確率: (cos²(θ/2), sin²(θ/2))。"""
    return float(np.cos(theta_rad / 2) ** 2), float(np.sin(theta_rad / 2) ** 2)


# =============================================================================
# Density-matrix fidelity (true state, φ-sensitive)
# =============================================================================

def compute_fidelity_logical(theta_rad: float, phi_rad: float, noise_model=None,
                                 backend_name=None, selected_qubits=None,
                                 noise_scale: float = 1.0,
                                 feedforward_latency_s: float = DEFAULT_FEEDFORWARD_LATENCY_S,
                                 scheduling_method: str | None = PAPER_SCHEDULING_METHOD,
                                 dd: bool = False, zz_hz: float | None = PAPER_ZZ_HZ,
                                 t2star_ratio: float | None = PAPER_T2STAR_RATIO,
                                 idle_fast: dict | None = PAPER_IDLE_FAST,
                                 method: str = PAPER_ESTIMATION_METHOD, shots=None) -> dict:
    """遅延測定回路 + DM シミュレーションによる真の論理フィデリティ（method="shots" なら動的回路をショットで。2026-10-28）。

    [[2,1,1]] 符号空間 span{|00⟩, |11⟩} に射影してから比較するため、
    error_detection_rate も同時に算出（符号空間外の確率重み）。

    遅延測定 → measurement なし → NoiseModel の readout error が効かないため、
    `backend_name` + `selected_qubits` が与えられたら DM に Pauli mixing で
    readout error を後付けする（Mode1 と同じ思想）。

    Args:
        theta_rad, phi_rad: 入力 Bloch 角
        noise_model: NoiseModel または None（理想）
        backend_name: readout error 抽出用バックエンド名（オプション）
        selected_qubits: 物理量子ビット (alice_0, alice_1, med_0, med_1, bob_0, bob_1)
        noise_scale: 他の noise channel と整合させるための readout スケール（Bob の待ち時間の T1・T2 も割る）
        feedforward_latency_s: Alice と Mediator の測定のあと、補正までの古典の遅れ（秒）。既定は IBM の約 600 ns

    Returns:
        dict: fidelity, error_detection_rate, p_measured_0/1, p_ideal_0/1
    """
    r = run_app(LOGICAL, {"theta": theta_rad, "phi": phi_rad}, backend_name, selected_qubits,
                noise_model=noise_model, noise_scale=noise_scale, feedforward_latency_s=feedforward_latency_s,
                scheduling_method=scheduling_method, dd=dd, zz_hz=zz_hz, t2star_ratio=t2star_ratio, idle_fast=idle_fast,
                method=method, shots=shots)
    p_ideal_0, p_ideal_1 = ideal_bloch_probs(theta_rad)
    return {
        'fidelity': _round6(r["fidelity"]),
        'error_detection_rate': round(r["error_detection_rate"], 4),
        'p_measured_0': round(r["p_measured_0"], 4),
        'p_measured_1': round(r["p_measured_1"], 4),
        'p_ideal_0': round(p_ideal_0, 4),
        'p_ideal_1': round(p_ideal_1, 4),
        **_estimation_keys(r),
    }


def _round6(x):
    return None if x is None else round(x, 6)


def _estimation_keys(r: dict) -> dict:
    """推定法の鍵（ショット統一。dm は sigma 0・shots None）"""
    return {'fidelity_sigma': _round6(r["sigma"]), 'acceptance': round(r["acceptance"], 6),
            'detected_at_readout': _round6(r["detected_at_readout"]), 'shots': r["shots"], 'seed': r["seed"], 'method': r["method"],
            'n_accepted_min': r["n_accepted_min"]}


def compute_fidelity_physical(theta_rad: float, phi_rad: float, noise_model=None,
                                  backend_name=None, selected_qubits=None,
                                  noise_scale: float = 1.0,
                                  feedforward_latency_s: float = DEFAULT_FEEDFORWARD_LATENCY_S,
                                  scheduling_method: str | None = PAPER_SCHEDULING_METHOD,
                                 dd: bool = False, zz_hz: float | None = PAPER_ZZ_HZ,
                                 t2star_ratio: float | None = PAPER_T2STAR_RATIO,
                                 idle_fast: dict | None = PAPER_IDLE_FAST,
                                 method: str = PAPER_ESTIMATION_METHOD, shots=None) -> dict:
    """遅延測定回路 + DM シミュレーションによる真の物理テレポーテーション（method="shots" なら動的回路をショットで。2026-10-28）
    フィデリティ。Mode1 と同じ readout error post-step を選択的に適用。

    Args:
        theta_rad, phi_rad: 入力 Bloch 角
        noise_model: NoiseModel または None（理想）
        backend_name: readout error 抽出用バックエンド名
        selected_qubits: 物理量子ビット (Alice_state, Alice_ancilla, Bob)
        noise_scale: 他の noise channel と整合させるための readout スケール（Bob の待ち時間の T1・T2 も割る）
        feedforward_latency_s: Alice の測定のあと、補正までの古典の遅れ（秒）。既定は IBM の約 600 ns

    Returns:
        dict: fidelity, p_measured_0/1
    """
    r = run_app(PHYSICAL, {"theta": theta_rad, "phi": phi_rad}, backend_name, selected_qubits,
                noise_model=noise_model, noise_scale=noise_scale, feedforward_latency_s=feedforward_latency_s,
                scheduling_method=scheduling_method, dd=dd, zz_hz=zz_hz, t2star_ratio=t2star_ratio, idle_fast=idle_fast,
                method=method, shots=shots)
    bob_arr = r["state"].data if r["state"] is not None else None
    return {
        'fidelity': _round6(r["fidelity"]),
        'p_measured_0': round(float(bob_arr[0, 0].real), 4) if bob_arr is not None else None,
        'p_measured_1': round(float(bob_arr[1, 1].real), 4) if bob_arr is not None else None,
        **_estimation_keys(r),
    }


def compute_dm_fidelity_logical(theta_rad: float, phi_rad: float, *args, **kwargs) -> dict:
    """密度行列版（method="dm"）の論理テレポ。compute_fidelity_logical の包み。dm の内部を確かめるテストと検算が使う（2026-10-28）"""
    kwargs["method"] = "dm"
    return compute_fidelity_logical(theta_rad, phi_rad, *args, **kwargs)


def compute_dm_fidelity_physical(theta_rad: float, phi_rad: float, *args, **kwargs) -> dict:
    """密度行列版（method="dm"）の物理テレポ。compute_fidelity_physical の包み"""
    kwargs["method"] = "dm"
    return compute_fidelity_physical(theta_rad, phi_rad, *args, **kwargs)
