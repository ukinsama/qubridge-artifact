"""Noise scaling sweep + Fig.5 stress test orchestration.

- ``find_crossover``        ノイズスイープ点列から物理→論理交差点を線形補間で抽出
- ``run_noise_sweep``       任意 (θ, φ) で noise scale を掃引し DM フィデリティ列を返す
- ``run_stress_test_sweep`` 論文 Fig.5 の |+⟩ vs |1⟩ selective protection 二段プロット
"""
import logging

import numpy as np

from services.mode3.fidelity import (
    compute_fidelity_logical,
    compute_fidelity_physical,
)

logger = logging.getLogger(__name__)


# Fig.5 (paper §IV.D / e3d_noise_scaling) の selective protection を再現するため、
# stress-test 状態 |+⟩ (Z-error 感受 → logical 不利) と |1⟩ (T1 decay → logical 有利)
# を同じノイズスケールで掃引する。CommonConfig.test_states と整合。
STRESS_TEST_STATES = (
    {"label": "+", "display": "|+>", "theta_deg": 90.0, "phi_deg": 0.0},
    {"label": "1", "display": "|1>", "theta_deg": 180.0, "phi_deg": 0.0},
)


def find_crossover(points: list) -> dict | None:
    """ノイズスイープから物理 → 論理のクロスオーバーポイントを検出する。

    物理フィデリティが論理フィデリティを下回る最初の点を線形補間で求める。

    Args:
        points: [{'noise_scale': float, 'physical_fidelity': float, 'logical_fidelity': float}, ...]

    Returns:
        {'noise_scale': float, 'fidelity': float} or None if no crossover
    """
    for i in range(1, len(points)):
        phys_prev = points[i - 1]['physical_fidelity']
        phys_curr = points[i]['physical_fidelity']
        logi_prev = points[i - 1]['logical_fidelity']
        logi_curr = points[i]['logical_fidelity']

        # 失敗点 (どちらかが None) は補間できないのでスキップ。
        # 呼び出し側のフィルタに依存せず、この関数単体でも None 混じりの
        # points を安全に扱えるようにしておく。
        if None in (phys_prev, phys_curr, logi_prev, logi_curr):
            continue

        diff_prev = phys_prev - logi_prev
        diff_curr = phys_curr - logi_curr

        if diff_prev > 0 and diff_curr <= 0:
            interp = diff_prev / (diff_prev - diff_curr) if diff_prev != diff_curr else 0.5
            scale = points[i - 1]['noise_scale'] + interp * (points[i]['noise_scale'] - points[i - 1]['noise_scale'])
            fid = phys_prev + interp * (phys_curr - phys_prev)
            return {'noise_scale': round(float(scale), 4), 'fidelity': round(float(fid), 4)}

    return None


def _sweep(backend_name: str, selected: list, noise_scales, gate_pulse_map: dict | None, states: list, shots=None) -> list:
    """scale ごとにノイズモデルを 1 回作り、各状態の物理・論理の F を返す（states の順に点列を並べた list）。

    ノイズモデルは状態（θ, φ）に依らないので、|+⟩ と |1⟩ の 2 状態でも scale ごとに 1 回で足りる
    （2026-10-20 まで状態ごとに作り直し、UI の 1 リクエストの約 2 割が重複だった）。
    noise_scale は fidelity の計算にも渡す: ゲートの雑音は noise_model でスケール済みだが、遅延測定回路の DM に後付けする
    読み出し誤り（runner の readout_flip_probabilities）と待ちの T1・T2 はこの引数でしかスケールされない。
    失敗した scale の点は両 fidelity が None（find_crossover はそれを飛ばす）。
    """
    from services.pipeline.noise import create_reduced_noise_model, DEFAULT_GATE_PULSE_MAP

    gpm = gate_pulse_map or DEFAULT_GATE_PULSE_MAP
    qubits_for_noise = selected[:6] if len(selected) >= 6 else list(range(6))
    series = [[] for _ in states]
    for ns in noise_scales:
        ns_val = float(ns)
        failed = {'noise_scale': round(ns_val, 4), 'physical_fidelity': None, 'logical_fidelity': None,
                  'physical_sigma': None, 'logical_sigma': None, 'logical_acceptance': None}
        try:
            noise_model = create_reduced_noise_model(backend_name, qubits_for_noise, noise_scale=ns_val, gate_pulse_map=gpm)
        except Exception as e:
            logger.warning("Noise sweep failed at scale=%.2f: %s", ns_val, e)
            for points in series:
                points.append(dict(failed))
            continue
        for points, (theta_rad, phi_rad) in zip(series, states):
            try:
                phys = compute_fidelity_physical(
                    theta_rad, phi_rad, noise_model=noise_model, backend_name=backend_name,
                    selected_qubits=selected[:3] if selected else None, noise_scale=ns_val, shots=shots,
                )
                logi = compute_fidelity_logical(
                    theta_rad, phi_rad, noise_model=noise_model, backend_name=backend_name,
                    selected_qubits=selected if selected else None, noise_scale=ns_val, shots=shots,
                )
                points.append({'noise_scale': round(ns_val, 4),
                               'physical_fidelity': phys['fidelity'], 'logical_fidelity': logi['fidelity'],
                               'physical_sigma': phys['fidelity_sigma'], 'logical_sigma': logi['fidelity_sigma'],
                               'logical_acceptance': logi['acceptance']})
            except Exception as e:
                logger.warning("Noise sweep failed at scale=%.2f: %s", ns_val, e)
                points.append(dict(failed))
    return series


def run_noise_sweep(params, num_scales: int, scale_min: float, scale_max: float,
                    gate_pulse_map: dict | None = None, shots=None) -> list:
    """ノイズスケールを変化させて物理/論理フィデリティを計算する（DM法）。

    Args:
        params: theta / phi（ラジアン）/ backend_name / selected_qubits を持つもの（services.input_resolution.PipelineInputs）
        num_scales: ノイズスケール点数
        scale_min, scale_max: スケール範囲
        gate_pulse_map: L3 で選択した per-gate pulse shape。None なら IBM 既定形状。
            noise model（ゲートノイズ）と fidelity（readout）の両方に同じ map を渡し、
            L3 のパルス形状選択が Results のフィデリティ曲線に反映されるようにする。

    Returns:
        list[dict]: noise_scale / physical_fidelity / logical_fidelity を持つ点列。失敗点は両 fidelity が None。
    """
    selected = list(getattr(params, 'selected_qubits', []) or [])
    return _sweep(params.backend_name, selected, np.linspace(scale_min, scale_max, num_scales), gate_pulse_map,
                  [(params.theta, params.phi)], shots=shots)[0]


def run_stress_test_sweep(params, num_scales: int, scale_min: float, scale_max: float,
                          gate_pulse_map: dict | None = None, shots=None) -> list:
    """|+⟩ と |1⟩ の 2 状態で noise sweep を回し、Fig.5 形式のデータを返す。

    Args:
        params: backend_name / selected_qubits を持つ namespace
        num_scales / scale_min / scale_max: 共通の noise scale 範囲
        gate_pulse_map: L3 で選択した per-gate pulse shape（両状態の sweep に共通適用）

    Returns:
        list[dict]: 各状態の panel データ
            [{"state_label": "+", "state_display": "|+>", "points": [...]}, ...]
    """
    states = [(float(np.radians(st["theta_deg"])), float(np.radians(st["phi_deg"]))) for st in STRESS_TEST_STATES]
    series = _sweep(params.backend_name, list(params.selected_qubits or []), np.linspace(scale_min, scale_max, num_scales),
                    gate_pulse_map, states, shots=shots)
    return [{"state_label": st["label"], "state_display": st["display"], "points": points}
            for st, points in zip(STRESS_TEST_STATES, series)]
