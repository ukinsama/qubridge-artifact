"""L3 のノイズモデル: 較正（RB の誤り・T1・T2・読み出し）とパルスの形から、Aer の NoiseModel を組む正本。

役割: ゲートごとに thermal_relaxation（T1・T2 の Kraus）∘ Pauli-Lindblad（RB の残り）を 1 つの QuantumError にして、
     選んだ物理 qubit（仮想 0..N−1 に対応）だけの NoiseModel を返す。形の効果は較正値への差分（shape_adjusted_*）。
境界: 較正は services/static_backend.py を読む。L2（配置）は qubits.py・target.py、待ち時間の緩和は target.relax_delays、
     読み出しを補正の誤りとして混ぜるのは runner.py。Target の誤り率も同じ shape_adjusted_* を使う（VF2 が見る数字と一致）。
存在理由: 旧 experiments/common/lindblad_synthesis.py＋noise_utils.py。2026-09-28 に pipeline へ。
契約: 公開名は tests/test_noise_public_names.py。物理の関数（decoherence_fractions・synth_*・_match_average_infidelity・
     get_2q_gate_time・clamp_t2）は knob_inventory の記録ノートがモジュール属性で差し替えるので名前と置き場を動かさない。
     論文の数字は tests/test_paper_numbers.py が 6 桁で固定する。
判断記録: docs/decisions/2026-10-16-l2-as-transpile.md、docs/surveys/2026-09-27_prior-work_pulse-shape-evidence.md

方法（Magnus M1 の Pauli-Lindblad 合成。Malekakhlagh et al., npj QI 2025, arXiv:2502.03462）:
  1. 物理の Lindbladian L = L_T1 + L_T2 + L_control をゲートの時間 τ_g だけ掛け、相互作用描像で Magnus の 1 次 Ω_1 を取る
  2. exp(Ω_1) を疎な Pauli-Lindblad 基底 Λ(ρ) = exp[Σ_k λ_k (P_k ρ P_k† − ρ)] に分解する
  ここでは T1・T2 は Qiskit の thermal_relaxation_error（全次数の Kraus）をそのまま使い、RB の残り（制御・漏れ・クロストーク）
  を物理の割合（amp damping → {X, Y, Z}、残りは一様）で Pauli 生成子に配る。合成の平均ゲート忠実度が RB の値に一致するよう
  Pauli 部分を _match_average_infidelity で合わせる。静的 ZZ は雑音モデルには入れない（較正に周波数が無い。tests/test_zz_model.py）。2026-10-20 から、実測の ζ（core.config.PAPER_ZZ_HZ）を
  時刻決め後の回路のパス（target.apply_zz）と Bob の待ち（runner._zz_during_wait）で RZZ として入れる（2026-10-21 から run_app の既定）。
  DD 無しの空きの T2*（PAPER_T2STAR_RATIO × T2）も雑音モデルでなく delay の緩和（target.relax_delays の t2_scale）と待ち（runner._finish）で当てる。

出典:
  - Malekakhlagh, Seif, Puzzuoli, Govia, van den Berg, npj QI 2025；van den Berg et al., Nat. Phys. 2023（sparse Pauli-Lindblad）
  - 1Q の形の効果: services/pipeline/pulse_shape_table.json（3 準位のパルス計算、T1 = T2 = ∞。experiments/knob_inventory/pulse_shape_delta_table.py、2026-09-28）
  - 2Q の形の効果: 同じ表の "cz"（experiments/knob_inventory/cz_pulse_to_aer.py の結合器の模型、Δ/2π = 200 MHz の仮定、2026-10-09）
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Dict, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import PTM, Pauli, pauli_basis
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error, thermal_relaxation_error

from core.config import Config
from services.static_backend import load_static_backend_data

logger = logging.getLogger(__name__)


# =============================================================================
# Pulse-shape constants（旧 noise_utils から。2026-09-28）
# =============================================================================

# 較正に長さが無い 2Q の辺の既定。torino は全 cz 辺に長さがあり通らない（2026-10-20 まで Config に無い属性を getattr で読んでいた）
DEFAULT_2Q_GATE_TIME_S = 400e-9

# 1Q の形の効果は pulse_shape_table.json（3準位のパルス計算の表）を pulse_leakage_error が読む。
PULSE_SHAPE_TABLE = Path(__file__).with_name("pulse_shape_table.json")

# 2Q の形の効果も pulse_shape_table.json の "cz" の表（cz_pulse_to_aer の模型、Δ/2π = 200 MHz の仮定）を
# pulse_2q_control_error が読む。2026-10-09 までは較正の誤りへの倍率（gaussian_square 1.0・drag 1.4・square 1.8、出典なし）だった。

# ゲートの長さは形によらず較正の値（2026-09-28 ケイの判断。IBM は 1Q に DRAG しか使ったことがなく、他の形の長さに
# 出典は作れない。2026-09-28 までは形ごとの倍率 drag 1.0・gaussian_square 1.3・square 0.8 で変えていた）


# IBM's default shapes (calibration snapshots; experiments/knob_inventory/ibm_default_pulses.py).
# The Heron CZ waveform is not published; "gaussian_square" here stands for the calibrated default.
DEFAULT_GATE_PULSE_MAP = {
    "sx": "drag", "x": "drag", "cz": "gaussian_square", "ecr": "gaussian_square",
    "measure": "gaussian_square", "id": "drag", "rz": "drag",
    # fractional gates。IBM の既定の形と同じ扱い
    "rx": "drag", "rzz": "gaussian_square",
}

# =============================================================================
# Pulse-shape physics: 1Q control error per shape and gate length (table lookup)
# =============================================================================

@lru_cache(maxsize=1)
def _pulse_shape_table() -> dict:
    """表を1回だけ読む。形（drag・gaussian・gaussian_square・square）× 長さ T の制御の誤り r_ctrl"""
    with open(PULSE_SHAPE_TABLE, encoding="utf-8") as f:
        return json.load(f)


def pulse_leakage_error(pulse_shape: str, gate_time_s: float, gate: str = "sx") -> float:
    """1Q ゲート1回の制御の誤り（漏れがほぼ全部。T1・T2 は含まない）を、表を長さ T で線形補間して返す。

    表は 3準位の transmon をパルスで解いた平均ゲート誤り（α/2π = 330 MHz、面積で振幅を較正、前後の仮想 Z を
    最適化、漏れは I/2 に戻す）。格子は形ごと（square・gaussian_square は T に対して周期 2π/|α| ≈ 3 ns で振動するので
    0.1 ns、drag・gaussian は 2 ns）。格子の外は端の値で止める。gate は "sx"（θ = π/2。rx もこちら）か "x"（θ = π）。
    2026-09-28 まで Gambetta 2011 型の係数 C2・C4（square・gaussian_square は出典なし）で square を約3倍に
    見積もっていた（記録ノート experiments/knob_inventory/pulse_to_aer.py。2026-10-20 に削除、git 履歴に残る）。
    """
    curves = _pulse_shape_table()["curves"]["x" if gate == "x" else "sx"]
    c = curves.get(pulse_shape) or curves["drag"]
    return float(min(np.interp(gate_time_s * 1e9, c["T_ns"], c["r_ctrl"]), 0.75))


def pulse_2q_control_error(pulse_shape: str, gate_time_s: float) -> float:
    """CZ 1 回の制御の誤り（漏れと入れ替え。T1・T2 は含まない）を、表を長さ T で線形補間して返す。

    表は固定周波数 transmon 2 つを結合器でつなぐ模型（experiments/knob_inventory/cz_pulse_to_aer.py）を T1 = T2 = ∞ で
    解いた平均ゲート誤り。Δ/2π = 200 MHz・α/2π = 330 MHz は較正に無い仮定（表の meta）。"drag" は gaussian の包絡なので
    gaussian の曲線、ecr・rzz も cz の曲線を使う（近似）。格子の外は端の値で止める。
    """
    curves = _pulse_shape_table()["curves"]["cz"]
    key = "gaussian" if pulse_shape == "drag" else pulse_shape
    c = curves.get(key) or curves["gaussian_square"]
    return float(min(np.interp(gate_time_s * 1e9, c["T_ns"], c["r_ctrl"]), 0.75))



# =============================================================================
# Lindblad synthesis: Pauli-Lindblad rate calculations
# =============================================================================

def shape_adjusted_1q_error(calib_err: float, shape: str, gate_time_s: float, gate: str = "sx") -> float:
    """1Q の RB 誤り（形の差分込み）。較正値 ＋ (r_ctrl(形, T) − r_ctrl(drag, T))。T は較正の長さで形によらない。
    較正値は IBM の既定（drag）で測った値なので、drag のときは足さない。雑音モデル（build_lindblad_noise_model）と
    Target（services/pipeline/target.py）の両方がこれを使う（VF2 が見る数字と雑音モデルの数字を一致させる。2026-10-16）"""
    if shape == "drag":
        return float(calib_err)
    return float(calib_err) + max(0.0, pulse_leakage_error(shape, gate_time_s, gate=gate) - pulse_leakage_error("drag", gate_time_s, gate=gate))


def shape_adjusted_2q_error(calib_err: float, shape: str, gate_time_s: float) -> float:
    """2Q の RB 誤り（形の差分込み）。較正値 ＋ (r_ctrl(形, T) − r_ctrl(gaussian_square, T))。既定の形のときは足さない"""
    return float(calib_err) + max(0.0, pulse_2q_control_error(shape, gate_time_s) - pulse_2q_control_error("gaussian_square", gate_time_s))


def clamp_t2(t1: float, t2: float) -> float:
    """Enforce the physical bound T2 ≤ 2T1.

    Until 2026-09-27 this also clamped T2 ≤ T1 "for Aer numerical stability"; Aer ran
    fine with T2 > T1 in the 2026-09-26 hardware comparison, so that clamp was removed.
    """
    return min(t2, 2 * t1)


def _get_gate_time(gate_properties: dict, gate_name: str, phys_q: int) -> float:
    """Calibrated 1Q gate duration, falling back to Config default."""
    gt = gate_properties.get(gate_name, {}).get(f"({phys_q},)", {}).get("duration")
    return gt if gt is not None else Config.DEFAULT_SINGLE_GATE_TIME


def get_2q_gate_time(gate_properties: dict, gate_name: str, q_a: int, q_b: int) -> Optional[float]:
    """Calibrated 2Q gate duration of the edge, stored under "(a, b)" in either direction; None if absent.

    Until 2026-09-27 2Q gates were looked up with the 1Q key "(q,)", missed, and every edge fell to
    the 60 ns 1Q default (ibm_torino CZ is 68-184 ns per edge, mostly 68 ns).
    """
    props = gate_properties.get(gate_name, {})
    for key in (f"({q_a}, {q_b})", f"({q_b}, {q_a})"):
        gt = props.get(key, {}).get("duration")
        if gt is not None:
            return gt
    return None


def decoherence_fractions(T1: float, T2: float) -> Tuple[float, float, float]:
    """Physics-derived fractions for {X, Y, Z} Pauli generators from T1/T2 decoherence.

    From Magnus M1 of amplitude damping + pure dephasing Lindbladians over duration τ:
        rate_X = rate_Y = γ_↓ τ / 4       (amp damping, symmetric X/Y redistribution)
        rate_Z = γ_↓ τ / 4 + γ_φ τ / 2    (amp damping Z-contribution + pure dephasing)
    where γ_↓ = 1/T1, γ_φ = max(0, 1/T2 - 1/(2 T1)).

    Returns normalized fractions (f_X, f_Y, f_Z) summing to 1.
    """
    if T1 <= 0 or T2 <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    gamma_down = 1.0 / T1
    gamma_phi = max(0.0, 1.0 / T2 - 1.0 / (2 * T1))
    # Shape-only (τ cancels when normalizing)
    rX = rY = gamma_down / 4
    rZ = gamma_down / 4 + gamma_phi / 2
    total = rX + rY + rZ
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (rX / total, rY / total, rZ / total)


def synth_1q_rates(rb_error: float, T1: float, T2: float) -> Dict[str, float]:
    """Pauli-Lindblad rates {X, Y, Z} for a 1Q gate.

    The physical Magnus-M1 rates from T1/T2 give a characteristic {X, Y, Z} fraction
    (see decoherence_fractions). We scale to the RB-calibrated total ε_RB:

        λ_P = f_P × ε_RB     for P ∈ {X, Y, Z}

    RB error absorbs: decoherence (captured by fractions), control miscalibration
    (spread uniformly across P), leakage (Z-like), crosstalk. Distributing by the
    decoherence fraction biases Z > X, Y (typical because T2 < 2T1), matching
    observed 1Q error spectra.
    """
    if rb_error <= 0:
        return {}
    f_X, f_Y, f_Z = decoherence_fractions(T1, T2)
    return {"X": f_X * rb_error, "Y": f_Y * rb_error, "Z": f_Z * rb_error}


def synth_2q_rates(rb_error_2q: float, T1a: float, T2a: float, T1b: float, T2b: float,
                   gate_time_s: float) -> Dict[str, float]:
    """Pauli-Lindblad rates for a 2Q gate (cz/ecr) across 15 non-identity Pauli strings.

    The 2Q gate's total RB error ε_RB_2Q is distributed across Pauli-Lindblad generators:

      - Individual qubit decoherence (amp damping + dephasing) on qubits a, b:
          λ_{IX}, λ_{IY}, λ_{IZ}  (qubit a)
          λ_{XI}, λ_{YI}, λ_{ZI}  (qubit b)
        distributed using the same decoherence fractions as 1Q.

      - Residual (control error, crosstalk, miscalibration): uniformly spread across
        the 2-local generators {XX, XY, XZ, YX, YY, YZ, ZX, ZY} so the total sum
        equals ε_RB_2Q. ZZ gets no share: static ZZ is not modeled, and keeping it out
        of the residual keeps every edge on the channel the former ZZ-free edges had.

    This gives a physics-structured channel that reduces to RB calibration in the sum.
    """
    if rb_error_2q <= 0:
        return {}

    # (1) Single-qubit decoherence shares on each qubit, weighted by individual error budgets
    fa_X, fa_Y, fa_Z = decoherence_fractions(T1a, T2a)
    fb_X, fb_Y, fb_Z = decoherence_fractions(T1b, T2b)
    # Each qubit's decoherence budget during the 2Q gate is the 1Q idle error over τ_g:
    deco_a = min((1.0 / T1a + max(0, 1.0 / T2a - 1.0 / (2 * T1a)) / 2) * gate_time_s, rb_error_2q * 0.5) if T1a > 0 and T2a > 0 else 0
    deco_b = min((1.0 / T1b + max(0, 1.0 / T2b - 1.0 / (2 * T1b)) / 2) * gate_time_s, rb_error_2q * 0.5) if T1b > 0 and T2b > 0 else 0
    # Cap individual decoherence to leave headroom for the residual
    deco_cap = rb_error_2q / 2
    deco_a = min(deco_a, deco_cap)
    deco_b = min(deco_b, deco_cap)

    rates: Dict[str, float] = {
        "IX": fa_X * deco_a, "IY": fa_Y * deco_a, "IZ": fa_Z * deco_a,
        "XI": fb_X * deco_b, "YI": fb_Y * deco_b, "ZI": fb_Z * deco_b,
    }

    # (2) Residual budget spread across remaining 2-local generators
    physics_sum = sum(rates.values())
    residual = max(0.0, rb_error_2q - physics_sum)
    residual_generators = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY"]
    per_gen = residual / len(residual_generators)
    for g in residual_generators:
        rates[g] = per_gen

    return rates


# =============================================================================
# Noise model construction
# =============================================================================

def _pauli_lindblad_error(rates: Dict[str, float], num_qubits: int):
    """Build a Pauli channel QuantumError from {pauli_string: rate}.

    First-order Pauli-Lindblad channel (λ_k small):
        Λ(ρ) = (1 - Σ λ_k) ρ + Σ λ_k P_k ρ P_k†

    We construct via ``pauli_error`` with explicit identity weight `1 - Σ λ_k`
    to guarantee the probabilities sum to exactly 1 (avoiding the floating-point
    drift observed with ``PauliLindbladError.to_quantum_error()`` which caused
    Qiskit Aer to silently skip un-normalized errors at specific noise scales,
    producing non-monotonic fidelity curves).
    """
    pauli_probs = []
    total = 0.0
    for pauli_str, rate in rates.items():
        if rate <= 0 or not np.isfinite(rate):
            continue
        if len(pauli_str) != num_qubits:
            continue
        pauli_probs.append((pauli_str, float(rate)))
        total += float(rate)
    if not pauli_probs:
        return None
    # Cap at 1 (total should be << 1 in our regime; safety floor against pathology)
    if total >= 1.0:
        # Normalize (keeps channel CPTP when rates unphysically large)
        scale = 0.999999 / total
        pauli_probs = [(p, r * scale) for p, r in pauli_probs]
        total = sum(r for _, r in pauli_probs)
    identity_prob = 1.0 - total
    ops = [("I" * num_qubits, identity_prob)] + pauli_probs
    return pauli_error(ops)


def _match_average_infidelity(thermal, rates: Dict[str, float], target: float,
                              num_qubits: int) -> Dict[str, float]:
    """Scale the Pauli part so that thermal ∘ Pauli has average gate infidelity equal to target.

    IBM's calibrated gate error is an RB average gate error
    (https://quantum.cloud.ibm.com/docs/en/guides/qpu-information), so the whole composite
    channel, not the Pauli part alone, must reproduce it. Until 2026-09-27 the Pauli
    probabilities summed to target, leaving the composite at 0.92-0.97 of the calibration.

    Closed form: with PTM diagonals A_Q (thermal) and 1 - 2k·a_Q (Pauli mixture scaled by k,
    a_Q = total probability of generators anticommuting with Q), the process fidelity is
    F_pro(k) = Σ_Q A_Q (1 - 2k a_Q) / d², linear in k. F_avg = (d F_pro + 1)/(d + 1) = 1 - target
    fixes k. If the thermal part alone already exceeds target, k = 0 (thermal only).
    """
    if not rates or target <= 0:
        return rates
    d = 2 ** num_qubits
    basis = pauli_basis(num_qubits)
    thermal_diag = (np.real(np.diag(PTM(thermal.to_quantumchannel()).data)) if thermal is not None
                    else np.ones(len(basis)))
    gens = [(Pauli(label), p) for label, p in rates.items() if p > 0]
    a = np.array([sum(p for g, p in gens if g.anticommutes(q)) for q in basis])
    f0 = thermal_diag.sum() / d**2
    slope = 2 * (thermal_diag * a).sum() / d**2
    f_target = 1 - (d + 1) * target / d
    if slope <= 0:
        return rates
    k = max(0.0, (f0 - f_target) / slope)
    return {label: p * k for label, p in rates.items()}


def add_readout_errors(noise_model, qp_map: dict, phys_to_virt: dict, ro_scale: float):
    """Attach calibrated readout confusion matrices to each qubit (unchanged from legacy)."""
    for phys_q, virt_q in phys_to_virt.items():
        qp = qp_map.get(phys_q)
        if qp is None:
            continue
        p0g1 = qp.get("prob_meas0_prep1")
        p1g0 = qp.get("prob_meas1_prep0")
        if p0g1 is not None and p1g0 is not None:
            p0g1_s = min(p0g1 * ro_scale, 0.5)
            p1g0_s = min(p1g0 * ro_scale, 0.5)
            try:
                noise_model.add_readout_error(
                    ReadoutError([[1 - p1g0_s, p1g0_s], [p0g1_s, 1 - p0g1_s]]), [virt_q])
            except (ValueError, RuntimeError):
                pass
        elif "readout_error" in qp:
            # IBM の readout_error は2つの向きの誤り率の平均なので、各向きにそのまま入れる
            # （2026-09-26 まで err/2 で入れ、読み出しの誤りを半分に見積もっていた）
            err = min(qp["readout_error"] * ro_scale, 0.5)
            if err > 0:
                try:
                    noise_model.add_readout_error(
                        ReadoutError([[1 - err, err], [err, 1 - err]]), [virt_q])
                except (ValueError, RuntimeError):
                    pass


def with_readout_scale(noise_model: NoiseModel, backend_name: str, qubits: Sequence[int], scale: float) -> NoiseModel:
    """受け取ったモデルの ReadoutError だけを倍率 scale で付け直した写しを返す（元は変えない。純粋）。
    ゲート誤りはそのまま写す。scale == 0 なら読み出し誤り無し。全 qubit（中間測定も Bob の最後の測定も）に一律に掛ける
    （L4 の knob。2026-10-28 ケイ「最後の読み出しにも入れておきましょう」。spec 2026-10-28-shots-estimation-design）。
    run_app は gate_pulse_map を知らないので組み直せず、この写しで L4 の倍率を当てる"""
    d = noise_model.to_dict()
    d["errors"] = [e for e in d.get("errors", []) if e.get("type") != "roerror"]
    out = NoiseModel.from_dict(d)
    backend_data = load_static_backend_data(backend_name)
    qp_map = {qp["qubit"]: qp for qp in backend_data.get("qubit_properties", [])}
    phys_to_virt = {pq: i for i, pq in enumerate(qubits)}
    if scale:
        add_readout_errors(out, qp_map, phys_to_virt, scale)
    return out


def _thermal(t1: float, t2: float, gate_time: float):
    """T1・T2 の Kraus（Aer が組めなければ None）"""
    try:
        return thermal_relaxation_error(t1, t2, gate_time)
    except (ValueError, RuntimeError):
        return None


def _thermal_2q(T1a: float, T2a: float, T1b: float, T2b: float, gate_time: float):
    """2 qubit それぞれの T1・T2 の Kraus のテンソル積（Aer が組めなければ None）"""
    try:
        return thermal_relaxation_error(T1a, T2a, gate_time).expand(thermal_relaxation_error(T1b, T2b, gate_time))
    except (ValueError, RuntimeError):
        return None


def _compose(thermal, pauli, fallback=None):
    """thermal ∘ Pauli を 1 つの QuantumError に。片方しか無ければそれ、両方無ければ fallback()（2Q の depolarizing）か None"""
    if thermal is not None and pauli is not None:
        try:
            return thermal.compose(pauli)
        except Exception:
            return thermal
    if pauli is not None:
        return pauli
    if thermal is not None:
        return thermal
    return fallback() if fallback is not None else None


def _depolarizing(error: float, num_qubits: int):
    try:
        return depolarizing_error(error, num_qubits)
    except (ValueError, RuntimeError):
        return None


def _add(noise_model: NoiseModel, error, gate: str, qubits: list) -> None:
    """Aer が受け付けない誤り（CPTP でない等）は黙って落とす（2026-09-28 までの形）"""
    if error is None:
        return
    try:
        noise_model.add_quantum_error(error, gate, qubits)
    except (ValueError, RuntimeError):
        pass


def _with_fractional_copies(backend_data: dict) -> tuple[dict, list, list]:
    """fractional（rx・rzz）を rx = 同じ qubit の sx、rzz = 同じ辺の cz の写しとして足した (gate_properties, basis_gates, errors_2q)。

    IBM の較正に fractional の誤りは無く、公式は「非 fractional の写し」と書く（https://quantum.cloud.ibm.com/docs/en/guides/fractional-gates）。
    誤りも時間も同じなので、モデルで効くのはゲートの数の違いだけ。較正に rzz・rx が既にある辺・qubit はその値を使い、無いものだけ写す
    （同じ辺に 2 本載せると Aer が合成して誤りが 2 本分になる。2026-09-28 の最終レビュー）
    """
    gate_properties = dict(backend_data.get("gate_properties", {}))
    basis_gates = list(backend_data.get("basis_gates", []))
    if "sx" in gate_properties:
        gate_properties["rx"] = {**gate_properties["sx"], **gate_properties.get("rx", {})}
        if "rx" not in basis_gates:
            basis_gates.append("rx")
    if "cz" in gate_properties:
        gate_properties["rzz"] = {**gate_properties["cz"], **gate_properties.get("rzz", {})}
    calibrated = backend_data.get("errors", [])
    has_rzz = {(e["qubit_from"], e["qubit_to"]) for e in calibrated if e.get("gate") == "rzz"}
    errors_2q = list(calibrated) + [
        {**e, "gate": "rzz"} for e in calibrated
        if e.get("gate") == "cz" and (e["qubit_from"], e["qubit_to"]) not in has_rzz
    ]
    return gate_properties, basis_gates, errors_2q


def _scaled_coherence(qp_map: dict, physical_qubits: List[int], noise_scale: float) -> dict:
    """qubit ごとの (T1 / noise_scale, clamp_t2(T2 / noise_scale))。T1 か T2 の無い qubit は入れない（誤りを付けない）。
    黙って飛ばすと、その qubit はゲート誤りが付かず読み出し誤りだけ残って F が楽観に出るので、飛ばした qubit を警告に出す
    （2026-10-27。論文の列挙 `qubits.baseline_paths` は T1・T2 の無い qubit を既に外している。例外にするかは TASKS）"""
    out, missing = {}, []
    for phys_q in physical_qubits:
        qp = qp_map.get(phys_q)
        if qp is None:
            missing.append(f"q{phys_q}（較正に無い）")
            continue
        lacking = [k for k in ("T1", "T2") if qp.get(k) is None]
        if lacking:
            missing.append(f"q{phys_q}（{'・'.join(lacking)} が無い）")
            continue
        t1 = qp["T1"] / noise_scale
        out[phys_q] = (t1, clamp_t2(t1, qp["T2"] / noise_scale))
    if missing:
        logger.warning("雑音モデルから外した qubit: %s。ゲート誤りが付かず読み出し誤りだけ残るので、この qubit を含む F は楽観に出る", "、".join(missing))
    return out


def _add_1q_errors(noise_model, phys_to_virt: dict, scaled_T: dict, gate_properties: dict, basis_gates: list,
                   gpm: dict, duration_scale: float, noise_scale: float) -> None:
    """1Q: sx・x・rx は thermal ∘ Pauli（RB の値に形の差分を足し、合成の平均忠実度を合わせる）、id・rz は thermal だけ（RB の較正が無い）"""
    for phys_q, virt_q in phys_to_virt.items():
        if phys_q not in scaled_T:
            continue
        t1, t2 = scaled_T[phys_q]
        for gate_name in ("sx", "x", "id", "rz", "rx"):
            if gate_name not in basis_gates:
                continue
            gate_time = _get_gate_time(gate_properties, gate_name, phys_q) * duration_scale   # 形によって長さは変えない
            thermal = _thermal(t1, t2, gate_time)
            if gate_name in ("id", "rz"):
                _add(noise_model, thermal, gate_name, [virt_q])
                continue
            calib_err = gate_properties.get(gate_name, {}).get(f"({phys_q},)", {}).get("error", 0)
            # 形の効果は差で入れる（正本は shape_adjusted_1q_error。Target も同じ関数で誤り率を出す）
            rb = min(shape_adjusted_1q_error(calib_err, gpm.get(gate_name, "drag"), gate_time, gate=gate_name) * noise_scale, 0.75)
            rates = _match_average_infidelity(thermal, synth_1q_rates(rb, t1, t2), rb, num_qubits=1)
            _add(noise_model, _compose(thermal, _pauli_lindblad_error(rates, num_qubits=1)), gate_name, [virt_q])


def _add_2q_errors(noise_model, phys_to_virt: dict, scaled_T: dict, gate_properties: dict, errors_2q: list,
                   gpm: dict, duration_scale: float, noise_scale: float) -> None:
    """2Q: 較正の辺ごとに (thermal ⊗ thermal) ∘ Pauli。両端が選んだ qubit の中にあり T1・T2 が取れる辺だけ"""
    for err in errors_2q:
        q_from, q_to = err["qubit_from"], err["qubit_to"]
        if q_from not in scaled_T or q_to not in scaled_T:   # scaled_T は選んだ qubit のうち T1・T2 のあるものだけ
            continue
        gate = err["gate"]
        gate_time = get_2q_gate_time(gate_properties, gate, q_from, q_to)
        if gate_time is None or gate_time <= 0:
            gate_time = DEFAULT_2Q_GATE_TIME_S
        # 形の効果は差で入れる: 較正値 ＋ (r_ctrl(形, T) − r_ctrl(gaussian_square, T))。T は較正の長さで形によらない。
        # 較正値は IBM の既定（gaussian_square の代わり）で測った値なので、既定の形のときは足さない
        rb = min(shape_adjusted_2q_error(err["error_rate"], gpm.get(gate, "gaussian_square"), gate_time) * duration_scale * noise_scale, 0.75)
        if rb <= 0:
            continue
        gate_time = gate_time * duration_scale
        (T1a, T2a), (T1b, T2b) = scaled_T[q_from], scaled_T[q_to]
        thermal = _thermal_2q(T1a, T2a, T1b, T2b, gate_time)
        rates = _match_average_infidelity(thermal, synth_2q_rates(rb, T1a, T2a, T1b, T2b, gate_time), rb, num_qubits=2)
        composite = _compose(thermal, _pauli_lindblad_error(rates, num_qubits=2), fallback=lambda: _depolarizing(rb, 2))
        _add(noise_model, composite, gate, [phys_to_virt[q_from], phys_to_virt[q_to]])


def build_lindblad_noise_model(backend_name: str, physical_qubits: List[int],
                                gate_pulse_map: Optional[dict] = None,
                                duration_scale: float = 1.0,
                                noise_scale: float = 1.0,
                                fractional_gates: bool = False,
                                readout_scale: float | None = None):
    """較正から、選んだ物理 qubit だけの NoiseModel を組む（Lindblad 合成、Magnus M1）。

    ゲートごとの誤りは thermal_a ⊗ thermal_b ⊗ … ∘ Pauli-Lindblad（RB の値に合わせた残り）の 1 つの QuantumError。

    Args:
        backend_name: 較正の名前（例 "ibm_marrakesh"）
        physical_qubits: 使う物理 qubit。仮想 0..N−1 がこの順に対応する
        gate_pulse_map: ゲートごとの形（DEFAULT_GATE_PULSE_MAP を上書き）。誤りは形の差分で入る。長さは変わらない
        duration_scale: ゲートの長さの倍率（長さの実験用）
        noise_scale: RB の誤りと読み出しを倍にし、T1・T2 を 1/noise_scale にする。0 ならゲート誤り無し（readout_scale も無ければ空のモデル）
        fractional_gates: True なら rzz・rx を cz・sx の写しとして足す（_with_fractional_copies）
        readout_scale: 読み出し誤りだけ別の倍率にする（None なら noise_scale。0 で読み出し誤り無し。ショット版の運転手の L4 の knob。2026-10-20）
    """
    readout = noise_scale if readout_scale is None else readout_scale
    if noise_scale == 0.0 and not readout:
        return NoiseModel()

    backend_data = load_static_backend_data(backend_name)
    if fractional_gates:
        gate_properties, basis_gates, errors_2q = _with_fractional_copies(backend_data)
    else:
        gate_properties = backend_data.get("gate_properties", {})
        basis_gates = backend_data.get("basis_gates", [])
        errors_2q = backend_data.get("errors", [])

    phys_to_virt = {pq: i for i, pq in enumerate(physical_qubits)}
    qp_map = {qp["qubit"]: qp for qp in backend_data.get("qubit_properties", [])}
    gpm = {**DEFAULT_GATE_PULSE_MAP, **(gate_pulse_map or {})}
    scaled_T = _scaled_coherence(qp_map, physical_qubits, noise_scale) if noise_scale != 0.0 else {}

    noise_model = NoiseModel()
    if noise_scale != 0.0:   # noise_scale=0 は「ゲート誤り無し」。readout_scale だけ与えれば読み出しの誤りだけのモデル（ショット版の検算用）
        _add_1q_errors(noise_model, phys_to_virt, scaled_T, gate_properties, basis_gates, gpm, duration_scale, noise_scale)
        _add_2q_errors(noise_model, phys_to_virt, scaled_T, gate_properties, errors_2q, gpm, duration_scale, noise_scale)
    # 読み出しは測定パルスの形によらず較正の値（IBM は既定の gaussian_square で較正し、形の倍率を支える実測が無い。2026-09-27 ケイの決定）
    add_readout_errors(noise_model, qp_map, phys_to_virt, readout)
    return noise_model


# 呼び手の多い名前（services 5・experiments・tests 11・記録ノート 20 本超）。同じ関数で、別名を保つだけ
create_reduced_noise_model = build_lindblad_noise_model


# =============================================================================
# 解析式（E1-H の dry-run と experiments/l3_tuning_figure.py の見積もり用。雑音モデルには使わない）
# =============================================================================

def decoherence_error(gate_time_s: float, t1: float, t2: float) -> float:
    """Gate error from decoherence (exact Lindblad coefficients, arXiv:2302.13885).

    epsilon = (2/3)*gamma_1*t + (4/3)*gamma_phi*t
    where gamma_1 = 1/T1, gamma_phi = 1/T2 - 1/(2*T1).
    """
    if t1 <= 0 or t2 <= 0:
        return 0.0
    gamma_1 = 1.0 / t1
    gamma_phi = max(0.0, 1.0 / t2 - 1.0 / (2 * t1))
    return min((2.0 / 3) * gamma_1 * gate_time_s + (4.0 / 3) * gamma_phi * gate_time_s, 0.75)


# =============================================================================
# Noisy simulator (moved from services/static_backend.py, 2026-09-28)
# =============================================================================

def create_noisy_simulator(backend_name: str, qubits: Optional[List[int]] = None):
    """Create a noisy AerSimulator from static backend data.

    Uses `create_reduced_noise_model` (RB baseline + pulse-shape physics),
    scoped to the given physical qubits. Defaults to [0..5] which covers
    Mode2 CHSH (≤4 qubits) and Mode3 logical teleportation (6 qubits).

    Args:
        backend_name: IBM Quantum backend name
        qubits: Physical qubits the circuit will be mapped to. Noise is built
            only for these qubits and applied virtual→physical via their order.

    Returns:
        AerSimulator with noise model, or None on failure.
    """
    try:
        noise_model = create_reduced_noise_model(backend_name, qubits or list(range(6)))
        return AerSimulator(noise_model=noise_model)
    except Exception as e:
        logger.warning("Noise model failed for %s: %s", backend_name, e)
        return None
