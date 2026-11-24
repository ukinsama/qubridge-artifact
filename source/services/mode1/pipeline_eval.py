"""Mode1 の層ごとの累積評価: 画面の L1〜L4 の決め事を 1 つの run_app に通し、段ごとの F を返す。

役割: baseline（何も決めていない状態）→ +L2（選んだ鎖）→ +L3（時刻決め・DD）→ +L4（読み出し・遅れ・後選択）の 4 段を
     同じ入力（θ, φ）で回し、F・Bloch 角・受理率を段ごとに返す。Results の滝グラフと Bloch の比較の入力。各段は前の段の決め事を
     全部持ち、1 つの層の決め事だけ足す。
baseline の定義（2026-10-28。L4 を足したときの提案）: 既定の経路（core.config.PAPER_DEFAULT_PATH）・時刻決め asap（Qiskit の既定）・
     DD なし・読み出しは較正どおり・遅れ 600 ns・後選択なし。パルスの形は knob ではない（2026-10-21 に L3 から外した）ので
     全段で IBM の既定。10/20 の版は baseline を All Square にしていたが、形は決め事でなくなったのでやめた。
境界: 計算は services/pipeline/runner.run_app。較正は噛まない。セッションは読まず渡し値だけ（2026-10-20、画面の層の整理）。
契約: after_l3（alap・DD なし）は simulate_dm の既定と同じ値、DD ありは simulate_dm(dd=True) と同じ値（tests/test_mode1_pipeline_eval.py）。
     L4 の既定（較正の読み出し・600 ns・後選択なし・実時間の補正）は論文の形で、baseline〜after_l3 は L4 を既定に固定する。読み出しの「整形」は較正値 ×0.9（2026-10-16 ケイ「0.9」）。
spec: docs/superpowers/specs/2026-10-20-mode1-pages-layers-l4-design.md
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from core.config import PAPER_ESTIMATION_METHOD, PAPER_SCHEDULING_METHOD
from services.apps.teleportation import PHYSICAL, physical_pauli_frame, physical_postselected
from services.mode1.error_decomposition import get_baseline_qubits
from services.mode1.simulation import _extract_bloch_angles
from services.pipeline import noise, runner
from services.pipeline.feedforward import DEFAULT_FEEDFORWARD_LATENCY_S

Readout = Literal["calibrated", "ideal", "shaped"]
READOUT_SCALE: dict[str, float] = {"calibrated": 1.0, "ideal": 0.0, "shaped": 0.9}
Schedule = Literal["asap", "alap"]
Correction = Literal["realtime", "frame"]   # 実時間（if_test で Bob を直す）／後処理（Bob はすぐ測り、結果を読み替える）


@dataclass(frozen=True)
class L3Knobs:
    """L3（いつ・待ちの間に何を打つか）の決め事。既定は論文の形（alap・DD なし）"""
    schedule: Schedule = PAPER_SCHEDULING_METHOD   # 時刻決め。delay の置き方（較正の長さ、T1・T2 の緩和つき）
    dd: bool = False                               # 待ちの間に X–X を打つ（静的な ZZ の位相を打ち消し、空きの減衰が T2* → T2 に戻る）


@dataclass(frozen=True)
class L4Knobs:
    """L4（どう測り、測った後どうするか）の決め事。既定は論文の形"""
    readout: Readout = "calibrated"                      # 読み違いの確率: 較正値 ／ 0 ／ ×0.9（整形）
    latency_s: float = DEFAULT_FEEDFORWARD_LATENCY_S     # 測定のあと補正までの古典の遅れ
    postselect: bool = False                             # Alice の結果が 00 の試行だけ残す（補正も待ちも無い）
    correction: Correction = "realtime"                  # 補正の実装。frame は待ち無し（physical_pauli_frame）。後選択なら効かない


BASELINE_L3 = L3Knobs(schedule="asap", dd=False)   # 手つかずの L3: Qiskit の既定の時刻決め、DD なし
PAPER_L3 = L3Knobs()
PAPER_L4 = L4Knobs()


def evaluate_step(theta_rad: float, phi_rad: float, backend_name: str, qubits: list[int], l3: L3Knobs, l4: L4Knobs, shots=None) -> dict:
    """1 つの設定で run_app を回し、F・Bloch 角・受理率を返す。形は IBM の既定（gate_pulse_map 無し）"""
    chain = list(qubits)
    nm = noise.create_reduced_noise_model(backend_name, chain)
    scale = READOUT_SCALE[l4.readout]
    if l4.postselect:
        flips = runner.readout_flip_probabilities(backend_name, chain, PHYSICAL.measured, scale)
        spec = physical_postselected(tuple(float(p) for p in flips))
    elif l4.correction == "frame":
        spec = physical_pauli_frame()
    else:
        spec = PHYSICAL
    r = runner.run_app(spec, {"theta": float(theta_rad), "phi": float(phi_rad)}, backend_name, chain,
                       noise_model=nm, feedforward_latency_s=l4.latency_s, scheduling_method=l3.schedule, dd=l3.dd,
                       readout_scale=scale, method=PAPER_ESTIMATION_METHOD, shots=shots)
    if r["state"] is None:   # 受理 0（後選択で全部落ちた）
        theta_out, phi_out, bloch_length = None, None, None
    else:
        theta_out, phi_out, bloch_length = _extract_bloch_angles(r["state"])
    return {"fidelity": None if r["fidelity"] is None else round(float(r["fidelity"]), 6),
            "fidelity_sigma": None if r["sigma"] is None else round(float(r["sigma"]), 6),
            "roX": theta_out, "roZ": phi_out, "bloch_length": bloch_length,
            "acceptance": round(float(r["acceptance"]), 4), "shots": r["shots"], "seed": r["seed"], "qubits": chain,
            "n_accepted_min": r.get("n_accepted_min")}


def evaluate_layers(theta_rad: float, phi_rad: float, backend_name: str, qubits: list[int],
                    l3: L3Knobs = PAPER_L3, l4: L4Knobs = PAPER_L4, shots=None) -> dict:
    """baseline → +L2 → +L3 → +L4 の累積の 4 段。各段は前の段の決め事を全部持ち、1 つの層の決め事だけ足す"""
    baseline_qubits = get_baseline_qubits(backend_name)
    steps = {
        "baseline": evaluate_step(theta_rad, phi_rad, backend_name, baseline_qubits, BASELINE_L3, PAPER_L4, shots),
        "after_l2": evaluate_step(theta_rad, phi_rad, backend_name, qubits, BASELINE_L3, PAPER_L4, shots),
        "after_l3": evaluate_step(theta_rad, phi_rad, backend_name, qubits, l3, PAPER_L4, shots),
        "after_l4": evaluate_step(theta_rad, phi_rad, backend_name, qubits, l3, l4, shots),
    }

    def pct(v):
        return None if v is None else round(v * 100, 2)

    return {
        "steps": steps,
        "layer_fidelities": {k: pct(v["fidelity"]) for k, v in steps.items()},
        "layer_sigmas": {k: pct(v["fidelity_sigma"]) for k, v in steps.items()},
        "settings": {"theta_deg": round(math.degrees(theta_rad), 2), "phi_deg": round(math.degrees(phi_rad), 2),
                     "backend_name": backend_name, "qubits": list(qubits), "l3": asdict(l3), "l4": asdict(l4)},
    }
