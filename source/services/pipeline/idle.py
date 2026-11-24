"""空きの減衰の正本: T1・T2 の緩和に、始めの数 µs で速く落ちる位相緩和を合成する。

役割: qubit が空いている間（Bob の待ち、時刻決めが回路に入れる delay）に状態がどう崩れるかを 1 か所で決める。
  Bob の待ちは feedforward.apply_feedforward が、回路の delay は target.dephase_idles（速い成分。DD が delay を割る前に
  空き 1 区間 1 回）と target.relax_delays（T1・T2）が使う。
境界: 較正の値（T1・T2）は呼び手が読んで渡す（ここは I/O をしない）。論文の代表値は core.config.PAPER_IDLE_FAST_TERMS。
式: 遅い成分は thermal_relaxation_error(T1, clamp_t2(T1, T2), t)（T2 は呼び手が DD 無しなら T2* = 0.8 × T2 にして渡す）。
  速い成分はコヒーレンスの係数 g(t) = 1 − Σ a_i (1 − e^{−t/τ_i})。Aer の phase_damping_error(λ) は非対角成分に √(1 − λ) を
  掛けるので λ = 1 − g² を渡す。2 つの channel は可換で、|1⟩ の占有は位相緩和では変わらない（2026-10-28 に数値で確認）。
  g は指数と違って足し合わせられない（g(t₁) g(t₂) ≠ g(t₁ + t₂)）ので、1 つの空きに 1 回だけ当てる。
出典: marrakesh の待ちの実測（experiments/hardware/idle_fast_fit.py、2026-10-27・28。DD 無しの 2 項化は idle_fast_fit_two_terms.py、2026-11-08）。
spec: docs/superpowers/specs/2026-10-28-idle-fast-dephasing-design.md（2026-10-28 ケイ「モデル直しましょう」）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit_aer.noise import QuantumError, phase_damping_error, thermal_relaxation_error

from core.config import PAPER_IDLE_FAST_TERMS
from services.pipeline.noise import clamp_t2

_SUM_CAP = 1.0 - 1e-9   # scaled で Σ a_i が 1 以上になったとき、全体をここまで縮める（g > 0 を保つ）


@dataclass(frozen=True)
class FastDephasing:
    """速い位相緩和の係数 g(t) = 1 − Σ a_i (1 − e^{−t/τ_i})。

    terms: ((a_i, τ_i [s]), …)。各 a_i ∈ [0, 1)、Σ a_i < 1、τ_i > 0
    t_min_s: これより短い空きには当てない（g = 1）。論文の値は 0（短い待ちも測った）。形は残す: 測った範囲の外に当てない知識の置き場
    """
    terms: tuple[tuple[float, float], ...]
    t_min_s: float = 0.0

    def __post_init__(self):
        terms = tuple((float(a), float(tau)) for a, tau in self.terms)
        for a, tau in terms:
            if not 0.0 <= a < 1.0:
                raise ValueError(f"a は [0, 1) の中（{a}）")
            if tau <= 0.0:
                raise ValueError(f"τ は正（{tau}）")
        if sum(a for a, _ in terms) >= 1.0:
            raise ValueError(f"Σ a_i は 1 未満（{sum(a for a, _ in terms)}）")
        if self.t_min_s < 0.0:
            raise ValueError(f"t_min_s は 0 以上（{self.t_min_s}）")
        object.__setattr__(self, "terms", terms)   # frozen なので __setattr__ を通す。型を float に揃える

    def coherence(self, t_s: float) -> float:
        """g(t)。t < t_min_s なら 1（当てない）"""
        if t_s < self.t_min_s:
            return 1.0
        return 1.0 - sum(a * (1.0 - math.exp(-t_s / tau)) for a, tau in self.terms)

    def phase_damping_param(self, t_s: float) -> float:
        """Aer の phase_damping_error に渡す λ = 1 − g(t)²"""
        return 1.0 - self.coherence(t_s) ** 2

    def scaled(self, noise_scale: float) -> "FastDephasing":
        """a_i を noise_scale 倍にする（T1・T2 を 1/noise_scale にするのと同じ向きで損が比例する）。Σ が 1 以上なら全体を縮める"""
        terms = [(a * noise_scale, tau) for a, tau in self.terms]
        total = sum(a for a, _ in terms)
        if total >= 1.0:
            terms = [(a * _SUM_CAP / total, tau) for a, tau in terms]
        return FastDephasing(tuple(terms), self.t_min_s)


def idle_error(t1_s: float, t2_s: float, t_s: float, fast: FastDephasing | None = None) -> QuantumError:
    """長さ t_s の空きの誤り: T1・T2 の緩和に、fast があれば速い位相緩和を合成する。fast=None なら緩和だけ（2026-10-21 までのモデル）"""
    if t_s < 0.0:
        raise ValueError(f"空きの長さは 0 以上（{t_s}）")
    err = thermal_relaxation_error(t1_s, clamp_t2(t1_s, t2_s), t_s)
    if fast is None:
        return err
    return err.compose(phase_damping_error(fast.phase_damping_param(t_s)))


def paper_idle_fast() -> dict[bool, FastDephasing]:
    """論文の代表値（core.config.PAPER_IDLE_FAST_TERMS）を包む。鍵は DD の有無"""
    return {dd: FastDephasing(tuple(tuple(x) for x in v["terms"]), float(v.get("t_min_s", 0.0)))
            for dd, v in PAPER_IDLE_FAST_TERMS.items()}


PAPER_IDLE_FAST: dict[bool, FastDephasing] = paper_idle_fast()
