"""Application configuration.

このモジュールは純粋な設定クラスを提供します。
セッション関連の関数は core.session_manager を直接使用してください。
"""
from __future__ import annotations

import logging
import os
from typing import ClassVar, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root directory (works in Docker and local environments)
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# 全ゲートをSquareパルスにする共有定数。
# mode1.py (ALL_SQUARE) と error_decomposition.py (_NO_TUNING_PULSE_MAP) で共用。
ALL_SQUARE_PULSE_MAP: dict[str, str] = {
    "sx": "square", "x": "square", "cz": "square", "ecr": "square",
    "measure": "square", "id": "square", "rz": "square",
}

# 論文の既定（experiments/config.py の CommonConfig と、services/mode1/error_decomposition.py が共有する。2026-09-28）
# 本番のコードが experiments を import しないよう、値はここに置き、CommonConfig がここを指す。
PAPER_BACKEND: str = "ibm_marrakesh"   # 2026-10-28 に torino から替えた（docs/decisions/2026-10-28-backend-marrakesh.md）。torino は記録として backend_errors.json に残る
PAPER_DEFAULT_PATH: tuple[int, ...] = (24, 23, 16)   # Baseline 398 経路（marrakesh、両向き）の seed 平均 F の 10th percentile（L3=All Square, |+>。2026-10-28 のショット推定で選び直し。41 位 / 398、F 0.806）
# 論文の L2: 較正を考慮したトランスパイラの選択（services/pipeline/target.paper_chain(PHYSICAL)。docs/decisions/2026-11-11-l2-trust-transpiler.md）。
# 値は手で写した凍結の値で、導出との一致は tests/test_scheduling_in_pipeline.py が確かめる
PAPER_BEST_PATH: tuple[int, ...] = (5, 6, 7)   # 論文の L2 ＝ トランスパイラの選択（paper_chain(PHYSICAL)。2026-11-11 ケイ「トランスパイルの結果を盲信する形で」。11/04〜11/11 は全格子で F が最高の 3–4–5 だった）
# 時刻決め（L2）の既定。run_app と UI の入口が同じ値を使う。None なら delay を入れない。経緯: docs/decisions/2026-10-16-l2-as-transpile.md
PAPER_SCHEDULING_METHOD: str | None = "alap"
# F の推定法の既定。"dm"（密度行列）か "shots"（動的回路をショットで。2026-10-28 ケイ「ショットでいいです」）。
# 凍結のやり直し（docs/superpowers/plans/2026-10-28-shots-estimation.md の Task 8）で "shots" に切り替えた。"dm" は検算と比較のために残す
PAPER_ESTIMATION_METHOD: str = "shots"   # 2026-10-28 の凍結のやり直しで切り替えた
# 常にかかる ZZ の代表値。2026-10-21 に ibm_marrakesh の cz 全 176 辺を ZZ Ramsey で測り（隣り合わない 5 組、実機 50 秒）、
# 動く 167 辺の |ζ| は中央値 3.43 kHz・q25〜q75 2.99〜3.81 kHz（experiments/hardware/results/zz_map_ibm_marrakesh_2026-10-21_0939.csv、
# 解析 zz_map_analysis.py）。cz 誤りとは無相関で、壊れた辺（cz 誤り 100%）だけ数十 kHz。同じ辺が日をまたいで 4.14 → 3.55 kHz と 15% 動く。
# 論文の機体 ibm_torino はアカウントに無く測れていないので、同じ Heron の代表値として置く（10/20 の 1 本 4.14 kHz から置き換え）。
# 2026-10-21 午後から run_app の既定（論文のモデル。凍結の更新と一緒に入れた。旧名 PROVISIONAL_ZZ_HZ）
PAPER_ZZ_HZ: float = 3.4e3
# DD 無しの待ち（と回路の中の delay）の減衰に使う T2* と較正の T2（Hahn echo）の比。2026-10-21 に ibm_marrakesh の 156 qubit を
# 隣り合わない 2 組の並列 Ramsey で測り（experiments/hardware/t2star_ramsey.py、実機 12 秒）、148 qubit の T2*/T2 の中央値 0.78
#（q25〜q75 0.56〜1.09、鎖 64–67 は 0.67〜0.98。解析 t2star_analysis.py）。減衰の形は指数なので待ちの損は時間に比例する。
# ケイ「中央値で入れてしまっていい」（2026-10-21）。DD ありは T2 のまま（X–X が echo の役をする）。None／1.0 で旧モデル
PAPER_T2STAR_RATIO: float = 0.8
# 空きの「始めの速い位相緩和」g(t) = 1 − Σ a_i (1 − e^{−t/τ_i})（services/pipeline/idle.py が包む。鍵は DD の有無）。
# 2026-10-27・28 に ibm_marrakesh の 156 qubit × 2 周で待ち 0.016〜60 µs のコヒーレンスを直接測り（experiments/hardware/short_wait.py、
# 解析 idle_fast_fit.py）、遅い指数（T2*・T2）で割った残りの中央値に当てた。DD 無し: a 0.047（95% 0.040〜0.056）・τ 0.33 µs
#（0.23〜0.55。0.016 µs からの点が決めた。段差でなく速い減衰）、X–X: a 0.034（0.022〜0.048）・τ 2.5 µs（1.5〜4.5）。
# a_off は t_w = 2.16 µs の DD の戻り（同じ qubit の損の差の中央値 1.69 pt）に合わせた。値は 2 桁に丸めている（当てはめは 0.0471・0.328 µs・0.0336・2.52 µs）。
# 2026-10-28 ケイ「モデル直しましょう」「現実の量子コンピュータの挙動をある程度模倣するモデルを作りたい」（それまでのモデルは戻り 0.36 pt）。
# t_min_s はこれより短い空きには当てない閾値（短い待ちも測ったので 0）。run_app(idle_fast=None) で 10/21 のモデル
# 2026-11-08 に DD 無しを 2 項にした（ケイ「結局他のすべての実験を回し直すのでお願いしてもらっていいですか」。experiments/hardware/idle_fast_fit_two_terms.py）。
# 1 項では DD 無しの 1〜10 µs の点が曲線より下に落ちていた（10/28 の判断記録の末尾）。2 つ目の項の τ₂ は X–X の τ_on に固定し
#（自由にすると a₁ と a₂ の配分が決まらない）、形は縞を外した曲線から a₁ 0.0240・τ₁ 0.140 µs・a₂ 0.0264、絶対値は同じ戻り 1.69 pt に
# 合わせる倍率 1.20 で a₁ 0.0288・a₂ 0.0317（χ² は 1 項の 1/3）。受け入れ 3 条件は 10/28 と同じく幅の中。配分の幅は広い（a₂ 0.011〜0.043）。
# 1 項の値（DD 無し a 0.047・τ 0.33 µs）は results/idle_fast_fit_ibm_marrakesh_2026-10-28.csv に残る
PAPER_IDLE_FAST_TERMS: dict[bool, dict] = {
    False: {"terms": ((0.029, 0.14e-6), (0.032, 2.5e-6)), "t_min_s": 0.0},   # DD 無し: 速い項 ＋ X–X と同じ τ の項
    True: {"terms": ((0.034, 2.5e-6),), "t_min_s": 0.0},                     # DD（X–X）あり
}


class Config:
    """Flask application configuration.

    このクラスは環境変数と定数のみを保持します。
    セッション関連の機能は core.session_manager を使用してください。
    """

    SECRET_KEY: ClassVar[Optional[str]] = os.getenv('SECRET_KEY')
    IBM_QUANTUM_TOKEN: ClassVar[Optional[str]] = os.getenv('IBM_QUANTUM_TOKEN')

    # Quantum computing constants
    MAX_QUBITS: ClassVar[int] = 200  # IBM Quantum最大デバイスサイズ（2026年時点: IBM Condor 133 qubits）
    DEFAULT_SINGLE_GATE_TIME: ClassVar[float] = 60e-9  # デフォルト1量子ゲート時間: 60 nanoseconds
    DEFAULT_BACKEND: ClassVar[str] = 'ibm_torino'  # デフォルトバックエンド

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.SECRET_KEY:
            raise ValueError("SECRET_KEY environment variable is not set.")
        if not cls.IBM_QUANTUM_TOKEN:
            logger.warning("IBM_QUANTUM_TOKEN is not set. Quantum features will be limited.")
            cls.IBM_QUANTUM_TOKEN = "DEVELOPMENT_MODE"
