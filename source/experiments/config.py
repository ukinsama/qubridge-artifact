"""論文 §IV の実験（E1-G・E1-H・E3-D）が共有する設定。

役割: 機体・best の経路・試す状態・Baseline の閾値を 1 か所に置き、3 本の runner と analyze が同じ値を読む。
境界: 本番（services）が要る値は core/config.py に置き、ここはそれを指す（services は experiments を import しない。
     tests/test_dependency_direction.py）。較正そのものは services/static_backend.py。
存在理由: 2026-01 の 13 本の実験（E1-A〜E4-A）の設定が 2026-10-20 まで残っていた。今の論文で回すのは 3 本だけなので、
     その 3 本と共通部だけにした（経緯は docs/decisions/2026-10-16-l2-as-transpile.md の追記）。
契約: CommonConfig.best_path はトランスパイラの選択（services/pipeline/target.paper_chain(PHYSICAL)）で、tests/test_scheduling_in_pipeline.py が導出と照合する（2026-11-11）。
     best_logical_path は services/pipeline/target.paper_chain の値を手で写した凍結の値で、tests/test_scheduling_in_pipeline.py が導出と較正の日付の一致を確かめる。
"""
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple

from core.config import PAPER_BACKEND, PAPER_BEST_PATH, PAPER_DEFAULT_PATH


@dataclass(frozen=True)
class L3Config:
    """L3 の決め事 1 つ（時刻決め × DD）。形は全段 IBM の既定（2026-10-28 ケイ「L3 を時刻決め＋DD に統一する」。
    docs/decisions/2026-10-28-l3-unified-scheduling-dd.md）"""
    name: str
    scheduling_method: str   # "asap"（Qiskit の既定）か "alap"
    dd: bool                 # 待ちの間に X–X を打つか

    def kwargs(self) -> dict:
        """services.mode1.simulate / run_app に渡す形（metadata.json の l3_configs もこの形）"""
        return {"scheduling_method": self.scheduling_method, "dd": self.dd}

    def knobs(self) -> dict:
        """services.mode1.pipeline_eval.L3Knobs に渡す形（鍵は schedule・dd。run_app の scheduling_method とは名前が違う）"""
        return {"schedule": self.scheduling_method, "dd": self.dd}


L3_UNTUNED = L3Config("asap, no DD", "asap", False)   # 手つかずの L3: Table I の段 1・2、E1-H stage 0・E1-G の列挙
L3_TUNED = L3Config("alap + DD", "alap", True)        # 整えた L3: Table I の段 3、E1-H stage 2、画面の +L3
L3_CONFIGS: Tuple[L3Config, ...] = (L3_UNTUNED, L3Config("alap, no DD", "alap", False), L3Config("asap + DD", "asap", True), L3_TUNED)


@dataclass(frozen=True)
class L4Config:
    """L4 の決め事 1 つ（補正の実装）。読み出し・遅れ・後選択は較正のモデルに固定（2026-11-04 ケイ「後選択の値を使いましょうか」→ AI が
    後処理の補正を勧め、ケイ「そうしてほしいです」。後選択は試行の 3/4 を捨てるので §IV.E の Log F × Accept の物差しと矛盾する）"""
    name: str
    correction: str   # "realtime"（if_test で Bob を直す。Bob は読み出し＋600 ns 待つ）か "frame"（Bob をすぐ測り、Alice の結果で読み替える）

    def kwargs(self) -> dict:
        """services.mode1.pipeline_eval.L4Knobs に渡す形"""
        return {"correction": self.correction}


L4_UNTUNED = L4Config("real-time feedforward", "realtime")   # 手つかずの L4: Table I の段 1〜3、E1-H stage 0〜2
L4_TUNED = L4Config("post-processed frame", "frame")          # 整えた L4: Table I の段 4、E1-H stage 3、画面の +L4
L4_CONFIGS: Tuple[L4Config, ...] = (L4_UNTUNED, L4_TUNED)


@dataclass(frozen=True)
class BaselineThresholds:
    """Baseline 構成に含まれる path の品質下限（"broken qubit" 除外基準）

    path が Baseline に含まれる iff 全 qubit が T1≥t1_min, T2≥t2_min,
    1Q≤gate1q_max を満たし、path 上の全 edge が 2Q≤gate2q_max を満たす。
    RO error の大きさは意図的に含めない（progressive filter stage で可視化するため）。
    ただし読み違いがどちらかの向きで readout_misread_max を超える qubit（逆に読むほうが多い＝読み出しが壊れている）は外す。
    雑音モデルは各向き 0.5 で頭打ちにするので、その qubit を含む鎖の F はコイン投げに近づき、配置の良し悪しを測らない
    （2026-11-02 ケイ「外したほうがいいでしょう」。marrakesh の q26・q48・q82・q119・q130）。
    """
    t1_min: float = 30e-6          # 30 μs
    t2_min: float = 15e-6           # 15 μs
    gate1q_max: float = 0.01       # 1%
    gate2q_max: float = 0.10       # 10%
    readout_misread_max: float = 0.5   # P(1|0)・P(0|1) の上限。超えたら読み出しが壊れた qubit


@dataclass(frozen=True)
class CommonConfig:
    """論文 §IV の全実験で共有する基準設定

    Why: Fig. 2(b) と Fig. 3(b) 等で同じ best path の Best F 数値が一致する
    ように、バックエンド / best path / シミュレーション方式を一元管理する。
    実験を追加する際は ``CommonConfig`` を参照することで、数値が自動整合する。
    """
    # バックエンド（全実験で統一）
    backend: str = PAPER_BACKEND
    # best path = 論文の L2 の選択 ＝ トランスパイラの選択 paper_chain(PHYSICAL)（値の正本は core.config.PAPER_BEST_PATH。2026-11-11 に全格子の最高から戻した。
    # 経緯は docs/decisions/2026-11-11-l2-trust-transpiler.md）。導出との一致は tests/test_scheduling_in_pipeline.py
    best_path: Tuple[int, ...] = PAPER_BEST_PATH
    # 論理テレポ（6 qubit）の配置も transpile の選択（paper_chain(LOGICAL)）。2026-10-20 ケイ「論理テレポも Qiskit の関数でレイアウトしましょう」。11/11 に F の規則を当てる試みは記録だけ（knob_inventory/logical_layout_select.py）。
    # 10/20 昼までは VF2 の完全部分グラフ（T 字）を雑音で採点した [60, 61, 62, 63, 73, 64]。経路づけの SWAP は run_app の transpile が入れる
    # 2026-10-28 夜: L2 の採点に測定を足した master（ab5d4a2）と合流し、paper_chain(LOGICAL, "ibm_marrakesh") は 97–89–87–88–107–90 →
    # 44–47–45–46–43–57 に動いた。E3-D は同日夜に回し直し（|1⟩ で論理 > 物理 に戻る。paper/numbers_canonical.md §1.5）
    best_logical_path: Tuple[int, ...] = (44, 47, 45, 46, 43, 57)   # ibm_marrakesh、較正 2026-10-28（paper_chain(LOGICAL, "ibm_marrakesh")、測定込みの採点）
    # Default path: Baseline 分布（marrakesh 398 本・両向き）の seed 平均 F の 10th percentile
    # (手つかずの L3、|+>) 代表点。Waterfall UI / Table I Step 1 の出発点。値の正本は core.config.PAPER_DEFAULT_PATH
    default_path: Tuple[int, ...] = PAPER_DEFAULT_PATH
    # L3 の手つかず／整えた状態（E1-G の列挙と E1-H の stage 0・2 が使う。形は IBM の既定で固定）
    l3_untuned: L3Config = L3_UNTUNED
    l3_tuned: L3Config = L3_TUNED
    # L4 の手つかず／整えた状態（E1-H の stage 0〜2 は手つかず、stage 3 が整えた。2026-11-04）
    l4_untuned: L4Config = L4_UNTUNED
    l4_tuned: L4Config = L4_TUNED
    # F の推定法: "shots"（動的回路をショットで回し、固定 3 基底から F。2026-10-28 ケイ「ショットでいいです」）か "dm"（密度行列）。
    # 本番の既定 core.config.PAPER_ESTIMATION_METHOD とは別に、論文の runner はこの値を明示して渡す
    estimation: str = "shots"
    simulation_method: str = "shots"   # metadata に書く名前（旧 "simulate_dm"）
    # 論文のショット数と seed（固定した複数値。seed ごとに 1 行書き、analyze が平均と散らばりを出す。experiments/common/seed_stats.py）
    paper_shots: int = 2 ** 18
    paper_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    # テスト状態 2 点: |+> と |1>
    # Why: [[2,1,1]] 符号は X エラー（bit flip）を検出するが Z エラー（phase flip）はできない
    # という選択的保護を持つ。この物理を paper Fig 5 で可視化するため、2 つの stress test 状態を使う:
    #   - |+> (equator): 最大 Z-error 感受 → [[2,1,1]] 保護なし → Logical < Physical
    #   - |1> (Z-eigenstate): T1 decay が X-error として検出される → Logical > Physical at 高 noise
    # この対比が paper §IV.D の "framework correctly characterizes selective protection" narrative。
    # Fig 2b/3b/4b は |+> だけ（E1-H は test_states[0] に固定し、E1-G の図は E1-H の段 0 の F を読む）、Fig 5（E3-D）は両方を回して比較。
    test_states: Tuple[Tuple[str, float, float], ...] = (
        ("+", math.pi / 2, 0.0),
        ("1", math.pi, 0.0),
    )
    # metadata.json に書く n_repeats と、E1-G の dry-run が 1 構成に書く行数（E1-H の dry-run は dry_run_repeats）。
    # 本番の行数は paper_seeds で決まる（2026-10-28 のショット推定から。seed ごとに 1 行）
    n_repeats: int = 3
    # Baseline 構成の品質閾値（broken 除外）
    baseline_thresholds: BaselineThresholds = field(default_factory=BaselineThresholds)


@dataclass(frozen=True)
class E1GConfig:
    """E1-G Cumulative Filter → Fidelity Band

    L2 フィルタを段階的に適用し、各段階での最良/最悪パスのフィデリティ帯
    (best - worst) が狭まる様子を測定。Fig. 3(b) の図を生成する。
    Baseline の絞り込み（T1・T2・1Q・2Q の下限）は common.baseline_thresholds。
    """
    common: CommonConfig = field(default_factory=CommonConfig)
    # Baseline の上に重ねる絞り込み: 読み出し誤り → 2Q 誤りを段階的に
    ro_threshold: float = 0.05
    gate2q_thresholds: Tuple[Tuple[float, str], ...] = (
        (0.01, "2Q<1%"),
        (0.005, "2Q<0.5%"),
        (0.003, "2Q<0.3%"),
    )


@dataclass(frozen=True)
class E3DConfig:
    """E3-D ノイズスケーリングによる Physical vs Logical テレポーテーション交差点分析の設定

    CommonConfig を参照することで、Fig.2(b)/3(b)/4(b) と完全整合:
      - 同じ backend / best_path（物理）/ best_logical_path（論理。どちらも transpile の選択）
      - 同じ推定法（CommonConfig.estimation = "shots"、paper_shots × paper_seeds。2026-10-28 に密度行列から統一）
      - 同じ test_states (CommonConfig.test_states = |+⟩・|1⟩)
      - 同じ n_repeats（metadata に書く値。行は seed ごと）
      - 同じ F の定義（services/mode3/fidelity.compute_fidelity_{physical,logical} → run_app。固定 3 基底のショットから F ± σ）
    """
    common: CommonConfig = field(default_factory=CommonConfig)
    # ノイズスケール値（0.5=軽ノイズ, 1.0=実機相当, 2.5=2.5倍ノイズ）
    # ns=0 (ideal F=1) は selective protection の story に寄与しないので除外し、
    # 0.5 刻み 5 点で F の差分領域を拡大して見せる
    noise_scales: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
    # dry-run用 (3 点だけ)
    dry_run_noise_scales: Tuple[float, ...] = (0.0, 1.0, 2.0)
    dry_run_repeats: int = 1

    # --- 互換エイリアス (CommonConfig 参照のショートカット) ---
    @property
    def backend(self) -> str:
        return self.common.backend

    @property
    def best_path(self) -> Tuple[int, ...]:
        return self.common.best_path

    @property
    def best_logical_path(self) -> Tuple[int, ...]:
        return self.common.best_logical_path

    @property
    def n_repeats(self) -> int:
        return self.common.n_repeats


@dataclass(frozen=True)
class E1HConfig:
    """E1-H Pipeline Convergence (累積フィデリティ帯収束)

    各パイプラインステージの判断を累積的に最適化していき、
    フィデリティ帯（best-worst間）が段階的に収束する様子を測定する。
    L2のE1-G（フィルタ収束）のパイプライン全体版。
    """
    common: CommonConfig = field(default_factory=CommonConfig)
    # +L2 の段で比べる L3 の 4 通り（時刻決め asap/alap × DD なし/あり）。2026-10-28 までは形 4 つ（All Square・GsSq・DRAG・Per-gate）だった。
    # 形の比較は paper/numbers_canonical.md §1.4 の側の結果として残す（10/28 午後の凍結 87cdb9c の CSV）
    l3_configs: Tuple[L3Config, ...] = L3_CONFIGS
    # +L2+L3 の段で比べる L4 の 2 通り（実時間の補正／後処理の補正。2026-11-04）
    l4_configs: Tuple[L4Config, ...] = L4_CONFIGS
    dry_run_repeats: int = 2

    # --- 互換エイリアス (CommonConfig 参照のショートカット) ---
    @property
    def backend(self) -> str:
        return self.common.backend

    @property
    def best_path(self) -> Tuple[int, ...]:
        return self.common.best_path

    @property
    def n_repeats(self) -> int:
        return self.common.n_repeats



# --- 出力ディレクトリ ---
# プロジェクトルートからの相対パス
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def get_experiment_metadata() -> dict:
    """実験メタデータを辞書で返す（結果JSONに埋め込み用）"""
    try:
        import qiskit
        qiskit_version = qiskit.__version__
    except ImportError:
        qiskit_version = "unknown"

    return {
        "timestamp": datetime.now().isoformat(),
        "qiskit_version": qiskit_version,
        "config_version": "1.0",
        "optimization_level": 3,           # 論文用に固定（transpile の最大最適化）
        "primary_backend": PAPER_BACKEND,
    }
