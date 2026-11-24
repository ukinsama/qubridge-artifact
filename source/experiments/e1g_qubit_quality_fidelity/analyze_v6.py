"""
E1-G v6: Cumulative Filter → Fidelity Band 分析

v5 と同形式の line chart (パイプラインの鎖の F / Worst F + band) を生成。
帯の上端はパイプラインの鎖（論文の L2 ＝ CommonConfig.best_path）の手つかずの F に固定する（2026-11-04 ケイ「2でお願いします」。
定義は Band(S) = Y(ω_S) − min Y、ω_S は S の中でパイプラインが選ぶ構成。E1-G では (P*, σ0, μ0)、Fig 2b では F_opt）。
10/21〜11/04 は b1（段の中の最高 − 最低）で、上端が段ごとに動きうる形だった。鎖が絞り込みで落ちる段があれば ValueError
（上端が集合の外に出る帯は作らない）。段の本当の最高は脚注のために残す。
段の全経路の F は E1-H stage 0 の CSV（同じ列挙、ショット推定の seed 平均）から取る（2026-10-28。
密度行列を生で呼ぶと論文の推定法と食い違う）。E1-G は L2 の切り離しの掃引なので、L3・L4 は手つかず（σ0, μ0）の行だけ読む
（2026-11-04 夕方から stage 0 は全格子 ＝ 経路 × L3 の 4 通り × L4 の 2 通り。全格子の帯は Table II の 1 行目で、ここでは使わない）。
数字の正本は paper/numbers_canonical.md §1.3。

Usage:
    python -m experiments.e1g_qubit_quality_fidelity.analyze_v6 --paper
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.cli import create_analyzer_parser
from experiments.common.paper_style import apply_style, save_figure, SINGLE_COL
from experiments.common.data_loader import load_experiment_data
from experiments.config import L3_UNTUNED, L4_UNTUNED, CommonConfig
from experiments.e1g_qubit_quality_fidelity.runner_v6 import BACKEND, E1G_DIR, EXPERIMENT_ID, get_filtered_paths

FIG_DIR = E1G_DIR / "figures"
E1H_CSV = E1G_DIR.parent / "e1h_pipeline_convergence" / "e1h_pipeline_convergence.csv"   # 段の全経路の F（手つかずの L3）はここ


def stage_extremes(stage_paths, f_of_path, anchor):
    """段ごとの帯。上端はパイプラインの鎖 anchor の F（全段で同じ値）、下端は段の最低で、band_pt = (anchor − worst) × 100。
    stage_paths は runner_v6.get_filtered_paths の形 [(label, [(path, e2q), ...]), ...]。f_of_path(path) は経路 1 本の F（手つかずの L3・L4）。
    anchor が段に残っていなければ ValueError（上端が集合の外に出る帯は作らない。10/21 に b2 を退けた理由と同じ）。
    max は段の本当の最高で、anchor より高い経路があってよい（脚注に書く。2026-11-04 ケイ「2でお願いします」）"""
    anchor = tuple(int(q) for q in anchor)
    f_anchor = f_of_path(anchor)
    rows = []
    for label, paths in stage_paths:
        keep = [tuple(int(q) for q in p) for p, _e in paths]
        if anchor not in keep:
            raise ValueError(f"段 {label!r} にパイプラインの鎖 {anchor} が残っていない（上端が集合の外に出る）")
        fs = [f_of_path(p) for p in keep]
        worst = min(fs)
        rows.append({"label": label, "n": len(fs), "anchor": f_anchor, "max": max(fs), "worst": worst,
                     "band_pt": (f_anchor - worst) * 100})
    return rows


def analyze(paper: bool = False, csv_path: str | None = None):
    apply_style(paper)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_experiment_data(EXPERIMENT_ID, csv_path)   # 既定は results/e1g_v6/e1g_v6.csv（無ければ FileNotFoundError）
    if df.empty:
        print("No CSV data found.")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

    # 段の経路は runner_v6 と同じ列挙（両向き）。E1-G の CSV は段ごとの best/worst の 2 経路しか持たないので、
    # 段の全経路の F は E1-H stage 0（同じ 398 本、手つかずの L3 ＝ asap・DD なし・既定の形、2^18 ショット × seed）の CSV から seed 平均で取る。
    # 10/28 までは simulate_dm（密度行列）を生で呼んでいたが、論文の推定はショットなので CSV と食い違っていた
    e1h = pd.read_csv(E1H_CSV)
    # 切り離しの掃引は (σ0, μ0) で読む。stage 0 が全格子（2026-11-04 夕方から）でも、手つかずの L3・L4 の行だけ取る
    s0 = e1h[(e1h.stage == 0) & (e1h.l3_config == L3_UNTUNED.name) & (e1h.l4_config == L4_UNTUNED.name)]
    per_path = s0.groupby(["path_q1", "path_q2", "path_q3"])["fidelity"]
    assert (per_path.size() == per_path.size().iloc[0]).all(), "経路ごとの seed の本数がそろっていない"
    base = per_path.mean()

    def f_untuned(path):
        return float(base.loc[tuple(int(q) for q in path)])

    # 帯の上端はパイプラインの鎖（論文の L2 ＝ CommonConfig.best_path）の手つかずの F に固定する（2026-11-04 ケイ「2でお願いします」）。
    # 10/21〜11/04 は b1（段の中の最高 − 最低）。10/20 までは上端を best の経路の All Square に固定していた（b2）が、
    # 当時の best の経路 66–65–64 は +RO<5% の段から落ちるので段の外の値が天井になっていた。今は鎖が落ちる段があれば止まる
    anchor_path = tuple(int(q) for q in CommonConfig().best_path)
    # E1-G の CSV の best の位置は同じ鎖のはず（runner_v6 が CommonConfig.best_path を書く）
    csv_best = {tuple(int(q) for q in str(s).strip("[]").split(",")) for s in df[df.position == "best"]["path"]}
    assert csv_best == {anchor_path}, (csv_best, anchor_path)
    rows = stage_extremes(get_filtered_paths(BACKEND), f_untuned, anchor_path)
    csv_labels = df.groupby("filter_stage").first()["filter_label"].tolist()
    assert [r["label"] for r in rows] == csv_labels, (csv_labels, [r["label"] for r in rows])
    labels = [r["label"] for r in rows]
    n_paths_list = [r["n"] for r in rows]
    best_means = np.array([r["anchor"] for r in rows])   # 上端（全段で同じ値）
    max_means = np.array([r["max"] for r in rows])       # 段の本当の最高（脚注に書く）
    worst_means = np.array([r["worst"] for r in rows])
    x_positions = list(range(len(rows)))

    bands = best_means - worst_means   # 上端 − 段の最低（小数）。表示は ×100 で pt

    # Shaded band（上端 ＝ パイプラインの鎖の F、下端 ＝ 段の最低。L3・L4 は手つかずに固定）
    ax.fill_between(x_positions, worst_means, best_means,
                    color="#94A3B8", alpha=0.3,
                    label="Fidelity band ($L_3$, $L_4$ untuned)", zorder=1)


    # パイプラインの鎖の F（全段で同じ。鎖はどの段にも残る）
    ax.plot(x_positions, best_means, color="#16A34A",
            marker="o", markersize=5, linewidth=1.4,
            markerfacecolor="white", markeredgewidth=1.2,
            label="Pipeline chain $F$ ($L_3$, $L_4$ untuned)", zorder=3)

    # Worst F (rising)
    ax.plot(x_positions, worst_means, color="#2563EB",
            marker="s", markersize=5, linewidth=1.4,
            markerfacecolor="white", markeredgewidth=1.2,
            label="Worst $F$", zorder=3)

    # Numeric labels only at endpoints
    n = len(x_positions)
    for i in [0, n - 1]:
        ax.annotate(f"{best_means[i]:.3f}", xy=(i, best_means[i]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.5, color="#16A34A")
        if abs(best_means[i] - worst_means[i]) > 0.003:
            ax.annotate(f"{worst_means[i]:.3f}", xy=(i, worst_means[i]),
                        xytext=(0, -8), textcoords="offset points",
                        ha="center", va="top", fontsize=5.5, color="#2563EB")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Fidelity $F$")
    y_lo = float(np.min(worst_means)) - 0.02
    y_hi = float(np.max(best_means)) + 0.04   # 上端の線の数字（+6 pt）が枠に切れない余白
    ax.set_ylim(y_lo, y_hi)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#CCCCCC")
    ax.set_axisbelow(True)

    if not paper:
        ax.set_title("L2: Cumulative Filtering Narrows Fidelity Band (shots, seed mean)")

    fig.tight_layout()
    save_figure(fig, str(FIG_DIR / "e1g_filter_band"), paper=paper)
    plt.close(fig)

    print("\n=== E1-G v6 Cumulative Filter Summary (shots, seed mean of E1-H stage 0) ===")
    for i, label in enumerate(labels):
        print(f"  {label:20s}: {n_paths_list[i]:3d} paths  "
              f"F_chain={best_means[i]:.4f}  F_max={max_means[i]:.4f}  F_worst={worst_means[i]:.4f}  "
              f"band={bands[i] * 100:.2f}%")

    print(f"\n  Band narrowing (chain − worst): {bands[0] * 100:.2f}% → {bands[-1] * 100:.2f}%  (chain {anchor_path})")
    print(f"  Worst F lift: {worst_means[0]:.4f} → {worst_means[-1]:.4f} "
          f"(+{worst_means[-1]-worst_means[0]:.4f})")
    print("Done.")


def main():
    parser = create_analyzer_parser("E1-G v6: Cumulative Filter Band Analysis")
    args = parser.parse_args()
    analyze(paper=args.paper, csv_path=args.csv)


if __name__ == "__main__":
    main()
