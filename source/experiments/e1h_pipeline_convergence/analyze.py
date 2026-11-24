"""
E1-H: Pipeline Convergence — Analysis & Figure Generation

Generates a band-width bar chart showing how the fidelity range narrows
as L2 (qubit selection), L3 (scheduling and dynamical decoupling) and L4 (correction implementation)
are cumulatively optimized（L4 の段は 2026-11-04）.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.common.cli import create_analyzer_parser
from experiments.common.data_loader import load_experiment_data
from experiments.config import L3_CONFIGS, L4_UNTUNED, RESULTS_DIR

EXPERIMENT_ID = "e1h_pipeline_convergence"
RESULTS_PATH = Path(RESULTS_DIR) / EXPERIMENT_ID
FIGURES_PATH = RESULTS_PATH / "figures"


def stage1_l3_rows(per_config: pd.DataFrame) -> list[dict]:
    """E1-H stage 1（best の経路 × L3 の 4 通り、L4 は手つかず）の seed 平均 ± std を L3_CONFIGS の順に並べる（Fig 4b・Table II の 2 行目）。
    足りない構成は KeyError（黙って飛ばさない）。l4_config の列があれば手つかずの L4 の行だけ（2026-11-04 までの CSV には列が無い）"""
    s1 = per_config[per_config["stage"] == 1]
    if "l4_config" in s1.columns:
        s1 = s1[s1["l4_config"] == L4_UNTUNED.name]
    s1 = s1.set_index("l3_config")
    rows = []
    for c in L3_CONFIGS:
        if c.name not in s1.index:
            raise KeyError(f"E1-H stage 1 に {c.name!r} が無い（あるのは {list(s1.index)}）")
        r = s1.loc[c.name]
        rows.append({"name": c.name, "fidelity": float(r["fidelity"]), "seed_std": float(r["seed_std"])})
    return rows


def plot_l3_configs(per_config: pd.DataFrame, paper: bool) -> None:
    """Fig 4b: best の経路での L3 の 4 通り（時刻決め × DD）。誤差棒は seed の std"""
    rows = stage1_l3_rows(per_config)
    fig, ax = plt.subplots(figsize=(3.5, 2.2) if paper else (6, 3.6))
    names = [r["name"] for r in rows]
    f = np.array([r["fidelity"] for r in rows])
    err = np.array([r["seed_std"] for r in rows])
    x = np.arange(len(rows))
    colors = ["#94A3B8", "#2563EB", "#60A5FA", "#16A34A"]   # 手つかず / alap だけ / DD だけ / 両方
    ax.bar(x, f, yerr=err, color=colors, width=0.6, capsize=2, linewidth=0)
    for i in range(len(rows)):
        ax.annotate(f"{f[i]:.3f}", xy=(x[i], f[i]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6 if paper else 8)
    ax.set_ylabel("Fidelity $F$")
    lo = float(np.floor((f.min() - 0.01) * 100) / 100)
    ax.set_ylim(lo, min(1.0, f.max() + 0.012))
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#CCCCCC")
    if not paper:
        ax.set_title("L3 on the chosen chain: scheduling × decoupling")
    plt.tight_layout()
    fig.savefig(FIGURES_PATH / "e1h_l3_configs.png", dpi=300)
    if paper:
        fig.savefig(FIGURES_PATH / "e1h_l3_configs.pdf")
    plt.close()
    print("L3 configs (stage 1): " + ", ".join(f"{r['name']} {r['fidelity']:.4f}±{r['seed_std']:.4f}" for r in rows))


PER_CONFIG_KEYS = ["stage", "stage_label", "path_q1", "path_q2", "path_q3", "l3_config"]


def per_config_means(df: pd.DataFrame) -> pd.DataFrame:
    """行は seed ごとなので、構成（段・経路・L3・L4）ごとに seed 平均・seed σ・本数を取る（帯はこの平均で出す）"""
    keys = PER_CONFIG_KEYS + (["l4_config"] if "l4_config" in df.columns else [])
    return df.groupby(keys, as_index=False).agg(fidelity=("fidelity", "mean"), seed_std=("fidelity", "std"),
                                               n_seeds=("fidelity", "size"))


def stage_summary(per_config: pd.DataFrame) -> pd.DataFrame:
    """段ごとの統計と、Best を F_opt に固定した帯（2026-11-04 ケイ「Bestの線を固定もしましょう」）。
    F_opt は最後の段の 1 構成の F。帯は F_opt − 段の最低なので、寄与 C_ℓ は最低の上がり幅と同じになり、足すと段 0 の帯に戻る。
    fidelity_max は段の本当の最高で、F_opt を超える構成があることを脚注に書くために残す"""
    last = per_config[per_config["stage"] == per_config["stage"].max()]
    if len(last) != 1:
        raise ValueError(f"最後の段は 1 構成のはず（{len(last)} 構成ある）")
    f_opt = float(last["fidelity"].iloc[0])
    rows = []
    for stage in sorted(per_config["stage"].unique()):
        sdf = per_config[per_config["stage"] == stage]
        worst = sdf["fidelity"].min()
        rows.append({
            "stage": int(stage),
            "stage_label": sdf["stage_label"].iloc[0],
            "n_configs": len(sdf),
            "n_seeds": int(sdf["n_seeds"].iloc[0]),
            "fidelity_mean": sdf["fidelity"].mean(),
            "fidelity_std": sdf["fidelity"].std(),
            "fidelity_min": worst,
            "fidelity_max": sdf["fidelity"].max(),
            "fidelity_best": f_opt,
            "band_pt": (f_opt - worst) * 100,
            "fidelity_q25": sdf["fidelity"].quantile(0.25),
            "fidelity_q75": sdf["fidelity"].quantile(0.75),
            "seed_std_mean": sdf["seed_std"].mean(),
        })
    return pd.DataFrame(rows)


def analyze(paper: bool = False, csv_path: str | None = None):
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    df = load_experiment_data(EXPERIMENT_ID, csv_path)   # 既定は results/e1h_pipeline_convergence/e1h_pipeline_convergence.csv

    # 2026-10-28 ショット推定: 行は seed ごとなので、まず構成（段・経路・形）ごとに seed 平均を取り、帯はその平均の最高 − 最低で出す
    # （構成が 1 つの段の帯は 0。seed 間の散らばりは seed_std に別に出す）
    per_config = per_config_means(df)
    summary = stage_summary(per_config)
    summary.to_csv(FIGURES_PATH / "e1h_summary.csv", index=False)

    print("\n=== Pipeline Convergence Summary ===")
    print(summary.to_string(index=False))
    print(f"\nBand widths (pt, F_opt − worst): {summary['band_pt'].round(2).tolist()}")

    # --- Figure: Band width bar chart ---
    if paper:
        plt.rcParams.update({
            "font.size": 8,
            "font.family": "serif",
            "figure.figsize": (3.5, 2.2),
            "figure.dpi": 300,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            # 全 paper figure で外周余白を最小化 (paper_style.apply_paper_style と揃える)
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        })
    else:
        plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 150})

    fig, ax = plt.subplots()

    labels = ["Baseline" if label == "baseline" else label for label in summary["stage_label"].values]
    worst = summary["fidelity_min"].values
    x = np.arange(len(labels))

    # Best の線は全段 F_opt に固定する（2026-11-04 ケイ「Bestの線を固定もしましょう」）。段の本当の最高は summary の fidelity_max
    best = summary["fidelity_best"].values

    # Shaded band between F_opt and the worst configuration of the stage (the converging region)
    ax.fill_between(x, worst, best, color="#94A3B8", alpha=0.3,
                    label="Fidelity band", zorder=1)

    # Best F line (fixed at F_opt)
    ax.plot(x, best, marker="o", color="#16A34A", linewidth=1.4,
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label="Best $F$ (fixed at $F_{\\mathrm{opt}}$)", zorder=3)

    # Worst F line (rises as layers are fixed)
    ax.plot(x, worst, marker="s", color="#2563EB", linewidth=1.4,
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label="Worst $F$", zorder=3)

    # Numeric value labels near each marker. Best は真上、Worst は右下に置く（Worst の線は右上がりなので、
    # 真下に置くと次の点へ向かう線と重なる。2026-11-04 ケイ「数値をなるべくかぶらせないで」）
    for i in range(len(labels)):
        ax.annotate(f"{best[i]:.3f}", xy=(x[i], best[i]),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color="#16A34A")
        if abs(best[i] - worst[i]) > 0.005:  # suppress when coincident
            ax.annotate(f"{worst[i]:.3f}", xy=(x[i], worst[i]),
                        xytext=(7, -9), textcoords="offset points",
                        ha="left", va="top", fontsize=6, color="#2563EB")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Fidelity $F$")
    # Paper-mode y-axis is tightened to actual data range (worst 0.867 → best 0.985)
    # with small annotation headroom. 以前の (0.70, 1.04) だと band が縦方向に
    # 潰れて見えたため絞り込み。再生成時もこの範囲を維持すること。
    ax.set_ylim(max(0.0, float(np.floor(worst.min() * 20) / 20) - 0.02), 1.0)   # 下限は Baseline の最低に合わせる（marrakesh は 0.475。2026-10-28）
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#CCCCCC")

    if not paper:
        ax.set_title("Pipeline Convergence: Best/Worst $F$ per Stage")

    plt.tight_layout()
    fig.savefig(FIGURES_PATH / "e1h_pipeline_convergence.png", dpi=300)
    if paper:
        fig.savefig(FIGURES_PATH / "e1h_pipeline_convergence.pdf")
    plt.close()

    plot_l3_configs(per_config, paper)
    print(f"\nFigures saved to: {FIGURES_PATH}")


def main():
    parser = create_analyzer_parser("E1-H: Pipeline Convergence Analysis")
    args = parser.parse_args()
    analyze(paper=args.paper, csv_path=args.csv)


if __name__ == "__main__":
    main()
