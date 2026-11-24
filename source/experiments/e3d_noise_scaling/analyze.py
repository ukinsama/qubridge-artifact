"""
E3-D: ノイズスケーリング テレポーテーション比較実験の分析・図表生成

Physical vs Logical のフィデリティ vs ノイズスケール折れ線、
状態別サブプロット、交差点分析、統計サマリーを生成する。
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# プロジェクトルートをパスに追加
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiments.config import RESULTS_DIR
from experiments.common.paper_style import apply_style, save_figure, get_figsize, paper_title, SINGLE_COL
from experiments.common.constants import CIRCUIT_TYPE_COLORS
from experiments.common.data_loader import load_experiment_data


def _find_crossover(noise_scales, phys_fid, logi_fid, min_magnitude: float = 0.001):
    """
    線形補間で Physical と Logical の交差点を探す。

    min_magnitude: 交差前後の差の大きさがこれ未満の場合は統計ノイズ内の疑似交差とみなし
                   無視する。best_path + VF2 best layout のように両者がほぼ 1.0 に張り付く
                   領域で、丸めノイズによる符号反転が crossover と誤検出されるのを防ぐ。
    """
    for i in range(len(noise_scales) - 1):
        diff_i = phys_fid[i] - logi_fid[i]
        diff_next = phys_fid[i + 1] - logi_fid[i + 1]

        if diff_i * diff_next < 0:
            if max(abs(diff_i), abs(diff_next)) < min_magnitude:
                continue
            t = diff_i / (diff_i - diff_next)
            crossover = noise_scales[i] + t * (noise_scales[i + 1] - noise_scales[i])
            return float(crossover)

    return None


def plot_noise_scaling_lines(df: pd.DataFrame, output_dir: str, paper: bool = False) -> None:
    """Physical vs Logical のフィデリティ vs ノイズスケール (state 別 2 subplot)。

    各 subplot は自分のデータレンジに独立 zoom する (sharey=False) ので、|1⟩ の
    小さな crossover も視覚的に強調される。

    [[2,1,1]] 符号の選択的保護を可視化:
      - |+⟩ (Z-error sensitive): Logical < Physical 全域 (code 保護なし)
      - |1⟩ (T1 decay → X-error): ns=1.5 で crossover し Logical > Physical
    """
    apply_style(paper)

    states = sorted(df["state_label"].unique())
    n_states = len(states)

    if paper:
        # Top-bottom (n_states rows × 1 col): each panel gets full column width for readability
        figsize = get_figsize("single", aspect=1.05)
        fig, axes = plt.subplots(n_states, 1, figsize=figsize, sharex=True, sharey=False)
    else:
        figsize = (6 * n_states, 6)
        fig, axes = plt.subplots(1, n_states, figsize=figsize, sharey=False)
    if n_states == 1:
        axes = [axes]

    state_descriptions = {
        "|+>": "Z-error dominant (undetectable)",
        "|1>": "T1 decay → X-error (detectable)",
    }

    PHYS_COLOR = CIRCUIT_TYPE_COLORS.get("physical", "#1f77b4")
    LOGI_COLOR = CIRCUIT_TYPE_COLORS.get("logical", "#d62728")
    SHADE_GOOD = "#4caf50"   # 緑 — Logical 有利
    SHADE_BAD  = "#f44336"   # 赤 — Logical 劣勢

    for ax, state in zip(axes, states):
        sdf_state = df[df["state_label"] == state]

        noise_scales = sorted(sdf_state["noise_scale"].unique())
        phys_means, logi_means = [], []
        for ns in noise_scales:
            p = sdf_state[(sdf_state["circuit_type"] == "physical") & (sdf_state["noise_scale"] == ns)]["fidelity"]
            lg = sdf_state[(sdf_state["circuit_type"] == "logical") & (sdf_state["noise_scale"] == ns)]["fidelity"]
            phys_means.append(p.mean() if not p.empty else np.nan)
            logi_means.append(lg.mean() if not lg.empty else np.nan)
        phys_arr = np.array(phys_means)
        logi_arr = np.array(logi_means)
        ns_arr = np.array(noise_scales)

        # ΔF の符号で塗り分け: Logical ≥ Physical → 緑, Logical < Physical → 赤
        ax.fill_between(ns_arr, phys_arr, logi_arr,
                        where=(logi_arr >= phys_arr), interpolate=True,
                        color=SHADE_GOOD, alpha=0.22, label=r"Logical $\geq$ Physical")
        ax.fill_between(ns_arr, phys_arr, logi_arr,
                        where=(logi_arr < phys_arr), interpolate=True,
                        color=SHADE_BAD, alpha=0.22, label="Logical < Physical")

        # 曲線本体
        ax.plot(ns_arr, phys_arr, marker="o", color=PHYS_COLOR,
                linewidth=2.2 if paper else 2.8,
                markersize=5.5 if paper else 8,
                markeredgecolor="white", markeredgewidth=0.6,
                label="Physical Teleport", zorder=3)
        ax.plot(ns_arr, logi_arr, marker="s", color=LOGI_COLOR,
                linewidth=2.2 if paper else 2.8,
                markersize=5.5 if paper else 8,
                markeredgecolor="white", markeredgewidth=0.6,
                label="Logical Teleport", zorder=3)

        # crossover 垂直線 + annotation
        crossover = _find_crossover(noise_scales, phys_means, logi_means)
        if crossover is not None and crossover > noise_scales[0] + 0.01:
            ax.axvline(x=crossover, color='#6a1b9a', linestyle='--',
                       linewidth=1.2, alpha=0.8, zorder=2)
            f_at_cross = np.interp(crossover, ns_arr, phys_arr)
            ax.annotate(f'crossover\nns={crossover:.2f}',
                        xy=(crossover, f_at_cross),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=7 if paper else 9,
                        color='#6a1b9a', fontweight='bold',
                        ha='left')

        # タイトル: 状態のみ(短く) — 物理背景は paper 版では caption で説明し、non-paper だけ 2 行目に出す
        desc = state_descriptions.get(state, "")
        title = f"State {state}"
        if not paper and desc:
            title += f"\n{desc}"
        ax.set_title(paper_title(title, paper), fontsize=9 if paper else 11,
                     pad=2)

        # X label only on bottom subplot for top-bottom layout
        if not paper or ax is axes[-1]:
            ax.set_xlabel("Noise Scale", fontsize=8 if paper else 10)
        ax.set_ylabel("Fidelity $F$", fontsize=8 if paper else 10)
        ax.tick_params(labelsize=7 if paper else 9)
        ax.grid(True, alpha=0.28, linewidth=0.6)
        ax.set_axisbelow(True)
        # paper 版: 凡例なし(caption / panel 配置で説明); non-paper のみ凡例表示
        if not paper:
            ax.legend(fontsize=9, loc="lower left",
                      framealpha=0.92, edgecolor="#cccccc")

        # Per-state tight y-zoom: |1⟩ の crossover が ~0.001 swing でも見えるように
        state_fids = np.concatenate([phys_arr, logi_arr])
        y_min = state_fids.min() - 0.0015
        y_max = min(1.0, state_fids.max() + 0.0015)
        ax.set_ylim(y_min, y_max)

    if not paper:
        fig.suptitle(
            "E3-D: [[2,1,1]] selective protection — shaded region sign indicates benefit",
            fontsize=13,
        )
    else:
        # Shared compact legend inside the figure, top-edge
        handles = [
            Line2D([0], [0], color=PHYS_COLOR, marker='o', linewidth=1.8,
                   markersize=4, label='Physical'),
            Line2D([0], [0], color=LOGI_COLOR, marker='s', linewidth=1.8,
                   markersize=4, label='Logical'),
            Patch(facecolor=SHADE_GOOD, alpha=0.22, label=r'Log.$\geq$Phys.'),
            Patch(facecolor=SHADE_BAD, alpha=0.22, label='Log.<Phys.'),
        ]
        fig.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.5, 1.0), ncol=4,
                   fontsize=7, frameon=False,
                   handletextpad=0.3, columnspacing=0.9,
                   borderpad=0.1)
    if paper:
        # Top-bottom compact: top reserved for shared legend
        fig.tight_layout(pad=0.25, h_pad=0.4, rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    path_stem = os.path.join(output_dir, "e3d_noise_scaling_lines")
    save_figure(fig, path_stem, paper)
    plt.close(fig)


def plot_state_breakdown(df: pd.DataFrame, output_dir: str, paper: bool = False) -> None:
    """状態別サブプロット（2×2）"""
    apply_style(paper)
    states = sorted(df["state_label"].unique())
    all_fid = df["fidelity"].dropna()
    y_min = max(0, all_fid.min() - 0.005)

    n_states = len(states)
    ncols = min(2, n_states)
    nrows = (n_states + ncols - 1) // ncols

    figsize = get_figsize("double", aspect=0.8) if paper else (12, 10)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for idx, state in enumerate(states):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        for ctype in ["physical", "logical"]:
            sdf = df[(df["circuit_type"] == ctype) & (df["state_label"] == state)]
            if sdf.empty:
                continue

            grouped = sdf.groupby("noise_scale")["fidelity"].agg(["mean", "std"]).reset_index()
            grouped = grouped.sort_values("noise_scale")

            color = CIRCUIT_TYPE_COLORS.get(ctype, "gray")
            marker = "o" if ctype == "physical" else "s"

            ax.errorbar(
                grouped["noise_scale"], grouped["mean"],
                yerr=grouped["std"],
                marker=marker,
                color=color,
                label=f"{ctype.capitalize()}",
                linewidth=1.2 if paper else 1.5,
                markersize=3 if paper else 5,
                capsize=2,
            )

        ax.set_title(f"State: {state}", fontsize=9 if paper else 11)
        ax.set_ylim(y_min, 1.002)
        ax.grid(True, alpha=0.3)

        if row == nrows - 1:
            ax.set_xlabel("Noise Scale")
        if col == 0:
            ax.set_ylabel("Fidelity")
        if idx == 0:
            ax.legend(fontsize=7 if paper else 9)

    # 余ったサブプロットを非表示
    for idx in range(n_states, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(paper_title("E3-D: Fidelity vs Noise Scale by State", paper),
                 fontsize=10 if paper else 14)
    fig.tight_layout()
    path_stem = os.path.join(output_dir, "e3d_state_breakdown")
    save_figure(fig, path_stem, paper)
    plt.close(fig)


def plot_success_vs_fidelity(df: pd.DataFrame, output_dir: str, paper: bool = False) -> None:
    """Post-selection success rate vs fidelity (logical mode only).

    Construct validity の補強: F 以外の指標(成功率)でも decision propagation が観測されることを示す。
    """
    apply_style(paper)
    log_df = df[df["circuit_type"] == "logical"].copy()
    if log_df.empty:
        print("  No logical data for success-vs-fidelity plot.")
        return

    log_df["success_rate"] = 1.0 - log_df["error_detection_rate"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))

    # noise_scale ごとに集計
    grouped = log_df.groupby("noise_scale").agg(
        fidelity_mean=("fidelity", "mean"),
        fidelity_std=("fidelity", "std"),
        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),
    ).reset_index()
    grouped = grouped.sort_values("noise_scale")
    grouped["fidelity_std"] = grouped["fidelity_std"].fillna(0)
    grouped["success_std"] = grouped["success_std"].fillna(0)

    # scatter with noise_scale as color
    sc = ax.scatter(
        grouped["success_mean"], grouped["fidelity_mean"],
        c=grouped["noise_scale"], cmap="RdYlGn_r",
        s=60, edgecolors="white", linewidth=0.8, zorder=3,
        vmin=0, vmax=grouped["noise_scale"].max(),
    )

    # error bars
    ax.errorbar(
        grouped["success_mean"], grouped["fidelity_mean"],
        xerr=grouped["success_std"], yerr=grouped["fidelity_std"],
        fmt="none", color="gray", alpha=0.4, capsize=2, zorder=2,
    )

    # annotate noise scale
    for _, row in grouped.iterrows():
        ax.annotate(
            f"{row['noise_scale']:.1f}$\\times$",
            (row["success_mean"], row["fidelity_mean"]),
            textcoords="offset points", xytext=(6, -4),
            fontsize=6, color="gray",
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Noise Scale", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel("Post-selection Success Rate")
    ax.set_ylabel("Teleportation Fidelity $F$")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path_stem = os.path.join(output_dir, "e3d_success_vs_fidelity")
    save_figure(fig, path_stem, paper)
    plt.close(fig)


def generate_summary(df: pd.DataFrame, output_dir: str) -> None:
    """統計サマリーCSV + 交差点分析を生成する"""
    noise_scales = sorted(df["noise_scale"].unique())

    # noise_scale × circuit_type の統計
    rows = []
    for ns in noise_scales:
        for ctype in ["physical", "logical"]:
            sdf = df[(df["noise_scale"] == ns) & (df["circuit_type"] == ctype)]
            fid = sdf["fidelity"].dropna()
            edr = sdf["error_detection_rate"].dropna()
            rows.append({
                "noise_scale": ns,
                "circuit_type": ctype,
                "count": len(fid),
                "fidelity_mean": round(fid.mean(), 4) if not fid.empty else None,
                "fidelity_std": round(fid.std(), 4) if len(fid) > 1 else None,
                "fidelity_min": round(fid.min(), 4) if not fid.empty else None,
                "fidelity_max": round(fid.max(), 4) if not fid.empty else None,
                "error_detection_mean": round(edr.mean(), 4) if not edr.empty else None,
            })

    summary_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "e3d_summary.csv")
    summary_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # 交差点分析
    phys_means = []
    logi_means = []
    for ns in noise_scales:
        phys_data = df[(df["circuit_type"] == "physical") & (df["noise_scale"] == ns)]["fidelity"]
        logi_data = df[(df["circuit_type"] == "logical") & (df["noise_scale"] == ns)]["fidelity"]
        phys_means.append(phys_data.mean() if not phys_data.empty else 0)
        logi_means.append(logi_data.mean() if not logi_data.empty else 0)

    crossover = _find_crossover(noise_scales, phys_means, logi_means)
    if crossover is not None:
        print(f"  Crossover point: noise_scale = {crossover:.3f}")
    else:
        print("  No crossover found in the scanned range")


def analyze_e3d(csv_path: str = None, paper: bool = False) -> None:
    """E3-D実験結果を分析し、全図表を生成する"""
    df = load_experiment_data("e3d_noise_scaling", csv_path)
    output_dir = os.path.join(RESULTS_DIR, "e3d_noise_scaling", "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("=== E3-D 分析 ===")
    print(f"Data: {len(df)} rows")
    print(f"Noise scales: {sorted(df['noise_scale'].unique())}")
    print(f"Output: {output_dir}")
    if paper:
        print("Style: IEEE paper")
    print()

    print("Noise scaling lines:")
    plot_noise_scaling_lines(df, output_dir, paper)
    print()

    print("State breakdown:")
    plot_state_breakdown(df, output_dir, paper)
    print()

    print("Success rate vs fidelity:")
    plot_success_vs_fidelity(df, output_dir, paper)
    print()

    print("Summary:")
    generate_summary(df, output_dir)


if __name__ == "__main__":
    from experiments.common.cli import create_analyzer_parser
    args = create_analyzer_parser("E3-D: 結果分析").parse_args()
    analyze_e3d(args.csv, paper=args.paper)
