"""
IEEE論文用の図表スタイル設定

IEEE投稿フォーマットに準拠したmatplotlib設定を提供する。
各analyze.pyから呼び出して使う共通モジュール。
"""
import os

import matplotlib
import matplotlib.pyplot as plt

# IEEE カラム幅 (inches)
SINGLE_COL = 3.5   # シングルカラム幅 (88mm)
DOUBLE_COL = 7.16   # ダブルカラム幅 (181mm)

# 解像度
DPI_PAPER = 300
DPI_SCREEN = 150


def apply_paper_style() -> None:
    """matplotlib rcParams を IEEE 論文用に設定する

    serif系フォント、小さめのフォントサイズ、高解像度にする。
    """
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": DPI_PAPER,
        "savefig.dpi": DPI_PAPER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "lines.markersize": 4,
    })


def apply_screen_style() -> None:
    """開発確認用のデフォルトスタイルに戻す"""
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": DPI_SCREEN,
        "savefig.dpi": DPI_SCREEN,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "lines.linewidth": 2,
        "axes.linewidth": 1.0,
        "lines.markersize": 6,
    })


def apply_style(paper: bool = False) -> None:
    """paper=True なら論文用、False なら通常スタイルを適用する"""
    if paper:
        apply_paper_style()
    else:
        apply_screen_style()


def get_figsize(width: str = "single", aspect: float = 0.75) -> tuple[float, float]:
    """IEEE カラム幅に合わせた figsize を返す

    Args:
        width: "single" (3.5in) または "double" (7.16in)
        aspect: 縦横比 (height = width * aspect)

    Returns:
        (width, height) のタプル
    """
    w = SINGLE_COL if width == "single" else DOUBLE_COL
    return (w, w * aspect)


def paper_title(full_title: str, paper: bool) -> str:
    """paper=True なら 'EX-Y: ' 接頭辞を除去する

    例: paper_title("E1-A: Circuit Depth by Strategy", paper=True)
         → "Circuit Depth by Strategy"
    """
    if paper and ": " in full_title:
        return full_title.split(": ", 1)[1]
    return full_title


def save_figure(fig: plt.Figure, path_without_ext: str, paper: bool = False) -> None:
    """図を保存する

    paper=True の場合: PDF (ベクター) + PNG (プレビュー) の両方を出力
    paper=False の場合: PNG のみ出力

    Args:
        fig: matplotlib Figure オブジェクト
        path_without_ext: 拡張子なしのファイルパス (例: "figures/e1c_box_depth")
        paper: 論文用出力かどうか
    """
    # 出力ディレクトリが存在しない場合は作成
    parent = os.path.dirname(path_without_ext)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if paper:
        # PDF (ベクター形式) - 論文投稿用
        pdf_path = f"{path_without_ext}.pdf"
        fig.savefig(pdf_path, format="pdf", dpi=DPI_PAPER, bbox_inches="tight")
        print(f"  Saved: {pdf_path}")

        # PNG (プレビュー用)
        png_path = f"{path_without_ext}.png"
        fig.savefig(png_path, format="png", dpi=DPI_PAPER, bbox_inches="tight")
        print(f"  Saved: {png_path}")
    else:
        # 通常出力: PNG のみ
        png_path = f"{path_without_ext}.png"
        fig.savefig(png_path, dpi=DPI_SCREEN)
        print(f"  Saved: {png_path}")
