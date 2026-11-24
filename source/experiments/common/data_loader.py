"""
実験データ読み込みヘルパー

CSV形式の実験結果を読み込む汎用ローダー。
3つの analyze.py で共通利用される。
"""
import os

import pandas as pd

from experiments.config import RESULTS_DIR


def load_experiment_data(experiment_name: str, csv_path: str = None) -> pd.DataFrame:
    """
    実験CSVを読み込む。

    Args:
        experiment_name: 実験名（例: "e1c_fidelity"）。
            csv_path 未指定時のデフォルトパス生成に使用。
        csv_path: CSVファイルパス。None の場合は RESULTS_DIR から自動解決。

    Returns:
        実験結果の DataFrame

    Raises:
        FileNotFoundError: CSVファイルが見つからない場合
    """
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, experiment_name, f"{experiment_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)
