"""実験結果の CSV 追記型ロガー（行を CSV に追記し、終わりに metadata.json を書く）。

役割: E1-G・E1-H・E3-D の runner が 1 実験 1 インスタンスで使う。CSV の列は各 runner が自分で持つ（E1G_COLUMNS・E1H_COLUMNS・E3D_COLUMNS）。
出力先の既定は experiments/common/output.default_output_dir（dry-run は results/dry_run/<id>）。
境界: 数値は作らない。読むのは experiments/common/data_loader.py。
"""
import csv
import json
import os
from typing import List, Dict, Any

from experiments.config import RESULTS_DIR, get_experiment_metadata


class ExperimentLogger:
    """
    実験結果のCSV追記型ロガー。

    1つの実験ごとに1インスタンスを生成し、
    行単位でCSVに追記していく。
    実験終了時に metadata.json も出力する。
    """

    def __init__(
        self,
        experiment_id: str,
        csv_columns: List[str],
        output_dir: str = None,
    ):
        """
        Args:
            experiment_id: 実験ID（例: "e1h_pipeline_convergence"）
            csv_columns: CSVのカラム名リスト
            output_dir: 出力先ディレクトリ（デフォルト: experiments/results/{experiment_id}/）
        """
        self.experiment_id = experiment_id
        self.csv_columns = csv_columns

        # 出力ディレクトリを確定
        if output_dir is None:
            self.output_dir = os.path.join(RESULTS_DIR, experiment_id)
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # CSVファイルパス
        self.csv_path = os.path.join(self.output_dir, f"{experiment_id}.csv")

        # 毎回新規作成（再実行時にデータが混在しないよう上書き）
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writeheader()

        self._row_count = 0

    def log_row(self, row: Dict[str, Any]) -> None:
        """CSVに1行追記する"""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writerow(row)
        self._row_count += 1

    def save_metadata(self, extra: Dict[str, Any] = None) -> str:
        """
        実験メタデータをJSONで保存する。

        Args:
            extra: 追加のメタデータ（実験固有のパラメータ等）

        Returns:
            メタデータファイルのパス
        """
        metadata = get_experiment_metadata()
        metadata["experiment_id"] = self.experiment_id
        metadata["total_rows"] = self._row_count
        metadata["csv_path"] = self.csv_path
        if extra:
            metadata.update(extra)

        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return meta_path

    @property
    def row_count(self) -> int:
        return self._row_count


def paper_estimation_meta(common, seeds=None) -> dict:
    """metadata.json の末尾に 3 本の runner が同じ順で書く推定の設定（CommonConfig の estimation・paper_shots・paper_seeds）。
    呼び手は extra の同じ位置に ** で広げる（鍵の順が古い metadata.json と変わらないように）。
    seeds を渡すと paper_seeds にそれを書く（別の seed で回し直した結果が自分の seed を名乗る）"""
    used = common.paper_seeds if seeds is None else seeds
    return {"estimation": common.estimation, "paper_shots": common.paper_shots, "paper_seeds": list(used)}
