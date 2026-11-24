"""実験の出力先の決め方（本番と dry-run を分ける）。

役割: 3 本の runner が ExperimentLogger に渡す output_dir の既定を 1 か所で決める。
境界: ディレクトリは作らない（作るのは ExperimentLogger）。CLI の --output-dir が与えられたら runner はこれを呼ばない。
存在理由: 2026-11-07 まで dry-run も本番と同じ results/<id>/<id>.csv を "w" で開いていて、論文の CSV を消す作りだった。
契約: tests/test_experiments/test_output.py。歯止め: tests/test_experiments/test_dry_run.py
"""
import os

from experiments.config import RESULTS_DIR

DRY_RUN_SUBDIR = "dry_run"
REPLICATION_SUBDIR = "replication"


def seeds_tag(seeds) -> str:
    """seed の組を置き場の名前にする。連番なら seeds5-9、そうでなければ seeds10_20"""
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) > 1 and all(b == a + 1 for a, b in zip(seeds, seeds[1:])):
        return f"seeds{seeds[0]}-{seeds[-1]}"
    return "seeds" + "_".join(str(s) for s in seeds)


def replication_output_dir(experiment_id: str, seeds) -> str:
    """別の seed で回し直した結果の置き場 results/replication/<seed の札>/<id>（論文の results/<id> とは別）"""
    return os.path.join(RESULTS_DIR, REPLICATION_SUBDIR, seeds_tag(seeds), experiment_id)


def default_output_dir(experiment_id: str, dry_run: bool) -> str:
    """本番は results/<id>、dry-run は results/dry_run/<id>（論文の CSV の隣には作らない）"""
    if dry_run:
        return os.path.join(RESULTS_DIR, DRY_RUN_SUBDIR, experiment_id)
    return os.path.join(RESULTS_DIR, experiment_id)
