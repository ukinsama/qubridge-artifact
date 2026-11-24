"""
CLI引数パーサー生成

runner用・analyze用の共通CLIパーサーを提供する。
各実験モジュールの __main__ ブロックで利用する。
"""
import argparse


def parse_seeds(text: str) -> tuple:
    """'5,6,7' → (5, 6, 7)。空と重複は ValueError（同じ seed を 2 回は回さない）"""
    seeds = tuple(int(s) for s in text.split(",") if s.strip())
    if not seeds:
        raise ValueError("--seeds が空")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"同じ seed が 2 回ある: {text}")
    return seeds


def create_runner_parser(description: str) -> argparse.ArgumentParser:
    """
    runner用の共通CLIパーサーを生成する。

    Args:
        description: パーサーの説明文（例: "E1-C: seed分布実験"）

    Returns:
        --dry-run, --backend, --verbose, --output-dir, --seeds を持つ ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dry-run", action="store_true", help="少数データで動作確認（results/dry_run/<id> に書く）")
    parser.add_argument("--backend", default=None, help="バックエンド名")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")
    parser.add_argument("--output-dir", default=None,
                        help="CSV と metadata.json の置き場（既定: results/<id>。--dry-run は results/dry_run/<id>）")
    parser.add_argument("--seeds", type=parse_seeds, default=None,
                        help="論文の seed（CommonConfig.paper_seeds）の代わりに回す seed。カンマ区切り（例: 5,6,7,8,9）。"
                             "再現の置き場は experiments.replication.run が決める")
    return parser


def create_analyzer_parser(description: str) -> argparse.ArgumentParser:
    """
    analyze用の共通CLIパーサーを生成する。

    Args:
        description: パーサーの説明文（例: "E1-C: 結果分析"）

    Returns:
        --csv, --paper を持つ ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--csv", default=None, help="CSVファイルパス")
    parser.add_argument("--paper", action="store_true", help="IEEE論文用高品質出力 (PDF+PNG)")
    return parser
