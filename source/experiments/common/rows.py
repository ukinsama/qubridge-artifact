"""E1-G と E1-H の runner が CSV の 1 行を組むときに共有する小物。

役割: ショット推定の 6 列の丸め、seed の並び、論文のショット設定、verbose 表示の ASCII 化。
境界: 列の順（CSV の fieldnames）は各 runner の *_COLUMNS が持つ。E3-D は丸めずに書き、dry-run も seed 0 の 1 回なので使わない。
存在理由: 2026-11-07 に runner.py と runner_v6.py の同じ 4 か所を畳んだ。
契約: tests/test_experiments/test_rows.py（値は畳む前の式で出した数をそのまま貼ってある）
"""


def shot_columns(r: dict, seed) -> dict:
    """evaluate_step / simulate の dict（fidelity・sigma・acceptance・n_accepted_min・shots）から CSV の 6 列。
    F と σ は 6 桁に丸めて None（受理 0）はそのまま、数は None を 0 に、seed は None（dry-run）を -1 に"""
    return {
        "fidelity": None if r["fidelity"] is None else round(r["fidelity"], 6),
        "sigma": None if r["sigma"] is None else round(r["sigma"], 6),
        "acceptance": round(r["acceptance"], 6),
        "n_accepted_min": r["n_accepted_min"] or 0,
        "shots": r["shots"] or 0,
        "seed": -1 if seed is None else seed,
    }


def seed_runs(dry_run: bool, n_repeats: int, paper_seeds) -> list:
    """(repeat, seed) の並び。本番は論文の seed ごとに 1 行、dry-run は seed 無し（None）を n_repeats 行"""
    return list(enumerate([None] * n_repeats if dry_run else paper_seeds))


def paper_shots_config(common, seed):
    """論文のショット数（CommonConfig.paper_shots）で seed を 1 つ指定した ShotsConfig。
    services.pipeline.shots は qiskit_aer を読むので、runner の import を軽く保つために呼ぶときに引く"""
    from services.pipeline.shots import ShotsConfig
    return ShotsConfig(shots=common.paper_shots, seed=seed)


def ascii_label(label: str) -> str:
    """verbose の print 用。ket の ⟩ など ASCII の外の文字を ? にする（端末の文字コードで落ちないように）"""
    return label.encode("ascii", "replace").decode()
