"""アプリ1つが走る流れ。読む順に:

1. app_spec.py     定義（何を走らせるか。回路・補正・残す qubit・指標）
2. qubits.py       経路と配置の選択（L2。鎖の列挙・Baseline・配置の採点）
3. noise.py        ノイズモデル（L3。較正から Lindblad 合成、パルスの形、fractional の写し）
4. idle.py         空きの減衰の正本（T1・T2 ＋ 始めの速い位相緩和。feedforward と target が使う。2026-10-28）
5. feedforward.py  待ち時間の減衰と誤り無しの補正
6. measurement.py  最後の測定の確率（混同行列つき）
7. runner.py       流れ（transpile → 時刻決め → 分岐: 密度行列 → 補正 → 読み出しの混ぜ方 → 指標 ／ shots.py）
   shots.py        推定段 shots（中間測定 → 待ち → if_test の補正 → 固定の 3 基底 → counts から F・σ・受理率。2026-10-28）

較正データは services/static_backend.py（ここには置かない。57 ファイルが使い、名前も役割どおり）。
アプリの定義と回路は services/apps/。experiments/ はこのパッケージを使う側で、逆向きには import しない
（歯止め tests/test_dependency_direction.py）。ここに再輸出は置かない（tests/test_pipeline_init.py）。
spec: docs/superpowers/specs/2026-09-28-pipeline-package-design.md
"""
