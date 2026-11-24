"""フィードフォワード（測定と古典の補正）を、補正の手前まで回した密度行列に当てる共通部品。

役割: Bob の待ち時間の減衰と、誤り無しの補正を順に当てる。services/pipeline/runner.py の run_app だけが使う。
境界: 較正の値は呼び出し側が読んで渡す（ここは I/O をしない）。ノイズモデルと読み出しの誤りの混ぜ方は runner が持つ。
契約: 補正は、遅延測定の原理で量子制御ゲートに置き換えた回路を誤り無しで当てる。待つのは補正の前
（2026-09-27。tests/test_feedforward.py・tests/test_mode3_feedforward.py）。
待ちの減衰の式は services/pipeline/idle.py の `idle_error`（T1・T2 の緩和 ＋ 速い位相緩和。2026-10-28）が正本で、ここは呼ぶだけ。
`idle_error` は最上位で名前を束ねるので、差し替えるならこのモジュールの `idle_error` を差し替える。
"""
from qiskit.quantum_info import Operator

from services.pipeline.idle import FastDephasing, idle_error

# IBM の公式の値「Feedforward latency is also down to around ~600ns」
# （https://www.ibm.com/quantum/blog/utility-scale-dynamic-circuits、2026-09-27 ケイの決定で既定にした）
DEFAULT_FEEDFORWARD_LATENCY_S = 600e-9


def readout_wait(qp_map: dict, measured_qubits, latency_s: float = 0.0) -> float:
    """Bob が待つ時間: 測る qubit を並行して読む時間（長い方の readout_length）＋フィードフォワードの遅れ"""
    return max((qp_map.get(q, {}).get('readout_length') or 0.0) for q in measured_qubits) + latency_s


def apply_feedforward(rho, correction, waiting, wait_s: float, positions=None, fast: FastDephasing | None = None):
    """Bob を待たせてから、補正を誤り無しで当てる。

    仮定: 補正の X・Z（打つかどうかは L4 の決め事、打つゲートの質は L3）には 1Q のゲート誤りを乗せない。
    ショット版の運転手（experiments/common/shots_runner.py）は if_test の X・Z に雑音モデルの 1Q 誤りが乗るので、
    差は 1Q 誤り 1 回分（torino で 0.03% 程度）。論文の数字はこちら（誤り無し）。2026-10-16 の層の境目の監査 3

    Args:
        rho: 補正の手前まで回した密度行列（qiskit の DensityMatrix）
        correction: 補正の回路（仮想の qubit 番号で書く）
        waiting: 待つ qubit ごとの (仮想の番号, T1, T2)。T1・T2 が無ければ減衰を掛けない
        wait_s: 待ち時間（秒）
        positions: 仮想の番号 → 密度行列の中の位置（transpile で並びが変わったとき）。None なら同じ並び
        fast: 待ちの速い位相緩和（空き 1 区間 1 回。run_app が DD の有無で選ぶ）。None なら T1・T2 の緩和だけ（2026-10-21 までのモデル）
    """
    pos = list(positions) if positions is not None else list(range(correction.num_qubits))
    for virtual, t1, t2 in waiting:
        if wait_s > 0 and t1 and t2:   # 較正に T1・T2 が無い qubit には速い成分も掛けない
            rho = rho.evolve(idle_error(t1, t2, wait_s, fast).to_quantumchannel(), qargs=[pos[virtual]])
    return rho.evolve(Operator(correction), qargs=pos)
