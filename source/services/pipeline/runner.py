"""共通の実行部品: アプリの定義（AppSpec）を、較正に合わせたノイズつきの密度行列で回す。

役割: 補正の手前まで回す → 待ち → 誤り無しの補正 → 選ぶ → 捨てる → 読み出しを補正の誤りとして混ぜる → 指標。
境界: ノイズモデルは呼び出し側が作って渡す（Mode1 は gate_pulse_map つき、Mode3・E3-D は noise_scale つき）。transpile 済みの回路に処理を挟む口は circuit_hook（既定 None。実験の時刻決め・delay の緩和用、2026-10-16）。
較正は services.static_backend を呼ぶときに読む（テストはこのモジュールの load_static_backend_data を差し替える）。
契約: 順番は 2026-09-27 までの simulate_dm・compute_dm_fidelity_physical/logical と同じ。変えると論文の数字が変わる
（tests/test_paper_numbers.py）。spec: docs/superpowers/specs/2026-09-27-multi-app-pipeline-design.md
"""
from __future__ import annotations

import logging
import math
from typing import Mapping, Sequence

import numpy as np
from qiskit import transpile as qk_transpile
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit_aer import AerSimulator

from core.config import PAPER_SCHEDULING_METHOD, PAPER_T2STAR_RATIO, PAPER_ZZ_HZ
from services import static_backend
from services.pipeline.app_spec import AppSpec, MetricInput, RESERVED_METRIC_KEYS
from services.pipeline.measurement import measurement_probabilities
from services.pipeline.idle import PAPER_IDLE_FAST, FastDephasing
from services.pipeline.target import apply_dd_on_chain, apply_zz, dephase_idles, durations_for_qubits, relax_delays, schedule_on_chain
from services.pipeline.feedforward import DEFAULT_FEEDFORWARD_LATENCY_S, apply_feedforward, readout_wait
from services.pipeline.noise import with_readout_scale
from services.pipeline.shots import ShotsConfig, ShotsResult, estimate_by_shots

logger = logging.getLogger(__name__)


def _backend_data(backend_name: str) -> dict:
    return static_backend.load_static_backend_data(backend_name)


def _qubit_props(backend_name: str) -> dict:
    """較正の qubit_properties を {物理番号: dict} で（待ち・読み出し・最後の測定が同じ表を見る）"""
    return {qp['qubit']: qp for qp in _backend_data(backend_name).get('qubit_properties', [])}


def transpile_for_noise(qc, noise_model, coupling_map=None, initial_layout=None):
    """ノイズの付いたゲート ∪ {rz} を基底にトランスパイルする（opt=3・seed=42）。

    noise_model.basis_gates には cx も入るが、誤りを付けているのは cz だけなので、basis_gates のままだと cx が
    残って誤りを避けてしまう。noise_instructions（実際に誤りの付いたゲート）に rz（仮想の Z、誤り不要）を足して
    基底にする。swap は基底に入れないので、経路の SWAP は cz に分解されて誤りを受ける。
    """
    if noise_model is None:
        return qc
    noise_gates = set(noise_model.noise_instructions) - {'measure', 'reset', 'barrier'}
    if not noise_gates:
        return qc
    return qk_transpile(qc, basis_gates=list(noise_gates | {'rz'}), coupling_map=coupling_map,
                        initial_layout=initial_layout, optimization_level=3, seed_transpiler=42)


def reduced_coupling_map(backend_name: str | None, qubits: Sequence[int] | None, n: int) -> list | None:
    """選んだ qubit の先頭 n 個の物理の隣接を、仮想の番号 0..n-1 に直す。n 個そろわなければ None。

    回路の辺のうち物理で隣り合わないものには、トランスパイラが SWAP を入れ、その誤りがノイズに載る。
    """
    if not (backend_name and qubits and len(qubits) >= n):
        return None
    q_to_idx = {q: i for i, q in enumerate(qubits[:n])}
    reduced = [[q_to_idx[a], q_to_idx[b]] for a, b in _backend_data(backend_name).get('coupling_map', [])
               if a in q_to_idx and b in q_to_idx]
    return reduced or None


def final_layout(qc) -> list[int]:
    """トランスパイル後の 仮想 → 密度行列の中の位置。取れなければ恒等。"""
    try:
        return qc.layout.final_index_layout() if qc.layout else list(range(qc.num_qubits))
    except Exception:
        return list(range(qc.num_qubits))


def readout_error(backend_name: str, qubit: int) -> float:
    """較正の読み出し誤り（測定パルスの形によらない。2026-09-27 ケイの決定）。取れなければ 0。
    記録ノート（knob_inventory の格子）がモジュール属性で差し替えるので、名前と署名を保つ"""
    try:
        return float(_qubit_props(backend_name).get(qubit, {}).get('readout_error', 0.0))
    except Exception:
        logger.debug("readout error 取得失敗 (backend=%s, qubit=%s)、0 を使用", backend_name, qubit, exc_info=True)
        return 0.0


def readout_probability(backend_name: str, qubits: Sequence[int], measured: Sequence[int],
                        noise_scale: float) -> float:
    """補正を間違える確率: 測る qubit の読み出し誤りの平均（noise_scale 倍）を 0.5 で頭打ちにする。"""
    return min(sum(readout_error(backend_name, qubits[i]) * noise_scale for i in measured) / len(measured), 0.5)


def readout_flip_probabilities(backend_name: str, qubits: Sequence[int], measured: Sequence[int],
                               noise_scale: float) -> tuple[float, ...]:
    """測定ごとに、その読み違いで補正を間違える確率（qubit の読み出し誤り × noise_scale を 0.5 で頭打ち）。

    op i は measured[i] の測定が決める補正なので、op ごとに別の確率を返す（2026-10-09 まで平均 p を全部の op に使っていた）。
    """
    return tuple(min(readout_error(backend_name, qubits[i]) * noise_scale, 0.5) for i in measured)


def apply_readout_mix(rho, ops, p):
    """op ごとの独立な確率 p_i で、ops の部分集合すべてを数えて混ぜる（2 op なら 4 項）。

    rho <- Σ_S Π_{i∈S} p_i Π_{i∉S} (1 − p_i) · (Π_{i∈S} op_i) rho (Π_{i∈S} op_i)†
    p は数（全部の op に同じ確率）か op ごとの並び。全部 0 以下なら rho をそのまま返す。
    """
    probs = [float(p)] * len(ops) if np.isscalar(p) else [float(v) for v in p]
    if len(probs) != len(ops):
        raise ValueError(f"ops が {len(ops)} 個なのに確率が {len(probs)} 個")
    if all(v <= 0 for v in probs):
        return rho
    arr = np.array(rho.data)
    out = np.zeros_like(arr)
    for mask in range(1 << len(ops)):
        w = 1.0
        op_total = None
        for i, (op, pi) in enumerate(zip(ops, probs)):
            if mask >> i & 1:
                w *= pi
                op_total = op if op_total is None else op @ op_total
            else:
                w *= 1 - pi
        if w == 0:
            continue
        out += w * arr if op_total is None else w * (op_total @ arr @ op_total.conj().T)
    return DensityMatrix(out)


def _density_matrix(qc, noise_model):
    """保存命令を足して Aer の密度行列法で回す（コピーしてから足す: core_circuit が同じ回路を返してもそれを書き換えない）"""
    qc = qc.copy()
    qc.save_density_matrix()
    return AerSimulator(method='density_matrix', noise_model=noise_model).run(qc).result().data()['density_matrix']


def run_app(spec: AppSpec, params: Mapping[str, float], backend_name: str | None = None,
            qubits: Sequence[int] | None = None, noise_model=None, noise_scale: float = 1.0,
            feedforward_latency_s: float = DEFAULT_FEEDFORWARD_LATENCY_S, circuit_hook=None,
            scheduling_method: str | None = PAPER_SCHEDULING_METHOD, readout_scale: float | None = None,
            dd: bool = False, zz_hz: float | None = PAPER_ZZ_HZ, t2star_ratio: float | None = PAPER_T2STAR_RATIO,
            idle_fast: Mapping[bool, FastDephasing] | None = PAPER_IDLE_FAST,
            method: str = "dm", shots: ShotsConfig | None = None) -> dict:
    """アプリを1回回し、指標・残した状態・受理率・測定の確率・残した qubit の物理番号を返す。
    method: F の推定法。"dm" は密度行列（補正を誤り無しの制御ゲートに置き換え、測定の副作用はチャネルで足す）、
        "shots" は動的回路（中間測定 → if_test の補正 → 固定の 3 基底）をショットで回して counts から出す（services/pipeline/shots.py）。
        段 0〜3 は共通で、relax_delays の手前で分かれる。戻りの鍵は両方で同じ（dm は sigma=0.0・shots=None）。
        run_app 自体の既定は "dm"（正確な推定器。検算・ノート・dm の内部のテストが使う）。論文と画面の既定 core.config.PAPER_ESTIMATION_METHOD は
        入口の包み（services/mode1/simulation.simulate・services/mode3/fidelity.compute_fidelity_*・pipeline_eval・experiments の runner）が渡す（2026-10-28）
    shots: method="shots" のショット数と seed（None なら ShotsConfig() の既定 8,192・seed 0）

    circuit_hook: transpile 済みの回路を受けて回路を返す関数。実験側から処理を挟む口（κ の格子）。hook で自前の時刻決めを挟むときは
        scheduling_method=None にする（delay と緩和が二重に乗る）。
    scheduling_method: 時刻決め（"asap"／"alap"）。較正の長さで delay を入れ、delay に T1・T2 の緩和を当てる（target.schedule_on_chain・relax_delays）。
        既定は論文の値（core.config.PAPER_SCHEDULING_METHOD）。None なら delay 無し（2026-10-16 の凍結の形）。
    readout_scale: L4 の読み出しの倍率（補正を間違える確率に掛ける。None なら noise_scale、0 で読み違い無し、0.9 で整形）。
        画面の L4 と κ の格子の knob（2026-10-16 ケイ「0.9」）。最後の測定の混同行列（measure_final）は noise_scale のまま。
    dd: L3 の DD（待ちの間に X–X）。回路の delay には target.apply_dd_on_chain、Bob の待ち（測定＋遅れ）には _zz_during_wait が
        X を 2 回置く。X の誤りは雑音モデルどおり乗り、静的な ZZ の位相だけ打ち消す
    zz_hz: 常にかかる ZZ（機体の性質）。回路の中の空きの重なりは target.apply_zz、Bob の待ちは _zz_during_wait。
        既定は論文の代表値 core.config.PAPER_ZZ_HZ（2026-10-21 に全辺の中央値。ケイ「②にしたい」）。None なら入れない（10/20 のモデル）
    t2star_ratio: DD 無しの空き（回路の delay と Bob の待ち）の減衰を T2 でなく T2* = ratio × T2 で当てる（2026-10-21 ケイ「中央値で入れて」、
        core.config.PAPER_T2STAR_RATIO）。DD ありは X–X が echo の役をするので T2 のまま。None なら T2（10/20 のモデル）
    idle_fast: 空きの始めの速い位相緩和（鍵は DD の有無。idle.FastDephasing）。回路の delay には target.dephase_idles が DD で割る前に
        空き 1 区間 1 回、Bob の待ちには _finish → apply_feedforward が 1 回当てる。既定は論文の代表値 core.config.PAPER_IDLE_FAST_TERMS
        （marrakesh の実測。2026-10-28 ケイ「モデル直しましょう」）。None で 10/21 のモデル。spec: docs/superpowers/specs/2026-10-28-idle-fast-dephasing-design.md
    """
    # 0. 選んだ qubit が定義の数に足りなければ、qubits[pos[i]] の IndexError になる前に止める
    if qubits and len(qubits) < spec.n_qubits:
        raise ValueError(f"spec {spec.id!r} は {spec.n_qubits} 個の qubit を使うが、qubits は {len(qubits)} 個")
    if circuit_hook is not None and scheduling_method is not None:
        raise ValueError("circuit_hook で時刻決めを挟むときは scheduling_method=None にする（delay と緩和が二重に乗る）")
    chain = list(qubits[:spec.n_qubits]) if qubits else None   # 回路の index k = chain[k]

    # 1–3. 補正の手前までを、ノイズの付いたゲートに翻訳して密度行列で回す
    qc = spec.core_circuit(params)
    cm = reduced_coupling_map(backend_name, qubits, spec.n_qubits) if spec.uses_layout else None
    qc = transpile_for_noise(qc, noise_model, coupling_map=cm,
                             initial_layout=list(range(qc.num_qubits)) if cm else None)
    layout = final_layout(qc) if spec.uses_layout else None   # 配置は transpile で決まる。後の時刻決めと緩和は .layout を保たないのでここで取る
    if circuit_hook is not None:
        qc = circuit_hook(qc)
    has_noise = noise_model is not None and len(noise_model.noise_instructions) > 0
    # 時刻決め（L2）と delay の緩和（L3 の待ち時間の扱い）。雑音が無いなら delay は何も変えないので入れない
    calibrated = has_noise and bool(backend_name) and bool(qubits)   # 較正の待ち・読み出しを当てる条件（理想の経路では当てない）
    edges = [tuple(e) for e in (reduced_coupling_map(backend_name, qubits, qc.num_qubits) or [])] if calibrated else []
    t2_scale = 1.0 if dd or t2star_ratio is None else t2star_ratio   # DD 無しの空きは T2*（Ramsey）、DD ありは T2（echo）で落ちる
    fast_off, fast_on = (idle_fast or {}).get(False), (idle_fast or {}).get(True)   # 空きの速い位相緩和（None なら無し）
    if scheduling_method is not None and calibrated:
        qc = schedule_on_chain(qc, backend_name, chain, scheduling_method)
        qc = dephase_idles(qc, backend_name, chain, fast_off, fast_on, dd, noise_scale)   # DD が delay を割る前に、空き 1 区間 1 回
        if dd:
            qc = apply_dd_on_chain(qc, backend_name, chain)
        if zz_hz:   # 回路の中で両端が同時に空く区間（テレポには無い。GHZ 等の長い鎖で出る）。緩和を当てる前に差す
            qc = apply_zz(qc, durations_for_qubits(backend_name, chain), edges, zz_hz)
    scheduled = scheduling_method is not None and calibrated
    if method == "shots":   # 推定段 shots: 待ち・補正・固定の基底変更・測定を回路に足し、delay の緩和はその中で当てる。
        # Bob の待ち（読み出し長＋遅れ）は dm の _finish と同じく scheduling_method=None でも入れる（calibrated だけで決める。
        # 2026-10-28 の最終レビュー Important 5: `calibrated and scheduled` だと時刻決め無しの呼び手で待ちが消え、dm より 4 pt 高く出ていた）
        if not spec.observables:
            raise ValueError(f"spec {spec.id!r} は observables が空なので method='shots' では推定できない")
        nm = with_readout_scale(noise_model, backend_name, qubits, readout_scale) if (readout_scale is not None and calibrated) else noise_model
        res = estimate_by_shots(spec, qc, layout, qubits, backend_name, nm, params, shots or ShotsConfig(),
                                feedforward_latency_s=feedforward_latency_s, dd=dd, zz_hz=zz_hz, noise_scale=noise_scale,
                                t2_scale=t2_scale, fast=fast_on if dd else fast_off, calibrated=calibrated, edges=edges)
        pos = layout if layout is not None else list(range(qc.num_qubits))
        kept_qubits = [qubits[pos[i]] for i in sorted(spec.kept, key=lambda i: pos[i])] if qubits else None
        return _result_from_shots(spec, res, kept_qubits)
    if method != "dm":
        raise ValueError(f"method={method!r}（\"dm\" か \"shots\"）")
    if scheduled:
        qc = relax_delays(qc, backend_name, chain, noise_scale, t2_scale)
    if zz_hz and calibrated:   # Bob の待ち（測定＋遅れ）の ZZ。dd なら X–X で打ち消す
        qc = _zz_during_wait(qc, spec, layout, qubits, backend_name, feedforward_latency_s, edges, zz_hz, dd)
    dm = _density_matrix(qc, noise_model)
    out = _finish(spec, params, dm, qc.num_qubits, layout, qubits, backend_name, noise_scale, feedforward_latency_s, calibrated,
                  noise_scale if readout_scale is None else readout_scale, t2_scale, fast_on if dd else fast_off)
    out.update(sigma=0.0, bloch=None, bloch_sigma=None, acceptance_sigma=0.0, detected_at_readout=None, shots=None, seed=None, method="dm",
               n_accepted_min=None)
    return out


_PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex), "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _result_from_shots(spec: AppSpec, res: ShotsResult, kept_qubits) -> dict:
    """ShotsResult を run_app の dict に直す。state は r から組んだ 1 qubit の密度行列 ½(I + r·σ)（|r| > 1 なら r を 1 に縮める。
    論理は復号後の 1 qubit の意味）。bloch・fidelity は生の値。論理の互換の鍵 error_detection_rate（= 1 − 中間シンドロームの受理率）・
    p_measured_0/1 も返す（決定③）。受理 0 なら fidelity・sigma・state は None"""
    r = {obs.name.rstrip("_L")[0]: b for obs, b in zip(spec.observables, res.bloch)}   # "X_L" → "X"
    state = None
    if res.fidelity is not None:
        vec = np.array([r.get("X", 0.0), r.get("Y", 0.0), r.get("Z", 0.0)])
        norm = float(np.linalg.norm(vec))
        if norm > 1.0:
            vec = vec / norm
        state = DensityMatrix(0.5 * (np.eye(2, dtype=complex) + sum(v * _PAULI[k] for v, k in zip(vec, "XYZ"))))
    out = {"score": res.fidelity, "fidelity": res.fidelity, "state": state, "acceptance": res.acceptance, "probabilities": None,
           "kept_qubits": kept_qubits, "sigma": res.sigma, "bloch": res.bloch, "bloch_sigma": res.bloch_sigma,
           "acceptance_sigma": res.acceptance_sigma, "detected_at_readout": res.detected_at_readout, "shots": res.shots,
           "seed": res.seed, "method": "shots", "n_accepted_min": res.n_accepted_min}
    if len(spec.kept) == 2:   # 論理の互換の鍵
        p0, p1 = res.p_measured if res.p_measured is not None else (0.0, 0.0)
        out.update(error_detection_rate=1.0 - res.acceptance, p_measured_0=p0, p_measured_1=p1)
    return out


def _zz_during_wait(qc, spec: AppSpec, layout, qubits, backend_name, feedforward_latency_s: float,
                    edges: Sequence[tuple[int, int]], zz_hz: float, dd: bool):
    """Bob の待ち（測る qubit の読み出し ＋ フィードフォワードの遅れ。_finish が緩和を当てる区間）の ZZ。
    測る qubit m と待つ qubit w が結合していれば、待ちの間に RZZ(2π ζ τ_wait) が掛かる。RZZ は m の Z 測定と可換なので、
    回路の末尾（測定の手前）に置いても待ちの間に置いたのと同じ。m の結果に条件づいた Z の回転が w に残る（フレームで直さない形）。
    dd=True は L3 の DD（待ちの間に w へ X–X）: 静的な ZZ の位相は打ち消すので RZZ を入れず、代わりに X を 2 回置いて
    x の誤りだけを乗せる（T1・T2 は DD で戻らないので _finish の緩和はそのまま）"""
    if spec.correction is None or not spec.measured or not spec.waiting:
        return qc
    pos = layout if layout is not None else list(range(qc.num_qubits))
    qp_map = _qubit_props(backend_name)
    wait_s = readout_wait(qp_map, [qubits[i] for i in spec.measured], feedforward_latency_s)
    coupled = {(min(a, b), max(a, b)) for a, b in edges}
    qc = qc.copy()
    for w in spec.waiting:
        pw = pos[w]
        if dd:
            qc.x(pw)
            qc.x(pw)
            continue
        for m in spec.measured:
            pm = pos[m]
            if (min(pm, pw), max(pm, pw)) in coupled:
                qc.rzz(2 * math.pi * zz_hz * wait_s, pm, pw)
    return qc


def _finish(spec: AppSpec, params: Mapping[str, float], dm, n_circuit: int, layout, qubits, backend_name,
            noise_scale: float, feedforward_latency_s: float, calibrated: bool, readout_scale: float | None = None,
            t2_scale: float = 1.0, fast: FastDephasing | None = None) -> dict:
    """補正の手前の密度行列から先（4. 待ちと補正 → 5–6. 後選択と partial trace → 7. 読み違えの混ぜ込み → 8. 指標）。
    run_app から切り出した（2026-10-16。run_app_on_target と共有していたが、そちらは 2026-10-20 に消した）。
    calibrated: 雑音あり・機体名あり・qubit ありのとき True。待ち（4）・読み違い（7）・最後の測定の混同（8）はこのときだけ当てる。
    readout_scale: 7. の読み違いの倍率（None なら noise_scale）
    t2_scale: 4. の待ちの T2 に掛ける倍率（DD 無しは T2* = 0.8 × T2。run_app が決める）
    fast: 4. の待ちの速い位相緩和（run_app が DD の有無で選ぶ。None なら無し）。a は noise_scale 倍（apply_feedforward の中で scaled）"""
    # 4. 測定とフィードフォワードの間だけ待たせ（T1・T2 は noise_scale で割る。T2 は t2_scale 倍）、補正を誤り無しで当てる
    if spec.correction is not None:
        waiting, wait_s = [], 0.0
        if calibrated and noise_scale > 0:
            qp_map = _qubit_props(backend_name)
            wait_s = readout_wait(qp_map, [qubits[i] for i in spec.measured], feedforward_latency_s)
            for v in spec.waiting:
                qp = qp_map.get(qubits[v], {})
                t1, t2 = qp.get('T1'), qp.get('T2')
                waiting.append((v, t1 / noise_scale if t1 else None, t2_scale * t2 / noise_scale if t2 else None))
        dm = apply_feedforward(dm, spec.correction(), waiting, wait_s, layout,
                               fast=fast.scaled(noise_scale) if (fast is not None and waiting) else None)

    # 5–6. （論理だけ）シンドロームで選び、残す qubit 以外を捨てる。partial_trace は位置の昇順で返すので、
    #      残した qubit の物理番号も同じ順で出す
    pos = layout if layout is not None else list(range(n_circuit))
    traced = [pos[i] for i in range(spec.n_qubits) if i not in spec.kept]
    kept_order = sorted(spec.kept, key=lambda i: pos[i])
    # 仮想の i は経路づけの SWAP で位置 pos[i] へ動くので、物理番号は qubits[pos[i]]（qubits[i] は動く前の番号）
    kept_qubits = [qubits[pos[i]] for i in kept_order] if qubits else None
    acceptance = 1.0
    if spec.postselect is not None:
        rho_post, p_valid = spec.postselect(dm.data, pos, n_circuit)
        if p_valid < 1e-12:
            dim = 2 ** len(spec.kept)
            kept = DensityMatrix(np.zeros((dim, dim), dtype=complex))
            rejected = 1.0
        else:
            rho_post /= p_valid
            rejected = max(0.0, 1.0 - p_valid)
            kept = partial_trace(DensityMatrix(rho_post), traced)
        acceptance = 1.0 - rejected
    else:
        kept = partial_trace(dm, traced)

    # 7. 読み違えた測定から補正を間違える分を混ぜる（readout.measured < n_qubits ≤ len(qubits) は AppSpec と 0. が保証する）
    if spec.readout is not None and calibrated:
        ps = readout_flip_probabilities(backend_name, qubits, spec.readout.measured,
                                        noise_scale if readout_scale is None else readout_scale)
        kept = apply_readout_mix(kept, spec.readout.ops, ps)

    # 8. 指標。run_app が足すキー（RESERVED_METRIC_KEYS）を metric が返したら、黙って上書きせずここで止める
    ideal = spec.ideal(params) if spec.ideal is not None else None
    probabilities = None
    if spec.measure_final:
        qp_map = _qubit_props(backend_name) if calibrated else None
        probabilities = measurement_probabilities(kept, qp_map, kept_qubits, noise_scale)
    metrics = spec.metric(MetricInput(kept, ideal, params, acceptance, probabilities, kept_qubits))
    reserved = RESERVED_METRIC_KEYS & metrics.keys()
    if reserved:
        raise ValueError(f"spec {spec.id!r} の metric が予約済みキー {sorted(reserved)} を返した")
    if "score" not in metrics:
        raise ValueError(f"spec {spec.id!r} の metric が score を返さない（0〜1 の指標が要る）")
    score = metrics["score"]
    # 数値の丸めで 1 をわずかに超える忠実度は通す（1e-9 の余裕）。NaN・無限大・範囲の外は比べる側が誤るので止める
    if not (math.isfinite(score) and -1e-9 <= score <= 1 + 1e-9):
        raise ValueError(f"spec {spec.id!r} の metric の score={score!r} が 0〜1 の外")
    return {**metrics, "state": kept, "acceptance": acceptance, "probabilities": probabilities,
            "kept_qubits": kept_qubits}
