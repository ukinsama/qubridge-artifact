"""ショット推定: 動的回路（中間測定 → if_test の補正）をそのまま Aer で回し、counts から F・σ・受理率を出す。

役割: run_app の推定段（method="shots"）。段 0〜3（翻訳・transpile・時刻決め・空きの位相緩和・DD・ZZ）は runner.py が共通に持ち、
      ここは「測って・待って・補正して・固定の基底で測って・数える」だけを受け持つ。
境界: 測る基底は入力（送った状態）に依存しない固定の組（AppSpec.observables）。送った状態の向き n は後処理でだけ使う
      （2026-10-28 ケイ「測定操作部分というか回路の中身はインプットに対して不変です」）。
契約: 補正は AppSpec.correction()（遅延測定の制御ゲートの回路）から機械的に翻訳する。cx(c, t) → if m_c: x(t)、cz(c, t) → if m_c: z(t)。
      c は measured、t は kept（Bob）に無ければ ValueError（tests/test_pipeline_shots.py）。
      後処理の補正（Pauli frame。waiting=() の spec、teleportation.physical_pauli_frame）: 待ちも if_test も置かず Bob をすぐ測り、
      counts の Bob の bit を Alice の結果で読み替える（X の補正は Z・Y 基底の結果を反転、Z の補正は X・Y 基底を反転。frame_corrected_bits）。
      2026-10-28 の L4 の knob「補正の実装」のショット版。
spec: docs/superpowers/specs/2026-10-28-shots-estimation-design.md
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import phase_damping_error

from services import static_backend
from services.pipeline.app_spec import AppSpec
from services.pipeline.feedforward import readout_wait
from services.pipeline.idle import FastDephasing
from services.pipeline.target import relax_delays


@dataclass(frozen=True)
class ShotsConfig:
    """1 回の推定に使うショット数と乱数の種。seed が None なら毎回変える（画面の既定）。論文は CommonConfig.paper_shots・paper_seeds"""
    shots: int = 8192
    seed: int | None = 0


@dataclass(frozen=True, eq=False)   # eq=False: ndarray を持つので値で比べない
class Observable:
    """Bob（kept）に対する固定の測り方 1 つ。入力が何でも同じ組を使う"""
    name: str                    # "X" / "Y" / "Z"（物理）、"Z_L" / "X_L" / "Y_L"（論理）
    bases: tuple[str, ...]       # kept の各 qubit を測る基底（"X"・"Y"・"Z"。kept の位置の昇順）
    parity_of: tuple[int, ...]   # 期待値に使う bit（kept の順の添字）。Z_L = Z⊗I なら (0,)、X_L = X⊗X なら (0, 1)
    matrix: np.ndarray           # kept の空間の演算子。理想状態から n_i = Tr(ρ_ideal · matrix) を出す


@dataclass(frozen=True)
class ShotsResult:
    """estimate_by_shots の戻り。fidelity と sigma は受理 0 のとき None"""
    fidelity: float | None
    sigma: float | None
    bloch: tuple[float, ...]            # ⟨O_i⟩（observables の順）
    bloch_sigma: tuple[float, ...]
    acceptance: float                   # 中間測定の後選択で残った割合（全観測量の合計）。後選択が無ければ 1
    acceptance_sigma: float
    detected_at_readout: float | None   # 論理だけ。Z_L の組で読めるパリティ違反の割合（F の条件にはしない）
    shots: int
    seed: int | None                    # 実際に使った種（cfg.seed が None なら引いた値）
    n_accepted_min: int                 # 観測量ごとの受理 shot 数の最小（σ の根拠）
    p_measured: tuple[float, float] | None   # 論理だけ。Z_L の組で偶パリティだった受理 shot のうち bit0 = 0 / 1 の割合


_GATE_TO_PAULI = {"cx": "x", "cz": "z"}
_SEED_MULT = 2654435761          # Knuth の乗法ハッシュ。隣り合う seed を Aer の種の空間で遠くに置く
_OBS_STRIDE = 1 << 21            # 観測量ごとの種の間隔（> 最大ショット数 2^20。run の shot の種が重ならない）
_SEED_MOD = 1 << 31


def simulator_seed(seed: int, obs_index: int) -> int:
    """使う側の seed と観測量の番号から Aer の seed_simulator を作る。
    Aer は中間測定のある回路を shot ごとに回し、shot k の乱数の種を seed_simulator + k にする（2026-10-28 に実測）。
    そのため seed を 1 ずつ変えると shot の種がほぼ重なって同じ counts になり、seed を複数使っても揺れが見えない
    （spread/σ ≈ 0.00）。種を 2^22 以上離すと二項分布の揺れに戻る（比 0.94〜1.25）。ここで離す"""
    return (seed * _SEED_MULT + obs_index * _OBS_STRIDE) % _SEED_MOD


def corrections_from_deferred(spec: AppSpec) -> tuple[tuple[int, str, int], ...]:
    """遅延測定の補正回路（制御ゲート）を、if_test で打つ Pauli の列 (measured の仮想番号, "x"|"z", waiting の仮想番号) に翻訳する。
    correction が None なら空"""
    if spec.correction is None:
        return ()
    out: list[tuple[int, str, int]] = []
    qc = spec.correction()
    for inst in qc.data:
        name = inst.operation.name
        if name not in _GATE_TO_PAULI:
            raise ValueError(f"{spec.id}: 補正の回路に {name} がある（遅延測定の形は cx・cz だけ）")
        control, target = (qc.find_bit(q).index for q in inst.qubits)
        if control not in spec.measured:
            raise ValueError(f"{spec.id}: 補正の制御 {control} が measured {spec.measured} に無い")
        if target not in spec.kept:
            raise ValueError(f"{spec.id}: 補正の標的 {target} が kept {spec.kept} に無い")
        out.append((control, _GATE_TO_PAULI[name], target))
    return tuple(out)


_FLIPPED_BY = {"x": ("Z", "Y"), "z": ("X", "Y")}   # その Pauli の補正が結果を反転させる測定基底（X は X と可換、Z は Z と可換）


def is_frame(spec: AppSpec) -> bool:
    """後処理の補正（Pauli frame）か: 補正はあるが待つ qubit が無い spec"""
    return spec.correction is not None and not spec.waiting


def frame_corrected_bits(f_bits: str, mids: Sequence[int], corrections: Sequence[tuple[int, str, int]],
                         measured: Sequence[int], kept_order: Sequence[int], bases: Sequence[str]) -> str:
    """後処理の補正: Alice の結果 mids（measured の順）で決まる Pauli を、Bob の測定結果 f_bits（kept_order の順）の反転に直す。
    ⟨P σ P⟩ = ±⟨σ⟩ なので、補正 P が σ と反可換な基底だけ bit を反転する。2 回反転すれば元に戻る（X と Z の両方で Y は不変）"""
    bits = list(f_bits)
    for c, gate, t in corrections:
        if mids[measured.index(c)] != 1:
            continue
        j = kept_order.index(t)
        if bases[j] in _FLIPPED_BY[gate]:
            bits[j] = "1" if bits[j] == "0" else "0"
    return "".join(bits)


def _basis_change(qc: QuantumCircuit, q: int, basis: str) -> None:
    """測る軸を Z へ回す（X: H、Y: S†H、Z: 無し）。入力によらず固定の組の一部"""
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)
    elif basis != "Z":
        raise ValueError(f"基底 {basis!r}（X・Y・Z だけ）")


def build_shot_circuit(spec: AppSpec, qc_sched: QuantumCircuit, obs, layout: list[int] | None, qubits: Sequence[int] | None,
                       backend_name: str | None, *, feedforward_latency_s: float, dd: bool, zz_hz: float | None,
                       noise_scale: float, t2_scale: float, fast: FastDephasing | None, calibrated: bool,
                       edges: Sequence[tuple[int, int]] = ()) -> QuantumCircuit:
    """1 観測量ぶんの動的回路: 段 0〜3 を終えた回路 → 中間測定 → 待ち（DD・ZZ・速い位相緩和）→ if_test の補正 →
    固定の基底変更 → 測定 → delay に T1・T2 の緩和。古典レジスタは mid（measured の順）と fin（kept の位置の昇順）。
    位置は pos[i]（経路づけの SWAP で仮想 i は pos[i] に居る。物理番号は qubits[pos[i]]）。
    calibrated が False（雑音なし・機体なし）なら待ちと緩和は入れない。qc_sched の delay はまだ緩和を当てていないこと（run_app が分岐点をそこにする）。
    edges は位置の空間の結合（run_app が reduced_coupling_map で出したもの。ここで runner を import すると循環するので受け取る）"""
    n = qc_sched.num_qubits
    pos = list(layout) if layout is not None else list(range(n))
    qr = QuantumRegister(n, "q")
    mid = ClassicalRegister(len(spec.measured), "mid")
    fin = ClassicalRegister(len(spec.kept), "fin")
    qc = QuantumCircuit(qr, mid, fin)
    qc.compose(qc_sched, qubits=list(range(n)), inplace=True)
    for k, m in enumerate(spec.measured):
        qc.measure(pos[m], mid[k])

    waits = calibrated and spec.correction is not None and bool(spec.measured) and bool(spec.waiting)
    if waits:
        data = static_backend.load_static_backend_data(backend_name)
        dt = float(data["timing"]["dt"])
        qp_map = {q["qubit"]: q for q in data.get("qubit_properties", [])}
        wait_s = readout_wait(qp_map, [qubits[pos[m]] for m in spec.measured], feedforward_latency_s)
        wait_dt = int(round(wait_s / dt))
        coupled = {(min(a, b), max(a, b)) for a, b in edges}
        for w in spec.waiting:
            pw = pos[w]
            if fast is not None:   # 待ち全体に速い位相緩和を 1 回（DD で 3 本に割っても 1 回。X と可換なので先頭でよい）
                lam = fast.scaled(noise_scale).phase_damping_param(wait_s)
                if lam > 0:
                    qc.append(phase_damping_error(lam).to_instruction(), [pw])
            if dd:   # 待ち/4 → X → 待ち/2 → X → 待ち/4。静的な ZZ の位相は打ち消すので RZZ は入れない（runner._zz_during_wait と同じ規則）
                q4, q2 = wait_dt // 4, wait_dt // 2
                qc.delay(q4, pw, unit="dt")
                qc.x(pw)
                qc.delay(q2, pw, unit="dt")
                qc.x(pw)
                qc.delay(wait_dt - q4 - q2, pw, unit="dt")
            else:
                qc.delay(wait_dt, pw, unit="dt")
                if zz_hz:
                    for m in spec.measured:
                        pm = pos[m]
                        if (min(pm, pw), max(pm, pw)) in coupled:
                            qc.rzz(2 * math.pi * zz_hz * wait_s, pm, pw)

    if not is_frame(spec):   # 後処理の補正なら if_test を置かず、counts の読み替えで済ませる（estimate_by_shots）
        for c, gate, t in corrections_from_deferred(spec):
            with qc.if_test((mid[spec.measured.index(c)], 1)):
                getattr(qc, gate)(pos[t])

    kept_order = sorted(spec.kept, key=lambda i: pos[i])
    for j, i in enumerate(kept_order):
        _basis_change(qc, pos[i], obs.bases[j])
        qc.measure(pos[i], fin[j])

    if waits:
        qc = relax_delays(qc, backend_name, list(qubits[:n]), noise_scale, t2_scale)
    return qc


def parse_counts(counts: dict, n_mid: int) -> dict[tuple[str, tuple[int, ...]], int]:
    """Aer の鍵 "fin mid"（レジスタの間は空白、各レジスタは bit 0 が右）を {(fin の bit 列（kept の順）, mid の値の組（measured の順）): 回数} に直す"""
    out: dict[tuple[str, tuple[int, ...]], int] = {}
    for key, c in counts.items():
        parts = key.split()
        f_bits = parts[0][::-1]
        mids = tuple(int(b) for b in parts[1][::-1]) if n_mid else ()
        out[(f_bits, mids)] = out.get((f_bits, mids), 0) + c
    return out


def expectation(kept: dict[str, int], parity_of: Sequence[int]) -> tuple[float, float]:
    """parity_of の bit の偶奇（1 が偶数個なら +1）の平均と、その二項分布の分散 (1 − e²)/N。受理 0 なら (0, inf)"""
    n = sum(kept.values())
    if n == 0:
        return 0.0, float("inf")
    e = sum(c * (1 if sum(int(bits[j]) for j in parity_of) % 2 == 0 else -1) for bits, c in kept.items()) / n
    return e, (1 - e * e) / n


def bloch_target(spec: AppSpec, params) -> tuple[float, ...]:
    """送った状態の向き n_i = Tr(ρ_ideal · O_i)（後処理でだけ使う。回路は入力を知らない）"""
    rho = np.asarray(spec.ideal(params).data)
    return tuple(float(np.trace(rho @ obs.matrix).real) for obs in spec.observables)


def estimate_by_shots(spec: AppSpec, qc_sched: QuantumCircuit, layout, qubits, backend_name, noise_model, params,
                      cfg: ShotsConfig, *, feedforward_latency_s: float, dd: bool, zz_hz: float | None, noise_scale: float,
                      t2_scale: float, fast: FastDephasing | None, calibrated: bool,
                      edges: Sequence[tuple[int, int]] = ()) -> ShotsResult:
    """観測量ごとに動的回路を回し、counts から F = ½(1 + Σ n_i⟨O_i⟩)・σ・受理率・読み出し時の検出を出す。
    seed が None なら 1 回引いて全観測量に使い、ShotsResult.seed に入れる。観測量 i の Aer の種は simulator_seed(seed, i)"""
    if not spec.observables:
        raise ValueError(f"{spec.id}: observables が空なので shots では推定できない")
    seed = cfg.seed if cfg.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    sim = AerSimulator(method="statevector")
    n_mid = len(spec.measured)
    pos = list(layout) if layout is not None else list(range(qc_sched.num_qubits))
    kept_order = sorted(spec.kept, key=lambda i: pos[i])          # fin の順（build_shot_circuit と同じ）
    frame_corr = corrections_from_deferred(spec) if is_frame(spec) else ()
    es, vs, accepted, total, n_min = [], [], 0, 0, None
    detected, p_measured = None, None
    for i, obs in enumerate(spec.observables):
        qc = build_shot_circuit(spec, qc_sched, obs, layout, qubits, backend_name, feedforward_latency_s=feedforward_latency_s,
                                dd=dd, zz_hz=zz_hz, noise_scale=noise_scale, t2_scale=t2_scale, fast=fast, calibrated=calibrated,
                                edges=edges)
        run_kw = {"shots": cfg.shots, "seed_simulator": simulator_seed(seed, i)}
        if noise_model is not None:
            run_kw["noise_model"] = noise_model
        raw = sim.run(qc, **run_kw).result().get_counts()
        kept: dict[str, int] = {}
        for (f_bits, mids), c in parse_counts(raw, n_mid).items():
            total += c
            if spec.accept_mid is not None and not spec.accept_mid(mids):
                continue
            accepted += c
            if frame_corr:
                f_bits = frame_corrected_bits(f_bits, mids, frame_corr, spec.measured, kept_order, obs.bases)
            kept[f_bits] = kept.get(f_bits, 0) + c
        n_acc = sum(kept.values())
        n_min = n_acc if n_min is None else min(n_min, n_acc)
        e, v = expectation(kept, obs.parity_of)
        es.append(e)
        vs.append(v)
        if len(spec.kept) == 2 and all(b == "Z" for b in obs.bases) and n_acc:   # 論理: Z_L の組で読めるパリティと、偶パリティの中の bit0 の割合
            odd = sum(c for bits, c in kept.items() if bits[0] != bits[1])
            even = n_acc - odd
            detected = odd / n_acc
            p_measured = ((sum(c for bits, c in kept.items() if bits == "00") / even,
                           sum(c for bits, c in kept.items() if bits == "11") / even) if even else (0.0, 0.0))
    n = bloch_target(spec, params)
    acceptance = accepted / total if total else 0.0
    acc_sigma = math.sqrt(acceptance * (1 - acceptance) / total) if total else 0.0
    if n_min == 0 or any(math.isinf(v) for v in vs):
        fidelity, sigma = None, None
    else:
        fidelity = 0.5 * (1 + sum(ni * ei for ni, ei in zip(n, es)))
        sigma = 0.5 * math.sqrt(sum(ni * ni * vi for ni, vi in zip(n, vs)))
    return ShotsResult(fidelity=fidelity, sigma=sigma, bloch=tuple(es), bloch_sigma=tuple(math.sqrt(v) if not math.isinf(v) else float("inf") for v in vs),
                       acceptance=acceptance, acceptance_sigma=acc_sigma, detected_at_readout=detected, shots=cfg.shots, seed=seed,
                       n_accepted_min=n_min or 0, p_measured=p_measured)
