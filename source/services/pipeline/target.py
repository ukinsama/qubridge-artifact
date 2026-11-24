"""Target（較正 ＋ L3 の形の差分）を組み、Qiskit の transpile に配置（L2）を選ばせる正本。

役割: L2 = Qiskit の transpile の入口。較正データから Target を組み、transpile が選んだ配置（仮想 → 物理）を読む。
境界: 雑音モデルは作らない（noise.py）。鎖の列挙・採点（experiments/common/chain_select.py）とは独立で、本番の UI の選択は触らない。
存在理由: 2026-10-16 ケイ「L2 を完全に transpile の中に入れたい」→「Target の較正データから組む部品を」。10/20「入れちゃいましょう」で採用。
     判断記録 docs/decisions/2026-10-16-l2-as-transpile.md（採用）。spec docs/superpowers/specs/2026-10-16-target-from-calibration-design.md
契約: compile_on_hardware は配置・経路づけ・翻訳・最適化・時刻決めを 1 回の qiskit.transpile に任せ（scheduling_method を渡したときだけ
     時刻決め）、自前で足すのは「使わない qubit を落とす」ことと「transpile が入れた delay に T1・T2 の緩和を当てる」ことだけ。
     鎖を渡す流れ（runner.run_app）の時刻決めはこのモジュールが正本: 較正の長さの表は durations_for_qubits（較正は gate_properties で
     雑音モデルと同じ関数）、時刻決めは schedule（PassManager の 3 パス: 単位を dt に → asap／alap で時刻 → 空きを delay か DD で埋める）、
     delay の緩和は relax_delays。experiments/common/shots_runner.py もこの 3 つを呼ぶ（2026-10-20 に畳んだ）。
     knob_inventory の記録ノート（layer_coupling_matrix_*）の durations_for・make_hook は記録の写しで、較正の instruction_durations
     の表を読む（値は gate_properties と 699 命令で一致、2026-10-20 確認。記録は回し直さない）。utils/scheduling.py は UI の表示用で別。
     時刻決めと緩和のパスは回路の .layout を保たないので、配置が要る呼び手は呼ぶ前に取っておく（run_app は transpile 直後に final_layout を取る）。
     論理の L2 の選択（CommonConfig.best_logical_path）は paper_chain が導出の正本で、値は較正とともに凍結として手で写す。物理の L2 は 2026-11-04 から
     全格子で F が最高の鎖（core.config.PAPER_BEST_PATH）で、paper_chain(PHYSICAL) は Qiskit の選択として本文で比べる相手に残る（docs/decisions/2026-11-04-l2-highest-fidelity.md）。
     配置の採点に渡す回路は layout_circuit（core ＋ measured・kept の測定）。VF2 は qubit ごとの平均誤りを「その仮想 qubit に載る
     1Q 命令の数」だけ掛けるので、測定の無い core では Bob（1Q 命令 0 個）の読み出し誤りが重み 0 になり、torino では読み出し 6.8% の
     q64 が Bob に落ちていた（2026-10-28）。計算に使う回路は core のまま（測定と補正は runner が別に当てる）。
     層の境目: L2 = transpile の既定の段（layout → routing → translation/optimization → scheduling の PadDelay まで）。
     DD（待ちの間に何を打つか）は L3 なので transpile の中に入れず、transpile の後に別のパス（apply_dd / apply_dd_on_chain）で足す。
     常にかかる ZZ は機体の性質で、時刻決めの後に apply_zz が delay の重なりに RZZ を差す（既定は無し。2026-10-20 仮）
     （docs/decisions/2026-10-16-layer-boundary-audit.md の 1）。
     build_target の sx・x・cz の誤り率は、同じ gate_pulse_map で build_lindblad_noise_model が使う RB 誤り（形の差分込み）と
     同じ関数（noise.shape_adjusted_1q_error / shape_adjusted_2q_error）で決める。VF2Layout が見る数字と雑音モデルの数字が一致する。
     measure は較正の読み出し誤りと読み出し長、rz は誤り 0・長さ 0、dt は較正の timing。土台は static_backend.create_target_from_static_data。
"""
from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionDurations, InstructionProperties, PassManager, Target
from qiskit.transpiler.passes import ALAPScheduleAnalysis, ASAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling, TimeUnitConversion
from qiskit_aer.noise import RelaxationNoisePass, phase_damping_error

from core.config import PAPER_SCHEDULING_METHOD
from services import static_backend
from services.pipeline.idle import FastDephasing
from services.pipeline.noise import (DEFAULT_GATE_PULSE_MAP, _get_gate_time, clamp_t2, get_2q_gate_time,
                                     shape_adjusted_1q_error, shape_adjusted_2q_error)


def build_target(backend_name: str, gate_pulse_map: Mapping[str, str] | None = None) -> Target:
    """較正データの Target に、L3 の形（gate_pulse_map）の差分を sx・x・cz の誤り率へ当てたものを返す。
    形が既定（sx・x は drag、cz は gaussian_square）なら較正値のまま。"""
    target = static_backend.create_target_from_static_data(backend_name)
    data = static_backend.load_static_backend_data(backend_name)
    gate_props = data.get("gate_properties", {})
    gpm = {**DEFAULT_GATE_PULSE_MAP, **(gate_pulse_map or {})}
    for gate in ("sx", "x"):
        if gate not in target.operation_names:
            continue
        shape = gpm.get(gate, "drag")
        for qargs_str, prop in gate_props.get(gate, {}).items():
            qargs = ast.literal_eval(qargs_str)
            if prop.get("error") is None:
                continue
            err = shape_adjusted_1q_error(prop["error"], shape, _get_gate_time(gate_props, gate, qargs[0]), gate=gate)
            target.update_instruction_properties(gate, qargs, InstructionProperties(duration=prop.get("duration"), error=err))
    if "cz" in target.operation_names:
        shape = gpm.get("cz", "gaussian_square")
        for qargs_str, prop in gate_props.get("cz", {}).items():
            qargs = ast.literal_eval(qargs_str)
            if prop.get("error") is None:
                continue
            t = get_2q_gate_time(gate_props, "cz", qargs[0], qargs[1]) or prop.get("duration") or 0.0
            err = shape_adjusted_2q_error(prop["error"], shape, t)
            target.update_instruction_properties("cz", qargs, InstructionProperties(duration=prop.get("duration"), error=err))
    return target


def transpile_on_target(qc: QuantumCircuit, target: Target, optimization_level: int = 3, seed: int = 42) -> QuantumCircuit:
    """Target で transpile する（配置 → 経路づけ → 翻訳・最適化。時刻決めは scheduling_method を渡した呼び手が足す）"""
    return transpile(qc, target=target, optimization_level=optimization_level, seed_transpiler=seed)


def layout_of(transpiled: QuantumCircuit, n_virtual: int, layout=None) -> list[int]:
    """transpile 後の回路から、元の回路の仮想 i → 最後に居る物理 qubit を読む（layout が無ければ恒等。最初の置き場は choose_layout）。
    layout は transpile 直後の回路の .layout（後のパスで消えるので、呼び手が持ち回ってよい）"""
    layout = layout or transpiled.layout
    if layout is None:
        return list(range(n_virtual))
    return [int(q) for q in layout.final_index_layout()[:n_virtual]]


PAPER_L2_STATE = {"theta": math.pi / 2, "phi": 0.0}   # 論文の L2 を決めた状態（|+⟩）。UI の「Qiskit L2」も同じ状態で選ぶ


def layout_circuit(spec, params: Mapping[str, float]) -> QuantumCircuit:
    """配置（L2）の採点に渡す回路: core（補正の手前まで）に measured と kept の測定を足したもの。
    測定を足すのは VF2 の採点に読み出し誤りを載せるため（Alice の 2 つは補正を誤らせ、Bob の最後の測定はショット版の F に直接乗る）。
    計算に使うのは core のまま（測定の副作用と補正は runner が当てる）。測定の基底は採点に効かないので Z で固定"""
    qc = spec.core_circuit(params).copy()
    qubits = list(dict.fromkeys((*spec.measured, *spec.kept)))      # 重なりを除き、順序は保つ
    creg = ClassicalRegister(len(qubits), "layout_meas")
    qc.add_register(creg)
    for k, v in enumerate(qubits):
        qc.measure(v, creg[k])
    return qc


def choose_layout(qc: QuantumCircuit, backend_name: str, gate_pulse_map: Mapping[str, str] | None = None,
                  optimization_level: int = 3, seed: int = 42) -> list[int]:
    """L2 の配置を Qiskit に選ばせる: Target（較正 ＋ 形の差分）で transpile し、仮想 i を最初に置いた物理 qubit の並びを返す。
    返した並びは、今の流れ（鎖 → create_reduced_noise_model → transpile_for_noise）の鎖としてそのまま渡せる。
    経路づけで SWAP が入る回路（論理テレポ）では最初の置き場と最後の位置が違う。鎖に要るのは最初の置き場
    （2026-10-20 まで最後の位置 layout_of を返していた。物理テレポは SWAP が無いので同じだった）"""
    tc = transpile_on_target(qc, build_target(backend_name, gate_pulse_map), optimization_level, seed)
    if tc.layout is None:
        return list(range(qc.num_qubits))
    return [int(q) for q in tc.layout.initial_index_layout()[:qc.num_qubits]]


@dataclass(frozen=True)
class Compiled:
    """transpile が機体の上に置いた回路を、使う qubit だけに縮めたもの"""
    circuit: QuantumCircuit            # 使う qubit だけの回路（index k = physical_qubits[k]）。時刻決めをしたなら delay 入り
    physical_qubits: list[int]         # 縮めた index → 物理 qubit。先頭 n 個は元の仮想 0..n−1 の初期配置、その後は経路づけが使った qubit
    final_positions: list[int]         # 元の仮想 i → 縮めた回路の中での最後の位置（SWAP が無ければ恒等）
    transpiled: QuantumCircuit         # 縮める前の、機体の幅のままの回路（確認用）


def compress_to_used(transpiled: QuantumCircuit, n_virtual: int, layout=None) -> Compiled:
    """機体の幅の回路から、ゲートか測定のある qubit だけを残す。delay しか無い qubit は落とす（時刻決めの詰め物）。
    layout は transpile 直後の .layout（DD のパスを通すと消えるので、compile_on_hardware が持ち回る）"""
    layout = layout or transpiled.layout
    used = _used_qubits(transpiled)
    initial = [int(q) for q in layout.initial_index_layout()[:n_virtual]] if layout else list(range(n_virtual))
    physical = initial + sorted(used - set(initial))
    index = {q: k for k, q in enumerate(physical)}
    out = QuantumCircuit(len(physical), transpiled.num_clbits, global_phase=transpiled.global_phase)
    for inst in transpiled.data:
        qs = [transpiled.find_bit(q).index for q in inst.qubits]
        if not all(q in index for q in qs):
            continue                                               # 使わない qubit の delay
        out.append(inst.operation, [index[q] for q in qs], [transpiled.find_bit(c).index for c in inst.clbits])
    final = layout_of(transpiled, n_virtual, layout)
    return Compiled(out, physical, [index[q] for q in final], transpiled)


def _used_qubits(circuit: QuantumCircuit) -> set[int]:
    """ゲートか測定のある qubit（delay しか無い qubit は時刻決めの詰め物なので数えない。dephase_idles が delay に差す
    channel 命令（quantum_channel）も詰め物の一部なので数えない）"""
    return {circuit.find_bit(q).index for inst in circuit.data
            if not isinstance(inst.operation, Delay) and inst.operation.name != "quantum_channel" for q in inst.qubits}


_ANALYSIS = {"asap": ASAPScheduleAnalysis, "alap": ALAPScheduleAnalysis}


def schedule(circuit: QuantumCircuit, durations: InstructionDurations, scheduling_method: str, pad=None) -> QuantumCircuit:
    """時刻決めの正本: 長さを dt に直し、asap／alap で各命令の時刻を決め、空きを pad（既定は delay）で埋める。ゲートは変えない。
    返る回路は .layout を持たない（PassManager の仕様）ので、配置が要る呼び手は呼ぶ前に取っておく"""
    if scheduling_method not in _ANALYSIS:
        raise ValueError(f"scheduling_method は {sorted(_ANALYSIS)} のどれか（{scheduling_method!r}）")
    pm = PassManager([TimeUnitConversion(durations), _ANALYSIS[scheduling_method](durations), pad or PadDelay()])
    return pm.run(circuit)


def apply_dd(transpiled: QuantumCircuit, target: Target, sequence: Sequence = (XGate(), XGate())) -> QuantumCircuit:
    """L3 のパス: 時刻決め済みの回路の delay に DD（既定は X–X）を詰める。transpile の外で呼ぶ（DD は L2 の transpile に含めない）。
    詰めるのは回路が使う qubit だけ（遊んでいる qubit に X を打つと「使う qubit」が増えてしまう）"""
    durations = target.durations()
    pad = PadDynamicalDecoupling(durations, list(sequence), qubits=sorted(_used_qubits(transpiled)),
                                 pulse_alignment=target.pulse_alignment or 1)
    return schedule(transpiled, durations, "alap", pad)


def apply_dd_on_chain(circuit: QuantumCircuit, backend_name: str, physical_qubits: Sequence[int],
                      sequence: Sequence = (XGate(), XGate())) -> QuantumCircuit:
    """L3 のパス（鎖を渡す流れ用）: 時刻決め済みの回路の delay に DD（既定は X–X）を詰める。apply_dd の Target 無し版で、
    長さは durations_for_qubits（較正の値）から取る。X には雑音モデルの x の誤りが乗る（DD は無料ではない）"""
    durations = durations_for_qubits(backend_name, physical_qubits)
    pad = PadDynamicalDecoupling(durations, list(sequence), qubits=sorted(_used_qubits(circuit)), pulse_alignment=1)
    return schedule(circuit, durations, "alap", pad)


def _instruction_duration_dt(op, qargs: list[int], durations: InstructionDurations) -> int:
    """命令の長さ（dt）。delay は命令が持つ長さ、barrier と rz は 0、それ以外は較正の表"""
    if isinstance(op, Delay):
        return int(op.duration) if op.unit == "dt" else int(round(float(op.duration) / durations.dt))
    if op.name in ("barrier", "rz"):
        return 0
    return int(durations.get(op.name, qargs, unit="dt"))


def apply_zz(circuit: QuantumCircuit, durations: InstructionDurations, edges: Sequence[tuple[int, int]],
             zz_hz: float | None) -> QuantumCircuit:
    """常にかかる ZZ（機体の性質。層の決め事ではない）を、時刻決め済み（delay 入り）の回路に入れる:
    結合した辺 (a, b) の両端が同時に空いている区間 τ ごとに RZZ(2π ζ τ) を差す（ζ は ZZ Ramsey の周波数差。核心は
    H = (2πζ/2)·ZZ/2 → 条件つき位相 2πζτ）。相手が測定中の区間と、測定のあとのフィードフォワードの待ち（_finish）には入れない。
    delay で埋めた回路は時刻が固まっているので、命令を順に再生するだけで開始時刻が出る（PadDelay の性質）。
    DD（apply_dd_on_chain）の後に呼ぶと、X で挟まれた区間の RZZ が符号を変えて打ち消し合う＝DD の効果がここで初めて F に出る。
    rzz は雑音モデルに誤りを持たない（fractional_gates=False）ので、ゲートの誤りは乗らない。
    ζ の正本は core.config.PAPER_ZZ_HZ（2026-10-21 に全辺の中央値。run_app の既定）。None か 0 なら何もしない"""
    if not zz_hz or not edges:
        return circuit
    n = circuit.num_qubits
    clock = [0] * n
    starts: list[int] = []
    idle: dict[int, list[tuple[int, int]]] = {q: [] for q in range(n)}
    for inst in circuit.data:
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        dur = _instruction_duration_dt(inst.operation, qargs, durations)
        s = max(clock[q] for q in qargs) if qargs else 0
        starts.append(s)
        for q in qargs:
            clock[q] = s + dur
        if isinstance(inst.operation, Delay) and dur > 0:
            idle[qargs[0]].append((s, s + dur))
    events: list[tuple[int, int, object]] = [(starts[i], 0, i) for i in range(len(circuit.data))]
    seen = set()
    for a, b in edges:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        for s1, e1 in idle[a]:
            for s2, e2 in idle[b]:
                s, e = max(s1, s2), min(e1, e2)
                if e > s:
                    events.append((s, 1, (a, b, 2 * math.pi * zz_hz * (e - s) * durations.dt)))
    events.sort(key=lambda x: (x[0], x[1], x[2] if x[1] == 0 else -1))
    out = circuit.copy_empty_like()
    for _, kind, payload in events:
        if kind == 0:
            inst = circuit.data[payload]
            out.append(inst.operation, [circuit.find_bit(q).index for q in inst.qubits],
                       [circuit.find_bit(c).index for c in inst.clbits])
        else:
            a, b, theta = payload
            out.rzz(theta, a, b)
    return out


def compile_on_hardware(qc: QuantumCircuit, backend_name: str, gate_pulse_map: Mapping[str, str] | None = None,
                        optimization_level: int = 3, scheduling_method: str | None = None, seed: int = 42,
                        dd: bool = False) -> Compiled:
    """L2 を丸ごと qiskit.transpile に任せる: Target（較正 ＋ 形の差分）で配置・経路づけ・翻訳・最適化（・時刻決め）を 1 回で行い、
    使う qubit だけに縮めて返す。雑音モデルは呼び手が physical_qubits から作る（順序: Target → transpile → 雑音モデル）。
    dd=True は時刻決めの後に L3 の DD のパス（apply_dd）を足す（scheduling_method が要る）"""
    target = build_target(backend_name, gate_pulse_map)
    tc = transpile(qc, target=target, optimization_level=optimization_level, scheduling_method=scheduling_method, seed_transpiler=seed)
    layout = tc.layout                       # DD のパスを通すと .layout が消えるので先に取る
    if dd:
        if scheduling_method is None:
            raise ValueError("DD は時刻決め（scheduling_method）の後に足す。scheduling_method を渡す")
        tc = apply_dd(tc, target)
    return compress_to_used(tc, qc.num_qubits, layout)


def dephase_idles(circuit: QuantumCircuit, backend_name: str, physical_qubits: Sequence[int],
                  fast_off: FastDephasing | None, fast_on: FastDephasing | None, dd: bool,
                  noise_scale: float = 1.0) -> QuantumCircuit:
    """時刻決め済み（delay 入り）の回路の各 delay に、空きの速い位相緩和（idle.FastDephasing）を 1 回ずつ差す（2026-10-28）。

    DD が delay を X で割る**前**に通す: g(t) は指数と違って足し合わせられないので、割った後に 1 本ずつ当てると重ねて数える。
    位相緩和は X と可換（X Z X = −Z でも Z ρ Z は同じ）なので、区間の中のどこに置いても同じ。T1・T2 の緩和は relax_delays が
    割った後の delay 1 本ずつに当てる（指数は足し合わせられる）。
    項の選び方: dd が真でその delay に X–X が入る長さ（≥ 2 × その qubit の x の長さ）なら fast_on、そうでなければ fast_off
    （詰められない短い空きは実機でも DD 無しの減衰をする）。選んだ項が None なら差さない。差すものが無ければ元の回路をそのまま返す。
    長さだけで選ぶので、PadDynamicalDecoupling が X を入れない先頭の空き（skip_reset_qubits）にも長ければ on の項が付くが、先頭の空きは
    |0⟩ のままで位相緩和が効かないので数値は変わらない（2026-10-28 のレビューで物理・論理 × asap/alap を確認）。較正に T1・T2 が無い
    qubit は見ていない（今は relax_delays が先に落ちる。Bob の待ちは apply_feedforward が飛ばす）。
    差す命令は Aer の quantum_channel。durations_for_qubits が長さ 0 を持つので、後の schedule・apply_dd_on_chain・apply_zz を通る"""
    if (fast_off is None and fast_on is None) or "delay" not in circuit.count_ops():
        return circuit
    durations = durations_for_qubits(backend_name, physical_qubits)
    gp = static_backend.load_static_backend_data(backend_name).get("gate_properties", {})
    x_len = {k: _get_gate_time(gp, "x", q) for k, q in enumerate(physical_qubits)}
    out = circuit.copy_empty_like()
    inserted = False
    for inst in circuit.data:
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        out.append(inst.operation, qargs, [circuit.find_bit(c).index for c in inst.clbits])
        if not isinstance(inst.operation, Delay):
            continue
        k = qargs[0]
        t_s = _instruction_duration_dt(inst.operation, qargs, durations) * durations.dt
        if t_s <= 0:
            continue
        fast = fast_on if (dd and t_s >= 2 * x_len[k]) else fast_off
        if fast is None:
            continue
        lam = fast.scaled(noise_scale).phase_damping_param(t_s)
        if lam <= 0:   # noise_scale = 0 などで a = 0 なら何も起きないので差さない
            continue
        out.append(phase_damping_error(lam).to_instruction(), [k])
        inserted = True
    return out if inserted else circuit


def relax_delays(circuit: QuantumCircuit, backend_name: str, physical_qubits: Sequence[int], noise_scale: float = 1.0,
                 t2_scale: float = 1.0) -> QuantumCircuit:
    """transpile（scheduling_method）が入れた delay に、その qubit の T1・T2 の緩和を当てる（Aer の RelaxationNoisePass）。
    delay が無ければそのまま返す。雑音モデルは delay に誤りを持たないので、ここで足す（ショット版の運転手と同じ）。
    t2_scale: T2 に掛ける倍率。DD 無しの空きは Ramsey の T2*（= PAPER_T2STAR_RATIO × T2）で落ちるので呼び手が 0.8 を渡す（2026-10-21）"""
    if "delay" not in circuit.count_ops():
        return circuit
    data = static_backend.load_static_backend_data(backend_name)
    qp = {q["qubit"]: q for q in data.get("qubit_properties", [])}
    t1s = [qp[q]["T1"] / noise_scale for q in physical_qubits]
    t2s = [clamp_t2(qp[q]["T1"] / noise_scale, t2_scale * qp[q]["T2"] / noise_scale) for q in physical_qubits]
    return RelaxationNoisePass(t1s, t2s, dt=float(data["timing"]["dt"]), op_types=Delay)(circuit)


def durations_for_qubits(backend_name: str, physical_qubits: Sequence[int]) -> InstructionDurations:
    """較正の長さの表（正本）。回路の index k = physical_qubits[k]。sx・x・id・measure は qubit ごと、rz は 0、
    cz は較正に辺がある全対を両向き（鎖の隣に限らない。経路づけが入った対にも長さが要る）。
    雑音モデルと同じ関数（noise._get_gate_time / get_2q_gate_time）で引くので、delay の長さとゲートの誤りが同じ較正を見る。
    dt は較正の timing。無い較正（is_offline_ready の必須項目）では時刻決めができないので止める"""
    data = static_backend.load_static_backend_data(backend_name)
    dt = data.get("timing", {}).get("dt")
    if dt is None:
        raise ValueError(f"{backend_name} の較正に timing.dt が無いので時刻決めはできない（scheduling_method=None にする）")
    gp = data.get("gate_properties", {})
    qp = {q["qubit"]: q for q in data.get("qubit_properties", [])}
    items = []
    for k, q in enumerate(physical_qubits):
        items += [(name, [k], _get_gate_time(gp, name, q), "s") for name in ("sx", "x", "id")]
        items.append(("rz", [k], 0.0, "s"))
        items.append(("quantum_channel", [k], 0.0, "s"))   # dephase_idles が delay に差す channel 命令。長さ 0 で時刻決め・DD・ZZ を通す
        if qp.get(q, {}).get("readout_length") is not None:
            items.append(("measure", [k], float(qp[q]["readout_length"]), "s"))
    for a, b in itertools.combinations(range(len(physical_qubits)), 2):
        d = get_2q_gate_time(gp, "cz", physical_qubits[a], physical_qubits[b])
        if d is not None:
            items += [("cz", [a, b], d, "s"), ("cz", [b, a], d, "s")]
    return InstructionDurations(items, dt=float(dt))


def schedule_on_chain(circuit: QuantumCircuit, backend_name: str, physical_qubits: Sequence[int], scheduling_method: str) -> QuantumCircuit:
    """鎖を渡す流れ（runner.run_app）と実験の circuit_hook の時刻決め: transpile 済みの回路（index k = physical_qubits[k]）に
    較正の長さで asap／alap の delay を入れる。delay の緩和は relax_delays が当てる。配置（.layout）は保たない"""
    return schedule(circuit, durations_for_qubits(backend_name, physical_qubits), scheduling_method)


def paper_chain(spec, backend_name: str, gate_pulse_map: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """Qiskit の配置の選択（論理の L2 の導出の正本。物理の論文の L2 は 2026-11-04 から core.config.PAPER_BEST_PATH）: |+⟩ の layout_circuit（core ＋ 測定）を Target（較正 ＋ 形の差分）・level 3・PAPER_SCHEDULING_METHOD で
    transpile した最初の置き場（仮想 i → 物理 qubit）。torino（較正 2026-01-16）では物理（3 qubit）66–65–55、論理（6 qubit。経路づけで
    SWAP が入る）65–66–68–67–69–74。2026-10-28 までは core だけで採点し、物理は Bob の読み出し 6.8% の 66–65–64 だった。
    CommonConfig.best_logical_path はこの値を較正の日付とともに手で写し、歯止めが一致を確かめる"""
    qc = layout_circuit(spec, PAPER_L2_STATE)
    return tuple(compile_on_hardware(qc, backend_name, gate_pulse_map, 3, PAPER_SCHEDULING_METHOD).physical_qubits)
