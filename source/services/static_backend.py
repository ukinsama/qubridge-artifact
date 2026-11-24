"""較正データ（static/data/backend_errors*.json）の読み口と、そこから Qiskit の Target・CouplingMap を組む正本。

役割: load_static_backend_data（機体 1 つの dict。キャッシュつき）、create_target_from_static_data、
     create_coupling_map_from_static_data、較正の一覧とオフライン可否、パルス表示用の抜粋（load_calibration_data）、
     機体の basis_gates（basis_gates_for。Mode1 の可視化と Mode3 の transpile が同じ関数を呼ぶ。2026-11-07）。
境界: ノイズモデルは services/pipeline/noise.py、形の差分を載せた Target は services/pipeline/target.build_target。
     ここは pipeline を import しない（tests/test_noise_public_names.py）。
契約: 読むのは static/data/backend_errors.json だけ（2026-10-28 に日付つきの較正と `名前@較正名` の仕組みを消した。古い較正は
     git の履歴にある）。テストは load_static_backend_data を差し替えるので、読む側はモジュール越しに呼ぶ。
"""

import ast
import json
import logging
import os
import threading
from typing import Dict, cast

logger = logging.getLogger(__name__)

# ファイルごとのキャッシュ（path → 中身）。マルチワーカーで同時に読んでも 1 回だけ読むようロックする
_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'data')
DEFAULT_FILE = os.path.join(DATA_DIR, 'backend_errors.json')

# 較正に basis_gates が無いときの既定（IBM Heron 系: CZ）
DEFAULT_BASIS_GATES = ['cz', 'id', 'rz', 'sx', 'x']

def _read(path: str) -> Dict:
    """ファイルを 1 回だけ読んでキャッシュする（無ければ FileNotFoundError）"""
    with _cache_lock:
        if path not in _cache:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Static backend data not found: {path}\n"
                    f"project_root={os.path.dirname(os.path.dirname(DATA_DIR))}, cwd={os.getcwd()}\n"
                    "Run 'python scripts/update_backend_data.py' to generate it."
                )
            logger.debug("Loading static backend data from %s", path)
            with open(path, 'r', encoding='utf-8') as f:
                _cache[path] = json.load(f)
        return _cache[path]


def load_static_backend_data(backend_name: str | None = None) -> Dict:
    """機体 1 つの較正の dict を返す。None なら既定ファイルの全体（{"backends": {...}, ...}）。無い機体は ValueError。"""
    data = _read(DEFAULT_FILE)
    if backend_name is None:
        return data
    backends = data.get('backends', {})
    if backend_name not in backends:
        raise ValueError(f"Backend '{backend_name}' not found in static data. Available: {list(backends)}")
    return cast(Dict, backends[backend_name])


def basis_gates_for(backend_name: str | None) -> list:
    """機体の basis_gates の写し。名前が空なら既定、較正に無い・読めないときも既定（読めないときは warning）"""
    if not backend_name:
        return list(DEFAULT_BASIS_GATES)
    try:
        backend_data = load_static_backend_data().get('backends', {}).get(backend_name)
    except Exception as e:
        logger.warning("Failed to load basis_gates from static data: %s", e)
        return list(DEFAULT_BASIS_GATES)
    if backend_data and 'basis_gates' in backend_data:
        return list(backend_data['basis_gates'])
    return list(DEFAULT_BASIS_GATES)


def is_offline_ready(backend_name: str) -> bool:
    """
    指定バックエンドがオフライン動作に必要なデータを持っているか確認。

    Returns:
        True if all required fields are present
    """
    try:
        backend_data = load_static_backend_data(backend_name)
        required_fields = ['timing', 'coupling_map', 'instruction_durations', 'qubit_properties']
        return all(field in backend_data for field in required_fields)
    except (FileNotFoundError, ValueError):
        return False


def create_target_from_static_data(backend_name: str):
    """
    静的データからQiskitのTargetオブジェクトを再構築。

    Page5のtranspile()とtimeline_drawer()に必要。

    Args:
        backend_name: バックエンド名

    Returns:
        qiskit.transpiler.Target: 再構築されたTargetオブジェクト
    """
    from qiskit.transpiler import Target, InstructionProperties
    from qiskit.circuit.library import standard_gates
    from qiskit.circuit import Measure, Delay, Reset

    backend_data = load_static_backend_data(backend_name)

    # タイミング情報を取得
    timing = backend_data.get('timing', {})

    # Targetを作成
    target = Target(
        description=f"Static target for {backend_name}",
        num_qubits=backend_data['n_qubits'],
        dt=timing.get('dt'),
        granularity=timing.get('granularity', 1),
        min_length=timing.get('min_length', 1),
        pulse_alignment=timing.get('pulse_alignment', 1),
        acquire_alignment=timing.get('acquire_alignment', 1)
    )

    # 標準ゲートのマッピング
    gate_map = {
        'ecr': standard_gates.ECRGate,
        'cx': standard_gates.CXGate,
        'cz': standard_gates.CZGate,
        'id': standard_gates.IGate,
        'rz': standard_gates.RZGate,
        'sx': standard_gates.SXGate,
        'x': standard_gates.XGate,
        'y': standard_gates.YGate,
        'z': standard_gates.ZGate,
        'h': standard_gates.HGate,
        's': standard_gates.SGate,
        'sdg': standard_gates.SdgGate,
        't': standard_gates.TGate,
        'tdg': standard_gates.TdgGate,
        'rzz': standard_gates.RZZGate,
        'rzx': standard_gates.RZXGate,
    }

    gate_properties = backend_data.get('gate_properties', {})
    basis_gates = backend_data.get('basis_gates', [])

    # 各basis_gateをTargetに追加
    for gate_name in basis_gates:
        if gate_name not in gate_map:
            continue

        gate_class = gate_map[gate_name]
        props = {}

        # gate_propertiesからプロパティを取得
        if gate_name in gate_properties:
            for qargs_str, prop_data in gate_properties[gate_name].items():
                try:
                    # "(0, 1)" -> (0, 1) または "(0,)" -> (0,)
                    # ast.literal_eval()で安全にパース（evalはコード実行の脆弱性があるため使用しない）
                    qargs = ast.literal_eval(qargs_str)
                    props[qargs] = InstructionProperties(
                        duration=prop_data.get('duration'),
                        error=prop_data.get('error')
                    )
                except (ValueError, SyntaxError):
                    # qargs_strのパースに失敗した場合はスキップ
                    continue

        try:
            # パラメータ付きゲート（RZ・RZZ・RZX）は引数が要る
            if gate_name in ('rz', 'rzz', 'rzx'):
                target.add_instruction(gate_class(0), props if props else None)
            else:
                target.add_instruction(gate_class(), props if props else None)
        except Exception as e:
            logger.warning("Could not add gate %s to target: %s", gate_name, e)

    # Measure命令を追加
    qubit_properties = backend_data.get('qubit_properties', [])
    measure_props = {}
    for qp in qubit_properties:
        q = qp['qubit']
        measure_props[(q,)] = InstructionProperties(
            duration=qp.get('readout_length'),
            error=qp.get('readout_error')
        )

    try:
        target.add_instruction(Measure(), measure_props if measure_props else None)
    except Exception as e:
        logger.warning("Could not add Measure to target: %s", e)

    # Delay と Reset は qubit ごとに登録する。全 qubit 共通（qargs=None）で登録すると target.qargs に None が混ざり、
    # VF2Layout（optimization_level 1〜3 の配置）が TypeError で落ちる（2026-10-16、L2 を transpile に揃える作業で実測）
    per_qubit = {(q,): None for q in range(backend_data['n_qubits'])}
    try:
        target.add_instruction(Delay(0), dict(per_qubit))
    except Exception:
        pass  # オプショナル: Delay命令がなくてもTargetは機能する
    try:
        target.add_instruction(Reset(), dict(per_qubit))
    except Exception:
        pass  # オプショナル: Reset命令がなくてもTargetは機能する

    return target


def create_coupling_map_from_static_data(backend_name: str):
    """較正の coupling_map から CouplingMap を作る（2 ファイル 6 機体とも coupling_map を持つ。is_offline_ready も必須にする）"""
    from qiskit.transpiler import CouplingMap

    backend_data = load_static_backend_data(backend_name)
    if 'coupling_map' not in backend_data:
        raise ValueError(f"No coupling map data available for {backend_name}")
    return CouplingMap([tuple(edge) for edge in backend_data['coupling_map']])


def load_calibration_data(backend_name: str) -> dict:
    """バックエンドのキャリブレーションデータを取得。

    dt, instruction_durations, gate_properties を返す。
    パルス可視化のタイミング・エラーレート表示に使用。

    Args:
        backend_name: IBM Quantum backend name

    Returns:
        dict with 'dt', 'instruction_durations', 'gate_properties' keys.
    """
    try:
        backend_info = load_static_backend_data(backend_name)
        if backend_info:
            return {
                'dt': backend_info.get('timing', {}).get('dt', 4e-9),
                'instruction_durations': backend_info.get('instruction_durations', []),
                'gate_properties': backend_info.get('gate_properties', {}),
            }
    except Exception:
        pass
    return {}


def clear_cache():
    """読み込んだ較正のキャッシュを全部消す（テスト用）"""
    with _cache_lock:
        _cache.clear()
