"""Error budget breakdown + waterfall layer fidelity for Mode1 Results.

論文 §III.B の error budget (SX / CZ / Readout 損失) と、L1/L2/L3 各層を加える
ごとに改善する fidelity の waterfall を返す。Mode1 Results UI と E1-H 実験で
共有される計算ロジック。

3 層 fidelity:
  baseline:  median-quality path、 全 gate Square (no tuning at any layer)
  after_l2:  user_qubits、         全 gate Square (qubit selection only)
  after_l3:  user_qubits、         user の per-gate pulse shapes (full L3 tuning)
"""
import logging
import threading

from core.config import ALL_SQUARE_PULSE_MAP as _NO_TUNING_PULSE_MAP   # 手つかずの baseline は全ゲート Square（正本は core.config）
from services.static_backend import load_static_backend_data
from services.pipeline.qubits import find_all_perfect_layouts, get_error_map
from services.mode1.simulation import simulate

logger = logging.getLogger(__name__)


def clamp_error(rate):
    """Clamp error rate to valid range, matching noise model filtering (0 < rate < 1)."""
    return rate if 0 < rate < 1 else 0


# =============================================================================
# Baseline qubit selection (10th-percentile path within Baseline distribution)
# =============================================================================

_baseline_cache: dict[str, list] = {}
_baseline_cache_lock = threading.Lock()


def get_baseline_qubits(backend_name: str) -> list:
    """Get the Default path: a representative point within the Baseline distribution.

    Baseline = paths meeting basic hardware quality thresholds
        (T1 ≥ 30μs, T2 ≥ 15μs, 1Q error ≤ 1%, 2Q edge error ≤ 10%).
    Default = the 10th-percentile path of the Baseline distribution under
        L3 = All Square (the untuned L3 state). Used as the Waterfall UI's
        Step 1 reference and Table I's starting point.

    For the paper backend (``core.config.PAPER_BACKEND``), ``core.config.PAPER_DEFAULT_PATH`` supplies
    the pre-computed path. Other backends fall back to a
    75th-percentile heuristic on (2Q + RO) error score.

    Result is cached per backend since static data does not change.

    Returns:
        list of 3 physical qubit indices.
    """
    with _baseline_cache_lock:
        if backend_name in _baseline_cache:
            return _baseline_cache[backend_name]

    # For the primary backend, use the pre-computed Default path from config.
    from core.config import PAPER_BACKEND, PAPER_DEFAULT_PATH
    if backend_name == PAPER_BACKEND:
        chosen = list(PAPER_DEFAULT_PATH)
        logger.info("Default qubits for %s: %s (10th-pct of Baseline, from core.config)",
                    backend_name, chosen)
        with _baseline_cache_lock:
            _baseline_cache[backend_name] = chosen
        return chosen

    # Fallback: compute a below-average representative for other backends.
    backend_data = load_static_backend_data(backend_name)
    error_map_2q = get_error_map(backend_data.get('errors', []))

    raw_paths = find_all_perfect_layouts(backend_name, [(0, 1), (1, 2)])
    if not raw_paths:
        return [0, 1, 2]

    ro_map = {p['qubit']: p.get('readout_error', 0) for p in backend_data.get('qubit_properties', [])}
    scored = []
    for path in raw_paths:
        a, b, c = path
        gate_score = error_map_2q.get((a, b), 0) + error_map_2q.get((b, c), 0)
        ro_score = ro_map.get(a, 0) + ro_map.get(b, 0)
        scored.append((gate_score + ro_score, list(path)))

    scored.sort()
    FALLBACK_PERCENTILE = 75
    idx = len(scored) * FALLBACK_PERCENTILE // 100
    chosen = scored[idx][1]
    logger.info("Default qubits for %s: %s (fallback %dth-pct, %d/%d paths)",
                backend_name, chosen, FALLBACK_PERCENTILE, idx, len(scored))

    with _baseline_cache_lock:
        _baseline_cache[backend_name] = chosen
    return chosen


# =============================================================================
# Error decomposition (analytical, from gate error rates)
# =============================================================================

def decompose_errors(qubits, gate_properties, qubit_properties, errors):
    """Decompose fidelity loss by error source for teleportation circuit.

    Args:
        qubits: list of 3 physical qubit indices [q0, q1, q2]
        gate_properties: dict from static backend data
        qubit_properties: list from static backend data
        errors: list of 2Q gate error entries from static backend data

    Returns:
        dict with sx_loss, cz_loss, ro_loss, estimated_fidelity
    """
    qp_map = {qp['qubit']: qp for qp in qubit_properties}

    two_q_gate = None
    for entry in errors:
        if entry.get('gate'):
            two_q_gate = entry['gate']
            break

    # Teleportation uses ~3 SX gates per qubit (H = RZ·SX·RZ decomposition)
    SX_GATES_PER_QUBIT = 3
    sx_fidelity = 1.0
    for q in qubits:
        sx_err = clamp_error(gate_properties.get('sx', {}).get(f'({q},)', {}).get('error', 0))
        sx_fidelity *= (1 - sx_err) ** SX_GATES_PER_QUBIT

    two_q_fidelity = 1.0
    if two_q_gate:
        for i in range(len(qubits) - 1):
            a, b = qubits[i], qubits[i + 1]
            gate_err = clamp_error(
                gate_properties.get(two_q_gate, {}).get(f'({a}, {b})', {}).get('error', 0)
                or gate_properties.get(two_q_gate, {}).get(f'({b}, {a})', {}).get('error', 0)
            )
            two_q_fidelity *= (1 - gate_err)

    ro_fidelity = 1.0
    for q in qubits[:2]:
        qp = qp_map.get(q, {})
        ro_fidelity *= (1 - clamp_error(qp.get('readout_error', 0)))

    return {
        'sx_loss': round((1 - sx_fidelity) * 100, 3),
        'cz_loss': round((1 - two_q_fidelity) * 100, 3),
        'ro_loss': round((1 - ro_fidelity) * 100, 3),
        'estimated_fidelity': round(sx_fidelity * two_q_fidelity * ro_fidelity * 100, 2),
    }


# =============================================================================
# High-level orchestration: error budget + 3-layer waterfall
# =============================================================================

def compute_error_decomposition(backend_name, user_qubits, theta, phi,
                                 gate_pulse_map=None, shots=None):
    """Compute full error decomposition with layer-by-layer fidelity waterfall.

    Args:
        backend_name: IBM Quantum backend name
        user_qubits: list of 3 user-selected physical qubit indices
        theta: Bloch sphere polar angle (radians)
        phi: Bloch sphere azimuthal angle (radians)
        gate_pulse_map: User's L3 per-gate pulse shape selection
            e.g. {"sx": "drag", "cz": "gaussian_square", "measure": "square"}

    Returns:
        dict with 'user', 'notuning', and optionally 'layer_fidelities'
    """
    backend_data = load_static_backend_data(backend_name)
    gate_properties = backend_data.get('gate_properties', {})
    qubit_properties = backend_data.get('qubit_properties', [])
    errors = backend_data.get('errors', [])

    notuning_qubits = get_baseline_qubits(backend_name)

    user_decomp = decompose_errors(user_qubits, gate_properties, qubit_properties, errors)
    user_decomp['qubits'] = user_qubits

    notuning_decomp = decompose_errors(notuning_qubits, gate_properties, qubit_properties, errors)
    notuning_decomp['qubits'] = notuning_qubits

    # Layer-by-layer fidelity for waterfall:
    #   baseline:  median-quality path, all-square (no tuning at any layer)
    #   after_l2:  user_qubits,         all-square (qubit selection only)
    #   after_l3:  user_qubits,         user's per-gate pulse shapes (full L3 tuning)
    result = {'user': user_decomp, 'notuning': notuning_decomp}
    try:
        notuning_f = simulate(theta, phi, backend_name, notuning_qubits, gate_pulse_map=_NO_TUNING_PULSE_MAP, shots=shots)["fidelity"]
        after_l2_f = simulate(theta, phi, backend_name, user_qubits, gate_pulse_map=_NO_TUNING_PULSE_MAP, shots=shots)["fidelity"]
        user_f = simulate(theta, phi, backend_name, user_qubits, gate_pulse_map=gate_pulse_map, shots=shots)["fidelity"]
        if None in (notuning_f, after_l2_f, user_f):
            raise ValueError("受理 0 の段がある")

        notuning_decomp['estimated_fidelity'] = round(notuning_f * 100, 2)
        user_decomp['estimated_fidelity'] = round(user_f * 100, 2)

        result['layer_fidelities'] = {
            'baseline': round(notuning_f * 100, 2),
            'after_l2': round(after_l2_f * 100, 2),
            'after_l3': round(user_f * 100, 2),
        }
    except Exception as e:
        logger.warning("density_matrix fidelity failed, using error rate estimate: %s", e)

    return result
