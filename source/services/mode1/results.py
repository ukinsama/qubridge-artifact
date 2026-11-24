"""Mode1 Results service helpers.

Keep route handlers focused on session I/O while this module owns the
state-vector fidelity decision logic for Results.
"""

import numpy as np

from core.config import ALL_SQUARE_PULSE_MAP
from services.mode1.error_decomposition import get_baseline_qubits
from services.mode1.simulation import simulate


def build_ideal_vector_setting(theta_rad: float, phi_rad: float) -> dict:
    """Return the user-facing Bloch angle setting in degrees."""
    return {
        "roX": round(float(np.degrees(theta_rad)), 2),
        "roZ": round(float(np.degrees(phi_rad)), 2),
    }


def compute_state_vector_entry(
    action: str,
    theta_rad: float,
    phi_rad: float,
    backend_name: str,
    selected_qubits: list,
    shots=None,
) -> dict:
    """Compute one Mode1 Results state-vector entry.

    Args:
        action: One of ``user_results``, ``l2_results``, or ``qiskit_results``.
        theta_rad: Bloch polar angle in radians.
        phi_rad: Bloch azimuth angle in radians.
        backend_name: Static backend name.
        selected_qubits: User-selected physical qubits for L2/L3 results.

    Returns:
        A serializable fidelity/Bloch summary.
    """
    if action == "user_results":
        qubits = selected_qubits
        gate_pulse_map = None
    elif action == "l2_results":
        qubits = selected_qubits
        gate_pulse_map = ALL_SQUARE_PULSE_MAP
    elif action == "qiskit_results":
        qubits = get_baseline_qubits(backend_name)
        gate_pulse_map = ALL_SQUARE_PULSE_MAP
    else:
        raise ValueError(f"Unsupported state-vector action: {action}")

    r = simulate(theta_rad, phi_rad, backend_name, qubits, gate_pulse_map=gate_pulse_map, shots=shots)
    return {
        "fidelity": None if r["fidelity"] is None else round(r["fidelity"], 4),
        "fidelity_sigma": None if r["sigma"] is None else round(r["sigma"], 6),
        "acceptance": round(r["acceptance"], 6),
        "shots": r["shots"],
        "seed": r["seed"],
        "roX": r["theta_deg"],
        "roZ": r["phi_deg"],
        "bloch_length": r["bloch_length"],
    }
