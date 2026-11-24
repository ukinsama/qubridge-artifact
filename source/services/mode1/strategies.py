"""Mode1 L2 の画面で見せる「選び方の比較」と「VF2 の配置の分布」。

役割: run_strategy_comparison は qubridge／random／Qiskit の 3 戦略を同じ backend・threshold・seed で回し、配置とスコアを並べる。
     compute_vf2_stats は全 VF2 の完全配置を採点し、閾値を通った中の best と全体のヒストグラムを返す。
境界: 列挙・採点・閾値の判定は services/pipeline/qubits.py が正本で、ここは呼ぶだけ。物理の「Qiskit に選ばせる」は
     services/pipeline/target.choose_layout（2026-10-20 に手組みの transpile から畳んだ。論文の物理の L2 は 2026-11-04 から別で、PAPER_BEST_PATH）。採点に渡す回路は
     paper_chain と同じ layout_circuit（core ＋ 測定。2026-10-28 から Bob の読み出し誤りも見る）。
契約: 物理の noise-aware と論理の default (Qiskit) は |+⟩ のテレポの core を Target で transpile した最初の置き場
     （seed=42 なら paper_chain と同じ。2026-10-20）。論理の qubridge（VF2 の完全部分グラフを雑音で採点）は比較用に残す。
"""
import logging

import numpy as np

from core.config import PAPER_BACKEND, PAPER_BEST_PATH
from services.apps.teleportation import LOGICAL, LOGICAL_CIRCUIT_EDGES, PHYSICAL
from services.pipeline.qubits import (
    error_maps,
    find_all_perfect_layouts,
    layouts_under_threshold,
    score_layout,
    select_qubits_qubridge,
    select_qubits_qubridge_logical,
    select_qubits_random,
)
from services.pipeline.target import PAPER_L2_STATE, choose_layout, layout_circuit
from services.static_backend import load_static_backend_data

logger = logging.getLogger(__name__)



def _try_strategy(name, color, fn, backend_name, score_edges=None):
    """戦略を 1 つ回して結果の dict にする（失敗は available=False で返し、他の戦略を止めない）"""
    try:
        qubits = fn()
        sc = score_layout(qubits, backend_name, score_edges)
        return {"strategy": name, "qubits": qubits, "score": round(sc, 6),
                "color": color, "available": True}
    except Exception as e:
        logger.debug("Strategy %s unavailable: %s", name, e)
        return {"strategy": name, "qubits": None, "score": None,
                "color": color, "available": False, "error": "Strategy unavailable"}


# =============================================================================
# Strategy comparison
# =============================================================================

def recommended_chain(backend_name: str, seed: int = 42) -> list:
    """L2 の「推奨の経路」: 論文の機体では論文の L2（core.config.PAPER_BEST_PATH。2026-11-04 から全格子で F が最高の鎖）。
    ほかの機体には全格子の F が無いので、Qiskit の transpile の配置（layout_circuit で採点。seed 42 なら paper_chain(PHYSICAL) と同じ）。
    UI の「推奨の経路を使う」ボタンが呼ぶ。Q1–Q3 に入れるだけで、確定は人が押す（2026-10-28 ケイ「ボタンでお願いします」）"""
    if backend_name == PAPER_BACKEND:
        return list(PAPER_BEST_PATH)
    return choose_layout(layout_circuit(PHYSICAL, PAPER_L2_STATE), backend_name, seed=seed)

def _default_logical_layout(backend_name: str, seed: int) -> list:
    """論理テレポの配置を Qiskit に選ばせる（論文の L2 と同じ target.choose_layout。seed 42 なら paper_chain(LOGICAL) と同じ）。
    採点に渡すのは測定込みの layout_circuit（2026-10-28）。2026-10-20 までは coupling_map と basis_gates だけの transpile（Target 無し）だった"""
    return choose_layout(layout_circuit(LOGICAL, PAPER_L2_STATE), backend_name, seed=seed)


def run_strategy_comparison(backend_name: str, threshold: float, seed: int, mode: str) -> dict:
    """Compare qubit selection strategies for a given backend.

    Args:
        backend_name: IBM Quantum backend name
        threshold: 2Q gate error threshold
        seed: random seed for reproducibility
        mode: "physical" (3Q linear) or "logical" (6Q VF2 DAG)

    Returns:
        dict with backend_name, threshold, seed, mode, strategies list
    """
    results = []

    if mode == 'physical':
        results.append(_try_strategy(
            "qubridge", "#2196F3",
            lambda: select_qubits_qubridge(backend_name, threshold, seed=seed),
            backend_name,
        ))
        results.append(_try_strategy(
            "random", "#FF9800",
            lambda: select_qubits_random(backend_name, seed=seed),
            backend_name,
        ))
        results.append(_try_strategy(
            "noise-aware (Qiskit L2)", "#9C27B0",
            lambda: recommended_chain(backend_name, seed),
            backend_name,
        ))
    else:
        # Logical mode: 6Q VF2 subgraph matching against LOGICAL_CIRCUIT_EDGES
        edges = LOGICAL_CIRCUIT_EDGES
        results.append(_try_strategy(
            "qubridge", "#2196F3",
            lambda: select_qubits_qubridge_logical(backend_name, edges, threshold, seed=seed),
            backend_name, score_edges=edges,
        ))
        results.append(_try_strategy(
            "random", "#FF9800",
            lambda: select_qubits_random(backend_name, seed=seed, n=6),
            backend_name, score_edges=edges,
        ))
        results.append(_try_strategy(
            "default (Qiskit)", "#4CAF50",
            lambda: _default_logical_layout(backend_name, seed),
            backend_name, score_edges=edges,
        ))

    return {
        "backend_name": backend_name,
        "threshold": threshold,
        "seed": seed,
        "mode": mode,
        "strategies": results,
    }


# =============================================================================
# VF2 layout statistics
# =============================================================================

def compute_vf2_stats(backend_name: str, threshold: float) -> dict:
    """Compute VF2 layout statistics for logical qubit selection.

    Args:
        backend_name: IBM Quantum backend name
        threshold: 2Q gate error threshold for filtering

    Returns:
        dict with total_layouts, filtered_layouts, scores, histogram, best_layout
    """
    circuit_edges = LOGICAL_CIRCUIT_EDGES

    all_layouts = find_all_perfect_layouts(backend_name, circuit_edges)
    if not all_layouts:
        return {
            "total_layouts": 0,
            "filtered_layouts": 0,
            "best_score": None,
            "worst_score": None,
            "median_score": None,
            "best_filtered_score": None,
            "histogram": [],
            "best_layout": None,
        }

    maps = error_maps(load_static_backend_data(backend_name))   # 較正の表は 1 回（配置は 400 本超）

    # Score all layouts and sort low-to-high (lower score = better)
    scored = sorted(
        ((score_layout(layout, backend_name, circuit_edges, maps=maps), layout) for layout in all_layouts),
        key=lambda x: x[0],
    )
    all_scores = [s for s, _ in scored]
    best_score = all_scores[0]
    worst_score = all_scores[-1]
    median_score = float(np.median(all_scores))

    # Histogram of all scores (10 bins)
    counts, bin_edges = np.histogram(all_scores, bins=10)
    histogram = [
        {
            "bin_start": round(float(bin_edges[i]), 6),
            "bin_end": round(float(bin_edges[i + 1]), 6),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    # Filter by threshold and take the best surviving layout
    filtered = layouts_under_threshold(all_layouts, circuit_edges, maps.two_q, threshold)
    if filtered:
        filtered_scored = sorted(
            ((score_layout(layout, backend_name, circuit_edges, maps=maps), layout) for layout in filtered),
            key=lambda x: x[0],
        )
        best_filtered_score = filtered_scored[0][0]
        best_layout = list(filtered_scored[0][1])
    else:
        best_filtered_score = None
        best_layout = list(scored[0][1])  # fallback to overall best

    return {
        "total_layouts": len(all_layouts),
        "filtered_layouts": len(filtered),
        "best_score": round(best_score, 6),
        "worst_score": round(worst_score, 6),
        "median_score": round(median_score, 6),
        "best_filtered_score": round(best_filtered_score, 6) if best_filtered_score is not None else None,
        "histogram": histogram,
        "best_layout": best_layout,
    }
