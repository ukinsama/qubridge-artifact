#!/usr/bin/env python3
"""Recompute the numbers reported in the paper from the CSV files in data/.

    python scripts/reproduce_numbers.py            # print every table and in-text number
    python scripts/reproduce_numbers.py --check    # also compare with the values printed in the paper (exit 1 on mismatch)

Only pandas is needed. Every CSV holds one row per simulator seed; the paper reports the mean over the five seeds,
rounded to three decimals for fidelities (four in Table III) and one decimal for percentage points.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

CHOSEN = (5, 6, 7)          # the chain assigned by the calibration-aware transpiler (P* in the paper)
DEFAULT = (24, 23, 16)      # the default chain from which the end-to-end trace starts
L3_SETTINGS = ("asap, no DD", "alap, no DD", "asap + DD", "alap + DD")   # sigma_0 ... sigma*
UNTUNED_L3, TUNED_L3 = L3_SETTINGS[0], L3_SETTINGS[3]
REALTIME, FRAME = "real-time feedforward", "post-processed frame"       # mu_0 and mu*
STATES = ("|+⟩", "|0⟩", "|1⟩")
APPS = ("Teleportation", "GHZ preparation", "Entanglement swapping", "CHSH")
SEEDS = 5

# The values as printed in the paper (tables, captions and running text). --check compares against these.
EXPECTED = {
    "table_pipeline_ablation": ((3072, 0.505, 47.1), (8, 0.924, 5.2), (2, 0.955, 2.1), (1, 0.976, 0.0)),
    "contributions_l2_l3_l4": (41.9, 3.1, 2.1),
    "table_l2_filter": ((384, 0.505, 41.9), (266, 0.78, 14.4), (236, 0.809, 11.5), (216, 0.809, 11.5), (164, 0.809, 11.5)),
    "l2_filter_upper_edge": 0.924,
    "l3_chosen_chain": (0.924, 0.939, 0.953, 0.955),
    "l3_chosen_chain_band": 3.1,
    "l3_decoupling_gain_under_asap_alap": (2.9, 1.6),
    "l3_alap_gain_without_with_decoupling": (1.5, 0.2),
    "l3_default_chain": (0.807, 0.821, 0.843, 0.847),
    "l3_default_chain_gain_total_alap_decoupling": (4.0, 1.4, 3.6),
    "l3_alap_plus_gain_zero_loss": (1.5, 0.8),
    "l2_l3_gain_difference": 0.9,
    "l2_l3_chains_differ_by_more_than_10_points": True,
    "l2_l3_interaction_below_1_percent": True,
    "l1_untuned": (0.924, 0.973, 0.962),
    "l1_untuned_band": 4.9,
    "l1_tuned": (0.955, 0.971, 0.961),
    "l1_tuned_band": 1.6,
    "l1_gain_from_l3_tuning": (3.1, -0.2, -0.1),
    "l4_realtime_frame_gain": (0.955, 0.976, 2.1),
    "l4_frame_without_decoupling": (0.976, 0.08),
    "l4_ideal_readout": (0.97, 1.5),
    "trace": (0.807, 0.924, 0.955, 0.976, 16.8),
    "default_chain_rank_from_bottom": (29, 384),
    "table_error_detected": (
        (0.5, 0.9681, 0.9216, 0.9606, 0.9769, 0.9821, 0.9598),
        (1.0, 0.9389, 0.8584, 0.9233, 0.9545, 0.9645, 0.9217),
        (1.5, 0.9117, 0.8041, 0.888, 0.9327, 0.9468, 0.8857),
        (2.0, 0.8858, 0.7573, 0.8546, 0.9117, 0.9291, 0.8515),
        (2.5, 0.8615, 0.7172, 0.823, 0.8912, 0.9115, 0.8195),
    ),
    "error_detected_gap_plus": (-0.046, -0.144),
    "error_detected_gap_one": (0.005, 0.02),
    "error_detected_acceptance_one_percent": (96.0, 81.9),
    "seed_spread_max": 0.0015,
    "interaction_l2_l3_max_without_chsh": 0.4,
    "interaction_l2_l3_chsh_share_and_floor": (4.8, 4.8),
    "interaction_l3_l4_max": 1.5,
    "interaction_l2_l4_range": (3.6, 4.0),
    "interaction_teleportation_l1_with_l2_l3_l4": (6.5, 4.0, 15.2),
    "interaction_higher_order_teleportation": 23,
    "interaction_higher_order_others_range": (0.6, 3.0),
    "shot_noise_total_range_without_chsh": (0.3, 1.2),
    "shot_noise_total_chsh": 9.8,
    "shot_noise_pairwise_floor_max": 0.3,
}


def _points(a: float, b: float) -> float:
    """Difference a - b in percentage points."""
    return (a - b) * 100


# ---------- progressive ablation (Table "Pipeline Band Narrowing", the trace, the L3 and L4 sections) ----------
def load_ablation(data: Path) -> pd.DataFrame:
    """One row per configuration (stage, chain, L3 setting, L4 setting): seed mean F and seed-to-seed std."""
    rows = pd.read_csv(data / "pipeline_ablation.csv")
    rows["chain"] = list(zip(rows.path_q1.astype(int), rows.path_q2.astype(int), rows.path_q3.astype(int)))
    cells = rows.groupby(["stage", "chain", "l3_config", "l4_config"], as_index=False).agg(
        F=("fidelity", "mean"), seeds=("seed", "count"), seed_std=("fidelity", "std"))
    if not (cells.seeds == SEEDS).all():
        raise ValueError("pipeline_ablation.csv: a configuration does not have five seeds")
    return cells


def pipeline_ablation(cells: pd.DataFrame) -> dict:
    """Band(S_k) = F_opt - min F over the configurations of stage k; C_L = band narrowing from fixing layer L."""
    f_opt = float(cells[cells.stage == 3].F.iloc[0])          # stage 3 is the single fully tuned configuration
    stages = cells.groupby("stage").agg(configs=("F", "size"), worst=("F", "min"))
    bands = [_points(f_opt, w) for w in stages.worst]
    table = tuple((int(n), round(w, 3), round(b, 1)) for n, w, b in zip(stages.configs, stages.worst, bands))
    contributions = (round(bands[0] - bands[1], 1), round(bands[1] - bands[2], 1), round(bands[2], 1))
    return {"table_pipeline_ablation": table, "contributions_l2_l3_l4": contributions}


def _four(cells: pd.DataFrame, stage: int, chain: tuple, l4: str) -> list[float]:
    """F at the four L3 settings, in the order of L3_SETTINGS."""
    sub = cells[(cells.stage == stage) & (cells.chain == chain) & (cells.l4_config == l4)].set_index("l3_config").F
    return [float(sub[name]) for name in L3_SETTINGS]


def l3_and_trace(cells: pd.DataFrame) -> dict:
    chosen = _four(cells, 1, CHOSEN, REALTIME)        # isolation sweep of L3 on the chosen chain, real-time correction
    default = _four(cells, 0, DEFAULT, REALTIME)      # the same four settings on the default chain (in the stage-0 grid)
    frame = _four(cells, 1, CHOSEN, FRAME)            # the chosen chain under the post-processed frame
    f_opt = float(cells[cells.stage == 3].F.iloc[0])
    untuned = cells[(cells.stage == 0) & (cells.l3_config == UNTUNED_L3) & (cells.l4_config == REALTIME)]
    ranked = untuned.sort_values("F").reset_index(drop=True)
    rank = int(ranked.index[ranked.chain == DEFAULT][0]) + 1
    return {
        "l3_chosen_chain": tuple(round(v, 3) for v in chosen),
        "l3_chosen_chain_band": round(_points(chosen[3], chosen[0]), 1),
        "l3_decoupling_gain_under_asap_alap": (round(_points(chosen[2], chosen[0]), 1), round(_points(chosen[3], chosen[1]), 1)),
        "l3_alap_gain_without_with_decoupling": (round(_points(chosen[1], chosen[0]), 1), round(_points(chosen[3], chosen[2]), 1)),
        "l3_default_chain": tuple(round(v, 3) for v in default),
        "l3_default_chain_gain_total_alap_decoupling": (
            round(_points(default[3], default[0]), 1), round(_points(default[1], default[0]), 1), round(_points(default[2], default[0]), 1)),
        "l4_realtime_frame_gain": (round(chosen[3], 3), round(f_opt, 3), round(_points(f_opt, chosen[3]), 1)),
        "l4_frame_without_decoupling": (round(frame[1], 3), round(_points(frame[1], frame[3]), 2)),
        "trace": (round(default[0], 3), round(chosen[0], 3), round(chosen[3], 3), round(f_opt, 3), round(_points(f_opt, default[0]), 1)),
        "default_chain_rank_from_bottom": (rank, len(ranked)),
    }


def l2_l3_interaction(data: Path) -> dict:
    """The two chains x four L3 settings read as shares of the variance of ln F (two-way layout, cell means)."""
    rows = pd.read_csv(data / "l3_gain_by_chain.csv")
    table = rows.groupby(["path", "l3"]).fidelity.mean().unstack("l3")[list(L3_SETTINGS)]
    gains = _points(table[TUNED_L3], table[UNTUNED_L3])
    gaps = (table.max() - table.min()) * 100                       # distance between the two chains at each setting
    log = table.apply(lambda column: column.map(math.log))
    grand = log.values.mean()
    total = ((log - grand) ** 2).values.sum()
    chains = log.shape[1] * ((log.mean(axis=1) - grand) ** 2).sum()
    settings = log.shape[0] * ((log.mean(axis=0) - grand) ** 2).sum()
    interaction = 100 * (total - chains - settings) / total
    return {
        "l2_l3_gain_difference": round(float(gains.max() - gains.min()), 1),
        "l2_l3_chains_differ_by_more_than_10_points": bool(gaps.min() > 10),
        "l2_l3_interaction_below_1_percent": bool(interaction < 1),
    }


# ---------- L2 filter cascade (Table "Cumulative L2 Filtering") ----------
def l2_filter(data: Path) -> dict:
    """Band = F of the chosen chain at the untuned setting - worst surviving configuration, per cumulative filter."""
    rows = pd.read_csv(data / "l2_filter_cascade.csv")
    rows = rows[rows.state_label.astype(str).str.strip() == "+"]        # the table and the figure are for |+>
    cells = rows.groupby(["filter_stage", "n_paths", "position"], as_index=False).agg(F=("fidelity", "mean"), seeds=("seed", "count"))
    if not (cells.seeds == SEEDS).all():
        raise ValueError("l2_filter_cascade.csv: a cell does not have five seeds")
    best = cells[cells.position == "best"].set_index("filter_stage")
    worst = cells[cells.position == "worst"].set_index("filter_stage")
    table = tuple((int(worst.n_paths[k]), round(worst.F[k], 3), round(_points(best.F[k], worst.F[k]), 1)) for k in worst.index)
    return {"table_l2_filter": table, "l2_filter_upper_edge": round(float(best.F.iloc[0]), 3)}


# ---------- L1 (three input states) ----------
def l1_states(data: Path) -> dict:
    mean = pd.read_csv(data / "l1_input_states.csv").groupby(["state", "l3"]).fidelity.mean()
    untuned = [float(mean[(s, UNTUNED_L3)]) for s in STATES]
    tuned = [float(mean[(s, TUNED_L3)]) for s in STATES]
    alap = [_points(mean[(s, "alap, no DD")], mean[(s, UNTUNED_L3)]) for s in STATES]
    return {
        "l1_untuned": tuple(round(v, 3) for v in untuned),
        "l1_untuned_band": round(_points(max(untuned), min(untuned)), 1),
        "l1_tuned": tuple(round(v, 3) for v in tuned),
        "l1_tuned_band": round(_points(max(tuned), min(tuned)), 1),
        "l1_gain_from_l3_tuning": tuple(round(_points(t, u), 1) for t, u in zip(tuned, untuned)),
        "l3_alap_plus_gain_zero_loss": (round(alap[0], 1), round(-alap[1], 1)),
    }


# ---------- L4 (readout error set to zero) ----------
def l4_ideal_readout(data: Path) -> dict:
    rows = pd.read_csv(data / "l4_correction_variants.csv").set_index("l4")
    base, ideal = rows.F["real-time correction (paper default)"], rows.F["ideal readout"]
    return {"l4_ideal_readout": (round(float(ideal), 3), round(_points(ideal, base), 1))}


# ---------- error-detected teleportation (Table "Teleportation Fidelity and Syndrome Acceptance") ----------
def error_detected(data: Path) -> dict:
    rows = pd.read_csv(data / "error_detected_teleportation.csv")
    rows["state"] = rows.state_label.str.strip("|>⟩ ")
    cells = rows.groupby(["noise_scale", "circuit_type", "state"]).agg(
        F=("fidelity", "mean"), accept=("acceptance", "mean"), seeds=("seed", "count"), seed_std=("fidelity", "std"))
    if not (cells.seeds == SEEDS).all():
        raise ValueError("error_detected_teleportation.csv: a cell does not have five seeds")
    table = []
    for scale in sorted(rows.noise_scale.unique()):
        cell = {(kind, state): cells.loc[(scale, kind, state)] for kind in ("physical", "logical") for state in ("+", "1")}
        table.append((float(scale),) + tuple(round(float(v), 4) for v in (
            cell[("physical", "+")].F, cell[("logical", "+")].F, cell[("logical", "+")].accept,
            cell[("physical", "1")].F, cell[("logical", "1")].F, cell[("logical", "1")].accept)))
    gap = {state: [float(cells.loc[(s, "logical", state)].F - cells.loc[(s, "physical", state)].F) for s in (0.5, 2.5)] for state in ("+", "1")}
    accept = [float(cells.loc[(s, "logical", "1")].accept) * 100 for s in (0.5, 2.5)]
    return {
        "table_error_detected": tuple(table),
        "error_detected_gap_plus": tuple(round(v, 3) for v in gap["+"]),
        "error_detected_gap_one": tuple(round(v, 3) for v in gap["1"]),
        "error_detected_acceptance_one_percent": tuple(round(v, 1) for v in accept),
        "_seed_std_max": float(cells.seed_std.max()),
    }


# ---------- separability of the layers (heat map and the shares quoted in the text) ----------
def interaction(data: Path) -> dict:
    share = pd.read_csv(data / "layer_interaction_shares.csv").set_index(["app", "pair"]).share_pct
    floor = pd.read_csv(data / "layer_interaction_noise_floor.csv").set_index(["app", "term"]).floor_pct
    with_l4 = APPS[:3]                                                   # CHSH has no L4 decision
    l2_l4 = [share[(app, "L2×L4")] for app in with_l4]
    higher_others = [share[(app, "higher")] for app in APPS[1:]]
    totals = [floor[(app, "total")] for app in with_l4]
    pairwise = [value for (app, term), value in floor.items() if "×" in term and (app, term) != ("CHSH", "L2×L3")]
    return {
        "interaction_l2_l3_max_without_chsh": round(max(share[(app, "L2×L3")] for app in with_l4), 1),
        "interaction_l2_l3_chsh_share_and_floor": (round(share[("CHSH", "L2×L3")], 1), round(floor[("CHSH", "L2×L3")], 1)),
        "interaction_l3_l4_max": round(max(share[(app, "L3×L4")] for app in with_l4), 1),
        "interaction_l2_l4_range": (round(min(l2_l4), 1), round(max(l2_l4), 1)),
        "interaction_teleportation_l1_with_l2_l3_l4": tuple(round(share[("Teleportation", f"L1×{layer}")], 1) for layer in ("L2", "L3", "L4")),
        "interaction_higher_order_teleportation": round(share[("Teleportation", "higher")]),
        "interaction_higher_order_others_range": (round(min(higher_others), 1), round(max(higher_others), 1)),
        "shot_noise_total_range_without_chsh": (round(min(totals), 1), round(max(totals), 1)),
        "shot_noise_total_chsh": round(floor[("CHSH", "total")], 1),
        "shot_noise_pairwise_floor_max": round(max(pairwise), 1),
    }


def _plain(value):
    """Turn numpy scalars into int / float / bool so that the report prints 4.8 rather than np.float64(4.8)."""
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    return value.item() if hasattr(value, "item") else value


def collect(data: Path = DATA) -> dict:
    """Every reported number, keyed like EXPECTED and rounded as in the paper."""
    cells = load_ablation(data)
    values = {}
    for part in (pipeline_ablation(cells), l3_and_trace(cells), l2_l3_interaction(data), l2_filter(data), l1_states(data),
                 l4_ideal_readout(data), error_detected(data), interaction(data)):
        values.update(part)
    # "seed-to-seed spread (at most 0.0015)": the largest std over the five seeds, over both seed-resolved experiments
    values["seed_spread_max"] = round(max(float(cells.seed_std.max()), values.pop("_seed_std_max")), 4)
    return {key: _plain(value) for key, value in values.items()}


def mismatches(values: dict) -> list[str]:
    """Keys whose recomputed value differs from the value printed in the paper."""
    def same(a, b) -> bool:
        if isinstance(a, tuple):
            return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        return abs(a - b) < 1e-9
    missing = sorted(set(EXPECTED) - set(values))
    return missing + [key for key, value in values.items() if key in EXPECTED and not same(value, EXPECTED[key])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA, help="directory holding the CSV files (default: data/)")
    parser.add_argument("--check", action="store_true", help="compare with the values printed in the paper; exit 1 on mismatch")
    args = parser.parse_args()
    values = collect(args.data)
    width = max(len(key) for key in values)
    for key, value in values.items():
        if isinstance(value, tuple) and value and isinstance(value[0], tuple):
            print(key)
            for row in value:
                print("    " + "  ".join(f"{cell:>8}" for cell in row))
        else:
            print(f"{key:<{width}}  {value}")
    if not args.check:
        return 0
    bad = mismatches(values)
    for key in bad:
        print(f"MISMATCH {key}: recomputed {values.get(key)}, paper {EXPECTED[key]}", file=sys.stderr)
    print(f"\n{len(EXPECTED) - len(bad)} of {len(EXPECTED)} reported values match the CSV data.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
