"""経路と配置の選択（L2 の候補を列挙し、較正の誤りで採点する）。

役割: 鎖の列挙（find_chains・baseline_paths・both_orientations）、VF2 の完全配置（find_all_perfect_layouts・score_layout）、
     UI の 3 戦略（select_qubits_qubridge・select_qubits_random・select_qubits_qubridge_logical）。
境界: 較正は services/static_backend.py を呼ぶときに読む（テストはそこを差し替える）。論文の配置は target.paper_chain
     （transpile に選ばせる）で、ここの select_* は UI の L2 の比較用。アプリの辺は呼び手が渡し、ここは特定のアプリを知らない。
存在理由: 旧 experiments/common/qubit_selection.py。2026-09-28 に pipeline へ移し、2026-10-20 に誤りの引き方を 1 つにした。
契約: find_chains の並びは E1-H・E1-G の CSV の行順を決めるので変えない。2Q 誤りは edge_error（(a,b) → (b,a) の順、
     0.0 は有効な値で、鍵が無いときだけ None）だけで引く。較正の表は error_maps で 1 回作り、採点はそれを受け取る。
判断記録: docs/decisions/2026-10-16-l2-as-transpile.md（L2 = transpile。UI の選択は触らない）
"""
import random as random_module
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rustworkx as rx
from rustworkx import vf2_mapping

from services import static_backend
from services.static_backend import load_static_backend_data


# =============================================================================
# 較正の表
# =============================================================================

def build_adjacency(coupling_map: list) -> dict:
    """カップリングマップから隣接リスト（無向グラフ）を構築する"""
    adj: dict = {}
    for edge in coupling_map:
        a, b = edge[0], edge[1]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def get_error_map(errors: list) -> dict:
    """エラーリストから (qubit_from, qubit_to) → error_rate の辞書を構築する"""
    error_map = {}
    for err in errors:
        key = (err["qubit_from"], err["qubit_to"])
        error_map[key] = err["error_rate"]
    return error_map


def edge_error(error_map: dict, a: int, b: int) -> Optional[float]:
    """辺 a–b の 2Q 誤りをどちら向きでも引く（(a,b) → (b,a) の順）。0.0 は有効な値で、鍵が無いときだけ None。

    較正は 3 機体とも全辺が両向きで同じ値を持つので、順は数字に効かない（2026-10-20 に実測）。
    `get((a,b)) or get((b,a))` の形は 0.0 を「無い」と読むので使わない。
    """
    e = error_map.get((a, b))
    return e if e is not None else error_map.get((b, a))


def single_qubit_error_map(gate_properties: dict, basis_gates: list) -> dict:
    """qubit ごとの 1Q ゲート（sx・x・id）の誤りの最大値 {qubit: max_1q_error}。いちばん悪いゲートを採点に効かせる。
    utils/coupling_map.py の画面の表示も同じ表を使う（2026-10-20 に畳んだ）"""
    single_q_gates = [g for g in ["sx", "x", "id"] if g in basis_gates]
    error_map: dict = {}
    for gate_name in single_q_gates:
        if gate_name not in gate_properties:
            continue
        for qargs_key, props in gate_properties[gate_name].items():
            # "(53,)" → 53
            qubit = int(qargs_key.strip("() ").rstrip(",").strip())
            err = props.get("error")
            if err is not None:
                error_map[qubit] = max(error_map.get(qubit, 0.0), err)
    return error_map


def _readout_error_map(qubit_properties: list) -> dict:
    """qubit ごとの読み出し誤り {qubit: readout_error}。ゲート誤りの 100〜300 倍あるので、経路の採点に入れる"""
    return {qp["qubit"]: qp["readout_error"] for qp in qubit_properties
            if qp is not None and qp.get("readout_error") is not None}


@dataclass(frozen=True)
class ErrorMaps:
    """採点に使う較正の表 3 つ。較正 1 回につき 1 回作り、配置ごとに作り直さない"""
    two_q: dict      # (a, b) → 2Q 誤り（edge_error で引く）
    one_q: dict      # qubit → 1Q 誤りの最大
    readout: dict    # qubit → 読み出し誤り


def error_maps(data: dict) -> ErrorMaps:
    """較正の dict（load_static_backend_data の返り値）から採点の表を作る"""
    return ErrorMaps(
        two_q=get_error_map(data.get("errors", [])),
        one_q=single_qubit_error_map(data.get("gate_properties", {}), data.get("basis_gates", [])),
        readout=_readout_error_map(data.get("qubit_properties", [])),
    )


# =============================================================================
# 鎖の列挙（E1-H・E1-G の Baseline）
# =============================================================================

def find_3qubit_paths(adj: dict) -> List[Tuple[int, int, int]]:
    """
    隣接リストから全ての3量子ビット連続パス (a-b-c) を列挙する。

    テレポーテーション回路は3量子ビットが線形に接続されている必要があるため、
    a-b と b-c の両方が接続されているパスを探す。
    """
    paths = []
    for b in adj:
        neighbors = sorted(adj[b])
        for i, a in enumerate(neighbors):
            for c in neighbors[i + 1:]:
                # a-b-c のパスが存在
                paths.append((a, b, c))
    return paths


def _chains_by_search(adj: dict, n: int) -> List[Tuple[int, ...]]:
    """深さ優先で n 個の単純な鎖を全部探す。向きは「最初 < 最後」の1つだけ残す。"""
    chains = []

    def extend(path):
        if len(path) == n:
            if path[0] < path[-1]:
                chains.append(tuple(path))
            return
        for q in sorted(adj[path[-1]]):
            if q not in path:
                extend(path + [q])

    for start in sorted(adj):
        extend([start])
    return chains


def find_chains(adj: dict, n: int) -> List[Tuple[int, ...]]:
    """n 個の qubit が一列につながる鎖を返す。向きは「最初 < 最後」にそろえる。

    n=3 は find_3qubit_paths をそのまま使う。E1-H・E1-G の CSV の行の並びはこの並び順で決まるので変えない。
    """
    if n < 2:
        # 1 個は鎖にならず、0 以下は探索が止まらない（長さが n に届かないまま全部の道を探す）
        raise ValueError(f"鎖の長さは 2 以上: n={n}")
    if n == 3:
        return find_3qubit_paths(adj)
    return _chains_by_search(adj, n)


def baseline_paths(backend_name: str, thresholds, n: int = 3) -> List[Tuple[Tuple[int, ...], float]]:
    """Baseline の条件を満たす鎖を、鎖の中の最大の 2Q 誤りの小さい順に返す。

    条件: 全 qubit が T1 ≥ t1_min・T2 ≥ t2_min・1Q 誤り ≤ gate1q_max・読み違いが各向き ≤ readout_misread_max、
    鎖の全辺が 2Q 誤り ≤ gate2q_max。
    読み出しの誤りの大きさは入れない（E1-G の最初の段で見せるため）。読み違いの上限は「読み出しが壊れた qubit」を外すだけ
    （逆に読むほうが多い qubit。2026-11-02）。E1-H と E1-G が共有する（2026-09-27 に畳んだ）。
    1Q の誤りは sx・x の較正、無ければ qubit_properties の gate_error_1q、それも無ければ 1.0（旧 E1-G の順。
    旧 E1-H は sx・x が無ければすぐ 1.0 だった。torino は全 qubit に sx があり数字は変わらない。2026-09-27 に畳んだ）。
    テストは static_backend.load_static_backend_data を差し替えるので、モジュール越しに呼ぶときに引く
    （名前で import した load_static_backend_data は差し替えを追わない。2026-09-28、削った関数内 import の代わり）。
    """
    data = static_backend.load_static_backend_data(backend_name)
    qp_map = {qp["qubit"]: qp for qp in data.get("qubit_properties", [])}
    gate_props = data.get("gate_properties", {})
    error_map = get_error_map(data.get("errors", []))

    def q_1q(q):
        for gate_name in ["sx", "x"]:
            props = gate_props.get(gate_name, {}).get(f"({q},)")
            if props is not None and props.get("error") is not None:
                return props["error"]
        return qp_map.get(q, {}).get("gate_error_1q", 1.0)

    def max_2q(chain):
        errs = [edge_error(error_map, a, b) for a, b in zip(chain, chain[1:])]
        return None if any(e is None for e in errs) else max(errs)   # 較正の無い辺を含む鎖は外す

    def readout_ok(q):
        # 読み違いの向きごとの確率（noise.add_readout_errors と同じ欄）。対が無ければ平均の readout_error、それも無ければ外さない
        p = qp_map.get(q, {})
        flips = [f for f in (p.get("prob_meas1_prep0"), p.get("prob_meas0_prep1")) if f is not None]
        if not flips and p.get("readout_error") is not None:
            flips = [p["readout_error"]]
        return all(f <= thresholds.readout_misread_max for f in flips)

    out = []
    for chain in find_chains(build_adjacency(data.get("coupling_map", [])), n):
        e = max_2q(chain)
        if e is None or e > thresholds.gate2q_max:
            continue
        if all(qp_map.get(q, {}).get("T1", 0.0) >= thresholds.t1_min
               and qp_map.get(q, {}).get("T2", 0.0) >= thresholds.t2_min
               and q_1q(q) <= thresholds.gate1q_max
               and readout_ok(q) for q in chain):
            out.append((chain, e))
    out.sort(key=lambda x: x[1])
    return out


def both_orientations(paths: List[Tuple[Tuple[int, ...], float]]) -> List[Tuple[Tuple[int, ...], float]]:
    """鎖の一覧に逆向きを足す（Alice と Bob をどちらの端に置くかも L2 の決め事。2026-10-16 の境目の監査 4）。
    正向き・逆向きを隣り合わせに並べ、誤りの値はそのまま。E1-H と E1-G の列挙が使う（2026-10-20、L2 = transpile の採用と一緒に）"""
    out = []
    for chain, e in paths:
        c = tuple(chain)
        out.append((c, e))
        out.append((c[::-1], e))
    return out


# =============================================================================
# UI の戦略 1: QuBridge（物理 3 qubit。閾値 → 鎖 → 3 段で絞る → seed で 1 つ）
# =============================================================================

def path_avg_error(path: Tuple[int, int, int], error_map: dict) -> Optional[float]:
    """鎖 (a, b, c) の 2Q 誤りの平均 (e_ab + e_bc) / 2。どちらかの辺に較正が無ければ None"""
    a, b, c = path
    e_ab, e_bc = edge_error(error_map, a, b), edge_error(error_map, b, c)
    if e_ab is None or e_bc is None:
        return None
    return (e_ab + e_bc) / 2


def _edges_under_threshold(coupling_map: list, error_map_2q: dict, threshold: float) -> list:
    """2Q 誤りが閾値以下の辺だけ残す（較正の無い辺は落とす）"""
    out = []
    for edge in coupling_map:
        e = edge_error(error_map_2q, edge[0], edge[1])
        if e is not None and e <= threshold:
            out.append(edge)
    return out


def _filter_top(paths: list, score_fn, keep_ratio: float = 0.5) -> list:
    """スコア（低いほど良い）の昇順に並べ、上位 keep_ratio 分（最低 1 つ）を残す"""
    scored = [(score_fn(p), p) for p in paths]
    scored.sort(key=lambda x: x[0])
    top_n = max(1, int(len(scored) * keep_ratio))
    return [p for _, p in scored[:top_n]]


def _narrow(paths: list, maps: ErrorMaps) -> list:
    """3 段で絞る: 2Q の平均で上位 30%（主軸）→ 1Q の平均で下位 20% を落とす → 読み出しの平均で下位 20% を落とす。
    2Q を主軸に厳選し、1Q・読み出しは外れ値の除外に留める（閾値 strict/loose の効果が結果に出るように）"""
    paths = _filter_top(paths, lambda p: path_avg_error(p, maps.two_q), keep_ratio=0.3)
    paths = _filter_top(paths, lambda p: sum(maps.one_q.get(q, 0.0) for q in p) / 3, keep_ratio=0.8)
    return _filter_top(paths, lambda p: sum(maps.readout.get(q, 0.0) for q in p) / 3, keep_ratio=0.8)


def _pick(candidates: list, seed: Optional[int]):
    """seed があれば乱数で 1 つ、無ければ先頭（候補はスコアの昇順に並んでいる）"""
    if seed is not None:
        return random_module.Random(seed).choice(candidates)
    return candidates[0]


def select_qubits_qubridge(backend_name: str, threshold: float = 0.01, seed: Optional[int] = None) -> List[int]:
    """QuBridge 戦略: 2Q 誤りが閾値以下の辺だけで 3 qubit の鎖を列挙し、threshold < 1.0 なら 3 段で絞って、seed で 1 つ選ぶ。

    Raises:
        ValueError: 閾値を満たす辺・鎖が無いとき
    """
    data = load_static_backend_data(backend_name)
    maps = error_maps(data)

    edges = _edges_under_threshold(data.get("coupling_map", []), maps.two_q, threshold)
    if not edges:
        raise ValueError(f"閾値 {threshold} を満たすエッジがありません（backend={backend_name}）")
    paths = find_3qubit_paths(build_adjacency(edges))
    if not paths:
        raise ValueError(f"閾値 {threshold} で3量子ビット連続パスが見つかりません（backend={backend_name}）")
    paths = [p for p in paths if path_avg_error(p, maps.two_q) is not None]   # 2Q の較正がそろう鎖だけ
    if not paths:
        raise ValueError("エラー情報が不足しています")

    if threshold < 1.0:   # strict: 3 段で絞る。loose（≥ 1.0）: 全部の鎖から選ぶ
        paths = _narrow(paths, maps)
    return list(_pick(paths, seed))


def select_qubits_random(backend_name: str, seed: Optional[int] = None, n: int = 3) -> List[int]:
    """ランダム戦略: 全 qubit から無作為に n 個（隣接を見ないので、SWAP が入る配置も出る。qubridge との比較用）"""
    n_qubits = load_static_backend_data(backend_name).get("n_qubits", 0)
    if n_qubits < n:
        raise ValueError(f"量子ビット数が不足しています: {n_qubits}（backend={backend_name}）")
    return random_module.Random(seed).sample(range(n_qubits), n)


# =============================================================================
# UI の戦略 2: VF2 の完全配置（論理 6 qubit。回路の辺に一致する部分グラフを全部列挙し、採点する）
# =============================================================================

def find_all_perfect_layouts(backend_name: str, circuit_edges: list, call_limit: int = 100000) -> list:
    """回路の辺に完全一致する（SWAP の要らない）物理 qubit の配置を VF2 で全部列挙する。

    Returns:
        list of list[int]: layout[i] = 論理 qubit i の物理番号
    """
    data = load_static_backend_data(backend_name)
    coupling_map = data.get("coupling_map", [])

    # 機体を無向グラフに（辺は 1 本ずつ）
    n_qubits = data.get("n_qubits", 0)
    backend_graph = rx.PyGraph()
    backend_graph.add_nodes_from(range(n_qubits))
    edge_set = set()
    for edge in coupling_map:
        a, b = edge[0], edge[1]
        key = (min(a, b), max(a, b))
        if key not in edge_set:
            edge_set.add(key)
            backend_graph.add_edge(a, b, None)

    # 回路の辺を無向グラフに
    n_logical = max(max(e) for e in circuit_edges) + 1
    circuit_graph = rx.PyGraph()
    circuit_graph.add_nodes_from(range(n_logical))
    for a, b in circuit_edges:
        circuit_graph.add_edge(a, b, None)

    mappings = vf2_mapping(backend_graph, circuit_graph, subgraph=True, call_limit=call_limit)

    # mapping: {backend_node: circuit_node} → layout[circuit_node] = backend_node
    layouts = []
    for mapping in mappings:
        layout = [0] * n_logical
        for backend_node, circuit_node in mapping.items():
            layout[circuit_node] = backend_node
        layouts.append(layout)
    return layouts


def _score(layout: list, maps: ErrorMaps, circuit_edges: list | None) -> float:
    """配置のノイズスコア（低いほど良い）= 2Q 誤りの平均 + 1Q 誤りの平均 + 読み出し誤りの平均。
    circuit_edges があれば回路が実際に使う辺だけ、無ければ配置の中の全隣接辺で 2Q を数える"""
    n = len(layout)
    avg_1q = sum(maps.one_q.get(q, 0.0) for q in layout) / n
    avg_ro = sum(maps.readout.get(q, 0.0) for q in layout) / n

    two_q_errors = []
    if circuit_edges is not None:
        for src, dst in circuit_edges:
            err = edge_error(maps.two_q, layout[src], layout[dst])
            if err is not None:
                two_q_errors.append(err)
    else:
        layout_set = set(layout)
        for (a, b), err in maps.two_q.items():
            if a in layout_set and b in layout_set:
                two_q_errors.append(err)
    avg_2q = sum(two_q_errors) / len(two_q_errors) if two_q_errors else 0.0
    return avg_2q + avg_1q + avg_ro


def score_layout(layout: list, backend_name: str, circuit_edges: list | None = None,
                 maps: ErrorMaps | None = None) -> float:
    """配置のノイズスコア（_score）。たくさんの配置を採点するときは maps を 1 回作って渡す（較正の表を作り直さない）"""
    if maps is None:
        maps = error_maps(load_static_backend_data(backend_name))
    return _score(layout, maps, circuit_edges)


def layouts_under_threshold(layouts: list, circuit_edges: list, error_map_2q: dict, threshold: float) -> list:
    """回路の辺に当たる 2Q 誤りが全部 threshold 以下の配置だけ残す（threshold ≥ 1.0 は素通し）。
    select_qubits_qubridge_logical と UI の compute_vf2_stats が共有する（2026-10-20 に畳んだ）"""
    if threshold >= 1.0:
        return list(layouts)
    out = []
    for layout in layouts:
        errs = [edge_error(error_map_2q, layout[src], layout[dst]) for src, dst in circuit_edges]
        if all(e is not None and e <= threshold for e in errs):
            out.append(layout)
    return out


def select_qubits_qubridge_logical(backend_name: str, circuit_edges: list, threshold: float = 0.01,
                                   seed: Optional[int] = None) -> list:
    """QuBridge 戦略（論理回路版）: VF2 の完全配置を 2Q 誤りの閾値で絞り、スコアの上位 30% から seed で 1 つ選ぶ。

    circuit_edges は回路の辺（テレポの論理なら services/apps/teleportation.LOGICAL.circuit_edges）。
    Raises:
        ValueError: 完全配置が無い、または閾値を満たす配置が無いとき
    """
    all_layouts = find_all_perfect_layouts(backend_name, circuit_edges)
    if not all_layouts:
        raise ValueError(f"パーフェクトレイアウトが見つかりません（backend={backend_name}）")

    maps = error_maps(load_static_backend_data(backend_name))
    layouts = layouts_under_threshold(all_layouts, circuit_edges, maps.two_q, threshold)
    if not layouts:
        raise ValueError(f"閾値 {threshold} を満たすレイアウトがありません（backend={backend_name}）")

    scored = sorted(((_score(layout, maps, circuit_edges), layout) for layout in layouts), key=lambda x: x[0])
    top_layouts = [layout for _, layout in scored[:max(1, int(len(scored) * 0.3))]]
    return _pick(top_layouts, seed)
