#!/usr/bin/env python3
"""
Record demonstrated bridge-building sequences as ranking training data.

This tool reuses bridge_twin.py for the exact same brick geometry, bases, gap,
placement legality, and COM stability checks. It does not train a model yet; it
turns a real or manually verified construction sequence into JSONL decision
cases that a ranking model can learn from.

Example:
    python demo_recorder.py --strategy aggressive --base 9 --out data/aggressive_demo.jsonl

Interactive commands:
    A L 12 effective_cantilever
    A' L 12 foundation_fill
    F R 42 rear_counterweight note text
    cands               show current legal stable candidates
    state               show current bridge state
    undo                remove last placed brick
    quit                exit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bridge_twin as twin


VERSION = 1

DEMONSTRATED_REASONS = {
    "effective_cantilever",
    "foundation_fill",
    "rear_counterweight",
    "close_bridge",
    "repair_weakness",
}

SLIM_CANDIDATE_KEYS = (
    "candidate_id",
    "brick",
    "side",
    "wx",
    "wz",
    "x0",
    "x1",
    "world_cx",
    "world_cz",
    "contact_width",
    "support_count",
    "base_contact_width",
    "brick_contact_width",
    "left_component_contact",
    "right_component_contact",
    "spans_both_components",
    "support_owner_count",
    "support_brick_count",
    "support_base_count",
    "support_efficiency",
    "is_foundation_move",
    "low_height_bonus",
    "height_penalty",
    "margin_after",
    "margin_delta",
    "reach_gain",
    "closure_gain",
    "gap_coverage_after",
    "balance_gain",
    "foundation_gain",
    "side_util_gain",
    "void_reduction",
    "trapped_void_penalty",
    "closes_bridge",
    "stable_by_model",
)


def finite_margin(value: float) -> float:
    return 999.0 if value == math.inf else float(value)


def side_for_candidate(bi: twin.BrickInst) -> str:
    return twin.side_for_x((bi.x0 + bi.x1) / 2.0)


def bridge_state_features(bridge: twin.Bridge) -> Dict[str, Any]:
    max_height = 0.0
    if bridge.bricks:
        max_height = max(b.wz + b.btype.height for b in bridge.bricks)
    return {
        "brick_count": len(bridge.bricks),
        "min_margin": finite_margin(bridge.min_margin),
        "danger": bridge.danger,
        "gap_coverage": bridge.gap_coverage,
        "remaining_gap": bridge.remaining_gap,
        "left_reach": bridge.left_reach,
        "right_reach": bridge.right_reach,
        "reach_balance": twin.reach_balance(bridge),
        "left_foundation_coverage": twin.foundation_coverage_for_side(bridge, "L"),
        "right_foundation_coverage": twin.foundation_coverage_for_side(bridge, "R"),
        "left_base_utilization": twin.base_cell_utilization_for_side(bridge, "L"),
        "right_base_utilization": twin.base_cell_utilization_for_side(bridge, "R"),
        "left_void_score": twin.base_trapped_void_score_for_side(bridge, "L"),
        "right_void_score": twin.base_trapped_void_score_for_side(bridge, "R"),
        "max_height": max_height,
        "no_progress_streak": twin.no_progress_streak(bridge),
        "bridge_closed": bridge.bridge_closed,
        "bridge_succeeded": bridge.bridge_succeeded,
        "brick_exhausted": bridge.brick_exhausted,
    }


def contact_payload(contacts: Sequence[twin.SurfaceSeg]) -> List[Dict[str, Any]]:
    return [
        {
            "x0": c.x0,
            "x1": c.x1,
            "z": c.z,
            "owner": c.owner,
            "width": max(0.0, c.x1 - c.x0),
        }
        for c in contacts
    ]


def brick_payload(bridge: twin.Bridge) -> List[Dict[str, Any]]:
    return [
        {
            "id": b.id,
            "brick": b.btype.key,
            "base_brick": b.btype.base_key,
            "flipped": b.btype.flipped,
            "wx": b.wx,
            "wz": b.wz,
            "x0": b.x0,
            "x1": b.x1,
            "world_cx": b.world_cx,
            "world_cz": b.world_cz,
            "actor": b.actor,
            "margin": finite_margin(b.margin),
            "contacts": contact_payload(b.contacts),
        }
        for b in bridge.bricks
    ]


def contact_graph_payload(bridge: twin.Bridge) -> Dict[str, Any]:
    nodes = [
        {"id": twin.BASE_LEFT, "kind": "base", "label": "left_base"},
        {"id": twin.BASE_RIGHT, "kind": "base", "label": "right_base"},
    ]
    nodes.extend(
        {
            "id": b.id,
            "kind": "brick",
            "brick": b.btype.key,
            "base_brick": b.btype.base_key,
            "flipped": b.btype.flipped,
            "wx": b.wx,
            "wz": b.wz,
        }
        for b in bridge.bricks
    )
    edges: List[Dict[str, Any]] = []
    for b in bridge.bricks:
        by_owner: Dict[int, float] = {}
        for c in b.contacts:
            by_owner[c.owner] = by_owner.get(c.owner, 0.0) + max(0.0, c.x1 - c.x0)
        for owner, width in sorted(by_owner.items()):
            edges.append({"upper": b.id, "lower": owner, "contact_width": width})
    return {"nodes": nodes, "edges": edges}


def candidate_contact_features(before: twin.Bridge, option: twin.BrickInst) -> Dict[str, Any]:
    owners = {c.owner for c in option.contacts}
    left_component = before.component_from(twin.BASE_LEFT)
    right_component = before.component_from(twin.BASE_RIGHT)
    base_width = sum(max(0.0, c.x1 - c.x0) for c in option.contacts if c.owner < 0)
    brick_width = sum(max(0.0, c.x1 - c.x0) for c in option.contacts if c.owner > 0)
    return {
        "support_count": len(option.contacts),
        "support_owner_count": len(owners),
        "support_brick_count": sum(1 for owner in owners if owner > 0),
        "support_base_count": sum(1 for owner in owners if owner < 0),
        "base_contact_width": base_width,
        "brick_contact_width": brick_width,
        "left_component_contact": bool(owners & left_component),
        "right_component_contact": bool(owners & right_component),
        "spans_both_components": bool(owners & left_component) and bool(owners & right_component),
    }


def candidate_features(before: twin.Bridge, option: twin.BrickInst) -> Dict[str, Any]:
    side = side_for_candidate(option)
    sim = before.clone()
    bi = twin.BrickInst(
        sim.next_id,
        option.btype,
        option.wx,
        option.wz,
        "demo",
        list(option.contacts),
    )
    sim.bricks.append(bi)
    sim.next_id += 1
    sim.analyse()
    closes_bridge = sim.is_closing_brick(bi.id)
    stable_by_model = closes_bridge or sim.min_margin >= twin.MIN_ACCEPT_MARGIN

    base_margin = before.min_margin if before.joints else 1.0
    side_util_before = twin.base_cell_utilization_for_side(before, side)
    side_util_after = twin.base_cell_utilization_for_side(sim, side)
    side_void_before = twin.base_trapped_void_score_for_side(before, side)
    side_void_after = twin.base_trapped_void_score_for_side(sim, side)
    foundation_before = twin.foundation_average_coverage(before)
    foundation_after = twin.foundation_average_coverage(sim)
    remaining_before = before.remaining_gap

    return {
        "brick": option.btype.key,
        "side": side,
        "wx": option.wx,
        "wz": option.wz,
        "x0": option.x0,
        "x1": option.x1,
        "world_cx": option.world_cx,
        "world_cz": option.world_cz,
        "contact_width": twin.contact_width(option.contacts),
        **candidate_contact_features(before, option),
        "support_efficiency": twin.foundation_support_efficiency(option),
        "is_foundation_move": twin.is_foundation_move(option),
        "low_height_bonus": twin.low_height_bonus(option),
        "height_penalty": twin.premature_height_penalty(before, option),
        "margin_after": finite_margin(sim.min_margin),
        "margin_delta": finite_margin(sim.min_margin) - finite_margin(base_margin),
        "reach_gain": twin.side_reach_gain(before, sim, side),
        "closure_gain": max(0.0, remaining_before - sim.remaining_gap),
        "gap_coverage_after": sim.gap_coverage,
        "balance_gain": twin.reach_balance(before) - twin.reach_balance(sim),
        "foundation_gain": max(0.0, foundation_after - foundation_before),
        "side_util_gain": max(0.0, side_util_after - side_util_before),
        "void_reduction": max(0.0, side_void_before - side_void_after),
        "trapped_void_penalty": max(0.0, side_void_after - side_void_before),
        "closes_bridge": closes_bridge,
        "stable_by_model": stable_by_model,
        "model_stability_threshold": twin.MIN_ACCEPT_MARGIN,
        "after_bridge_closed": sim.bridge_closed,
        "after_bridge_succeeded": sim.bridge_succeeded,
        "contacts": contact_payload(option.contacts),
    }


def generate_candidates(
    bridge: twin.Bridge,
    stable_only: bool = True,
    complete_edges: bool = True,
    max_options: int = 64,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, float, float]] = set()
    for side in ("L", "R"):
        for bt in twin.BTYPES.values():
            for wx in twin.candidate_x_positions(bridge, bt, side):
                for option in bridge.try_place_options(
                    bt,
                    wx,
                    "demo",
                    max_options=max_options,
                    complete_edges=complete_edges,
                ):
                    key = (bt.key, round(option.wx, 6), round(option.wz, 6))
                    if key in seen:
                        continue
                    features = candidate_features(bridge, option)
                    if stable_only and not features["stable_by_model"]:
                        continue
                    seen.add(key)
                    rows.append(features)
    rows.sort(
        key=lambda r: (
            r["side"],
            r["brick"],
            round(r["wx"], 6),
            -r["wz"],
        )
    )
    for idx, row in enumerate(rows):
        row["candidate_id"] = idx
    return rows


def generate_stable_candidates(bridge: twin.Bridge) -> List[Dict[str, Any]]:
    return generate_candidates(bridge, stable_only=True)


def match_candidate(
    candidates: Sequence[Dict[str, Any]],
    brick: str,
    side: Optional[str],
    wx: float,
    wz: Optional[float],
) -> Optional[int]:
    matches = [
        i
        for i, c in enumerate(candidates)
        if c["brick"] == brick
        and abs(c["wx"] - wx) <= 1e-5
        and (side is None or c["side"] == side)
        and (wz is None or abs(c["wz"] - wz) <= 1e-5)
    ]
    if not matches:
        return None
    if wz is None and len(matches) > 1:
        matches.sort(key=lambda i: -candidates[i]["wz"])
    return matches[0]


def place_from_candidate(bridge: twin.Bridge, candidate: Dict[str, Any]) -> twin.BrickInst:
    placed = bridge.place(
        twin.BTYPES[candidate["brick"]],
        candidate["wx"],
        "demo",
        candidate["wz"],
    )
    if not placed:
        raise RuntimeError("Candidate was generated but could not be placed.")
    return placed


def write_case(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    slim_path = slim_path_for(out_path)
    with slim_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(slim_case(payload), ensure_ascii=False) + "\n")


def success_reason(bridge: twin.Bridge) -> str:
    if bridge.bridge_succeeded:
        return "closing_brick_connected_both_sides"
    if bridge.brick_exhausted:
        return "brick_exhausted"
    return "in_progress"


def slim_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: candidate[key] for key in SLIM_CANDIDATE_KEYS if key in candidate}


def slim_case(case: Dict[str, Any]) -> Dict[str, Any]:
    if case.get("type") == "candidate_annotation":
        return {
            "type": "candidate_annotation_slim",
            "version": case["version"],
            "episode_id": case.get("episode_id"),
            "strategy": case["strategy"],
            "base_span": case["base_span"],
            "step_index": case["step_index"],
            "annotation": case["annotation"],
            "reason": case.get("reason", "hard_negative_unspecified"),
            "before_state": case["before_state"],
            "candidate": slim_candidate(case["candidate"]),
        }
    return {
        "type": "decision_case_slim",
        "version": case["version"],
        "episode_id": case.get("episode_id"),
        "strategy": case["strategy"],
        "base_span": case["base_span"],
        "step_index": case["step_index"],
        "label": case["label"],
        "reason": case.get("reason", "unspecified"),
        "before_state": case["before_state"],
        "chosen_index": case["chosen_index"],
        "chosen": slim_candidate(case["chosen"]),
        "candidates": [slim_candidate(c) for c in case["candidates"]],
        "episode_done": case["episode_done"],
        "success_reason": case.get("success_reason", "in_progress"),
    }


def slim_path_for(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_slim" + out_path.suffix)


def save_slim_cases(out_path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    slim_path = slim_path_for(out_path)
    slim_path.parent.mkdir(parents=True, exist_ok=True)
    with slim_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(slim_case(case), ensure_ascii=False) + "\n")


def print_state(bridge: twin.Bridge) -> None:
    st = bridge_state_features(bridge)
    print(
        "state:",
        f"bricks={st['brick_count']}",
        f"margin={st['min_margin']:.2f}",
        f"gap={st['gap_coverage']:.2f}",
        f"L={st['left_reach']:.1f}",
        f"R={st['right_reach']:.1f}",
        f"closed={st['bridge_closed']}",
    )


def print_candidates(candidates: Sequence[Dict[str, Any]], limit: int = 20) -> None:
    if not candidates:
        print("No stable candidates.")
        return
    for c in candidates[:limit]:
        print(
            f"{c['candidate_id']:>3}. {c['side']} {c['brick']} "
            f"wx={c['wx']:.1f} wz={c['wz']:.3f} "
            f"margin={c['margin_after']:.2f} reach=+{c['reach_gain']:.1f} "
            f"close={c['closes_bridge']}"
        )
    if len(candidates) > limit:
        print(f"... {len(candidates) - limit} more")


def parse_move(parts: Sequence[str]) -> Tuple[str, Optional[str], float, Optional[float], str, str]:
    if not parts:
        raise ValueError("empty command")
    brick = parts[0].upper()
    if brick not in twin.BTYPES:
        raise ValueError("first token must be a brick key A-F, optionally with ' for flipped")

    idx = 1
    side: Optional[str] = None
    if idx < len(parts) and parts[idx].upper() in {"L", "R"}:
        side = parts[idx].upper()
        idx += 1
    if idx >= len(parts):
        raise ValueError("missing wx")
    wx = twin.snap_place_x(float(parts[idx]))
    idx += 1

    wz: Optional[float] = None
    if idx < len(parts):
        try:
            wz = float(parts[idx])
            idx += 1
        except ValueError:
            wz = None

    reason = ""
    note = ""
    if idx < len(parts):
        reason = parts[idx]
        idx += 1
    if reason not in DEMONSTRATED_REASONS:
        raise ValueError(
            "missing/invalid reason; use one of: "
            + ", ".join(sorted(DEMONSTRATED_REASONS))
        )
    if idx < len(parts):
        note = " ".join(parts[idx:])
    return brick, side, wx, wz, reason, note


def interactive(args: argparse.Namespace) -> None:
    twin.LAYOUT.base_span = args.base
    bridge = twin.Bridge()
    history: List[twin.BrickInst] = []
    out_path = Path(args.out)

    print("Aggressive demo recorder")
    print(f"base={args.base}, strategy={args.strategy}, out={out_path}")
    print("Enter moves like: A L 12 effective_cantilever")
    print("Commands: cands, state, undo, quit")

    while True:
        raw = input(f"step {len(history) + 1}> ").strip()
        if not raw:
            continue
        cmd = raw.lower()
        if cmd in {"q", "quit", "exit"}:
            break
        if cmd == "state":
            print_state(bridge)
            continue
        if cmd.startswith("cands"):
            candidates = generate_stable_candidates(bridge)
            print_candidates(candidates)
            continue
        if cmd == "undo":
            removed = bridge.undo_last()
            if removed and history:
                history.pop()
            print(f"undo: {removed.btype.key if removed else 'nothing'}")
            continue

        try:
            brick, side, wx, wz, reason, note = parse_move(raw.split())
        except Exception as exc:
            print(f"Invalid command: {exc}")
            continue

        before_state = bridge_state_features(bridge)
        before_bricks = brick_payload(bridge)
        before_contact_graph = contact_graph_payload(bridge)
        candidates = generate_stable_candidates(bridge)
        chosen_index = match_candidate(candidates, brick, side, wx, wz)
        if chosen_index is None:
            print("This move is not in the current stable legal candidate set.")
            print("Nearest candidates with same brick/side:")
            near = [
                c for c in candidates
                if c["brick"] == brick and (side is None or c["side"] == side)
            ]
            print_candidates(sorted(near, key=lambda c: abs(c["wx"] - wx))[:10], limit=10)
            continue

        chosen = candidates[chosen_index]
        placed = place_from_candidate(bridge, chosen)
        history.append(placed)
        after_state = bridge_state_features(bridge)

        payload = {
            "type": "decision_case",
            "version": VERSION,
            "episode_id": args.episode_id,
            "source": args.source,
            "strategy": args.strategy,
            "base_span": args.base,
            "gap_mm": twin.GAP_MM,
            "cell_mm": twin.CELL_MM,
            "step_index": len(history) - 1,
            "raw_input": raw,
            "label": "demonstrated",
            "reason": reason,
            "note": note,
            "before_state": before_state,
            "before_bricks": before_bricks,
            "before_contact_graph": before_contact_graph,
            "candidates": candidates,
            "chosen_index": chosen_index,
            "chosen": chosen,
            "after_state": after_state,
            "after_bricks": brick_payload(bridge),
            "contact_graph": contact_graph_payload(bridge),
            "episode_done": bridge.bridge_succeeded or bridge.brick_exhausted,
            "success_reason": success_reason(bridge),
        }
        write_case(out_path, payload)
        print(
            f"recorded {placed.btype.key} {chosen['side']} "
            f"wx={placed.wx:.1f} wz={placed.wz:.3f} "
            f"margin={after_state['min_margin']:.2f} "
            f"candidates={len(candidates)} -> {out_path}"
        )
        if bridge.bridge_succeeded:
            print("Bridge closed successfully. You can quit or continue recording.")
        elif bridge.brick_exhausted:
            print("Brick limit reached without closure.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record demonstrated bridge decisions as JSONL.")
    parser.add_argument("--strategy", default="aggressive", choices=("aggressive", "conservative"))
    parser.add_argument("--base", type=int, default=9, choices=twin.BASE_PRESETS)
    parser.add_argument("--out", default="data/aggressive_demo.jsonl")
    parser.add_argument("--source", default="manual_real_bridge")
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args()
    if args.episode_id is None:
        args.episode_id = Path(args.out).stem
    interactive(args)


if __name__ == "__main__":
    main()
