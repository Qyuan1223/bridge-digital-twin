#!/usr/bin/env python3
"""Feature extraction for bridge candidate ranking.

The functions here are deliberately dependency-free. They operate on the JSONL
records produced by visual_demo_recorder.py / demo_recorder.py and return plain
Python dictionaries suitable for a simple ranker or for export to another ML
toolchain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BRICK_KEYS = ("A", "A'", "B", "B'", "C", "C'", "D", "E", "E'", "F", "F'")
DEMO_REASONS = (
    "effective_cantilever",
    "foundation_fill",
    "rear_counterweight",
    "close_bridge",
    "repair_weakness",
)
BAD_REASONS = (
    "wastes_cantilever_space",
    "ineffective_vertical_stack",
    "weak_connection",
    "blocks_future_placement",
)


STATE_NUMERIC = (
    "brick_count",
    "min_margin",
    "danger",
    "gap_coverage",
    "remaining_gap",
    "left_reach",
    "right_reach",
    "reach_balance",
    "left_foundation_coverage",
    "right_foundation_coverage",
    "left_base_utilization",
    "right_base_utilization",
    "left_void_score",
    "right_void_score",
    "max_height",
    "no_progress_streak",
)


CANDIDATE_NUMERIC = (
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
    "support_owner_count",
    "support_brick_count",
    "support_base_count",
    "support_efficiency",
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
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_many(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def scaled(name: str, value: Any) -> float:
    v = finite(value)
    if name in {"min_margin", "margin_after", "margin_delta", "balance_gain"}:
        return clamp(v, -6.0, 6.0) / 6.0
    if name in {"remaining_gap", "left_reach", "right_reach", "reach_balance", "wx", "x0", "x1", "world_cx"}:
        return v / 60.0
    if name in {"wz", "world_cz", "max_height", "height_penalty"}:
        return v / 14.0
    if name in {"brick_count", "no_progress_streak", "support_count", "support_owner_count", "support_brick_count", "support_base_count"}:
        return v / 20.0
    if name in {"contact_width", "base_contact_width", "brick_contact_width", "reach_gain", "closure_gain"}:
        return v / 12.0
    return v


def one_hot(features: Dict[str, float], prefix: str, value: str, options: Iterable[str]) -> None:
    for option in options:
        features[f"{prefix}_{option}"] = 1.0 if value == option else 0.0


def brick_map(case: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(b["id"]): b for b in case.get("before_bricks", []) if "id" in b}


def candidate_support_owner_ids(candidate: Dict[str, Any]) -> List[int]:
    owners: List[int] = []
    seen = set()
    for c in candidate.get("contacts", []):
        owner = int(c.get("owner", 0))
        if owner not in seen:
            owners.append(owner)
            seen.add(owner)
    return owners


def candidate_contact_widths_by_owner(candidate: Dict[str, Any]) -> Dict[int, float]:
    widths: Dict[int, float] = {}
    for c in candidate.get("contacts", []):
        owner = int(c.get("owner", 0))
        width = max(0.0, finite(c.get("width", finite(c.get("x1")) - finite(c.get("x0")))))
        widths[owner] = widths.get(owner, 0.0) + width
    return widths


def support_is_overhanging(owner_brick: Dict[str, Any], state: Dict[str, Any]) -> bool:
    # A support brick is treated as "overhanging" when it extends into the gap
    # beyond the current base front. This is intentionally approximate: the
    # physical legality remains handled by the digital twin.
    x0 = finite(owner_brick.get("x0"))
    x1 = finite(owner_brick.get("x1"))
    cx = (x0 + x1) / 2.0
    left_reach = finite(state.get("left_reach"))
    right_reach = finite(state.get("right_reach"))
    # For left-side pieces, x1 extension contributes to left_reach. For
    # right-side pieces, x0 extension contributes to right_reach. We do not know
    # the exact base fronts here, so use nonzero reach plus contact to brick as
    # a conservative signal.
    return (cx < 30.0 and left_reach > 0.25) or (cx >= 30.0 and right_reach > 0.25)


def owner_brick_centers(case: Dict[str, Any], owner_ids: Sequence[int]) -> List[float]:
    bmap = brick_map(case)
    centers: List[float] = []
    for owner in owner_ids:
        b = bmap.get(owner)
        if not b:
            continue
        centers.append((finite(b.get("x0")) + finite(b.get("x1"))) / 2.0)
    return centers


def lock_span_quality(case: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """How clearly the candidate bridges two lower bricks.

    A real locking move should have contact to at least two brick owners, enough
    brick-contact width, and those owners should not be nearly coincident in x.
    """
    by_owner = candidate_contact_widths_by_owner(candidate)
    brick_owners = [owner for owner, width in by_owner.items() if owner > 0 and width >= 0.20]
    if len(brick_owners) < 2:
        return 0.0
    centers = owner_brick_centers(case, brick_owners)
    if len(centers) < 2:
        return 0.0
    owner_span = max(centers) - min(centers)
    brick_contact = sum(width for owner, width in by_owner.items() if owner > 0)
    span_score = clamp(owner_span / 7.0, 0.0, 1.0)
    contact_score = clamp(brick_contact / 6.0, 0.0, 1.0)
    return span_score * contact_score


def center_foundation_island_penalty(case: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Penalty for low single-brick center placement that wastes base space."""
    state = case.get("before_state", {})
    if not candidate.get("is_foundation_move"):
        return 0.0
    if finite(candidate.get("support_brick_count")) > 0.0:
        return 0.0
    if finite(candidate.get("support_base_count")) != 1.0:
        return 0.0
    reach_gain = max(0.0, finite(candidate.get("reach_gain")))
    closure_gain = max(0.0, finite(candidate.get("closure_gain")))
    if reach_gain > 0.25 or closure_gain > 0.25:
        return 0.0

    side = str(candidate.get("side", ""))
    util = finite(state.get("left_base_utilization" if side == "L" else "right_base_utilization"))
    if util >= 0.45:
        return 0.0

    x0 = finite(candidate.get("x0"))
    x1 = finite(candidate.get("x1"))
    cx = finite(candidate.get("world_cx"), (x0 + x1) / 2.0)
    # In the normalized 9-span layout, the side center can be approximated by
    # the midpoint between the side's current low structure and candidate span.
    same_side = [
        b for b in case.get("before_bricks", [])
        if (side == "L" and (finite(b.get("x0")) + finite(b.get("x1"))) / 2.0 < 30.0)
        or (side == "R" and (finite(b.get("x0")) + finite(b.get("x1"))) / 2.0 >= 30.0)
    ]
    if same_side:
        sx0 = min([x0] + [finite(b.get("x0")) for b in same_side])
        sx1 = max([x1] + [finite(b.get("x1")) for b in same_side])
    else:
        # Empty-side opening moves: center-ish placements are usually poor
        # because they split the available base into two unusable fragments.
        sx0, sx1 = x0 - 4.5, x1 + 4.5
    center = (sx0 + sx1) / 2.0
    half = max(1e-6, (sx1 - sx0) / 2.0)
    centered = 1.0 - clamp(abs(cx - center) / half, 0.0, 1.0)
    width_pressure = clamp((x1 - x0) / 9.0, 0.0, 1.0)
    return centered * width_pressure * clamp(1.0 - util / 0.45, 0.0, 1.0)


def history_features(context: Optional[Dict[str, Any]]) -> Dict[str, float]:
    context = context or {}
    prev_reason = context.get("previous_reason", "")
    return {
        "prev_effective_cantilever": 1.0 if prev_reason == "effective_cantilever" else 0.0,
        "prev_repair_weakness": 1.0 if prev_reason == "repair_weakness" else 0.0,
        "prev_foundation_fill": 1.0 if prev_reason == "foundation_fill" else 0.0,
        "prev_rear_counterweight": 1.0 if prev_reason == "rear_counterweight" else 0.0,
        "prev_close_bridge": 1.0 if prev_reason == "close_bridge" else 0.0,
    }


def downhill_stair_penalty(case: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Approximate whether the candidate worsens a descending frontier.

    We look at brick centers on the candidate side, include the candidate, sort
    from the base toward the gap, and measure how often height drops while
    moving outward. A strong descending stair is undesirable for the user's
    aggressive strategy because it tends to waste future interlock space.
    """
    state = case.get("before_state", {})
    side = candidate.get("side", "L")
    points: List[Tuple[float, float]] = []
    for b in case.get("before_bricks", []):
        x = (finite(b.get("x0")) + finite(b.get("x1"))) / 2.0
        z = finite(b.get("wz")) + 0.5
        if (side == "L" and x < 30.0) or (side == "R" and x >= 30.0):
            points.append((x, z))
    points.append((finite(candidate.get("world_cx")), finite(candidate.get("wz")) + 0.5))
    if len(points) < 3:
        return 0.0
    points.sort(key=lambda p: p[0], reverse=(side == "R"))
    drops = 0
    total = 0
    drop_amount = 0.0
    for (_x0, z0), (_x1, z1) in zip(points, points[1:]):
        dz = z1 - z0
        total += 1
        if dz < -0.15:
            drops += 1
            drop_amount += -dz
    if total <= 0:
        return 0.0
    return clamp((drops / total) * 0.6 + drop_amount / 10.0, 0.0, 1.0)


def aggressive_rule_components(
    case: Dict[str, Any],
    candidate: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Human-readable bridge-building priors for the aggressive strategy.

    These are not legality checks. They summarize the design knowledge the
    demonstrations are meant to teach: fill useful low support space, make real
    cantilever progress, interlock with more than one support when possible, and
    avoid building vertical towers or descending frontier staircases.
    """
    state = case.get("before_state", {})
    context = context or {}
    previous_reason = str(context.get("previous_reason", ""))

    reach_gain = max(0.0, finite(candidate.get("reach_gain")))
    closure_gain = max(0.0, finite(candidate.get("closure_gain")))
    margin_after = finite(candidate.get("margin_after"))
    wz = finite(candidate.get("wz"))
    contact_width = max(0.0, finite(candidate.get("contact_width")))
    brick_contact = max(0.0, finite(candidate.get("brick_contact_width")))
    base_contact = max(0.0, finite(candidate.get("base_contact_width")))
    support_owner_count = max(0.0, finite(candidate.get("support_owner_count")))
    no_progress_streak = max(0.0, finite(state.get("no_progress_streak")))
    base_util = (
        finite(state.get("left_base_utilization")) + finite(state.get("right_base_utilization"))
    ) / 2.0
    foundation_pressure = clamp(1.0 - base_util, 0.0, 1.0)

    support_bricks = [
        owner for owner in candidate_support_owner_ids(candidate)
        if owner > 0
    ]
    bmap = brick_map(case)
    locks_two_supports = 1.0 if len(support_bricks) >= 2 else 0.0
    locks_overhang = 1.0 if any(
        owner in bmap and support_is_overhanging(bmap[owner], state)
        for owner in support_bricks
    ) else 0.0
    lock_span = lock_span_quality(case, candidate)
    center_island = center_foundation_island_penalty(case, candidate)

    is_foundation = 1.0 if candidate.get("is_foundation_move") else 0.0
    closes_bridge = 1.0 if candidate.get("closes_bridge") else 0.0
    stable_bonus = 1.0 if candidate.get("stable_by_model") else -0.35
    margin_bonus = clamp(margin_after, -2.0, 4.0) / 4.0
    downhill = downhill_stair_penalty(case, candidate)

    low_foundation_fill = foundation_pressure * is_foundation
    high_before_foundation = foundation_pressure * max(0.0, (wz - 3.2) / 8.0) * (1.0 - min(1.0, reach_gain / 2.0))
    vertical_stack = 1.0 if reach_gain <= 0.25 and closure_gain <= 0.25 and wz > 4.0 else 0.0
    progress_pressure = clamp(no_progress_streak / 3.0, 0.0, 1.5)
    progress_after_stall = progress_pressure * min(1.0, reach_gain / 3.0)
    stall_penalty = progress_pressure * (1.0 if reach_gain <= 0.25 and not closes_bridge else 0.0)

    reinforce_after_cantilever = (
        1.0 if previous_reason == "effective_cantilever" and brick_contact > 0.25 else 0.0
    )
    cantilever_after_reinforce = (
        1.0 if previous_reason == "repair_weakness" and reach_gain > 0.25 else 0.0
    )

    contact_quality = clamp(contact_width / 9.0, 0.0, 1.5)
    brick_lock_quality = clamp(brick_contact / 8.0, 0.0, 1.5)
    multi_support_quality = clamp(support_owner_count / 3.0, 0.0, 1.5)

    score = (
        1.10 * stable_bonus
        + 0.90 * margin_bonus
        + 2.80 * min(1.0, reach_gain / 4.0)
        + 2.20 * min(1.0, closure_gain / 6.0)
        + 1.20 * low_foundation_fill
        + 1.10 * locks_two_supports
        + 1.00 * locks_overhang
        + 1.60 * lock_span
        + 0.75 * brick_lock_quality
        + 0.55 * contact_quality
        + 0.45 * multi_support_quality
        + 0.80 * reinforce_after_cantilever
        + 0.90 * cantilever_after_reinforce
        + 1.10 * progress_after_stall
        + 2.50 * closes_bridge
        - 1.20 * high_before_foundation
        - 3.80 * center_island
        - 1.20 * vertical_stack
        - 1.80 * downhill
        - 1.00 * stall_penalty
    )

    return {
        "rule_aggressive_score": score / 8.0,
        "rule_stable_bonus": stable_bonus,
        "rule_margin_bonus": margin_bonus,
        "rule_reach_gain": min(1.0, reach_gain / 4.0),
        "rule_closure_gain": min(1.0, closure_gain / 6.0),
        "rule_low_foundation_fill": low_foundation_fill,
        "rule_locks_two_supports": locks_two_supports,
        "rule_locks_overhang": locks_overhang,
        "rule_lock_span_quality": lock_span,
        "rule_center_foundation_island_penalty": center_island,
        "rule_brick_lock_quality": brick_lock_quality,
        "rule_contact_quality": contact_quality,
        "rule_multi_support_quality": multi_support_quality,
        "rule_reinforce_after_cantilever": reinforce_after_cantilever,
        "rule_cantilever_after_reinforce": cantilever_after_reinforce,
        "rule_progress_after_stall": progress_after_stall,
        "rule_high_before_foundation": high_before_foundation,
        "rule_vertical_stack": vertical_stack,
        "rule_downhill_stair_penalty": downhill,
        "rule_stall_penalty": stall_penalty,
        "rule_base_contact": clamp(base_contact / 9.0, 0.0, 1.5),
    }


def conservative_rule_components(
    case: Dict[str, Any],
    candidate: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Human-readable priors for the conservative strategy.

    Conservative does not mean "never cantilever". It means using base space,
    preserving stability margin, repairing weak structures, and only then
    making periodic progress toward closure.
    """
    state = case.get("before_state", {})
    context = context or {}
    previous_reason = str(context.get("previous_reason", ""))

    reach_gain = max(0.0, finite(candidate.get("reach_gain")))
    closure_gain = max(0.0, finite(candidate.get("closure_gain")))
    margin_after = finite(candidate.get("margin_after"))
    margin_delta = finite(candidate.get("margin_delta"))
    wz = finite(candidate.get("wz"))
    contact_width = max(0.0, finite(candidate.get("contact_width")))
    brick_contact = max(0.0, finite(candidate.get("brick_contact_width")))
    base_contact = max(0.0, finite(candidate.get("base_contact_width")))
    support_owner_count = max(0.0, finite(candidate.get("support_owner_count")))
    no_progress_streak = max(0.0, finite(state.get("no_progress_streak")))

    side = str(candidate.get("side", ""))
    side_util = finite(state.get("left_base_utilization" if side == "L" else "right_base_utilization"))
    foundation_pressure = clamp(1.0 - side_util, 0.0, 1.0)
    low_foundation_fill = foundation_pressure * (1.0 if candidate.get("is_foundation_move") else 0.0)
    foundation_gain = max(0.0, finite(candidate.get("foundation_gain")))
    side_util_gain = max(0.0, finite(candidate.get("side_util_gain")))
    void_reduction = max(0.0, finite(candidate.get("void_reduction")))
    trapped_void = max(0.0, finite(candidate.get("trapped_void_penalty")))
    height_penalty = max(0.0, finite(candidate.get("height_penalty")))

    support_bricks = [
        owner for owner in candidate_support_owner_ids(candidate)
        if owner > 0
    ]
    bmap = brick_map(case)
    locks_two_supports = 1.0 if len(support_bricks) >= 2 else 0.0
    locks_overhang = 1.0 if any(
        owner in bmap and support_is_overhanging(bmap[owner], state)
        for owner in support_bricks
    ) else 0.0
    lock_span = lock_span_quality(case, candidate)
    center_island = center_foundation_island_penalty(case, candidate)

    closes_bridge = 1.0 if candidate.get("closes_bridge") else 0.0
    stable_bonus = 1.0 if candidate.get("stable_by_model") else -1.0
    margin_bonus = clamp(margin_after, -1.0, 5.0) / 5.0
    margin_delta_bonus = clamp(margin_delta, -3.0, 3.0) / 3.0
    contact_quality = clamp(contact_width / 9.0, 0.0, 1.5)
    brick_lock_quality = clamp(brick_contact / 8.0, 0.0, 1.5)
    base_contact_quality = clamp(base_contact / 9.0, 0.0, 1.5)
    multi_support_quality = clamp(support_owner_count / 3.0, 0.0, 1.5)
    downhill = downhill_stair_penalty(case, candidate)

    progress_pressure = clamp(no_progress_streak / 3.0, 0.0, 1.3)
    periodic_progress = progress_pressure * min(1.0, reach_gain / 2.5)
    no_progress_penalty = progress_pressure * (1.0 if reach_gain <= 0.20 and not closes_bridge else 0.0)
    premature_height = foundation_pressure * max(0.0, (wz - 3.2) / 8.0)
    weak_margin_penalty = clamp((0.8 - margin_after) / 1.4, 0.0, 1.0)

    repair_after_cantilever = (
        1.0 if previous_reason == "effective_cantilever" and (brick_contact > 0.25 or margin_delta > 0.05) else 0.0
    )
    controlled_cantilever_after_repair = (
        1.0 if previous_reason == "repair_weakness" and 0.20 < reach_gain <= 3.0 else 0.0
    )

    score = (
        1.50 * stable_bonus
        + 2.10 * margin_bonus
        + 0.95 * margin_delta_bonus
        + 2.30 * low_foundation_fill
        + 1.80 * foundation_gain
        + 1.80 * side_util_gain
        + 1.65 * void_reduction
        + 1.25 * base_contact_quality
        + 1.35 * contact_quality
        + 1.50 * locks_two_supports
        + 1.25 * lock_span
        + 1.10 * locks_overhang
        + 0.95 * brick_lock_quality
        + 0.60 * multi_support_quality
        + 0.85 * min(1.0, reach_gain / 3.5)
        + 0.90 * min(1.0, closure_gain / 5.0)
        + 0.90 * repair_after_cantilever
        + 0.75 * controlled_cantilever_after_repair
        + 0.90 * periodic_progress
        + 2.50 * closes_bridge
        - 1.85 * weak_margin_penalty
        - 1.70 * center_island
        - 1.60 * trapped_void
        - 1.50 * premature_height
        - 1.20 * height_penalty
        - 1.20 * no_progress_penalty
        - 1.00 * downhill
    )

    return {
        "rule_conservative_score": score / 9.0,
        "rule_cons_stable_bonus": stable_bonus,
        "rule_cons_margin_bonus": margin_bonus,
        "rule_cons_margin_delta_bonus": margin_delta_bonus,
        "rule_cons_low_foundation_fill": low_foundation_fill,
        "rule_cons_foundation_gain": foundation_gain,
        "rule_cons_side_util_gain": side_util_gain,
        "rule_cons_void_reduction": void_reduction,
        "rule_cons_base_contact_quality": base_contact_quality,
        "rule_cons_contact_quality": contact_quality,
        "rule_cons_locks_two_supports": locks_two_supports,
        "rule_cons_lock_span_quality": lock_span,
        "rule_cons_locks_overhang": locks_overhang,
        "rule_cons_brick_lock_quality": brick_lock_quality,
        "rule_cons_reach_gain": min(1.0, reach_gain / 3.5),
        "rule_cons_closure_gain": min(1.0, closure_gain / 5.0),
        "rule_cons_repair_after_cantilever": repair_after_cantilever,
        "rule_cons_controlled_cantilever_after_repair": controlled_cantilever_after_repair,
        "rule_cons_periodic_progress": periodic_progress,
        "rule_cons_weak_margin_penalty": weak_margin_penalty,
        "rule_cons_center_foundation_island_penalty": center_island,
        "rule_cons_trapped_void_penalty": trapped_void,
        "rule_cons_premature_height_penalty": premature_height,
        "rule_cons_height_penalty": height_penalty,
        "rule_cons_no_progress_penalty": no_progress_penalty,
        "rule_cons_downhill_stair_penalty": downhill,
    }


def extract_features(
    case: Dict[str, Any],
    candidate: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    state = case.get("before_state", {})
    features: Dict[str, float] = {}

    for name in STATE_NUMERIC:
        features[f"state_{name}"] = scaled(name, state.get(name, 0.0))
    for name in CANDIDATE_NUMERIC:
        features[f"cand_{name}"] = scaled(name, candidate.get(name, 0.0))

    one_hot(features, "brick", str(candidate.get("brick", "")), BRICK_KEYS)
    one_hot(features, "side", str(candidate.get("side", "")), ("L", "R"))

    for name in (
        "left_component_contact",
        "right_component_contact",
        "spans_both_components",
        "is_foundation_move",
        "closes_bridge",
        "stable_by_model",
    ):
        features[f"cand_{name}"] = 1.0 if candidate.get(name) else 0.0

    owners = candidate_support_owner_ids(candidate)
    support_bricks = [owner for owner in owners if owner > 0]
    bmap = brick_map(case)
    locks_two_supports = len(support_bricks) >= 2
    locks_overhang = any(
        owner in bmap and support_is_overhanging(bmap[owner], state)
        for owner in support_bricks
    )
    features["eng_locks_two_supports"] = 1.0 if locks_two_supports else 0.0
    features["eng_locks_overhanging_support"] = 1.0 if locks_overhang else 0.0
    features["eng_lock_span_quality"] = lock_span_quality(case, candidate)
    features["eng_center_foundation_island_penalty"] = center_foundation_island_penalty(case, candidate)

    h = history_features(context)
    features.update(h)
    features["eng_reinforce_after_cantilever"] = (
        1.0 if h["prev_effective_cantilever"] and candidate.get("brick_contact_width", 0.0) > 0.25 else 0.0
    )
    features["eng_cantilever_after_reinforce"] = (
        1.0 if h["prev_repair_weakness"] and finite(candidate.get("reach_gain")) > 0.25 else 0.0
    )

    features["eng_downhill_stair_penalty"] = downhill_stair_penalty(case, candidate)
    features["eng_foundation_pressure"] = max(
        0.0,
        1.0 - (
            finite(state.get("left_base_utilization")) + finite(state.get("right_base_utilization"))
        )
        / 2.0,
    )
    features["eng_low_foundation_fill"] = (
        features["eng_foundation_pressure"] * (1.0 if candidate.get("is_foundation_move") else 0.0)
    )
    features["eng_aggressive_progress"] = max(0.0, scaled("reach_gain", candidate.get("reach_gain", 0.0)))
    features.update(aggressive_rule_components(case, candidate, context))
    features.update(conservative_rule_components(case, candidate, context))

    return features


def group_by_episode(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    episodes: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        eid = str(row.get("episode_id") or "episode")
        episodes.setdefault(eid, []).append(row)
    return episodes


def previous_reasons_by_step(rows: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    prev = ""
    result: Dict[int, str] = {}
    for row in rows:
        if row.get("type") != "decision_case":
            continue
        step = int(row.get("step_index", 0))
        result[step] = prev
        prev = str(row.get("reason", ""))
    return result
