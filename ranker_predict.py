#!/usr/bin/env python3
"""Inspect a trained bridge ranker on recorded JSONL episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ranker_features import (
    aggressive_rule_components,
    conservative_rule_components,
    extract_features,
    load_jsonl,
    previous_reasons_by_step,
)
from train_ranker import dot, rule_score_key


def load_model(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score(
    weights: Dict[str, float],
    case: Dict[str, Any],
    candidate: Dict[str, Any],
    previous_reason: str,
    rule_blend: float,
    strategy: str,
) -> float:
    x = extract_features(case, candidate, {"previous_reason": previous_reason})
    return dot(weights, x) + rule_blend * x.get(rule_score_key(strategy), 0.0)


def describe_candidate(c: Dict[str, Any]) -> str:
    return (
        f"{c.get('brick')} {c.get('side')} "
        f"wx={float(c.get('wx', 0.0)):.1f} wz={float(c.get('wz', 0.0)):.3f} "
        f"reach={float(c.get('reach_gain', 0.0)):+.1f} "
        f"margin={float(c.get('margin_after', 0.0)):.2f} "
        f"contact={float(c.get('contact_width', 0.0)):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ranker predictions.")
    parser.add_argument("--model", default="models/aggressive_ranker.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--rule-blend", type=float, default=None)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    model = load_model(Path(args.model))
    weights = model["weights"]
    strategy = str(model.get("strategy", "aggressive"))
    rule_blend = float(model.get("rule_blend_weight", 0.0) if args.rule_blend is None else args.rule_blend)
    rows = load_jsonl(Path(args.data))
    prev_by_step = previous_reasons_by_step(rows)

    shown = 0
    for row in rows:
        if row.get("type") != "decision_case":
            continue
        if args.max_steps and shown >= args.max_steps:
            break
        shown += 1
        step = int(row["step_index"])
        previous_reason = prev_by_step.get(step, "")
        scored: List[Tuple[int, float]] = [
            (i, score(weights, row, candidate, previous_reason, rule_blend, strategy))
            for i, candidate in enumerate(row["candidates"])
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        chosen_index = int(row["chosen_index"])
        rank = [idx for idx, _s in scored].index(chosen_index) + 1
        print(f"\nstep {step} reason={row.get('reason')} chosen_rank={rank}/{len(scored)}")
        print("chosen:", describe_candidate(row["chosen"]))
        for place, (idx, value) in enumerate(scored[: args.top], 1):
            marker = "*" if idx == chosen_index else " "
            print(f"{marker}{place}. score={value:+.3f} {describe_candidate(row['candidates'][idx])}")
            if args.explain:
                if strategy == "conservative":
                    parts = conservative_rule_components(row, row["candidates"][idx], {"previous_reason": previous_reason})
                    keep = [
                        "rule_conservative_score",
                        "rule_cons_margin_bonus",
                        "rule_cons_low_foundation_fill",
                        "rule_cons_side_util_gain",
                        "rule_cons_locks_two_supports",
                        "rule_cons_lock_span_quality",
                        "rule_cons_repair_after_cantilever",
                        "rule_cons_periodic_progress",
                        "rule_cons_weak_margin_penalty",
                        "rule_cons_premature_height_penalty",
                    ]
                else:
                    parts = aggressive_rule_components(row, row["candidates"][idx], {"previous_reason": previous_reason})
                    keep = [
                        "rule_aggressive_score",
                        "rule_reach_gain",
                        "rule_low_foundation_fill",
                        "rule_locks_two_supports",
                        "rule_locks_overhang",
                        "rule_lock_span_quality",
                        "rule_brick_lock_quality",
                        "rule_progress_after_stall",
                        "rule_center_foundation_island_penalty",
                        "rule_downhill_stair_penalty",
                        "rule_vertical_stack",
                    ]
                explain = " ".join(f"{k}={parts[k]:.2f}" for k in keep)
                print(f"    {explain}")


if __name__ == "__main__":
    main()
