#!/usr/bin/env python3
"""Train a dependency-free bridge candidate ranker.

This is intentionally a small, inspectable first model: a linear pairwise
ranker trained with logistic loss. It is not meant to be the final ML ceiling;
it gives us a reproducible baseline and a model file that can be used inside
the digital twin without installing sklearn / numpy / pandas.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ranker_features import (
    BAD_REASONS,
    DEMO_REASONS,
    extract_features,
    group_by_episode,
    load_many,
    previous_reasons_by_step,
)


FeatureVector = Dict[str, float]


def rule_score_key(strategy: str) -> str:
    if strategy == "conservative":
        return "rule_conservative_score"
    return "rule_aggressive_score"


def dot(weights: Dict[str, float], x: FeatureVector) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in x.items())


def sigmoid_stable(x: float) -> float:
    if x >= 35.0:
        return 1.0
    if x <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def subtract(a: FeatureVector, b: FeatureVector) -> FeatureVector:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) - v
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def discover_data(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            parent = p.parent if str(p.parent) else Path(".")
            for child in parent.glob(p.name):
                if child.is_file() and not child.name.endswith("_slim.jsonl") and "_enriched" not in child.name:
                    paths.append(child)
        elif p.is_file():
            paths.append(p)
    return sorted(dict.fromkeys(paths))


def hard_negatives_by_step(rows: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("type") != "candidate_annotation":
            continue
        step = int(row.get("step_index", 0))
        grouped.setdefault(step, []).append(row)
    return grouped


def sample_unchosen(
    candidates: Sequence[Dict[str, Any]],
    chosen_index: int,
    limit: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    pool = [c for i, c in enumerate(candidates) if i != chosen_index]
    if len(pool) <= limit:
        return pool
    # Keep some hard-looking candidates near high progress/stability and sample
    # the rest. This avoids training only against trivial weak moves.
    pool.sort(
        key=lambda c: (
            -float(c.get("stable_by_model", False)),
            -float(c.get("reach_gain", 0.0)),
            -float(c.get("contact_width", 0.0)),
        )
    )
    elite = pool[: max(4, limit // 4)]
    rest = pool[max(4, limit // 4):]
    rng.shuffle(rest)
    return (elite + rest)[:limit]


def build_pairs(
    episodes: Dict[str, List[Dict[str, Any]]],
    train_episode_ids: Sequence[str],
    negatives_per_step: int,
    rng: random.Random,
) -> List[FeatureVector]:
    pairs: List[FeatureVector] = []
    for eid in train_episode_ids:
        rows = episodes[eid]
        prev_by_step = previous_reasons_by_step(rows)
        bad_by_step = hard_negatives_by_step(rows)
        for row in rows:
            if row.get("type") != "decision_case":
                continue
            step = int(row.get("step_index", 0))
            context = {"previous_reason": prev_by_step.get(step, "")}
            chosen = row["chosen"]
            chosen_x = extract_features(row, chosen, context)

            hard = [ann["candidate"] for ann in bad_by_step.get(step, [])]
            sampled = sample_unchosen(row["candidates"], int(row["chosen_index"]), negatives_per_step, rng)
            negatives = hard + sampled
            for neg in negatives:
                neg_x = extract_features(row, neg, context)
                pairs.append(subtract(chosen_x, neg_x))
    rng.shuffle(pairs)
    return pairs


def train_linear_ranker(
    pairs: Sequence[FeatureVector],
    epochs: int = 30,
    lr: float = 0.08,
    l2: float = 0.0005,
    seed: int = 7,
) -> Dict[str, float]:
    rng = random.Random(seed)
    weights: Dict[str, float] = {}
    order = list(range(len(pairs)))
    for epoch in range(epochs):
        rng.shuffle(order)
        eta = lr / math.sqrt(1.0 + epoch * 0.25)
        for idx in order:
            x = pairs[idx]
            score = dot(weights, x)
            grad_scale = 1.0 - sigmoid_stable(score)
            for k, v in x.items():
                old = weights.get(k, 0.0)
                weights[k] = old * (1.0 - eta * l2) + eta * grad_scale * v
    return weights


def score_candidate(
    weights: Dict[str, float],
    case: Dict[str, Any],
    candidate: Dict[str, Any],
    previous_reason: str,
    rule_blend: float = 0.0,
    strategy: str = "aggressive",
) -> float:
    x = extract_features(case, candidate, {"previous_reason": previous_reason})
    return dot(weights, x) + rule_blend * x.get(rule_score_key(strategy), 0.0)


def rank_candidates(
    weights: Dict[str, float],
    case: Dict[str, Any],
    previous_reason: str,
    rule_blend: float = 0.0,
    strategy: str = "aggressive",
) -> List[Tuple[int, float]]:
    scored = [
        (i, score_candidate(weights, case, c, previous_reason, rule_blend, strategy))
        for i, c in enumerate(case.get("candidates", []))
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def evaluate_episode(
    weights: Dict[str, float],
    rows: Sequence[Dict[str, Any]],
    rule_blend: float = 0.0,
    strategy: str = "aggressive",
) -> Dict[str, Any]:
    prev_by_step = previous_reasons_by_step(rows)
    bad_by_step = hard_negatives_by_step(rows)
    ranks: List[int] = []
    top1 = top3 = 0
    hard_ranks: List[int] = []
    hard_count = 0
    for row in rows:
        if row.get("type") != "decision_case":
            continue
        step = int(row.get("step_index", 0))
        ranking = rank_candidates(weights, row, prev_by_step.get(step, ""), rule_blend, strategy)
        ordered_ids = [idx for idx, _score in ranking]
        chosen_index = int(row["chosen_index"])
        try:
            rank = ordered_ids.index(chosen_index) + 1
        except ValueError:
            rank = len(ordered_ids) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 3:
            top3 += 1

        for ann in bad_by_step.get(step, []):
            hard_count += 1
            hard_score = score_candidate(weights, row, ann["candidate"], prev_by_step.get(step, ""), rule_blend, strategy)
            # Rank among the decision candidates by insertion point.
            worse_or_equal = sum(1 for _idx, score in ranking if score >= hard_score)
            hard_ranks.append(max(1, worse_or_equal))

    n = max(1, len(ranks))
    return {
        "steps": len(ranks),
        "top1": top1 / n,
        "top3": top3 / n,
        "mrr": sum(1.0 / r for r in ranks) / n if ranks else 0.0,
        "mean_rank": sum(ranks) / n if ranks else 0.0,
        "hard_negative_count": hard_count,
        "hard_negative_mean_rank": (sum(hard_ranks) / len(hard_ranks)) if hard_ranks else None,
    }


def leave_one_episode_out(
    episodes: Dict[str, List[Dict[str, Any]]],
    negatives_per_step: int,
    epochs: int,
    seed: int,
    rule_blend: float,
    strategy: str,
) -> Dict[str, Any]:
    episode_ids = sorted(episodes)
    folds: Dict[str, Any] = {}
    for test_id in episode_ids:
        train_ids = [eid for eid in episode_ids if eid != test_id]
        pairs = build_pairs(episodes, train_ids, negatives_per_step, random.Random(seed))
        weights = train_linear_ranker(pairs, epochs=epochs, seed=seed)
        folds[test_id] = evaluate_episode(weights, episodes[test_id], rule_blend, strategy)

    total_steps = sum(f["steps"] for f in folds.values()) or 1
    aggregate = {
        "top1": sum(f["top1"] * f["steps"] for f in folds.values()) / total_steps,
        "top3": sum(f["top3"] * f["steps"] for f in folds.values()) / total_steps,
        "mrr": sum(f["mrr"] * f["steps"] for f in folds.values()) / total_steps,
        "mean_rank": sum(f["mean_rank"] * f["steps"] for f in folds.values()) / total_steps,
        "steps": total_steps,
    }
    hard_values = [f["hard_negative_mean_rank"] for f in folds.values() if f["hard_negative_mean_rank"] is not None]
    aggregate["hard_negative_mean_rank"] = sum(hard_values) / len(hard_values) if hard_values else None
    return {"folds": folds, "aggregate": aggregate}


def top_weights(weights: Dict[str, float], n: int = 30) -> Dict[str, List[Tuple[str, float]]]:
    ordered = sorted(weights.items(), key=lambda kv: kv[1])
    return {
        "most_negative": ordered[:n],
        "most_positive": list(reversed(ordered[-n:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bridge candidate ranker.")
    parser.add_argument("--data", nargs="+", default=["data/aggressive_*.jsonl"])
    parser.add_argument("--out", default="models/aggressive_ranker.json")
    parser.add_argument("--report", default="models/aggressive_ranker_report.json")
    parser.add_argument("--strategy", default="aggressive")
    parser.add_argument("--negatives-per-step", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--rule-blend", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    paths = discover_data(args.data)
    if not paths:
        raise SystemExit("No JSONL data files found.")
    rows = load_many(paths)
    episodes = group_by_episode(rows)
    episode_ids = sorted(episodes)
    rng = random.Random(args.seed)

    report = leave_one_episode_out(episodes, args.negatives_per_step, args.epochs, args.seed, args.rule_blend, args.strategy)
    pairs = build_pairs(episodes, episode_ids, args.negatives_per_step, rng)
    weights = train_linear_ranker(pairs, epochs=args.epochs, seed=args.seed)

    model = {
        "model_type": "linear_pairwise_ranker",
        "version": 1,
        "strategy": args.strategy,
        "data_files": [str(p) for p in paths],
        "episode_ids": episode_ids,
        "negatives_per_step": args.negatives_per_step,
        "epochs": args.epochs,
        "rule_blend_weight": args.rule_blend,
        "rule_score_key": rule_score_key(args.strategy),
        "weights": weights,
        "feature_count": len(weights),
    }

    report["training"] = {
        "data_files": [str(p) for p in paths],
        "episodes": len(episode_ids),
        "rows": len(rows),
        "decision_cases": sum(1 for r in rows if r.get("type") == "decision_case"),
        "hard_negative_annotations": sum(1 for r in rows if r.get("type") == "candidate_annotation"),
        "pair_count": len(pairs),
        "feature_count": len(weights),
        "rule_blend_weight": args.rule_blend,
        "rule_score_key": rule_score_key(args.strategy),
        "top_weights": top_weights(weights, 25),
    }

    out_path = Path(args.out)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    agg = report["aggregate"]
    print(f"trained {out_path}")
    print(f"report {report_path}")
    print(
        "LOEO:",
        f"top1={agg['top1']:.3f}",
        f"top3={agg['top3']:.3f}",
        f"mrr={agg['mrr']:.3f}",
        f"mean_rank={agg['mean_rank']:.1f}",
    )
    print("pairs", len(pairs), "features", len(weights))


if __name__ == "__main__":
    main()
