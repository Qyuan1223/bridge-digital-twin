# Bridge Ranker Workflow

This folder contains a small, inspectable ranking pipeline for the bridge
digital twin. It learns from recorded successful bridge demonstrations and
hard negative candidate annotations.

## Data

Record demonstrations with `visual_demo_recorder.py`.

Each demonstrated placement must include one reason:

- `effective_cantilever`
- `foundation_fill`
- `rear_counterweight`
- `close_bridge`
- `repair_weakness`

Each bad candidate must include one reason:

- `wastes_cantilever_space`
- `ineffective_vertical_stack`
- `weak_connection`
- `blocks_future_placement`

The strategy label is episode-level. Do not mix conservative and aggressive
bridges in one training run.

## Train Aggressive Model

```powershell
python train_ranker.py --data data/aggressive_*.jsonl --strategy aggressive --out models/aggressive_ranker.json --report models/aggressive_ranker_report.json --epochs 45 --negatives-per-step 80 --rule-blend 2.0
```

The model is a pairwise linear ranker plus an explicit aggressive rule blend.
The learned part compares demonstrated moves against unchosen and marked-bad
candidates. The rule part keeps human design priors visible: foundation use,
real cantilever progress, multi-support locking, repair-after-cantilever,
cantilever-after-repair, and penalties for vertical stacking or downhill
frontiers.

## Inspect Predictions

```powershell
python ranker_predict.py --model models/aggressive_ranker.json --data data/aggressive_001.jsonl --top 5 --max-steps 8 --explain
```

Useful fields:

- `chosen_rank`: where the demonstrated move ranked among all candidates.
- `rule_reach_gain`: reward for extending toward closure.
- `rule_low_foundation_fill`: reward for using low support space while the base
  is still underused.
- `rule_locks_two_supports`: whether the candidate is supported by multiple
  bricks.
- `rule_locks_overhang`: whether it locks a support brick that is already
  overhanging.
- `rule_downhill_stair_penalty`: penalty for worsening a descending frontier.
- `rule_vertical_stack`: penalty for non-progress vertical stacking.

## Current Baseline

With the first 10 aggressive bridge episodes:

- episodes: 10
- decision cases: 153
- hard negative annotations: 99
- leave-one-episode-out top1: about 8.5%
- leave-one-episode-out top3: about 21.6%
- leave-one-episode-out mean chosen rank: about 19.2

These metrics are strict because each step can have hundreds of legal
candidates and many are nearly equivalent. For design review, inspect the top
5-10 recommendations and the explanation fields, not only top1 accuracy.

## Train Conservative Later

After recording conservative demonstrations:

```powershell
python train_ranker.py --data data/conservative_*.jsonl --strategy conservative --out models/conservative_ranker.json --report models/conservative_ranker_report.json --epochs 45 --negatives-per-step 80 --rule-blend 2.0
```

The conservative model can use the same code path, but its data should express
the different episode-level goal.

## Record Conservative Bridges

Use one JSONL file per successful bridge. Keep the episode strategy fixed as
`conservative`.

```powershell
python visual_demo_recorder.py --strategy conservative --base 9 --out data/conservative_001.jsonl --episode-id conservative_001
```

Recommended naming:

```text
data/conservative_001.jsonl
data/conservative_002.jsonl
...
data/conservative_010.jsonl
```

For each demonstrated placement, select one reason before clicking:

- `1` effective cantilever
- `2` foundation fill
- `3` rear counterweight
- `4` close bridge
- `5` repair weakness

For bad candidates, select one reason before pressing `X`:

- `6` wastes cantilever space
- `7` ineffective vertical stack
- `8` weak connection
- `9` blocks future placement

For conservative demonstrations, prioritize marking bad candidates that are
stable but strategically poor:

- early high-layer stacking while the base still has usable space
- center-island low placements that split the base into unusable fragments
- weak one-owner contacts
- moves that reduce margin without creating useful progress
- moves that block future low-layer repair or support placements

After recording the 10 bridges:

```powershell
python train_ranker.py --data data/conservative_*.jsonl --strategy conservative --out models/conservative_ranker.json --report models/conservative_ranker_report.json --epochs 45 --negatives-per-step 80 --rule-blend 2.0
```

Inspect one episode:

```powershell
python ranker_predict.py --model models/conservative_ranker.json --data data/conservative_001.jsonl --top 5 --max-steps 8 --explain
```
