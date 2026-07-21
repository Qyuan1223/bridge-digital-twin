#!/usr/bin/env python3
"""
Visual recorder for demonstrated bridge-building sequences.

This is a data-collection twin, not the human/robot interaction twin. The user
directly places bricks in a Pygame canvas, and each accepted placement writes a
training decision case:

    before_state + all stable legal candidates + chosen candidate + after_state

The geometry, bases, gap, candidate generation, and stability checks are reused
from bridge_twin.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

import bridge_twin as twin
import demo_recorder as recorder


DEMONSTRATED_REASONS = {
    "1": "effective_cantilever",
    "2": "foundation_fill",
    "3": "rear_counterweight",
    "4": "close_bridge",
    "5": "repair_weakness",
}

BAD_REASONS = {
    "6": "wastes_cantilever_space",
    "7": "ineffective_vertical_stack",
    "8": "weak_connection",
    "9": "blocks_future_placement",
}


def save_cases(path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    recorder.save_slim_cases(path, cases)


def geometric_options_for_hover(
    bridge: twin.Bridge,
    bt: twin.BrickType,
    wx: float,
) -> List[twin.BrickInst]:
    return sorted(
        bridge.try_place_options(bt, wx, "demo", max_options=64, complete_edges=True),
        key=lambda o: (o.wz, -twin.contact_width(o.contacts)),
    )


def _candidate_drop_seeds(bridge: twin.Bridge, bt: twin.BrickType, wx: float) -> List[float]:
    candidates: List[float] = []

    def add(value: float) -> None:
        if not any(abs(value - old) <= 1e-7 for old in candidates):
            candidates.append(value)

    surfaces = bridge.all_surfaces()
    supports = twin.support_polys(bridge.bricks)
    for sx in surfaces:
        for bx0, bx1, bbz in bt.bottom_edges:
            if twin.overlap(wx + bx0, wx + bx1, sx.x0, sx.x1):
                add(sx.z - bbz)
    for _owner, spoly in supports:
        for bx, bz in bt.poly[:-1]:
            bwx = wx + bx
            for sx, sz in spoly[:-1]:
                if abs(bwx - sx) <= twin.ALIGN_TOL:
                    add(sz - bz)
    for bp0, bp1 in twin.poly_edges(bt.poly):
        bslope = twin.edge_slope(bp0, bp1)
        if bslope is None:
            continue
        bx0 = wx + min(bp0[0], bp1[0])
        bx1 = wx + max(bp0[0], bp1[0])
        for _owner, spoly in supports:
            for sp0, sp1 in twin.poly_edges(spoly):
                sslope = twin.edge_slope(sp0, sp1)
                if sslope is None or abs(bslope - sslope) > 0.02:
                    continue
                x0 = max(bx0, min(sp0[0], sp1[0]))
                x1 = min(bx1, max(sp0[0], sp1[0]))
                if x1 - x0 <= 0.20:
                    continue
                x = (x0 + x1) / 2
                bz = twin.segment_z_at(bp0, bp1, x - wx)
                sz = twin.segment_z_at(sp0, sp1, x)
                if bz is not None and sz is not None:
                    add(sz - bz)
    return sorted(candidates, reverse=True)


def debug_hover_reasons(bridge: twin.Bridge, bt: twin.BrickType, wx: Optional[float]) -> List[str]:
    if wx is None:
        return ["cursor outside canvas"]
    if abs(wx - twin.snap_place_x(wx)) > 1e-6:
        return ["not on projected hex-lattice grid"]
    live_options = bridge.try_place_options(bt, wx, "debug", max_options=64, complete_edges=True)
    if live_options:
        lattice_supported = any(
            any(seg.owner != twin.BASE_LEFT and seg.owner != twin.BASE_RIGHT for seg in option.contacts)
            for option in live_options
        )
        mode = "lattice/cell" if lattice_supported else "base/lattice"
        return [f"{len(live_options)} geometric placements available", f"mode: {mode}"]

    seeds = _candidate_drop_seeds(bridge, bt, wx)
    if not seeds:
        return ["no candidate height: no aligned edge/vertex contact"]
    expanded = list(seeds)
    for seed in list(seeds):
        for i in range(-12, 1):
            value = seed + i * (twin.HEX_RISE / 2.0)
            if not any(abs(value - old) <= 1e-7 for old in expanded):
                expanded.append(value)
    seeds = sorted(expanded, reverse=True)

    supports = twin.support_polys(bridge.bricks)
    overlap_count = 0
    no_contact_count = 0
    path_block_count = 0
    legal_count = 0
    samples: List[str] = []
    low_samples: List[str] = []

    def owner_name(owner: int) -> str:
        if owner == twin.BASE_LEFT:
            return "left base"
        if owner == twin.BASE_RIGHT:
            return "right base"
        for b in bridge.bricks:
            if b.id == owner:
                return f"{b.btype.key}#{owner}"
        return str(owner)

    def overlap_detail(candidate_poly: Sequence[Tuple[float, float]]) -> str:
        details: List[Tuple[float, str]] = []
        for owner, spoly in supports:
            area = twin.sampled_overlap_area(candidate_poly, spoly)
            if area > twin.OVERLAP_AREA_TOL:
                details.append((area, owner_name(owner)))
        if not details:
            return "unknown"
        details.sort(reverse=True)
        area, name = details[0]
        return f"{name} area={area:.4f}"

    def support_detail(candidate_poly: Sequence[Tuple[float, float]]) -> str:
        raw = [
            (seg, ex0, ex1)
            for owner, spoly in supports
            for seg, ex0, ex1 in twin._raw_boundary_contact_segments(candidate_poly, spoly, owner)
        ]
        if not raw:
            return "raw=0"
        pieces = [seg for seg, _ex0, _ex1 in raw]
        total_width = sum(max(0.0, seg.x1 - seg.x0) for seg in pieces)
        span = max(seg.x1 for seg in pieces) - min(seg.x0 for seg in pieces)
        owners = sorted({owner_name(seg.owner) for seg in pieces})
        edges = {
            (round(ex0, 3), round(ex1, 3), round(seg.z, 3))
            for seg, ex0, ex1 in raw
        }
        return f"raw={len(raw)} owners={len(owners)} edges={len(edges)} width={total_width:.2f} span={span:.2f}"

    for wz in seeds:
        candidate_poly = twin.world_poly(bt, wx, wz)
        reason = ""
        if any(twin.polys_overlap_area(candidate_poly, spoly) for _owner, spoly in supports):
            overlap_count += 1
            reason = f"wz {wz:.3f}: overlap {overlap_detail(candidate_poly)}"
            if len(samples) < 3:
                samples.append(reason)
        else:
            contacts = twin.complete_candidate_edge_contacts(candidate_poly, supports)
            if twin.contact_width(contacts) < 0.2:
                no_contact_count += 1
                reason = f"wz {wz:.3f}: no support {support_detail(candidate_poly)}"
                if len(samples) < 3:
                    samples.append(reason)
            elif twin.vertical_drop_path_blocked(bt, wx, wz, bridge.bricks):
                path_block_count += 1
                reason = f"wz {wz:.3f}: blocked by 3-layer insertion rule"
                if len(samples) < 3:
                    samples.append(reason)
            else:
                legal_count += 1
        if reason:
            low_samples.append(reason)

    if legal_count:
        return [f"{legal_count} geometric placements available"]
    summary = [
        f"tested heights: {len(seeds)}",
        f"overlap: {overlap_count}",
        f"no support: {no_contact_count}",
        f"path blocked: {path_block_count}",
    ]
    return summary + samples + ["lowest tested:"] + low_samples[-6:]


class RecorderUI:
    def __init__(self, screen: pygame.Surface):
        self.sc = screen
        self.canvas = twin.UI(screen)
        self.f12 = pygame.font.Font(None, 18)
        self.f14 = pygame.font.Font(None, 21)
        self.f16 = pygame.font.Font(None, 24)
        self.f20 = pygame.font.Font(None, 30)
        self.f28 = pygame.font.Font(None, 42)

    def render(
        self,
        bridge: twin.Bridge,
        selected: str,
        strategy: str,
        out_path: Path,
        demo_reason: Optional[str],
        bad_reason: Optional[str],
        cases: Sequence[Dict[str, Any]],
        message: str,
        hover_wx: Optional[float],
        hover_option: Optional[twin.BrickInst],
        hover_count: int,
        drop_index: int,
        debug_lines: Sequence[str],
    ) -> None:
        self.canvas.draw_canvas(bridge, [], selected, None, "", "")
        if hover_option is not None:
            self.canvas.draw_ghost(hover_option.btype, hover_option.wx, hover_option.wz, 34, outline_alpha=135)
        self.draw_panel(
            bridge,
            selected,
            strategy,
            out_path,
            demo_reason,
            bad_reason,
            cases,
            message,
            hover_wx,
            hover_option,
            hover_count,
            drop_index,
            debug_lines,
        )
        pygame.display.flip()

    def draw_panel(
        self,
        bridge: twin.Bridge,
        selected: str,
        strategy: str,
        out_path: Path,
        demo_reason: Optional[str],
        bad_reason: Optional[str],
        cases: Sequence[Dict[str, Any]],
        message: str,
        hover_wx: Optional[float],
        hover_option: Optional[twin.BrickInst],
        hover_count: int,
        drop_index: int,
        debug_lines: Sequence[str],
    ) -> None:
        pygame.draw.rect(self.sc, twin.PANEL_BG, (twin.PANEL_X, 0, twin.WIN_W - twin.PANEL_X, twin.WIN_H))
        pygame.draw.line(self.sc, twin.PANEL_LINE, (twin.PANEL_X, 0), (twin.PANEL_X, twin.WIN_H), 1)
        x = twin.PANEL_X + 22
        y = 24
        width = twin.WIN_W - twin.PANEL_X - 44

        def row(text: str, font=None, col=twin.PANEL_TEXT, gap=6) -> None:
            nonlocal y
            surf = (font or self.f14).render(text, True, col)
            self.sc.blit(surf, (x, y))
            y += surf.get_height() + gap

        def sep(space: int = 12) -> None:
            nonlocal y
            y += 4
            pygame.draw.line(self.sc, twin.PANEL_LINE, (x, y), (x + width, y), 1)
            y += space

        row("Demo Recorder", self.f28, twin.PANEL_TEXT, 4)
        row("visual ranking-data capture", self.f12, twin.PANEL_MUTED)
        sep()

        strat_col = twin.ORANGE if strategy == "aggressive" else twin.GREEN
        row(f"Strategy: {strategy}", self.f16, strat_col)
        row(f"Base preset: {twin.LAYOUT.base_label}-span", self.f12, twin.PANEL_MUTED)
        row(f"Output: {out_path}", self.f12, twin.PANEL_MUTED)
        decision_count = sum(1 for case in cases if case.get("type") == "decision_case")
        bad_count = sum(1 for case in cases if case.get("type") == "candidate_annotation")
        row(f"Recorded steps: {decision_count}   bad marks: {bad_count}", self.f12, twin.BLUE)
        sep()

        mm = "--" if bridge.min_margin == float("inf") else f"{bridge.min_margin:.2f}"
        row("Current Structure", self.f16)
        row(f"margin {mm}   gap {bridge.gap_coverage * 100:.0f}%", self.f12, twin.PANEL_TEXT)
        row(f"L {bridge.left_reach:.1f} / R {bridge.right_reach:.1f}   bricks {len(bridge.bricks)}/{twin.MAX_BRICKS}", self.f12, twin.PANEL_MUTED)
        if bridge.bridge_succeeded:
            row("status: closed / success", self.f12, twin.GREEN)
        elif bridge.brick_exhausted:
            row("status: brick exhausted", self.f12, twin.RED)
        else:
            row("status: recording", self.f12, twin.PANEL_MUTED)
        sep()

        row("Placement", self.f16)
        row(f"selected brick: {selected}", self.f12, twin.PANEL_TEXT)
        demo_text = demo_reason or "required: press 1-5"
        bad_text = bad_reason or "required for X: press 6-9"
        row(f"place reason: {demo_text}", self.f12, twin.BLUE if demo_reason else twin.ORANGE)
        row(f"bad reason: {bad_text}", self.f12, twin.BLUE if bad_reason else twin.ORANGE)
        if hover_wx is not None:
            row(f"hover wx: {hover_wx:.1f}", self.f12, twin.PANEL_MUTED)
        if hover_option is not None:
            row(
                f"drop {drop_index + 1}/{hover_count}: wz={hover_option.wz:.3f}",
                self.f12,
                twin.GREEN,
            )
        else:
            row("no geometric placement under cursor", self.f12, twin.ORANGE)
        if message:
            row(message, self.f12, twin.ORANGE if "not" in message.lower() or "invalid" in message.lower() else twin.GREEN)
        if debug_lines:
            row("Debug", self.f16, twin.PANEL_TEXT)
            for line in debug_lines[:12]:
                row(line, self.f12, twin.PANEL_MUTED, 3)
        sep()

        row("Brick Library", self.f16)
        bx = x
        by = y
        item_w = (width - 10) // 2
        item_h = 42
        for k in twin.BRICK_LIBRARY_KEYS:
            rect = pygame.Rect(bx, by, item_w, item_h)
            self.canvas.draw_brick_icon(twin.BTYPES[k], rect, k == selected)
            bx += item_w + 10
            if bx + item_w > twin.WIN_W - 18:
                bx = x
                by += item_h + 10
        y = by + item_h + 10
        sep()

        for line in (
            "Click: record placement",
            "X mark hover candidate bad",
            "Wheel / Up Down: switch drop",
            "1 effective cantilever   2 base fill",
            "3 rear weight   4 close   5 repair",
            "6 waste space   7 stack   8 weak   9 block",
            "A-F brick   T flip",
            "Z undo   base set by --base",
            "strategy set by --strategy   R reset   Q quit",
        ):
            row(line, self.f12, twin.PANEL_MUTED, 4)


def make_case(
    bridge: twin.Bridge,
    candidates: Sequence[Dict[str, Any]],
    chosen_index: int,
    before_state: Dict[str, Any],
    before_bricks: Sequence[Dict[str, Any]],
    before_contact_graph: Dict[str, Any],
    after_state: Dict[str, Any],
    args: argparse.Namespace,
    label: str,
    reason: str,
    raw_input: str,
) -> Dict[str, Any]:
    return {
        "type": "decision_case",
        "version": recorder.VERSION,
        "episode_id": args.episode_id,
        "source": args.source,
        "strategy": args.strategy,
        "base_span": args.base,
        "gap_mm": twin.GAP_MM,
        "cell_mm": twin.CELL_MM,
        "step_index": len(bridge.bricks) - 1,
        "raw_input": raw_input,
        "label": label,
        "reason": reason,
        "note": "",
        "before_state": before_state,
        "before_bricks": list(before_bricks),
        "before_contact_graph": before_contact_graph,
        "candidates": list(candidates),
        "chosen_index": chosen_index,
        "chosen": candidates[chosen_index],
        "after_state": after_state,
        "after_bricks": recorder.brick_payload(bridge),
        "contact_graph": recorder.contact_graph_payload(bridge),
        "episode_done": bridge.bridge_succeeded or bridge.brick_exhausted,
        "success_reason": recorder.success_reason(bridge),
    }


def make_bad_annotation(
    bridge: twin.Bridge,
    candidate: twin.BrickInst,
    args: argparse.Namespace,
    reason: str,
) -> Dict[str, Any]:
    candidate_features = recorder.candidate_features(bridge, candidate)
    return {
        "type": "candidate_annotation",
        "version": recorder.VERSION,
        "episode_id": args.episode_id,
        "source": args.source,
        "strategy": args.strategy,
        "base_span": args.base,
        "gap_mm": twin.GAP_MM,
        "cell_mm": twin.CELL_MM,
        "step_index": len(bridge.bricks),
        "annotation": "hard_negative",
        "reason": reason,
        "before_state": recorder.bridge_state_features(bridge),
        "before_bricks": recorder.brick_payload(bridge),
        "before_contact_graph": recorder.contact_graph_payload(bridge),
        "candidate": candidate_features,
        "episode_done": bridge.bridge_succeeded or bridge.brick_exhausted,
        "success_reason": recorder.success_reason(bridge),
    }


def set_base(span: int, bridge: twin.Bridge, cases: List[Dict[str, Any]], out_path: Path) -> None:
    twin.LAYOUT.base_span = span
    bridge.reset()
    cases.clear()
    save_cases(out_path, cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual bridge demo recorder.")
    parser.add_argument("--strategy", default="aggressive", choices=("aggressive", "conservative"))
    parser.add_argument("--base", type=int, default=9, choices=twin.BASE_PRESETS)
    parser.add_argument("--out", default="data/aggressive_visual_demo.jsonl")
    parser.add_argument("--source", default="visual_real_bridge")
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args()
    if args.episode_id is None:
        args.episode_id = Path(args.out).stem

    twin.LAYOUT.base_span = args.base
    out_path = Path(args.out)
    cases: List[Dict[str, Any]] = []
    save_cases(out_path, cases)

    pygame.init()
    screen = pygame.display.set_mode((twin.WIN_W, twin.WIN_H))
    pygame.display.set_caption("Bridge Demo Recorder")
    clock = pygame.time.Clock()
    ui = RecorderUI(screen)

    bridge = twin.Bridge()
    selected = "A"
    current_demo_reason: Optional[str] = None
    current_bad_reason: Optional[str] = None
    message = "ready"
    last_hover_key: Optional[Tuple[str, float, int]] = None
    drop_index = 0

    while True:
        mx, my = pygame.mouse.get_pos()
        hover_wx_raw = twin.p2w(mx, my)[0] if mx < twin.PANEL_X else None
        hover_wx: Optional[float] = None
        hover_options: List[twin.BrickInst] = []
        hover_option: Optional[twin.BrickInst] = None
        debug_lines: List[str] = []
        if hover_wx_raw is not None:
            bt = twin.BTYPES[selected]
            hover_wx = twin.snap_place_x(hover_wx_raw - twin.brick_bounds_center(bt))
            hover_options = geometric_options_for_hover(bridge, bt, hover_wx)
            debug_lines = debug_hover_reasons(bridge, bt, hover_wx)
            hover_key = (selected, hover_wx, len(bridge.bricks))
            if hover_key != last_hover_key:
                drop_index = 0
                last_hover_key = hover_key
            if hover_options:
                drop_index %= len(hover_options)
                hover_option = hover_options[drop_index]

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEWHEEL and hover_options:
                drop_index = (drop_index - ev.y) % len(hover_options)
            if ev.type == pygame.KEYDOWN:
                key = ev.unicode.upper()
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if key in twin.BTYPES:
                    selected = key
                    drop_index = 0
                    message = f"selected {selected}"
                elif ev.key == pygame.K_t:
                    selected = twin.toggle_flip_key(selected)
                    drop_index = 0
                    message = f"selected {selected}"
                elif ev.key == pygame.K_UP and hover_options:
                    drop_index = (drop_index - 1) % len(hover_options)
                elif ev.key == pygame.K_DOWN and hover_options:
                    drop_index = (drop_index + 1) % len(hover_options)
                elif ev.unicode in DEMONSTRATED_REASONS:
                    current_demo_reason = DEMONSTRATED_REASONS[ev.unicode]
                    message = f"place reason set to {current_demo_reason}"
                elif ev.unicode in BAD_REASONS:
                    current_bad_reason = BAD_REASONS[ev.unicode]
                    message = f"bad reason set to {current_bad_reason}"
                elif ev.key == pygame.K_z:
                    if cases and cases[-1].get("type") == "candidate_annotation":
                        removed_case = cases.pop()
                        save_cases(out_path, cases)
                        message = f"removed bad mark {removed_case.get('candidate', {}).get('brick', '')}"
                    else:
                        removed = bridge.undo_last()
                        if removed and cases:
                            cases.pop()
                            save_cases(out_path, cases)
                            message = f"undo {removed.btype.key}; rewrote output"
                        else:
                            message = "nothing to undo"
                elif ev.key == pygame.K_x and hover_option is not None:
                    if current_bad_reason is None:
                        message = "select bad reason first: 6-9"
                    else:
                        annotation = make_bad_annotation(bridge, hover_option, args, current_bad_reason)
                        cases.append(annotation)
                        save_cases(out_path, cases)
                        message = (
                            f"marked bad {hover_option.btype.key} "
                            f"wx={hover_option.wx:.1f} wz={hover_option.wz:.3f}"
                        )
                        current_bad_reason = None
                elif ev.key == pygame.K_x:
                    message = "no candidate to mark bad"
                elif ev.key == pygame.K_r:
                    bridge.reset()
                    cases.clear()
                    save_cases(out_path, cases)
                    message = "session reset"

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hover_option is not None:
                if current_demo_reason is None:
                    message = "select placement reason first: 1-5"
                    continue
                before_state = recorder.bridge_state_features(bridge)
                before_bricks = recorder.brick_payload(bridge)
                before_contact_graph = recorder.contact_graph_payload(bridge)
                candidates = recorder.generate_candidates(
                    bridge,
                    stable_only=False,
                    complete_edges=True,
                    max_options=64,
                )
                chosen_index = recorder.match_candidate(
                    candidates,
                    hover_option.btype.key,
                    recorder.side_for_candidate(hover_option),
                    hover_option.wx,
                    hover_option.wz,
                )
                if chosen_index is None:
                    chosen = recorder.candidate_features(bridge, hover_option)
                    chosen["candidate_id"] = len(candidates)
                    candidates.append(chosen)
                    chosen_index = len(candidates) - 1
                placed = recorder.place_from_candidate(bridge, candidates[chosen_index])
                after_state = recorder.bridge_state_features(bridge)
                case = make_case(
                    bridge,
                    candidates,
                    chosen_index,
                    before_state,
                    before_bricks,
                    before_contact_graph,
                    after_state,
                    args,
                    "demonstrated",
                    current_demo_reason,
                    f"{placed.btype.key} {candidates[chosen_index]['side']} {placed.wx:.1f} {placed.wz:.6f}",
                )
                cases.append(case)
                save_cases(out_path, cases)
                message = (
                    f"recorded {placed.btype.key} {candidates[chosen_index]['side']} "
                    f"wx={placed.wx:.1f} wz={placed.wz:.3f}"
                )
                current_demo_reason = None
                drop_index = 0

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hover_wx is not None and hover_option is None:
                message = "no geometric placement here"
                twin.log_event(
                    "DEBUG",
                    f"hover reject brick={selected} wx={hover_wx:.2f} | " + " | ".join(debug_lines),
                )

        ui.render(
            bridge,
            selected,
            args.strategy,
            out_path,
            current_demo_reason,
            current_bad_reason,
            cases,
            message,
            hover_wx,
            hover_option,
            len(hover_options),
            drop_index,
            debug_lines,
        )
        clock.tick(30)


if __name__ == "__main__":
    main()
