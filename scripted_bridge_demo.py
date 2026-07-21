#!/usr/bin/env python3
"""Scripted human/robot bridge demo using a recorded successful episode.

This is a no-algorithm integration demo. It replays a recorded bridge sequence
from JSONL and alternates responsibility:

- odd-numbered placements: human/CV side
- even-numbered placements: robot side

The program does not recommend human placements. It only evaluates whether the
human's latest placement matches the scripted bridge closely enough, is merely
acceptable, or must be removed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pygame

import bridge_twin as twin


SCRIPT_TOL_X = 0.05
SCRIPT_TOL_Z = 0.05


@dataclass(frozen=True)
class ScriptStep:
    index: int
    actor: str
    brick: str
    side: str
    wx: float
    wz: float
    reason: str
    success_reason: str

    @property
    def label(self) -> str:
        return f"{self.index:02d} {self.actor} {self.brick} {self.side} x={self.wx:.1f}"


def load_script(path: Path) -> List[ScriptStep]:
    steps: List[ScriptStep] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "decision_case":
                continue
            chosen = row["chosen"]
            index = int(row["step_index"]) + 1
            steps.append(
                ScriptStep(
                    index=index,
                    actor="human" if index % 2 == 1 else "robot",
                    brick=str(chosen["brick"]),
                    side=str(chosen["side"]),
                    wx=float(chosen["wx"]),
                    wz=float(chosen["wz"]),
                    reason=str(row.get("reason", "")),
                    success_reason=str(row.get("success_reason", "in_progress")),
                )
            )
    return steps


def geometric_options_for_hover(
    bridge: twin.Bridge,
    bt: twin.BrickType,
    wx: float,
) -> List[twin.BrickInst]:
    return sorted(
        bridge.try_place_options(bt, wx, "human", max_options=64, complete_edges=True),
        key=lambda option: (option.wz, -twin.contact_width(option.contacts)),
    )


def place_exact(bridge: twin.Bridge, step: ScriptStep, actor: str) -> Optional[twin.BrickInst]:
    bt = twin.BTYPES[step.brick]
    return bridge.place(bt, step.wx, actor, step.wz)


def same_script_move(placed: twin.BrickInst, step: ScriptStep) -> bool:
    return (
        placed.btype.key == step.brick
        and abs(placed.wx - step.wx) <= SCRIPT_TOL_X
        and abs(placed.wz - step.wz) <= SCRIPT_TOL_Z
    )


def acceptable_move(before: twin.Bridge, after: twin.Bridge, placed: twin.BrickInst) -> bool:
    if after.is_closing_brick(placed.id):
        return True
    if after.min_margin < twin.MIN_ACCEPT_MARGIN:
        return False
    if not before.joints:
        return True
    # Stable but not exactly scripted is allowed to remain in the interaction
    # model, but this scripted demo cannot advance until the expected move is
    # observed because robot steps are fixed to the recorded bridge.
    return after.min_margin >= twin.MIN_ACCEPT_MARGIN


def scripted_reaction(before: twin.Bridge, after: twin.Bridge, placed: twin.BrickInst, expected: ScriptStep) -> Tuple[str, str, bool]:
    if same_script_move(placed, expected):
        if after.bridge_succeeded:
            return "nod", "Human move accepted: bridge closed", True
        return "nod", f"Human move accepted: {expected.label}", True
    if not acceptable_move(before, after, placed):
        return "remove", "Robot removes: unacceptable placement", False
    return "", "Acceptable but off-script; undo or place scripted move", False


def scripted_robot_place(bridge: twin.Bridge, step: ScriptStep) -> Tuple[str, str, Optional[twin.BrickInst]]:
    placed = place_exact(bridge, step, "robot")
    if not placed:
        twin.log_event("SCRIPT", f"robot failed {step.label}")
        return "shake", f"Robot failed scripted step {step.index}", None
    if bridge.bridge_succeeded:
        return "nod", f"Robot placed {step.brick}; bridge closed", placed
    return "nod", f"Robot placed {step.brick}: {step.label}", placed


class ScriptedUI:
    def __init__(self, screen: pygame.Surface):
        self.sc = screen
        self.canvas = twin.UI(screen)
        self.f12 = pygame.font.Font(None, 18)
        self.f14 = pygame.font.Font(None, 21)
        self.f16 = pygame.font.Font(None, 24)
        self.f22 = pygame.font.Font(None, 34)

    def render(
        self,
        bridge: twin.Bridge,
        script: Sequence[ScriptStep],
        cursor: int,
        selected: str,
        hover_option: Optional[twin.BrickInst],
        hover_count: int,
        drop_index: int,
        message: str,
        reaction: str,
    ) -> None:
        self.sc.fill(twin.BG)
        self.draw_canvas(bridge, hover_option, hover_count, drop_index, message, reaction)
        self.draw_panel(bridge, script, cursor, selected, message, reaction)
        pygame.display.flip()

    def draw_canvas(
        self,
        bridge: twin.Bridge,
        hover_option: Optional[twin.BrickInst],
        hover_count: int,
        drop_index: int,
        message: str,
        reaction: str,
    ) -> None:
        # Reuse the existing twin canvas renderer with no recommendations.
        self.canvas.draw_canvas(bridge, [], "", None, message, reaction)
        if hover_option is not None:
            self.canvas.draw_ghost(hover_option.btype, hover_option.wx, hover_option.wz, 24, outline_alpha=120)
            if hover_count > 1:
                label = self.f12.render(f"drop {drop_index + 1}/{hover_count}", True, twin.PANEL_MUTED)
                px, py = twin.w2p(hover_option.world_cx, hover_option.wz + hover_option.btype.height + 0.45)
                self.sc.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))

    def row(self, text: str, x: int, y: int, font=None, col=None) -> int:
        surf = (font or self.f14).render(text, True, col or twin.PANEL_TEXT)
        self.sc.blit(surf, (x, y))
        return y + surf.get_height() + 6

    def draw_panel(
        self,
        bridge: twin.Bridge,
        script: Sequence[ScriptStep],
        cursor: int,
        selected: str,
        message: str,
        reaction: str,
    ) -> None:
        pygame.draw.rect(self.sc, twin.PANEL_BG, (twin.PANEL_X, 0, twin.WIN_W - twin.PANEL_X, twin.WIN_H))
        pygame.draw.line(self.sc, twin.PANEL_LINE, (twin.PANEL_X, 0), (twin.PANEL_X, twin.WIN_H), 1)
        x = twin.PANEL_X + 22
        y = 24

        y = self.row("Scripted Bridge Demo", x, y, self.f22)
        y = self.row("aggressive_001 / no recommendation", x, y, self.f12, twin.PANEL_MUTED)
        y += 10

        status = "success" if bridge.bridge_succeeded else "building"
        if cursor >= len(script):
            status = "complete"
        y = self.row(f"Status: {status}", x, y, self.f16, twin.GREEN if bridge.bridge_succeeded else twin.PANEL_TEXT)
        y = self.row(f"Bricks: {len(bridge.bricks)}/{len(script)}", x, y, self.f12, twin.PANEL_MUTED)
        mm = "--" if bridge.min_margin == math.inf else f"{bridge.min_margin:.2f}"
        y = self.row(f"Margin: {mm}   coverage {bridge.gap_coverage * 100:.0f}%", x, y, self.f12, twin.PANEL_MUTED)
        y += 12

        if cursor < len(script):
            step = script[cursor]
            color = twin.BLUE if step.actor == "human" else twin.ORANGE
            y = self.row(f"Next: {step.actor.upper()}", x, y, self.f16, color)
            y = self.row(f"{step.index}. brick {step.brick} {step.side}", x, y, self.f14, twin.PANEL_TEXT)
            y = self.row(f"wx={step.wx:.1f}  wz={step.wz:.3f}", x, y, self.f12, twin.PANEL_MUTED)
            y = self.row(f"reason: {step.reason}", x, y, self.f12, twin.PANEL_MUTED)
        else:
            y = self.row("All scripted steps completed", x, y, self.f16, twin.GREEN)
        y += 14

        y = self.row("Robot reaction", x, y, self.f16)
        col = {"nod": twin.GREEN, "remove": twin.RED, "shake": twin.ORANGE, "": twin.PANEL_MUTED}.get(reaction, twin.BLUE)
        y = self.row(message or "Waiting", x, y, self.f12, col)
        y += 14

        y = self.row("Controls", x, y, self.f16)
        for line in (
            "A-F select brick, T flip",
            "Mouse wheel / Up Down: drop",
            "Left click: human/CV placement",
            "Space: execute robot step",
            "Z undo   R reset   Q quit",
        ):
            y = self.row(line, x, y, self.f12, twin.PANEL_MUTED)

        y += 10
        y = self.row("Recent script", x, y, self.f16)
        start = max(0, cursor - 3)
        for i in range(start, min(len(script), start + 7)):
            step = script[i]
            prefix = ">" if i == cursor else " "
            color = twin.BLUE if i == cursor else twin.PANEL_MUTED
            self.row(f"{prefix} {step.index:02d} {step.actor[:1]} {step.brick} x={step.wx:.1f}", x, y, self.f12, color)
            y += 20


def run(args: argparse.Namespace) -> None:
    script = load_script(Path(args.script))
    if not script:
        raise SystemExit(f"No decision_case rows found in {args.script}")

    twin.LAYOUT.base_span = args.base
    pygame.init()
    screen = pygame.display.set_mode((twin.WIN_W, twin.WIN_H))
    pygame.display.set_caption("Bridge Scripted Demo - aggressive_001")
    clock = pygame.time.Clock()
    ui = ScriptedUI(screen)

    bridge = twin.Bridge()
    cursor = 0
    selected = script[0].brick if script else "A"
    message = "Waiting for first human placement"
    reaction = ""
    drop_index = 0
    pending_robot_at: Optional[int] = None
    last_hover_key: Optional[Tuple[str, float, int]] = None

    while True:
        if (
            pending_robot_at is not None
            and pygame.time.get_ticks() >= pending_robot_at
            and cursor < len(script)
            and script[cursor].actor == "robot"
        ):
            reaction, message, placed = scripted_robot_place(bridge, script[cursor])
            if placed:
                cursor += 1
                if cursor < len(script):
                    selected = script[cursor].brick
            pending_robot_at = None

        mx, my = pygame.mouse.get_pos()
        hover_option: Optional[twin.BrickInst] = None
        hover_options: List[twin.BrickInst] = []
        hover_wx_raw = twin.p2w(mx, my)[0] if mx < twin.PANEL_X else None
        if hover_wx_raw is not None and selected in twin.BTYPES:
            bt = twin.BTYPES[selected]
            hover_wx = twin.snap_place_x(hover_wx_raw - twin.brick_bounds_center(bt))
            hover_options = geometric_options_for_hover(bridge, bt, hover_wx)
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
                elif ev.key == pygame.K_t:
                    selected = twin.toggle_flip_key(selected)
                    drop_index = 0
                elif ev.key == pygame.K_UP and hover_options:
                    drop_index = (drop_index - 1) % len(hover_options)
                elif ev.key == pygame.K_DOWN and hover_options:
                    drop_index = (drop_index + 1) % len(hover_options)
                elif ev.key == pygame.K_z:
                    removed = bridge.undo_last()
                    if removed and cursor > 0:
                        cursor -= 1
                    message = f"Undo {removed.btype.key}" if removed else "Nothing to undo"
                    reaction = ""
                    pending_robot_at = None
                elif ev.key == pygame.K_r:
                    bridge.reset()
                    cursor = 0
                    selected = script[0].brick
                    message = "Scene reset"
                    reaction = ""
                    pending_robot_at = None
                elif ev.key == pygame.K_SPACE:
                    pending_robot_at = None
                    if cursor >= len(script):
                        message = "Script already complete"
                        reaction = "nod"
                    elif script[cursor].actor != "robot":
                        message = "Waiting for human step"
                        reaction = ""
                    else:
                        reaction, message, placed = scripted_robot_place(bridge, script[cursor])
                        if placed:
                            cursor += 1
                            if cursor < len(script):
                                selected = script[cursor].brick

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and mx < twin.PANEL_X:
                if cursor >= len(script):
                    message = "Script already complete"
                    reaction = "nod"
                    continue
                expected = script[cursor]
                if expected.actor != "human":
                    message = "Robot turn: press Space"
                    reaction = ""
                    continue
                if hover_option is None:
                    message = "No geometric placement here"
                    reaction = "shake"
                    continue

                before = bridge.clone()
                placed = twin.BrickInst(
                    bridge.next_id,
                    hover_option.btype,
                    hover_option.wx,
                    hover_option.wz,
                    "human",
                    list(hover_option.contacts),
                )
                bridge.bricks.append(placed)
                bridge.next_id += 1
                bridge.analyse()

                reaction, message, advance = scripted_reaction(before, bridge, placed, expected)
                if reaction == "remove":
                    bridge.remove_id(placed.id)
                elif advance:
                    cursor += 1
                    if cursor < len(script):
                        selected = script[cursor].brick
                        if script[cursor].actor == "robot" and not args.manual_robot:
                            pending_robot_at = pygame.time.get_ticks() + args.robot_delay_ms
                drop_index = 0

        ui.render(bridge, script, cursor, selected, hover_option, len(hover_options), drop_index, message, reaction)
        clock.tick(30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-algorithm scripted bridge demo.")
    parser.add_argument("--script", default="data/aggressive_001.jsonl")
    parser.add_argument("--base", type=int, default=9, choices=twin.BASE_PRESETS)
    parser.add_argument("--manual-robot", action="store_true", help="Require Space to execute robot turns.")
    parser.add_argument("--robot-delay-ms", type=int, default=700)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
