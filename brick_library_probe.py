#!/usr/bin/env python
"""
Interactive recorder for fixed brick-library pickup poses.

This script is meant to be used together with ../probe_position.py:

1. Run the JetMax probe tool in one terminal and manually jog the gripper
   to the brick pickup pose.
2. Copy the printed position/yaw into this script with the `record` command.
3. Repeat until every fixed supply slot has a measured pose.

The output JSON is intentionally simple so later robot scripts can load it
without importing pygame/ROS/digital-twin code.
"""

from __future__ import print_function

import argparse
import io
import json
import os
import shlex
from copy import deepcopy
from datetime import datetime


BRICKS = ("A", "B", "C", "D", "E", "F")

SLOT_COUNTS = {
    "A": 6,
    "B": 6,
    "C": 6,
    "D": 5,
    "E": 5,
    "F": 5,
}

ORIENTATIONS = ("normal", "flipped")
DEFAULT_OUT = os.path.join("calibration", "brick_library_positions.json")


class SlotSpec(object):
    def __init__(self, slot_id, brick, orientation, brick_key, index):
        self.slot_id = slot_id
        self.brick = brick
        self.orientation = orientation
        self.brick_key = brick_key
        self.index = index


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def key_for(brick, orientation):
    return brick if orientation == "normal" else "{}'".format(brick)


def build_slot_specs():
    specs = []
    for brick in BRICKS:
        for orientation in ORIENTATIONS:
            suffix = "N" if orientation == "normal" else "F"
            for index in range(1, SLOT_COUNTS[brick] + 1):
                specs.append(
                    SlotSpec(
                        slot_id="{}_{:s}_{:02d}".format(brick, suffix, index),
                        brick=brick,
                        orientation=orientation,
                        brick_key=key_for(brick, orientation),
                        index=index,
                    )
                )
    return specs


def blank_pose():
    return {"x": None, "y": None, "z": None, "yaw": None}


def blank_state():
    return {
        "version": 1,
        "unit": "mm",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "source": "brick_library_probe.py",
        "notes": "",
        "origin": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "gripper": {
            "default_yaw": None,
            "open_angle": None,
            "close_angle": None,
            "pickup_z_offset": 0.0,
            "safe_z": None,
        },
        "slot_counts": deepcopy(SLOT_COUNTS),
        "orientation_aliases": {
            "normal": "unflipped/original brick orientation",
            "flipped": "180-degree flipped brick orientation",
        },
        "slots": [],
    }


def slot_to_record(spec):
    return {
        "id": spec.slot_id,
        "brick": spec.brick,
        "orientation": spec.orientation,
        "brick_key": spec.brick_key,
        "slot_index": spec.index,
        "pose": blank_pose(),
        "relative_pose": blank_pose(),
        "recorded": False,
        "recorded_at": None,
        "note": "",
    }


def ensure_state_shape(state):
    """Merge an existing file with the current slot table without dropping data."""
    base = blank_state()
    for key, value in state.items():
        if key != "slots":
            base[key] = value

    existing = {slot.get("id"): slot for slot in state.get("slots", []) if slot.get("id")}
    merged = []
    for spec in build_slot_specs():
        fresh = slot_to_record(spec)
        old = existing.get(spec.slot_id)
        if old:
            fresh.update(old)
            fresh["id"] = spec.slot_id
            fresh["brick"] = spec.brick
            fresh["orientation"] = spec.orientation
            fresh["brick_key"] = spec.brick_key
            fresh["slot_index"] = spec.index
            fresh.setdefault("pose", blank_pose())
            fresh.setdefault("relative_pose", blank_pose())
            fresh.setdefault("recorded", pose_complete(fresh["pose"]))
        merged.append(fresh)
    base["slots"] = merged
    return base


def load_state(path):
    if not os.path.exists(path):
        return ensure_state_shape(blank_state())
    with io.open(path, "r", encoding="utf-8") as fh:
        return ensure_state_shape(json.load(fh))


def save_state(path, state):
    state["updated_at"] = now_iso()
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
        fh.write(u"\n")


def as_float(value):
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("not a number: {}".format(value))


def pose_complete(pose):
    return all(pose.get(k) is not None for k in ("x", "y", "z", "yaw"))


def relative_pose(pose, origin):
    return {
        "x": round(pose["x"] - float(origin.get("x", 0.0)), 4),
        "y": round(pose["y"] - float(origin.get("y", 0.0)), 4),
        "z": round(pose["z"] - float(origin.get("z", 0.0)), 4),
        "yaw": round(pose["yaw"] - float(origin.get("yaw", 0.0)), 4),
    }


def parse_pose(args, default_yaw):
    if len(args) not in (3, 4):
        raise ValueError("record needs: x y z [yaw]")
    x, y, z = (as_float(v) for v in args[:3])
    if len(args) == 4:
        yaw = as_float(args[3])
    elif default_yaw is not None:
        yaw = float(default_yaw)
    else:
        raise ValueError("yaw is required unless `default-yaw <value>` is set")
    return {"x": x, "y": y, "z": z, "yaw": yaw}


def find_slot_index(slots, tokens):
    if not tokens:
        raise ValueError("goto needs a slot id, e.g. A_N_01, or brick orientation index")

    first = tokens[0].upper().replace("'", "_F")
    for i, slot in enumerate(slots):
        if slot["id"].upper() == first:
            return i

    if len(tokens) < 3:
        raise ValueError("goto examples: goto A normal 3 | goto B flipped 2 | goto A_N_01")
    brick = tokens[0].upper()
    orientation = tokens[1].lower()
    if orientation in ("n", "normal", "orig", "original"):
        orientation = "normal"
    elif orientation in ("f", "flip", "flipped", "reverse"):
        orientation = "flipped"
    else:
        raise ValueError("orientation must be normal/flipped")
    index = int(tokens[2])
    for i, slot in enumerate(slots):
        if slot["brick"] == brick and slot["orientation"] == orientation and slot["slot_index"] == index:
            return i
    raise ValueError("slot not found")


def first_missing_index(slots):
    for i, slot in enumerate(slots):
        if not slot.get("recorded"):
            return i
    return 0


def current_label(slot):
    orient = "normal" if slot["orientation"] == "normal" else "flipped"
    return "{}  brick={}  orientation={}  slot={}".format(
        slot["id"], slot["brick_key"], orient, slot["slot_index"]
    )


def print_current(slot, cursor, total):
    status = "recorded" if slot.get("recorded") else "missing"
    print("\n[{}/{}] {}  ({})".format(cursor + 1, total, current_label(slot), status))
    print("pose: {}".format(slot.get("pose", blank_pose())))
    if slot.get("recorded"):
        print("relative: {}".format(slot.get("relative_pose", blank_pose())))
    if slot.get("note"):
        print("note: {}".format(slot["note"]))


def print_help():
    print(
        """
Commands:
  record x y z yaw       save current slot, then advance
  record x y z           save using default yaw if set
  default-yaw yaw        set yaw used by later 3-number records
  origin x y z [yaw]     set measurement origin; relative poses update on save
  gripper open close     store pwm_servo2 open/close angles
  safe-z z               store global safe travel z
  note text...           attach note to current slot
  delete                 clear current slot
  next / prev            move cursor
  goto A normal 3        jump to a slot
  goto A_F_03            jump by slot id
  missing                list missing slots
  list [A|B|C|D|E|F]     list recorded/missing slots
  save                   write JSON now
  help                   show this help
  quit                   save and exit
"""
    )


def summarize(slots):
    all_slots = list(slots)
    done = sum(1 for slot in all_slots if slot.get("recorded"))
    return done, len(all_slots)


def update_all_relative(state):
    origin = state.get("origin", {})
    for slot in state["slots"]:
        pose = slot.get("pose", {})
        if pose_complete(pose):
            slot["relative_pose"] = relative_pose(pose, origin)


def prompt_text(text):
    try:
        return raw_input(text)
    except NameError:
        return input(text)


def run(path, autosave):
    state = load_state(path)
    slots = state["slots"]
    cursor = first_missing_index(slots)
    print("Brick library probe recorder -> {}".format(path))
    print("Total slots: {} (A-C: 6 each orientation, D-F: 5 each orientation)".format(len(slots)))
    print_help()
    print_current(slots[cursor], cursor, len(slots))

    while True:
        done, total = summarize(slots)
        raw = prompt_text("\n{} [{}/{}]> ".format(slots[cursor]["id"], done, total)).strip()
        if not raw:
            continue
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            print(f"parse error: {exc}")
            continue
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("help", "h", "?"):
                print_help()
            elif cmd in ("quit", "q", "exit"):
                update_all_relative(state)
                save_state(path, state)
                print("saved {}".format(path))
                break
            elif cmd in ("save", "write", "w"):
                update_all_relative(state)
                save_state(path, state)
                print("saved {}".format(path))
            elif cmd in ("next", "n"):
                cursor = min(len(slots) - 1, cursor + 1)
                print_current(slots[cursor], cursor, len(slots))
            elif cmd in ("prev", "back", "p"):
                cursor = max(0, cursor - 1)
                print_current(slots[cursor], cursor, len(slots))
            elif cmd == "goto":
                cursor = find_slot_index(slots, args)
                print_current(slots[cursor], cursor, len(slots))
            elif cmd == "record":
                pose = parse_pose(args, state.get("gripper", {}).get("default_yaw"))
                slot = slots[cursor]
                slot["pose"] = pose
                slot["relative_pose"] = relative_pose(pose, state["origin"])
                slot["recorded"] = True
                slot["recorded_at"] = now_iso()
                print("recorded {} -> {}".format(slot["id"], pose))
                if autosave:
                    save_state(path, state)
                if cursor < len(slots) - 1:
                    cursor += 1
                    print_current(slots[cursor], cursor, len(slots))
            elif cmd == "delete":
                slot = slots[cursor]
                slot["pose"] = blank_pose()
                slot["relative_pose"] = blank_pose()
                slot["recorded"] = False
                slot["recorded_at"] = None
                print("cleared {}".format(slot["id"]))
                if autosave:
                    save_state(path, state)
            elif cmd == "origin":
                if len(args) not in (3, 4):
                    raise ValueError("origin needs: x y z [yaw]")
                origin = {
                    "x": as_float(args[0]),
                    "y": as_float(args[1]),
                    "z": as_float(args[2]),
                    "yaw": as_float(args[3]) if len(args) == 4 else float(state["origin"].get("yaw", 0.0)),
                }
                state["origin"] = origin
                update_all_relative(state)
                print("origin set to {}".format(origin))
                if autosave:
                    save_state(path, state)
            elif cmd in ("default-yaw", "yaw"):
                if len(args) != 1:
                    raise ValueError("default-yaw needs one number")
                state["gripper"]["default_yaw"] = as_float(args[0])
                print("default yaw = {}".format(state["gripper"]["default_yaw"]))
                if autosave:
                    save_state(path, state)
            elif cmd == "gripper":
                if len(args) != 2:
                    raise ValueError("gripper needs: open close")
                state["gripper"]["open_angle"] = as_float(args[0])
                state["gripper"]["close_angle"] = as_float(args[1])
                print("gripper open/close = {} / {}".format(args[0], args[1]))
                if autosave:
                    save_state(path, state)
            elif cmd == "safe-z":
                if len(args) != 1:
                    raise ValueError("safe-z needs one number")
                state["gripper"]["safe_z"] = as_float(args[0])
                print("safe_z = {}".format(state["gripper"]["safe_z"]))
                if autosave:
                    save_state(path, state)
            elif cmd == "note":
                slots[cursor]["note"] = " ".join(args)
                print("note saved for {}".format(slots[cursor]["id"]))
                if autosave:
                    save_state(path, state)
            elif cmd == "missing":
                missing = [slot["id"] for slot in slots if not slot.get("recorded")]
                print("missing {}:".format(len(missing)))
                print(", ".join(missing) if missing else "none")
            elif cmd == "list":
                brick_filter = args[0].upper() if args else None
                for i, slot in enumerate(slots):
                    if brick_filter and slot["brick"] != brick_filter:
                        continue
                    mark = "ok" if slot.get("recorded") else "--"
                    print("{:02d}. {} {}".format(i + 1, mark, current_label(slot)))
            elif cmd in ("current", "c"):
                print_current(slots[cursor], cursor, len(slots))
            else:
                print("unknown command: {}; type `help`".format(cmd))
        except (ValueError, IndexError) as exc:
            print("error: {}".format(exc))


def parse_args():
    parser = argparse.ArgumentParser(description="Record fixed pickup poses for the bridge brick library.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output JSON path (default: {})".format(DEFAULT_OUT))
    parser.add_argument("--no-autosave", action="store_true", help="only save when `save` or `quit` is entered")
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.out, autosave=not args.no_autosave)


if __name__ == "__main__":
    main()
