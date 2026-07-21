#!/usr/bin/env python3
"""
Measure real-world placement poses for aggressive_001.

Workflow for each of the 19 digital-twin steps:
  1. Robot picks the required brick from the measured left/right supply grids.
  2. Robot returns to home while holding the brick.
  3. You jog the arm to the correct physical placement pose.
  4. Press `n` to save, release the brick, return home, and continue.

Run on JetMax:
    rosrun jetmax_demos tune_aggressive001_place_poses.py

Keys while tuning a placement:
    w/s : y +/-
    d/a : x +/-
    r/f : z +/-
    j/l : yaw -/+
    o/c : gripper open/close
    [/] or [/ : move step -/+   minimum 0.2mm; / also increases step
    ,/. : angle step -/+  minimum 1deg
    p   : print current pose
    h   : return home while holding
    x   : redo current step: home, release/refill, re-pick, tune again
    n   : save placement, release brick, return home, next step
    b   : previous step
    q   : save and quit
"""

from __future__ import print_function

import json
import math
import os
import select
import sys
import termios
import tty
import argparse
from datetime import datetime


ROS_NODE_NAME = "tune_aggressive001_place_poses"
OUT_PATH = os.path.join("calibration", "aggressive_001_place_poses.json")

PICK_OPEN_ANGLE = 100.0
CLOSE_ANGLE = 75.0
RELEASE_ANGLE = 100.0
HOME_GRIPPER = 55.0

HOME_X = 0.0
HOME_Y = -162.9
HOME_Z = 212.8
HOME_YAW = 50.0

SAFE_Z = 180.0
APPROACH_LIFT = 35.0
MOVE_STEP = 5.0
ANGLE_STEP = 5.0
MIN_MOVE_STEP = 0.2
MIN_ANGLE_STEP = 1.0
GRIPPER_STEP = 5.0
GRIP_HOLD_SECONDS = 1.0
RELEASE_HOLD_SECONDS = 1.0


# Digital-twin aggressive_001 sequence. Odd steps were human in the interaction
# demo, even steps were robot, but this measurement script can record all 19.
AGGRESSIVE_001 = [
    {"step": 1, "brick": "D", "side": "L", "wx": 14.0, "wz": 1.5642247142857144},
    {"step": 2, "brick": "D", "side": "L", "wx": 5.0, "wz": 1.5642247142857144},
    {"step": 3, "brick": "C", "side": "R", "wx": 47.0, "wz": 1.5642247142857144},
    {"step": 4, "brick": "E'", "side": "R", "wx": 37.5, "wz": 1.5642247142857144},
    {"step": 5, "brick": "E", "side": "L", "wx": 15.0, "wz": 2.8014038625491984},
    {"step": 6, "brick": "B", "side": "L", "wx": 8.0, "wz": 2.8014038625491984},
    {"step": 7, "brick": "F", "side": "L", "wx": 8.0, "wz": 4.038583010812682},
    {"step": 8, "brick": "F'", "side": "L", "wx": 15.5, "wz": 4.657172584944424},
    {"step": 9, "brick": "C", "side": "L", "wx": 12.5, "wz": 5.894351733207907},
    {"step": 10, "brick": "B", "side": "R", "wx": 45.5, "wz": 3.4199934366809397},
    {"step": 11, "brick": "D", "side": "R", "wx": 36.5, "wz": 3.4199934366809397},
    {"step": 12, "brick": "D", "side": "R", "wx": 45.5, "wz": 4.657172584944423},
    {"step": 13, "brick": "D", "side": "L", "wx": 20.0, "wz": 6.5129413073396485},
    {"step": 14, "brick": "F'", "side": "R", "wx": 33.5, "wz": 4.657172584944424},
    {"step": 15, "brick": "E'", "side": "R", "wx": 42.0, "wz": 5.894351733207907},
    {"step": 16, "brick": "E'", "side": "L", "wx": 15.0, "wz": 7.131530881471391},
    {"step": 17, "brick": "D", "side": "R", "wx": 32.0, "wz": 6.512941307339649},
    {"step": 18, "brick": "B", "side": "R", "wx": 38.0, "wz": 7.750120455603134},
    {"step": 19, "brick": "C'", "side": "L", "wx": 27.5, "wz": 7.131530881471391},
]


LEFT_GRID = {
    "r0c0": ( -250.0, -107.9,  97.8, 160.0),
    "r0c1": ( -210.0, -109.6,  99.5, 166.7),
    "r0c2": ( -170.0, -111.2, 101.1, 173.3),
    "r0c3": ( -130.0, -112.9, 102.8, 180.0),
    "r1c0": ( -247.5,  -22.9, 100.3, 142.5),
    "r1c1": ( -206.7,  -25.4, 102.0, 145.8),
    "r1c2": ( -165.8,  -27.9, 103.6, 149.2),
    "r1c3": ( -125.0,  -30.4, 105.3, 152.5),
    "r2c0": ( -245.0,   62.1, 102.8, 125.0),
    "r2c1": ( -203.3,   58.8, 104.5, 125.0),
    "r2c2": ( -161.7,   55.4, 106.1, 125.0),
    "r2c3": ( -120.0,   52.1, 107.8, 125.0),
}


RIGHT_GRID = {
    "r0c0": (125.0, -137.9,  97.8,  90.0),
    "r0c1": (168.333, -137.9,  98.633,  99.167),
    "r0c2": (211.667, -137.9,  99.467, 108.333),
    "r0c3": (250.0, -132.9,  97.8, 110.0),
    "r1c0": (125.0,  -55.4,  99.05, 113.75),
    "r1c1": (168.333,  -55.4,  99.883, 117.917),
    "r1c2": (211.667,  -55.4, 100.717, 122.083),
    "r1c3": (255.0,  -55.4, 101.55, 126.25),
    "r2c0": (115.0,   27.1, 100.3, 147.5),
    "r2c1": (163.333,   22.1, 101.133, 146.667),
    "r2c2": (216.667,   22.1, 101.967, 145.833),
    # r2c3 tune record was home; keep generated right-top value.
    "r2c3": (255.0,   27.1, 102.8, 145.0),
}


SOURCE_POOLS = {
    "A":  [("L", "r0c0"), ("L", "r0c1")],
    "A'": [("L", "r0c2"), ("L", "r0c3")],
    "C":  [("L", "r1c0"), ("L", "r1c1")],
    "C'": [("L", "r1c2"), ("L", "r1c3")],
    "F":  [("L", "r2c0"), ("L", "r2c1")],
    "F'": [("L", "r2c2"), ("L", "r2c3")],
    "B":  [("R", "r0c0"), ("R", "r0c1")],
    "B'": [("R", "r0c2"), ("R", "r0c3")],
    # D is symmetric in the digital model; all four D-row slots are usable.
    "D":  [("R", "r1c0"), ("R", "r1c1"), ("R", "r1c2"), ("R", "r1c3")],
    "D'": [("R", "r1c0"), ("R", "r1c1"), ("R", "r1c2"), ("R", "r1c3")],
    "E":  [("R", "r2c0"), ("R", "r2c1")],
    "E'": [("R", "r2c2"), ("R", "r2c3")],
}


rospy = None
hiwonder = None
jetmax = None
current_yaw = HOME_YAW
current_gripper = HOME_GRIPPER
move_step = MOVE_STEP
angle_step = ANGLE_STEP


def init_robot():
    global rospy, hiwonder, jetmax
    import rospy as rospy_module
    import hiwonder as hiwonder_module

    rospy = rospy_module
    hiwonder = hiwonder_module
    jetmax = hiwonder.JetMax()


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def getch(timeout=0.5):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def approach_z(work_z):
    return min(SAFE_Z, work_z + APPROACH_LIFT)


def travel_time(from_xyz, to_xyz, min_time=0.35):
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return max(dist / 150.0, min_time)


def set_yaw(yaw, duration=0.25):
    global current_yaw
    current_yaw = max(0.0, min(180.0, float(yaw)))
    hiwonder.pwm_servo1.set_position(int(round(current_yaw)), duration)
    rospy.sleep(duration + 0.05)


def set_gripper(angle, duration=0.25):
    global current_gripper
    current_gripper = max(0.0, min(180.0, float(angle)))
    hiwonder.pwm_servo2.set_position(int(round(current_gripper)), duration)
    rospy.sleep(duration + 0.05)


def move_xyz(x, y, z, duration=0.35):
    jetmax.set_position((x, y, z), duration)
    rospy.sleep(duration + 0.08)


def reset_home(holding=False):
    print("reset home holding={}".format(holding))
    set_yaw(HOME_YAW, 0.3)
    if not holding:
        set_gripper(HOME_GRIPPER, 0.3)
    move_xyz(HOME_X, HOME_Y, HOME_Z, 1.2)


def source_pose(source_side, grid_id):
    grid = LEFT_GRID if source_side == "L" else RIGHT_GRID
    return grid[grid_id]


def source_for_step(move, use_counts):
    brick = move["brick"]
    pool = SOURCE_POOLS[brick]
    used = use_counts.get(brick, 0)
    source_side, grid_id = pool[used % len(pool)]
    use_counts[brick] = used + 1
    refill = used >= len(pool)
    return source_side, grid_id, refill


def load_state():
    steps = []
    use_counts = {}
    for move in AGGRESSIVE_001:
        source_side, grid_id, refill = source_for_step(move, use_counts)
        x, y, z, yaw = source_pose(source_side, grid_id)
        record = dict(move)
        record.update({
            "operator": "human" if move["step"] % 2 == 1 else "robot",
            "source_side": source_side,
            "source_grid_id": grid_id,
            "source_pose": {"x": x, "y": y, "z": z, "yaw": yaw},
            "source_refill_required": refill,
            "place_pose": None,
            "place_gripper": None,
            "updated_at": None,
        })
        steps.append(record)

    state = {
        "version": 1,
        "updated_at": now_iso(),
        "notes": "Measured real-world placement poses for aggressive_001.",
        "steps": steps,
    }
    if not os.path.exists(OUT_PATH):
        return state
    with open(OUT_PATH, "r") as fh:
        old = json.load(fh)
    old_steps = {s["step"]: s for s in old.get("steps", [])}
    for step in state["steps"]:
        if step["step"] in old_steps:
            old_step = old_steps[step["step"]]
            for key in ("place_pose", "place_gripper", "updated_at", "note"):
                if key in old_step:
                    step[key] = old_step[key]
    return state


def save_state(state):
    state["updated_at"] = now_iso()
    folder = os.path.dirname(OUT_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with open(OUT_PATH, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")


def first_unmeasured_index(state):
    for i, step in enumerate(state["steps"]):
        if not step.get("place_pose"):
            return i
    return 0


def print_step(step):
    refill = " REFILL SOURCE SLOT" if step.get("source_refill_required") else ""
    print("\nStep {step:02d} [{operator}] brick={brick} side={side} wx={wx:.3f} wz={wz:.3f}".format(**step))
    print("source {} {} pose={}{}".format(
        step["source_side"], step["source_grid_id"], step["source_pose"], refill
    ))
    if step.get("place_pose"):
        print("saved place_pose = {}".format(step["place_pose"]))


def pick_source(step):
    pose = step["source_pose"]
    x, y, z, yaw = pose["x"], pose["y"], pose["z"], pose["yaw"]
    local_safe_z = approach_z(z)
    print("\nPICK step {step:02d} {brick} from {source_side} {source_grid_id}".format(**step))
    print("pick pose ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} approach_z={:.1f}".format(x, y, z, yaw, local_safe_z))
    if step.get("source_refill_required"):
        input("This brick type reuses a source slot. Refill {} {} with {} then press Enter...".format(
            step["source_side"], step["source_grid_id"], step["brick"]
        ))
    set_gripper(PICK_OPEN_ANGLE, 0.4)
    set_yaw(yaw, 0.3)
    cur = jetmax.position
    move_xyz(x, y, local_safe_z, travel_time(cur, (x, y, local_safe_z)))
    move_xyz(x, y, z, 0.6)
    set_gripper(CLOSE_ANGLE, 0.5)
    print("hold grip for {:.1f}s".format(GRIP_HOLD_SECONDS))
    rospy.sleep(GRIP_HOLD_SECONDS)
    move_xyz(x, y, local_safe_z, 0.7)
    reset_home(holding=True)


def print_current_pose(step):
    x, y, z = jetmax.position
    print("current step {step:02d}: position=({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f} move_step={:.1f} angle_step={:.1f}".format(
        x, y, z, current_yaw, current_gripper, move_step, angle_step, **step
    ))


def save_current_place(state, index):
    step = state["steps"][index]
    x, y, z = jetmax.position
    step["place_pose"] = {
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "z": round(float(z), 3),
        "yaw": round(float(current_yaw), 3),
    }
    step["place_gripper"] = round(float(current_gripper), 3)
    step["updated_at"] = now_iso()
    save_state(state)
    print("saved step {step:02d} place_pose={place_pose}".format(**step))


def release_and_home():
    x, y, z = jetmax.position
    local_safe_z = approach_z(z)
    print("release brick, hold {:.1f}s, lift to {:.1f}, home".format(RELEASE_HOLD_SECONDS, local_safe_z))
    set_gripper(RELEASE_ANGLE, 0.6)
    rospy.sleep(RELEASE_HOLD_SECONDS)
    move_xyz(x, y, local_safe_z, 0.7)
    reset_home(holding=False)


def redo_current_step(step):
    print("redo step {step:02d}: return home, release/refill, re-pick".format(**step))
    reset_home(holding=True)
    input("At home while holding. Take/remove the brick safely, refill source if needed, then press Enter to open gripper...")
    set_gripper(RELEASE_ANGLE, 0.6)
    rospy.sleep(RELEASE_HOLD_SECONDS)
    reset_home(holding=False)
    input("Confirm source slot {} {} contains brick {}, then press Enter to re-pick...".format(
        step["source_side"], step["source_grid_id"], step["brick"]
    ))
    pick_source(step)
    print("Redo pick complete. Jog to the correct placement pose, then press n to save.")
    print_current_pose(step)


def print_help():
    print(__doc__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune aggressive_001 real-world placement poses.")
    parser.add_argument("--start-step", type=int, default=None, help="start from this aggressive_001 step number")
    parser.add_argument("--redo-existing", action="store_true", help="ignore existing saved place poses when choosing start step")
    return parser.parse_args()


def main():
    global move_step, angle_step
    args = parse_args()
    init_robot()
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO, anonymous=True)
    state = load_state()
    if args.start_step is not None:
        index = max(0, min(len(state["steps"]) - 1, args.start_step - 1))
        if args.redo_existing:
            for step in state["steps"][index:]:
                step["place_pose"] = None
                step["place_gripper"] = None
                step["updated_at"] = None
            save_state(state)
    else:
        index = 0 if args.redo_existing else first_unmeasured_index(state)
    print_help()
    reset_home(holding=False)

    while not rospy.is_shutdown():
        step = state["steps"][index]
        print_step(step)
        input("Press Enter to pick this brick and start placement tuning...")
        pick_source(step)
        print("Jog to the correct placement pose, then press n to save/release/next.")
        print_current_pose(step)

        while not rospy.is_shutdown():
            c = getch(0.5)
            if c is None:
                continue
            if c == "q":
                save_current_place(state, index)
                save_state(state)
                print("saved {}".format(OUT_PATH))
                return
            elif c == "p":
                print_current_pose(step)
            elif c == "h":
                reset_home(holding=True)
                print_current_pose(step)
            elif c == "[":
                if move_step > 1.0:
                    move_step = max(1.0, move_step - 1.0)
                else:
                    move_step = max(MIN_MOVE_STEP, move_step - 0.2)
                print("move_step = {:.1f}".format(move_step))
            elif c == "]" or c == "/":
                if move_step < 1.0:
                    move_step = min(1.0, move_step + 0.2)
                else:
                    move_step += 1.0
                print("move_step = {:.1f}".format(move_step))
            elif c == ",":
                angle_step = max(MIN_ANGLE_STEP, angle_step - 1.0)
                print("angle_step = {:.1f}".format(angle_step))
            elif c == ".":
                angle_step += 1.0
                print("angle_step = {:.1f}".format(angle_step))
            elif c in ("w", "s", "a", "d", "r", "f"):
                x, y, z = jetmax.position
                if c == "w":
                    y += move_step
                elif c == "s":
                    y -= move_step
                elif c == "d":
                    x += move_step
                elif c == "a":
                    x -= move_step
                elif c == "r":
                    z += move_step
                elif c == "f":
                    z -= move_step
                move_xyz(x, y, z, 0.2)
                print_current_pose(step)
            elif c == "j":
                set_yaw(current_yaw - angle_step, 0.2)
                print_current_pose(step)
            elif c == "l":
                set_yaw(current_yaw + angle_step, 0.2)
                print_current_pose(step)
            elif c == "o":
                set_gripper(current_gripper + GRIPPER_STEP, 0.2)
                print_current_pose(step)
            elif c == "c":
                set_gripper(current_gripper - GRIPPER_STEP, 0.2)
                print_current_pose(step)
            elif c == "x":
                redo_current_step(step)
            elif c == "b":
                save_current_place(state, index)
                release_and_home()
                index = max(0, index - 1)
                break
            elif c == "n":
                save_current_place(state, index)
                release_and_home()
                if index >= len(state["steps"]) - 1:
                    print("All 19 steps measured. saved {}".format(OUT_PATH))
                    return
                index += 1
                break


if __name__ == "__main__":
    main()
