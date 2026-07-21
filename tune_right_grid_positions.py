#!/usr/bin/env python3
"""
Interactive tuner for the 12 right-zone pickup poses.

It visits each generated right-grid point, lets you jog x/y/z/yaw/gripper like
probe_position.py, and saves the adjusted poses to a JSON file.

Run on JetMax:
    rosrun jetmax_demos tune_right_grid_positions.py

Keys:
    w/s : y +/-
    d/a : x +/-
    r/f : z +/-
    j/l : yaw -/+
    o/c : gripper open/close
    [/] : step -/+
    p   : print current pose
    m   : move to current saved/generated point
    n   : save current pose and go next
    b   : save current pose and go previous
    q   : save file and quit
"""

from __future__ import print_function

import json
import math
import os
import select
import sys
import termios
import tty
from datetime import datetime

import rospy
import hiwonder


OUT_PATH = os.path.join("calibration", "right_grid_tuned.json")
ROS_NODE_NAME = "tune_right_grid_positions"

SAFE_Z = 180.0
APPROACH_LIFT = 35.0
STEP = 5.0
ANGLE_STEP = 5.0
GRIPPER_STEP = 5.0

HOME_X = 0.0
HOME_Y = -162.9
HOME_Z = 212.8
HOME_YAW = 50.0
HOME_GRIPPER = 55.0

PICK_OPEN_ANGLE = 100.0

ROWS = 3
COLS = 4

RIGHT_ZONE_LEFT_BOTTOM = (125.0, -137.9, 97.8, 90.0)
RIGHT_ZONE_RIGHT_TOP = (255.0, 27.1, 102.8, 145.0)
RIGHT_ZONE_RIGHT_BOTTOM_OVERRIDE = (250.0, -132.9, 107.8, 110.0)


jetmax = hiwonder.JetMax()
current_yaw = HOME_YAW
current_gripper = HOME_GRIPPER
step = STEP


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def lerp(a, b, t):
    return a + (b - a) * t


def build_grid():
    points = []
    lb = RIGHT_ZONE_LEFT_BOTTOM
    rt = RIGHT_ZONE_RIGHT_TOP
    for row in range(ROWS):
        v = float(row) / float(ROWS - 1)
        for col in range(COLS):
            u = float(col) / float(COLS - 1)
            x = lerp(lb[0], rt[0], u)
            y = lerp(lb[1], rt[1], v)
            diagonal_t = (u + v) * 0.5
            z = lerp(lb[2], rt[2], diagonal_t)
            yaw = lerp(lb[3], rt[3], diagonal_t)
            if row == 0 and col == COLS - 1:
                x, y, z, yaw = RIGHT_ZONE_RIGHT_BOTTOM_OVERRIDE
            points.append({
                "id": "r{}c{}".format(row, col),
                "row": row,
                "col": col,
                "generated": {"x": x, "y": y, "z": z, "yaw": yaw},
                "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                "gripper": PICK_OPEN_ANGLE,
                "updated_at": None,
            })
    return points


def load_state():
    state = {
        "version": 1,
        "updated_at": now_iso(),
        "notes": "Tuned right-zone pickup poses. Units are mm/degrees.",
        "points": build_grid(),
    }
    if not os.path.exists(OUT_PATH):
        return state
    with open(OUT_PATH, "r") as fh:
        old = json.load(fh)
    old_points = {p["id"]: p for p in old.get("points", [])}
    for p in state["points"]:
        if p["id"] in old_points:
            p.update(old_points[p["id"]])
    state.update({k: v for k, v in old.items() if k != "points"})
    state["points"] = state["points"]
    return state


def save_state(state):
    state["updated_at"] = now_iso()
    folder = os.path.dirname(OUT_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with open(OUT_PATH, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")


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


def approach_z(work_z):
    return min(SAFE_Z, work_z + APPROACH_LIFT)


def travel_time(from_xyz, to_xyz, min_time=0.35):
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return max(dist / 150.0, min_time)


def move_to_pose(point):
    pose = point["pose"]
    x, y, z, yaw = pose["x"], pose["y"], pose["z"], pose["yaw"]
    local_safe_z = approach_z(z)
    print("\nmove to {}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} approach_z={:.1f}".format(
        point["id"], x, y, z, yaw, local_safe_z
    ))
    set_gripper(point.get("gripper", PICK_OPEN_ANGLE), 0.3)
    set_yaw(yaw, 0.3)
    cur = jetmax.position
    move_xyz(x, y, local_safe_z, travel_time(cur, (x, y, local_safe_z)))
    move_xyz(x, y, z, 0.5)


def print_pose(point):
    x, y, z = jetmax.position
    print("current {}  position=({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f} step={:.1f}".format(
        point["id"], x, y, z, current_yaw, current_gripper, step
    ))


def save_current_point(state, index):
    point = state["points"][index]
    x, y, z = jetmax.position
    point["pose"] = {
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "z": round(float(z), 3),
        "yaw": round(float(current_yaw), 3),
    }
    point["gripper"] = round(float(current_gripper), 3)
    point["updated_at"] = now_iso()
    save_state(state)
    print("saved {} -> position=({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        point["id"], x, y, z, current_yaw, current_gripper
    ))


def reset_home():
    print("reset home")
    set_yaw(HOME_YAW, 0.3)
    set_gripper(HOME_GRIPPER, 0.3)
    move_xyz(HOME_X, HOME_Y, HOME_Z, 1.2)


def print_help():
    print(__doc__)


def main():
    global step
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO, anonymous=True)
    state = load_state()
    index = 0
    print_help()
    reset_home()
    move_to_pose(state["points"][index])
    print_pose(state["points"][index])

    while not rospy.is_shutdown():
        c = getch(0.5)
        if c is None:
            continue
        point = state["points"][index]
        if c == "q":
            save_current_point(state, index)
            reset_home()
            print("saved {}".format(OUT_PATH))
            break
        elif c == "m":
            move_to_pose(point)
        elif c == "p":
            print_pose(point)
        elif c == "h":
            reset_home()
        elif c == "[":
            step = max(0.5, step - 0.5)
            print("step = {:.1f}".format(step))
        elif c == "]":
            step += 0.5
            print("step = {:.1f}".format(step))
        elif c in ("w", "s", "a", "d", "r", "f"):
            x, y, z = jetmax.position
            if c == "w":
                y += step
            elif c == "s":
                y -= step
            elif c == "d":
                x += step
            elif c == "a":
                x -= step
            elif c == "r":
                z += step
            elif c == "f":
                z -= step
            move_xyz(x, y, z, 0.2)
            print_pose(point)
        elif c == "j":
            set_yaw(current_yaw - ANGLE_STEP, 0.2)
            print_pose(point)
        elif c == "l":
            set_yaw(current_yaw + ANGLE_STEP, 0.2)
            print_pose(point)
        elif c == "o":
            set_gripper(current_gripper + GRIPPER_STEP, 0.2)
            print_pose(point)
        elif c == "c":
            set_gripper(current_gripper - GRIPPER_STEP, 0.2)
            print_pose(point)
        elif c == "n":
            save_current_point(state, index)
            index = min(len(state["points"]) - 1, index + 1)
            reset_home()
            move_to_pose(state["points"][index])
            print_pose(state["points"][index])
        elif c == "b":
            save_current_point(state, index)
            index = max(0, index - 1)
            reset_home()
            move_to_pose(state["points"][index])
            print_pose(state["points"][index])


if __name__ == "__main__":
    main()
