#!/usr/bin/env python3
"""
Run a fully robotic physical test of aggressive_001.

Placement poses:
  - steps 1-6 use measured real-world poses supplied by the user
  - steps 7-19 are predicted from a rough affine fit from digital wx/wz

This is a first-pass smoke test, not a final calibrated build.

Run on JetMax:
    rosrun jetmax_demos run_aggressive001_physical_demo.py --step-confirm
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys


ROS_NODE_NAME = "run_aggressive001_physical_demo"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLACE_POSE_JSON = os.path.join(SCRIPT_DIR, "calibration", "aggressive_001_place_poses.json")

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
GRIP_HOLD_SECONDS = 1.0
RELEASE_HOLD_SECONDS = 1.0
HOME_HOLD_SECONDS = 0.8


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


MEASURED_PLACE = {
    1: (-100.0, -202.9, 112.8, 115.0),
    2: (-155.0, -207.9, 112.8, 105.0),
    3: (125.0, -202.9, 112.8, 165.0),
    4: (80.0, -212.9, 117.8, 155.0),
    5: (-95.0, -207.9, 117.8, 115.0),
    6: (-160.0, -202.9, 127.8, 105.0),
}

# Least-squares fit from the six measured steps:
# value = a*wx + b*wz + c
FIT = {
    "x": (6.951329, -12.184690, -173.306046),
    "y": (-0.024837, 0.721772, -207.136343),
    "z": (0.019553, 7.299730, 102.125651),
    "yaw": (1.492364, -2.867225, 100.870071),
}


LEFT_GRID = {
    "r0c0": (-250.0, -107.9, 97.8, 160.0), "r0c1": (-210.0, -109.6, 99.5, 166.7),
    "r0c2": (-170.0, -111.2, 101.1, 173.3), "r0c3": (-130.0, -112.9, 102.8, 180.0),
    "r1c0": (-247.5, -22.9, 100.3, 142.5), "r1c1": (-206.7, -25.4, 102.0, 145.8),
    "r1c2": (-165.8, -27.9, 103.6, 149.2), "r1c3": (-125.0, -30.4, 105.3, 152.5),
    "r2c0": (-245.0, 62.1, 102.8, 125.0), "r2c1": (-203.3, 58.8, 104.5, 125.0),
    "r2c2": (-161.7, 55.4, 106.1, 125.0), "r2c3": (-120.0, 52.1, 107.8, 125.0),
}

RIGHT_GRID = {
    "r0c0": (125.0, -137.9, 97.8, 90.0), "r0c1": (168.333, -137.9, 98.633, 99.167),
    "r0c2": (211.667, -137.9, 99.467, 108.333), "r0c3": (250.0, -132.9, 97.8, 110.0),
    "r1c0": (125.0, -55.4, 99.05, 113.75), "r1c1": (168.333, -55.4, 99.883, 117.917),
    "r1c2": (211.667, -55.4, 100.717, 122.083), "r1c3": (255.0, -55.4, 101.55, 126.25),
    "r2c0": (115.0, 27.1, 100.3, 147.5), "r2c1": (163.333, 22.1, 101.133, 146.667),
    "r2c2": (216.667, 22.1, 101.967, 145.833), "r2c3": (255.0, 27.1, 102.8, 145.0),
}

SOURCE_POOLS = {
    "A": [("L", "r0c0"), ("L", "r0c1")], "A'": [("L", "r0c2"), ("L", "r0c3")],
    "C": [("L", "r1c0"), ("L", "r1c1")], "C'": [("L", "r1c2"), ("L", "r1c3")],
    "F": [("L", "r2c0"), ("L", "r2c1")], "F'": [("L", "r2c2"), ("L", "r2c3")],
    "B": [("R", "r0c0"), ("R", "r0c1")], "B'": [("R", "r0c2"), ("R", "r0c3")],
    "D": [("R", "r1c0"), ("R", "r1c1"), ("R", "r1c2"), ("R", "r1c3")],
    "D'": [("R", "r1c0"), ("R", "r1c1"), ("R", "r1c2"), ("R", "r1c3")],
    "E": [("R", "r2c0"), ("R", "r2c1")], "E'": [("R", "r2c2"), ("R", "r2c3")],
}

rospy = None
hiwonder = None
jetmax = None


def init_robot():
    global rospy, hiwonder, jetmax
    import rospy as rospy_module
    import hiwonder as hiwonder_module
    rospy = rospy_module
    hiwonder = hiwonder_module
    jetmax = hiwonder.JetMax()


def fit_value(name, wx, wz):
    a, b, c = FIT[name]
    return a * wx + b * wz + c


def place_pose(move):
    if move["step"] in MEASURED_PLACE:
        return MEASURED_PLACE[move["step"]], "measured"
    wx, wz = move["wx"], move["wz"]
    return (
        fit_value("x", wx, wz),
        fit_value("y", wx, wz),
        fit_value("z", wx, wz),
        fit_value("yaw", wx, wz),
    ), "fit"


def tuple_from_pose_dict(pose):
    return (float(pose["x"]), float(pose["y"]), float(pose["z"]), float(pose["yaw"]))


def load_json_plan(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as fh:
        data = json.load(fh)
    steps = data.get("steps", [])
    if len(steps) != 19:
        raise ValueError("expected 19 steps in {}, got {}".format(path, len(steps)))
    plan = []
    for step in sorted(steps, key=lambda s: int(s["step"])):
        if not step.get("place_pose"):
            raise ValueError("step {} has no place_pose in {}".format(step.get("step"), path))
        if not step.get("source_pose"):
            raise ValueError("step {} has no source_pose in {}".format(step.get("step"), path))
        item = {
            "step": int(step["step"]),
            "brick": step["brick"],
            "side": step.get("side", "?"),
            "wx": float(step.get("wx", 0.0)),
            "wz": float(step.get("wz", 0.0)),
            "source_side": step.get("source_side", "?"),
            "source_grid_id": step.get("source_grid_id", "?"),
            "source_pose": tuple_from_pose_dict(step["source_pose"]),
            "refill": bool(step.get("source_refill_required", False)),
            "place_pose": tuple_from_pose_dict(step["place_pose"]),
            "place_source": "json",
            "place_gripper": step.get("place_gripper"),
        }
        plan.append(item)
    return plan


def source_plan(pose_json_path=None):
    json_plan = load_json_plan(pose_json_path)
    if json_plan is not None:
        return json_plan

    counts = {}
    plan = []
    for move in AGGRESSIVE_001:
        brick = move["brick"]
        pool = SOURCE_POOLS[brick]
        used = counts.get(brick, 0)
        side, grid_id = pool[used % len(pool)]
        counts[brick] = used + 1
        pose = (LEFT_GRID if side == "L" else RIGHT_GRID)[grid_id]
        item = dict(move)
        item.update({"source_side": side, "source_grid_id": grid_id, "source_pose": pose, "refill": used >= len(pool)})
        item["place_pose"], item["place_source"] = place_pose(move)
        plan.append(item)
    return plan


def approach_z(z, args):
    return min(args.safe_z, z + args.approach_lift)


def travel_time(from_xyz, to_xyz, min_time=0.35):
    return max(math.hypot(to_xyz[0] - from_xyz[0], to_xyz[1] - from_xyz[1]) / 150.0, min_time)


def set_servo1(yaw, duration):
    hiwonder.pwm_servo1.set_position(int(round(max(0, min(180, yaw)))), duration)
    rospy.sleep(duration + 0.05)


def set_gripper(angle, duration):
    hiwonder.pwm_servo2.set_position(int(round(max(0, min(180, angle)))), duration)
    rospy.sleep(duration + 0.05)


def move_xyz(x, y, z, duration):
    jetmax.set_position((x, y, z), duration)
    rospy.sleep(duration + 0.08)


def reset_home(args, holding):
    print("HOME holding={}".format(holding))
    set_servo1(args.home_yaw, 0.3)
    if not holding:
        set_gripper(args.home_gripper, 0.3)
    move_xyz(args.home_x, args.home_y, args.home_z, 1.2)
    rospy.sleep(args.home_hold)


def move_to_work_pose(x, y, z, yaw, args, label):
    local_z = approach_z(z, args)
    print("{} ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} approach_z={:.1f}".format(label, x, y, z, yaw, local_z))
    set_servo1(yaw, 0.3)
    cur = jetmax.position
    move_xyz(x, y, local_z, travel_time(cur, (x, y, local_z)))
    move_xyz(x, y, z, 0.6)
    return local_z


def pick_brick(item, args):
    x, y, z, yaw = item["source_pose"]
    if item["refill"]:
        input("REFILL {} {} with {} then Enter...".format(item["source_side"], item["source_grid_id"], item["brick"]))
    set_gripper(args.pick_open, 0.4)
    local_z = move_to_work_pose(x, y, z, yaw, args, "PICK step {:02d} {} from {} {}".format(
        item["step"], item["brick"], item["source_side"], item["source_grid_id"]
    ))
    set_gripper(args.close, 0.5)
    rospy.sleep(args.grip_hold)
    move_xyz(x, y, local_z, 0.7)
    reset_home(args, holding=True)


def place_brick(item, args):
    x, y, z, yaw = item["place_pose"]
    local_z = move_to_work_pose(x, y, z, yaw, args, "PLACE step {:02d} {} [{}]".format(
        item["step"], item["brick"], item["place_source"]
    ))
    set_gripper(args.release, 0.6)
    rospy.sleep(args.release_hold)
    move_xyz(x, y, local_z, 0.7)
    reset_home(args, holding=False)


def print_plan(plan):
    print("Aggressive 001 physical demo plan:")
    for item in plan:
        px, py, pz, pyaw = item["place_pose"]
        print("{step:02d} {brick:>2} src={source_side}:{source_grid_id} place=({:.1f},{:.1f},{:.1f}) yaw={:.1f} {}{}".format(
            px, py, pz, pyaw, item["place_source"], " REFILL" if item["refill"] else "", **item
        ))


def parse_args():
    p = argparse.ArgumentParser(description="Run full-machine aggressive_001 physical smoke test.")
    p.add_argument("--pose-json", default=DEFAULT_PLACE_POSE_JSON, help="measured aggressive_001 placement JSON")
    p.add_argument("--start-step", type=int, default=1)
    p.add_argument("--end-step", type=int, default=19)
    p.add_argument("--step-confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--safe-z", type=float, default=SAFE_Z)
    p.add_argument("--approach-lift", type=float, default=APPROACH_LIFT)
    p.add_argument("--pick-open", type=float, default=PICK_OPEN_ANGLE)
    p.add_argument("--close", type=float, default=CLOSE_ANGLE)
    p.add_argument("--release", type=float, default=RELEASE_ANGLE)
    p.add_argument("--grip-hold", type=float, default=GRIP_HOLD_SECONDS)
    p.add_argument("--release-hold", type=float, default=RELEASE_HOLD_SECONDS)
    p.add_argument("--home-hold", type=float, default=HOME_HOLD_SECONDS)
    p.add_argument("--home-x", type=float, default=HOME_X)
    p.add_argument("--home-y", type=float, default=HOME_Y)
    p.add_argument("--home-z", type=float, default=HOME_Z)
    p.add_argument("--home-yaw", type=float, default=HOME_YAW)
    p.add_argument("--home-gripper", type=float, default=HOME_GRIPPER)
    return p.parse_args()


def main():
    args = parse_args()
    plan = [i for i in source_plan(args.pose_json) if args.start_step <= i["step"] <= args.end_step]
    print_plan(plan)
    if args.dry_run:
        return
    init_robot()
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO, anonymous=True)
    reset_home(args, holding=False)
    for item in plan:
        if args.step_confirm:
            input("Press Enter to run step {:02d} {}...".format(item["step"], item["brick"]))
        pick_brick(item, args)
        place_brick(item, args)
    print("done")


if __name__ == "__main__":
    main()
