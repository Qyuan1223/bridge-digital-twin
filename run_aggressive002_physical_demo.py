#!/usr/bin/env python3
"""
Run a fully robotic physical test of aggressive_002.

Placement poses:
  - all steps are predicted from the physical mapping model:
    base(wx, wz, side) + brick-specific offset

This is a model-based smoke test, not a final measured build.

Run on JetMax:
    rosrun jetmax_demos run_aggressive002_physical_demo.py --step-confirm
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys


ROS_NODE_NAME = "run_aggressive002_physical_demo"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_JSON = os.path.join(SCRIPT_DIR, "calibration", "physical_mapping_model.json")

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


AGGRESSIVE_002 = [
    {"step": 1, "brick": "B", "side": "L", "wx": 17.0, "wz": 1.5642247142857144},
    {"step": 2, "brick": "D", "side": "L", "wx": 8.0, "wz": 1.5642247142857144},
    {"step": 3, "brick": "E", "side": "L", "wx": 6.0, "wz": 2.801403862549198},
    {"step": 4, "brick": "F'", "side": "L", "wx": 14.0, "wz": 2.801403862549198},
    {"step": 5, "brick": "E", "side": "L", "wx": 10.5, "wz": 4.657172584944423},
    {"step": 6, "brick": "C", "side": "L", "wx": 20.0, "wz": 4.0385830108126815},
    {"step": 7, "brick": "E'", "side": "R", "wx": 37.5, "wz": 1.5642247142857144},
    {"step": 8, "brick": "C", "side": "R", "wx": 47.0, "wz": 1.5642247142857144},
    {"step": 9, "brick": "F'", "side": "L", "wx": 15.5, "wz": 5.894351733207906},
    {"step": 10, "brick": "D", "side": "R", "wx": 42.5, "wz": 3.41999343668094},
    {"step": 11, "brick": "F'", "side": "R", "wx": 35.0, "wz": 4.0385830108126815},
    {"step": 12, "brick": "A", "side": "R", "wx": 46.5, "wz": 4.657172584944424},
    {"step": 13, "brick": "D", "side": "R", "wx": 39.5, "wz": 5.894351733207906},
    {"step": 14, "brick": "D", "side": "R", "wx": 32.0, "wz": 6.5129413073396485},
    {"step": 15, "brick": "E'", "side": "R", "wx": 39.0, "wz": 7.131530881471391},
    {"step": 16, "brick": "F", "side": "L", "wx": 23.0, "wz": 7.750120455603132},
]


EMBEDDED_MODEL = {
    "features": [
        "intercept", "wx", "wz", "side_R", "brick_A", "brick_A'", "brick_B", "brick_B'",
        "brick_C", "brick_C'", "brick_E", "brick_E'", "brick_F", "brick_F'",
    ],
    "targets": {
        "x": {"coefficients": {
            "intercept": -200.10713206672514, "wx": 7.199623217449162, "wz": 0.19437420385248713,
            "side_R": 6.502043268695211, "brick_A": -21.936377892833704, "brick_A'": -13.238891713268593,
            "brick_B": -17.706375158768243, "brick_B'": -6.619005204022711, "brick_C": -19.008249445627047,
            "brick_C'": -37.21899247515629, "brick_E": -14.729026882162442, "brick_E'": 1.0531528012043911,
            "brick_F": -13.167234980995852, "brick_F'": -10.864114670526861,
        }},
        "y": {"coefficients": {
            "intercept": -203.0888935142299, "wx": -0.043181413609019045, "wz": -0.6763204347914029,
            "side_R": 0.23556931254733932, "brick_A": 1.0675385095259813, "brick_A'": 0.665379923117133,
            "brick_B": 0.05483597147447948, "brick_B'": -0.36044146868332294, "brick_C": 1.7416425724445652,
            "brick_C'": -1.6425790727577045, "brick_E": -1.2919623178932458, "brick_E'": -1.5589793086548842,
            "brick_F": 0.8815901367987046, "brick_F'": -0.2649105899780822,
        }},
        "z": {"coefficients": {
            "intercept": 102.83144835742779, "wx": -0.12417551137543446, "wz": 7.630503017616334,
            "side_R": 3.232967128971759, "brick_A": 5.620532884351033, "brick_A'": -0.6035410511886223,
            "brick_B": -0.7502393890411344, "brick_B'": 1.423885146176485, "brick_C": 1.1641707940907176,
            "brick_C'": 0.38961119491708374, "brick_E": 0.22832091243895503, "brick_E'": 0.8502599717743272,
            "brick_F": 4.357943825217439, "brick_F'": 3.0413095840468416,
        }},
        "yaw": {"coefficients": {
            "intercept": 97.13942644313242, "wx": 1.4388014477795024, "wz": 0.11700230848465533,
            "side_R": 5.07622627772818, "brick_A": -6.442845192669086, "brick_A'": -4.391792229068835,
            "brick_B": -4.329968029633576, "brick_B'": -3.6983155182354186, "brick_C": -4.050893672963348,
            "brick_C'": -7.857166088659777, "brick_E": -3.6905073334357734, "brick_E'": -0.6047855102047653,
            "brick_F": -2.843668282238843, "brick_F'": -0.2275205371439498,
        }},
    },
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


def load_mapping_model(path):
    if path and os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh), "model-json"
    return EMBEDDED_MODEL, "embedded-model"


def feature_value(feature, move):
    if feature == "intercept":
        return 1.0
    if feature == "wx":
        return float(move["wx"])
    if feature == "wz":
        return float(move["wz"])
    if feature == "side_R":
        return 1.0 if move["side"] == "R" else 0.0
    if feature.startswith("brick_"):
        return 1.0 if move["brick"] == feature[len("brick_"):] else 0.0
    raise KeyError(feature)


def predict_target(model, target, move):
    features = model["features"]
    coefs = model["targets"][target]["coefficients"]
    total = 0.0
    for feature in features:
        total += float(coefs.get(feature, 0.0)) * feature_value(feature, move)
    return total


def place_pose(move, model, model_source):
    return (
        predict_target(model, "x", move),
        predict_target(model, "y", move),
        predict_target(model, "z", move),
        predict_target(model, "yaw", move),
    ), model_source


def tuple_from_pose_dict(pose):
    return (float(pose["x"]), float(pose["y"]), float(pose["z"]), float(pose["yaw"]))


def source_plan(model, model_source):
    counts = {}
    plan = []
    for move in AGGRESSIVE_002:
        brick = move["brick"]
        pool = SOURCE_POOLS[brick]
        used = counts.get(brick, 0)
        side, grid_id = pool[used % len(pool)]
        counts[brick] = used + 1
        pose = (LEFT_GRID if side == "L" else RIGHT_GRID)[grid_id]
        item = dict(move)
        item.update({"source_side": side, "source_grid_id": grid_id, "source_pose": pose, "refill": used >= len(pool)})
        item["place_pose"], item["place_source"] = place_pose(move, model, model_source)
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
    print("Aggressive 002 physical demo plan:")
    for item in plan:
        px, py, pz, pyaw = item["place_pose"]
        print("{step:02d} {brick:>2} src={source_side}:{source_grid_id} place=({:.1f},{:.1f},{:.1f}) yaw={:.1f} {}{}".format(
            px, py, pz, pyaw, item["place_source"], " REFILL" if item["refill"] else "", **item
        ))


def parse_args():
    p = argparse.ArgumentParser(description="Run full-machine aggressive_002 physical smoke test.")
    p.add_argument("--model-json", default=DEFAULT_MODEL_JSON, help="physical mapping model JSON")
    p.add_argument("--start-step", type=int, default=1)
    p.add_argument("--end-step", type=int, default=16)
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
    model, model_source = load_mapping_model(args.model_json)
    print("Using placement model: {}".format(model_source))
    plan = [i for i in source_plan(model, model_source) if args.start_step <= i["step"] <= args.end_step]
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
