#!/usr/bin/env python3
"""
One-brick real-robot smoke test: pick A_N_01 and place it at one target pose.

Run on JetMax:
    python test_pick_place_A.py

Optional:
    python test_pick_place_A.py --dry-run
    python test_pick_place_A.py --safe-z 190 --target-z 116
"""

from __future__ import print_function

import argparse
import io
import json
import math
import os

ROS_NODE_NAME = "test_pick_place_A"

# Measured brick-library pickup pose. The script will load calibration JSON first;
# these constants are used only if calibration/brick_library_positions.json is absent.
DEFAULT_PICK_X = -245.0
DEFAULT_PICK_Y = 67.1
DEFAULT_PICK_Z = 125.0
DEFAULT_PICK_YAW = 125.0

# Target placement pose supplied by the user.
DEFAULT_TARGET_X = -155.0
DEFAULT_TARGET_Y = -202.9
DEFAULT_TARGET_Z = 115.0
DEFAULT_TARGET_YAW = 105.0

# Gripper angles.
PICK_OPEN_ANGLE = 100.0
CLOSE_ANGLE = 75.0
PLACE_OPEN_ANGLE = 100.0
GRIP_HOLD_SECONDS = 1.0

# Field-measured neutral pose. Override with args if needed.
HOME_X = 0.0
HOME_Y = -162.9
HOME_Z = 212.8
HOME_YAW = 50.0
HOME_GRIPPER = 55.0
SAFE_Z = 180.0
APPROACH_LIFT = 35.0


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


def load_a_pick_pose(path):
    if not os.path.exists(path):
        return None
    with io.open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for slot in data.get("slots", []):
        if slot.get("id") == "A_N_01" and slot.get("recorded"):
            pose = slot.get("pose", {})
            if all(pose.get(k) is not None for k in ("x", "y", "z", "yaw")):
                return (
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["z"]),
                    float(pose["yaw"]),
                )
    return None


def travel_time(from_xyz, to_xyz, min_time):
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return max(dist / 150.0, min_time)


def approach_z(work_z, args):
    """Use a local travel height so far-away points do not become unreachable."""
    return min(args.safe_z, work_z + args.approach_lift)


def set_servo1(yaw, duration):
    hiwonder.pwm_servo1.set_position(int(round(yaw)), duration)
    rospy.sleep(duration + 0.05)


def set_gripper(angle, duration):
    hiwonder.pwm_servo2.set_position(int(round(angle)), duration)
    rospy.sleep(duration + 0.05)


def move_xyz(x, y, z, duration):
    jetmax.set_position((x, y, z), duration)
    rospy.sleep(duration + 0.08)


def reset_pose(home_x, home_y, home_z, home_yaw, home_gripper, duration):
    print("reset pose: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        home_x, home_y, home_z, home_yaw, home_gripper
    ))
    set_servo1(home_yaw, 0.3)
    set_gripper(home_gripper, 0.3)
    move_xyz(home_x, home_y, home_z, duration)


def pick(x, y, z, yaw, args):
    print("pick A_N_01: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(x, y, z, yaw))
    set_gripper(PICK_OPEN_ANGLE, 0.4)
    set_servo1(yaw, 0.3)

    local_safe_z = approach_z(z, args)
    print("approach_z = {:.1f}".format(local_safe_z))
    cur = jetmax.position
    t = travel_time(cur, (x, y, local_safe_z), 0.4)
    move_xyz(x, y, local_safe_z, t)
    move_xyz(x, y, z, 0.6)
    set_gripper(args.close, 0.5)
    print("hold grip for {:.1f}s".format(args.grip_hold))
    rospy.sleep(args.grip_hold)
    move_xyz(x, y, local_safe_z, 0.7)


def place(x, y, z, yaw, args):
    print("place target: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} release={:.1f}".format(x, y, z, yaw, args.release))
    set_servo1(yaw, 0.3)

    local_safe_z = approach_z(z, args)
    print("target approach_z = {:.1f}".format(local_safe_z))
    cur = jetmax.position
    t = travel_time(cur, (x, y, local_safe_z), 0.4)
    move_xyz(x, y, local_safe_z, t)
    move_xyz(x, y, z + 20.0, 0.5)
    move_xyz(x, y, z, 0.5)
    set_gripper(args.release, 0.6)
    rospy.sleep(1.0)
    move_xyz(x, y, local_safe_z, 0.7)


def parse_args():
    parser = argparse.ArgumentParser(description="Pick A_N_01 and place one measured target pose.")
    parser.add_argument("--calibration", default=os.path.join("calibration", "brick_library_positions.json"))
    parser.add_argument("--safe-z", type=float, default=SAFE_Z)
    parser.add_argument("--approach-lift", type=float, default=APPROACH_LIFT, help="local lift above each work z; prevents far high points from becoming unreachable")
    parser.add_argument("--pick-z", type=float, default=None, help="override measured pickup z")
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET_X)
    parser.add_argument("--target-y", type=float, default=DEFAULT_TARGET_Y)
    parser.add_argument("--target-z", type=float, default=DEFAULT_TARGET_Z)
    parser.add_argument("--target-yaw", type=float, default=DEFAULT_TARGET_YAW)
    parser.add_argument("--pick-open", type=float, default=PICK_OPEN_ANGLE)
    parser.add_argument("--close", type=float, default=CLOSE_ANGLE)
    parser.add_argument("--grip-hold", type=float, default=GRIP_HOLD_SECONDS, help="seconds to wait after closing gripper before lifting")
    parser.add_argument("--release", type=float, default=PLACE_OPEN_ANGLE)
    parser.add_argument("--home-x", type=float, default=HOME_X)
    parser.add_argument("--home-y", type=float, default=HOME_Y)
    parser.add_argument("--home-z", type=float, default=HOME_Z)
    parser.add_argument("--home-yaw", type=float, default=HOME_YAW)
    parser.add_argument("--home-gripper", type=float, default=HOME_GRIPPER)
    parser.add_argument("--no-reset", action="store_true", help="do not move to measured home before/after")
    parser.add_argument("--use-calibration", action="store_true", help="load A_N_01 from calibration JSON instead of the built-in measured pose")
    parser.add_argument("--dry-run", action="store_true", help="print the planned poses without moving")
    return parser.parse_args()


def main():
    global PICK_OPEN_ANGLE
    args = parse_args()
    PICK_OPEN_ANGLE = args.pick_open

    pose = load_a_pick_pose(args.calibration) if args.use_calibration else None
    if pose is None:
        pose = (DEFAULT_PICK_X, DEFAULT_PICK_Y, DEFAULT_PICK_Z, DEFAULT_PICK_YAW)
        print("using built-in measured A_N_01 pose")
    pick_x, pick_y, pick_z, pick_yaw = pose
    if args.pick_z is not None:
        pick_z = args.pick_z

    print("Plan:")
    print("  pick    ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(pick_x, pick_y, pick_z, pick_yaw))
    print("  place   ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(
        args.target_x, args.target_y, args.target_z, args.target_yaw
    ))
    print("  gripper pick-open/close/release = {:.1f}/{:.1f}/{:.1f}".format(
        args.pick_open, args.close, args.release
    ))
    print("  grip hold after close = {:.1f}s".format(args.grip_hold))
    print("  home    ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper
    ))
    print("  safe_z = {:.1f}".format(args.safe_z))

    if args.dry_run:
        return

    init_robot()
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO)
    if not args.no_reset:
        reset_pose(args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper, 1.5)

    pick(pick_x, pick_y, pick_z, pick_yaw, args)
    place(args.target_x, args.target_y, args.target_z, args.target_yaw, args)

    if not args.no_reset:
        reset_pose(args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper, 1.2)
    print("done")


if __name__ == "__main__":
    main()
