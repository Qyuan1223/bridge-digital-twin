#!/usr/bin/env python3
"""
Pick all 12 ideal-grid points in the left supply zone and place each one at
one common target. This is a calibration test for the "regular matrix" guess.

Run on JetMax:
    rosrun jetmax_demos test_pick_place_left_grid.py --step-confirm
"""

from __future__ import print_function

import argparse
import math

ROS_NODE_NAME = "test_pick_place_left_grid"

PICK_OPEN_ANGLE = 100.0
CLOSE_ANGLE = 75.0
RELEASE_ANGLE = 100.0
HOME_GRIPPER = 55.0

HOME_X = 0.0
HOME_Y = -162.9
HOME_Z = 212.8
HOME_YAW = 50.0

SAFE_Z = 180.0
GRIP_HOLD_SECONDS = 1.0
RELEASE_HOLD_SECONDS = 1.0
HOME_HOLD_SECONDS = 1.0

TARGET_X = -155.0
TARGET_Y = -202.9
TARGET_Z = 115.0
TARGET_YAW = 105.0

# Four measured corners of the left 3x4 supply zone.
LEFT_BOTTOM = (-250.0, -107.9, 97.8, 160.0)
RIGHT_BOTTOM = (-130.0, -112.9, 102.8, 180.0)
LEFT_TOP = (-245.0, 62.1, 102.8, 125.0)
RIGHT_TOP = (-120.0, 52.1, 107.8, 125.0)

ROWS = 3
COLS = 4


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


def lerp(a, b, t):
    return a + (b - a) * t


def bilinear(lb, rb, lt, rt, u, v):
    bottom = [lerp(lb[i], rb[i], u) for i in range(4)]
    top = [lerp(lt[i], rt[i], u) for i in range(4)]
    return tuple(lerp(bottom[i], top[i], v) for i in range(4))


def build_grid():
    points = []
    for row in range(ROWS):
        v = float(row) / float(ROWS - 1)
        for col in range(COLS):
            u = float(col) / float(COLS - 1)
            x, y, z, yaw = bilinear(LEFT_BOTTOM, RIGHT_BOTTOM, LEFT_TOP, RIGHT_TOP, u, v)
            points.append(("r{}c{}".format(row, col), row, col, x, y, z, yaw))
    return points


def travel_time(from_xyz, to_xyz, min_time):
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return max(dist / 150.0, min_time)


def set_servo1(yaw, duration):
    hiwonder.pwm_servo1.set_position(int(round(yaw)), duration)
    rospy.sleep(duration + 0.05)


def set_gripper(angle, duration):
    hiwonder.pwm_servo2.set_position(int(round(angle)), duration)
    rospy.sleep(duration + 0.05)


def move_xyz(x, y, z, duration):
    jetmax.set_position((x, y, z), duration)
    rospy.sleep(duration + 0.08)


def reset_pose(args, duration):
    print("reset: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper
    ))
    set_servo1(args.home_yaw, 0.3)
    set_gripper(args.home_gripper, 0.3)
    move_xyz(args.home_x, args.home_y, args.home_z, duration)
    print("home hold for {:.1f}s".format(args.home_hold))
    rospy.sleep(args.home_hold)


def wait_for_enter(args, label):
    if not args.step_confirm:
        return
    input("At home after {}. Press Enter to continue...".format(label))


def pick(label, x, y, z, yaw, args):
    print("\nPICK {}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(label, x, y, z, yaw))
    set_gripper(args.pick_open, 0.4)
    set_servo1(yaw, 0.3)

    cur = jetmax.position
    t = travel_time(cur, (x, y, args.safe_z), 0.4)
    move_xyz(x, y, args.safe_z, t)
    move_xyz(x, y, z, 0.6)
    set_gripper(args.close, 0.5)
    print("hold grip for {:.1f}s".format(args.grip_hold))
    rospy.sleep(args.grip_hold)
    move_xyz(x, y, args.safe_z, 0.7)


def place(label, args):
    print("PLACE {}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(
        label, args.target_x, args.target_y, args.target_z, args.target_yaw
    ))
    set_servo1(args.target_yaw, 0.3)

    cur = jetmax.position
    t = travel_time(cur, (args.target_x, args.target_y, args.safe_z), 0.4)
    move_xyz(args.target_x, args.target_y, args.safe_z, t)
    move_xyz(args.target_x, args.target_y, args.target_z + 20.0, 0.5)
    move_xyz(args.target_x, args.target_y, args.target_z, 0.5)
    set_gripper(args.release, 0.6)
    print("hold release for {:.1f}s".format(args.release_hold))
    rospy.sleep(args.release_hold)
    move_xyz(args.target_x, args.target_y, args.safe_z, 0.7)


def parse_args():
    parser = argparse.ArgumentParser(description="Pick 12 interpolated points from the left supply-zone grid.")
    parser.add_argument("--safe-z", type=float, default=SAFE_Z)
    parser.add_argument("--pick-open", type=float, default=PICK_OPEN_ANGLE)
    parser.add_argument("--close", type=float, default=CLOSE_ANGLE)
    parser.add_argument("--release", type=float, default=RELEASE_ANGLE)
    parser.add_argument("--grip-hold", type=float, default=GRIP_HOLD_SECONDS)
    parser.add_argument("--release-hold", type=float, default=RELEASE_HOLD_SECONDS)
    parser.add_argument("--home-hold", type=float, default=HOME_HOLD_SECONDS)
    parser.add_argument("--step-confirm", action="store_true")

    parser.add_argument("--target-x", type=float, default=TARGET_X)
    parser.add_argument("--target-y", type=float, default=TARGET_Y)
    parser.add_argument("--target-z", type=float, default=TARGET_Z)
    parser.add_argument("--target-yaw", type=float, default=TARGET_YAW)

    parser.add_argument("--home-x", type=float, default=HOME_X)
    parser.add_argument("--home-y", type=float, default=HOME_Y)
    parser.add_argument("--home-z", type=float, default=HOME_Z)
    parser.add_argument("--home-yaw", type=float, default=HOME_YAW)
    parser.add_argument("--home-gripper", type=float, default=HOME_GRIPPER)

    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def print_plan(args, points):
    print("Plan: pick {} ideal-grid points, place each at ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(
        len(points), args.target_x, args.target_y, args.target_z, args.target_yaw
    ))
    print("grid order: row0 bottom -> row2 top, col0 left -> col3 right")
    print("home = ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper
    ))
    for label, row, col, x, y, z, yaw in points:
        print("  {} row={} col={}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(
            label, row, col, x, y, z, yaw
        ))


def main():
    args = parse_args()
    points = build_grid()
    print_plan(args, points)
    if args.dry_run:
        return

    init_robot()
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO)
    if not args.no_reset:
        reset_pose(args, 1.5)
        wait_for_enter(args, "initial reset")

    for label, row, col, x, y, z, yaw in points:
        pick(label, x, y, z, yaw, args)
        place(label, args)
        if not args.no_reset:
            reset_pose(args, 1.2)
            wait_for_enter(args, label)

    print("done")


if __name__ == "__main__":
    main()
