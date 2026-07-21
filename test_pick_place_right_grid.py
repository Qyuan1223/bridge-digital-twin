#!/usr/bin/env python3
"""
Pick all 12 ideal-grid points in the right supply zone and place each one at
one common target.

Default assumption:
  the right supply zone is a regular 3x4 matrix. The measured left-bottom and
  right-top corners define the rectangular x/y span. z and yaw are interpolated
  smoothly between those two measured corners for a first validation pass.
"""

from __future__ import print_function

import argparse
import math


ROS_NODE_NAME = "test_pick_place_right_grid"

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
HOME_HOLD_SECONDS = 1.0

TARGET_X = -155.0
TARGET_Y = -202.9
TARGET_Z = 115.0
TARGET_YAW = 105.0

# Two measured diagonal corners of the right 3x4 supply zone.
RIGHT_ZONE_LEFT_BOTTOM = (125.0, -137.9, 97.8, 90.0)
RIGHT_ZONE_RIGHT_TOP = (255.0, 27.1, 102.8, 145.0)
RIGHT_ZONE_RIGHT_BOTTOM_OVERRIDE = (250.0, -132.9, 107.8, 110.0)

# Tuned pickup poses from tune_right_grid_positions.py. r2c3 is intentionally
# omitted because the provided record was the home pose, not the pickup pose.
RIGHT_GRID_TUNED_OVERRIDES = {
    "r0c0": (125.0, -137.9, 97.8, 90.0),
    "r0c1": (168.333, -137.9, 98.633, 99.167),
    "r0c2": (211.667, -137.9, 99.467, 108.333),
    "r0c3": (250.0, -132.9, 97.8, 110.0),
    "r1c0": (125.0, -55.4, 99.05, 113.75),
    "r1c1": (168.333, -55.4, 99.883, 117.917),
    "r1c2": (211.667, -55.4, 100.717, 122.083),
    "r1c3": (255.0, -55.4, 101.55, 126.25),
    "r2c0": (115.0, 27.1, 100.3, 147.5),
    "r2c1": (163.333, 22.1, 101.133, 146.667),
    "r2c2": (216.667, 22.1, 101.967, 145.833),
}

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


def build_grid(args):
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
            label = "r{}c{}".format(row, col)
            source = "generated"
            if args.use_tuned and label in RIGHT_GRID_TUNED_OVERRIDES:
                x, y, z, yaw = RIGHT_GRID_TUNED_OVERRIDES[label]
                source = "tuned"
            points.append((label, row, col, x, y, z, yaw, source))
    return points


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


def place(label, args):
    print("PLACE {}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(
        label, args.target_x, args.target_y, args.target_z, args.target_yaw
    ))
    set_servo1(args.target_yaw, 0.3)

    local_safe_z = approach_z(args.target_z, args)
    print("target approach_z = {:.1f}".format(local_safe_z))
    cur = jetmax.position
    t = travel_time(cur, (args.target_x, args.target_y, local_safe_z), 0.4)
    move_xyz(args.target_x, args.target_y, local_safe_z, t)
    move_xyz(args.target_x, args.target_y, args.target_z + 20.0, 0.5)
    move_xyz(args.target_x, args.target_y, args.target_z, 0.5)
    set_gripper(args.release, 0.6)
    print("hold release for {:.1f}s".format(args.release_hold))
    rospy.sleep(args.release_hold)
    move_xyz(args.target_x, args.target_y, local_safe_z, 0.7)


def parse_args():
    parser = argparse.ArgumentParser(description="Pick 12 regular right-zone grid points.")

    parser.add_argument("--safe-z", type=float, default=SAFE_Z)
    parser.add_argument("--approach-lift", type=float, default=APPROACH_LIFT, help="local lift above each work z; prevents far high points from becoming unreachable")
    parser.add_argument("--pick-open", type=float, default=PICK_OPEN_ANGLE)
    parser.add_argument("--close", type=float, default=CLOSE_ANGLE)
    parser.add_argument("--release", type=float, default=RELEASE_ANGLE)
    parser.add_argument("--grip-hold", type=float, default=GRIP_HOLD_SECONDS)
    parser.add_argument("--release-hold", type=float, default=RELEASE_HOLD_SECONDS)
    parser.add_argument("--home-hold", type=float, default=HOME_HOLD_SECONDS)
    parser.add_argument("--step-confirm", action="store_true")
    parser.add_argument("--use-generated", action="store_true", help="ignore hardcoded tuned poses and use generated grid poses")

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
    args = parser.parse_args()
    args.use_tuned = not args.use_generated
    return args


def print_plan(args, points):
    print("Plan: pick {} regular right-zone points, place each at ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(
        len(points), args.target_x, args.target_y, args.target_z, args.target_yaw
    ))
    print("right-zone LB = ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(*RIGHT_ZONE_LEFT_BOTTOM))
    print("right-zone RT = ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(*RIGHT_ZONE_RIGHT_TOP))
    print("right-zone RB override = ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(*RIGHT_ZONE_RIGHT_BOTTOM_OVERRIDE))
    print("use_tuned = {} (r2c3 tuned record ignored because it was home pose)".format(args.use_tuned))
    print("grid order: row0 bottom -> row2 top, col0 left -> col3 right")
    print("home = ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper
    ))
    for label, row, col, x, y, z, yaw, source in points:
        print("  {} row={} col={}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} [{}]".format(
            label, row, col, x, y, z, yaw, source
        ))


def main():
    args = parse_args()
    points = build_grid(args)
    print_plan(args, points)
    if args.dry_run:
        return

    init_robot()
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO)
    if not args.no_reset:
        reset_pose(args, 1.5)
        wait_for_enter(args, "initial reset")

    for label, row, col, x, y, z, yaw, source in points:
        pick(label, x, y, z, yaw, args)
        place(label, args)
        if not args.no_reset:
            reset_pose(args, 1.2)
            wait_for_enter(args, label)

    print("done")


if __name__ == "__main__":
    main()
