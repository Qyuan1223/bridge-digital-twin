#!/usr/bin/env python3
"""
Pick bricks from the four measured corners of the left supply zone and place
each one at one common test target.

Run on JetMax:
    rosrun jetmax_demos test_pick_place_corners.py

Useful overrides:
    rosrun jetmax_demos test_pick_place_corners.py --lt-yaw 155 --rt-yaw 175
    rosrun jetmax_demos test_pick_place_corners.py --target-z 102 --grip-hold 1.5
"""

from __future__ import print_function

import argparse
import math

import rospy
import hiwonder


ROS_NODE_NAME = "test_pick_place_corners"

# Gripper angles measured on site.
PICK_OPEN_ANGLE = 100.0
CLOSE_ANGLE = 75.0
RELEASE_ANGLE = 100.0
HOME_GRIPPER = 55.0

# Field-measured neutral pose.
HOME_X = 0.0
HOME_Y = -162.9
HOME_Z = 212.8
HOME_YAW = 50.0

SAFE_Z = 180.0
APPROACH_LIFT = 35.0
GRIP_HOLD_SECONDS = 1.0
RELEASE_HOLD_SECONDS = 1.0
HOME_HOLD_SECONDS = 1.0

# Common placement target for this corner test.
TARGET_X = -155.0
TARGET_Y = -202.9
TARGET_Z = 115.0
TARGET_YAW = 105.0


jetmax = hiwonder.JetMax()


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


def pick(name, x, y, z, yaw, args):
    print("\nPICK {}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(name, x, y, z, yaw))
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


def place(name, x, y, z, yaw, args):
    print("PLACE {}: ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f}".format(name, x, y, z, yaw))
    set_servo1(yaw, 0.3)

    local_safe_z = approach_z(z, args)
    print("target approach_z = {:.1f}".format(local_safe_z))
    cur = jetmax.position
    t = travel_time(cur, (x, y, local_safe_z), 0.4)
    move_xyz(x, y, local_safe_z, t)
    move_xyz(x, y, z + 20.0, 0.5)
    move_xyz(x, y, z, 0.5)
    set_gripper(args.release, 0.6)
    print("hold release for {:.1f}s".format(args.release_hold))
    rospy.sleep(args.release_hold)
    move_xyz(x, y, local_safe_z, 0.7)


def parse_args():
    parser = argparse.ArgumentParser(description="Pick four supply-zone corner bricks and place each at a common target.")

    parser.add_argument("--safe-z", type=float, default=SAFE_Z)
    parser.add_argument("--approach-lift", type=float, default=APPROACH_LIFT, help="local lift above each work z; prevents far high points from becoming unreachable")
    parser.add_argument("--pick-open", type=float, default=PICK_OPEN_ANGLE)
    parser.add_argument("--close", type=float, default=CLOSE_ANGLE)
    parser.add_argument("--release", type=float, default=RELEASE_ANGLE)
    parser.add_argument("--grip-hold", type=float, default=GRIP_HOLD_SECONDS)
    parser.add_argument("--release-hold", type=float, default=RELEASE_HOLD_SECONDS)
    parser.add_argument("--home-hold", type=float, default=HOME_HOLD_SECONDS)
    parser.add_argument("--step-confirm", action="store_true", help="pause at home after each block until Enter is pressed")

    parser.add_argument("--target-x", type=float, default=TARGET_X)
    parser.add_argument("--target-y", type=float, default=TARGET_Y)
    parser.add_argument("--target-z", type=float, default=TARGET_Z)
    parser.add_argument("--target-yaw", type=float, default=TARGET_YAW)

    parser.add_argument("--home-x", type=float, default=HOME_X)
    parser.add_argument("--home-y", type=float, default=HOME_Y)
    parser.add_argument("--home-z", type=float, default=HOME_Z)
    parser.add_argument("--home-yaw", type=float, default=HOME_YAW)
    parser.add_argument("--home-gripper", type=float, default=HOME_GRIPPER)

    parser.add_argument("--lb-yaw", type=float, default=160.0, help="left-bottom yaw")
    parser.add_argument("--lt-yaw", type=float, default=125.0, help="left-top yaw")
    parser.add_argument("--rt-yaw", type=float, default=125.0, help="right-top yaw")
    parser.add_argument("--rb-yaw", type=float, default=180.0, help="right-bottom yaw")

    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def corner_plan(args):
    return [
        ("left-bottom", -250.0, -107.9, 97.8, args.lb_yaw),
        ("left-top", -245.0, 62.1, 102.8, args.lt_yaw),
        ("right-top", -120.0, 52.1, 107.8, args.rt_yaw),
        ("right-bottom", -130.0, -112.9, 102.8, args.rb_yaw),
    ]


def print_plan(args, corners):
    print("Plan: pick four corners, place each at ({:.1f}, {:.1f}, {:.1f})".format(
        args.target_x, args.target_y, args.target_z
    ))
    print("gripper open/close/release = {:.1f}/{:.1f}/{:.1f}".format(
        args.pick_open, args.close, args.release
    ))
    print("home = ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} gripper={:.1f}".format(
        args.home_x, args.home_y, args.home_z, args.home_yaw, args.home_gripper
    ))
    print("safe_z = {:.1f}, grip_hold = {:.1f}s, release_hold = {:.1f}s".format(
        args.safe_z, args.grip_hold, args.release_hold
    ))
    print("home_hold = {:.1f}s, step_confirm = {}".format(args.home_hold, args.step_confirm))
    for name, x, y, z, yaw in corners:
        place_yaw = yaw if args.target_yaw is None else args.target_yaw
        print("  {}: pick ({:.1f}, {:.1f}, {:.1f}) yaw={:.1f} -> place yaw={:.1f}".format(
            name, x, y, z, yaw, place_yaw
        ))


def main():
    args = parse_args()
    corners = corner_plan(args)
    print_plan(args, corners)
    if args.dry_run:
        return

    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO)
    if not args.no_reset:
        reset_pose(args, 1.5)
        wait_for_enter(args, "initial reset")

    for name, x, y, z, yaw in corners:
        pick(name, x, y, z, yaw, args)
        place_yaw = yaw if args.target_yaw is None else args.target_yaw
        place(name, args.target_x, args.target_y, args.target_z, place_yaw, args)
        if not args.no_reset:
            reset_pose(args, 1.2)
            wait_for_enter(args, name)

    print("done")


if __name__ == "__main__":
    main()
