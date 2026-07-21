#!/usr/bin/env python3
"""
Fine coordinate probing tool for JetMax.

Keys:
    w/s : Y +/-
    a/d : X -/+
    r/f : Z +/-
    [/] : move step -/+   default 5mm, minimum 0.2mm
    ,/. : angle step -/+  default 5deg, minimum 1deg
    j/l : pwm_servo1 yaw -/+
    o/c : pwm_servo2 gripper open/close
    h   : home
    p   : print pose
    q   : quit
"""

import select
import sys
import termios
import tty

import rospy
import hiwonder


STEP = 5.0
ANGLE_STEP = 5.0
MIN_STEP = 0.2
MIN_ANGLE_STEP = 1.0

jetmax = hiwonder.JetMax()
servo1_angle = 50.0
gripper_angle = 55.0


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


def print_position():
    x, y, z = jetmax.position
    print("position = ({:.1f}, {:.1f}, {:.1f})   servo1(yaw)={:.1f}  gripper={:.1f}  step={:.1f}  angle_step={:.1f}".format(
        x, y, z, servo1_angle, gripper_angle, STEP, ANGLE_STEP
    ))


def move_by(dx, dy, dz):
    x, y, z = jetmax.position
    jetmax.set_position((x + dx, y + dy, z + dz), 0.2)
    rospy.sleep(0.22)
    print_position()


def decrease_step():
    global STEP
    if STEP > 1.0:
        STEP = max(1.0, STEP - 1.0)
    else:
        STEP = max(MIN_STEP, STEP - 0.2)
    print("step =", STEP)


def increase_step():
    global STEP
    if STEP < 1.0:
        STEP = min(1.0, STEP + 0.2)
    else:
        STEP += 1.0
    print("step =", STEP)


def main():
    global ANGLE_STEP, servo1_angle, gripper_angle
    rospy.init_node("probe_position_fine", log_level=rospy.INFO, anonymous=True)
    jetmax.go_home(1.5)
    rospy.sleep(1.6)
    hiwonder.pwm_servo1.set_position(int(round(servo1_angle)), 0.3)
    rospy.sleep(0.35)
    print(__doc__)
    print_position()

    while not rospy.is_shutdown():
        c = getch(0.5)
        if c is None:
            continue
        if c == "q":
            break
        elif c == "w":
            move_by(0, STEP, 0)
        elif c == "s":
            move_by(0, -STEP, 0)
        elif c == "d":
            move_by(STEP, 0, 0)
        elif c == "a":
            move_by(-STEP, 0, 0)
        elif c == "r":
            move_by(0, 0, STEP)
        elif c == "f":
            move_by(0, 0, -STEP)
        elif c == "[":
            decrease_step()
        elif c == "]":
            increase_step()
        elif c == ",":
            ANGLE_STEP = max(MIN_ANGLE_STEP, ANGLE_STEP - 1.0)
            print("angle_step =", ANGLE_STEP)
        elif c == ".":
            ANGLE_STEP += 1.0
            print("angle_step =", ANGLE_STEP)
        elif c == "j":
            servo1_angle = max(0.0, servo1_angle - ANGLE_STEP)
            hiwonder.pwm_servo1.set_position(int(round(servo1_angle)), 0.2)
            rospy.sleep(0.22)
            print_position()
        elif c == "l":
            servo1_angle = min(180.0, servo1_angle + ANGLE_STEP)
            hiwonder.pwm_servo1.set_position(int(round(servo1_angle)), 0.2)
            rospy.sleep(0.22)
            print_position()
        elif c == "o":
            gripper_angle = min(180.0, gripper_angle + ANGLE_STEP)
            hiwonder.pwm_servo2.set_position(int(round(gripper_angle)), 0.2)
            rospy.sleep(0.22)
            print_position()
        elif c == "c":
            gripper_angle = max(0.0, gripper_angle - ANGLE_STEP)
            hiwonder.pwm_servo2.set_position(int(round(gripper_angle)), 0.2)
            rospy.sleep(0.22)
            print_position()
        elif c == "h":
            jetmax.go_home(1.2)
            rospy.sleep(1.3)
            print_position()
        elif c == "p":
            print_position()


if __name__ == "__main__":
    main()
