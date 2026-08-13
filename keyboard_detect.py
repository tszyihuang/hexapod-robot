#!/usr/bin/env python3
"""键盘控制六足机器人：W 前进、S 后退、A 左移、D 右移、Q/E 旋转，松开即停。

主循环持续检测键盘状态并覆盖打印（按下为大写、松开为小写）。W/S/A/D
表示前后左右移动，Q/E 表示原地旋转（Q 逆时针、E 顺时针）。任一按键从
松开变为按下时，只启动一次对应的步态线程；按键松开或切换动作时停止
当前步态并回到自然站立姿态。斜向组合：W+A 左前方、W+D 右前方、S+A
左后方、S+D 右后方（斜向时两个分量各取速度的 √2/2，合速度与单方向
一致）。按 ESC 退出，退出前自动失能全部舵机。
"""

import argparse
import math
import sys
import threading
import time

import keyboard
import serial

import servo_driver

# 需要持续检测的按键
KEYS = ["w", "a", "s", "d", "q", "e"]
REFRESH_INTERVAL = 0.02


def desired_action(pressed: dict[str, bool], ser: serial.Serial,
                   speed: float, turn_speed: float):
    """返回 (方向名, 参与按键列表, 步态函数, 参数)；无方向键按下返回 None。

    斜向组合优先于单方向键：W+A 左前方、W+D 右前方、S+A 左后方、
    S+D 右后方。斜向两个分量各取 speed*√2/2，保证合速度与单方向一致。
    """
    diag = speed * math.sqrt(2.0) / 2.0
    if pressed["w"] and pressed["a"]:
        return ("左前方", ["w", "a"], servo_driver.cmd_move,
                (ser, diag, diag, None))
    if pressed["w"] and pressed["d"]:
        return ("右前方", ["w", "d"], servo_driver.cmd_move,
                (ser, diag, -diag, None))
    if pressed["s"] and pressed["d"]:
        return ("右后方", ["s", "d"], servo_driver.cmd_move,
                (ser, -diag, -diag, None))
    if pressed["s"] and pressed["a"]:
        return ("左后方", ["s", "a"], servo_driver.cmd_move,
                (ser, -diag, diag, None))
    if pressed["w"]:
        return "前进", ["w"], servo_driver.cmd_walk, (ser, speed, None)
    if pressed["s"]:
        return "后退", ["s"], servo_driver.cmd_walk, (ser, -speed, None)
    if pressed["a"]:
        return "左移", ["a"], servo_driver.cmd_strafe, (ser, speed, None)
    if pressed["d"]:
        return "右移", ["d"], servo_driver.cmd_strafe, (ser, -speed, None)
    if pressed["q"]:
        return "逆时针旋转", ["q"], servo_driver.cmd_turn, (ser, turn_speed, None)
    if pressed["e"]:
        return "顺时针旋转", ["e"], servo_driver.cmd_turn, (ser, -turn_speed, None)
    return None


def main():
    parser = argparse.ArgumentParser(description="键盘控制六足机器人移动")
    parser.add_argument("--port", default=servo_driver.PORT,
                        help=f"串口设备 (默认 {servo_driver.PORT})")
    parser.add_argument("--baud", type=int, default=servo_driver.BAUD,
                        help=f"波特率 (默认 {servo_driver.BAUD})")
    parser.add_argument("--speed", type=float,
                        default=servo_driver.WALK_SPEED_DEFAULT,
                        help="移动速度 mm/s，作用于前进/后退/左右"
                             f" (默认 {servo_driver.WALK_SPEED_DEFAULT:g})")
    parser.add_argument("--turn-speed", type=float,
                        default=servo_driver.TURN_SPEED_DEFAULT,
                        help="旋转角速度 deg/s，作用于 Q/E"
                             f" (默认 {servo_driver.TURN_SPEED_DEFAULT:g})")
    args = parser.parse_args()

    try:
        ser = serial.Serial(port=args.port, baudrate=args.baud, bytesize=8,
                            parity="N", stopbits=1, timeout=0.2,
                            write_timeout=1)
    except serial.SerialException as exc:
        print(f"无法打开串口 {args.port}: {exc}")
        sys.exit(1)

    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)

    # 当前步态线程、它的停止信号、以及它对应的方向按键组合
    gait_thread: threading.Thread | None = None
    gait_stop: threading.Event | None = None
    active_keys: tuple[str, ...] | None = None
    # 机器人是否已处于自然站立姿态：首次按方向键需要先站立，
    # 之后每次步态结束都会回到站立，可直接进入步态循环，省掉约 1.8 秒
    robot_standing = False

    print("键盘控制已启动：W 前进、S 后退、A 左移、D 右移、"
          "Q 逆时针、E 顺时针；W+A 左前、W+D 右前、S+A 左后、S+D 右后，"
          "松开停止，按 ESC 退出。")
    try:
        while True:
            pressed = {key: keyboard.is_pressed(key) for key in KEYS}
            state = [key.upper() if pressed[key] else key for key in KEYS]
            print(" ".join(state), end="\r", flush=True)

            desired = desired_action(pressed, ser, args.speed,
                                     args.turn_speed)
            desired_keys = tuple(desired[1]) if desired else None

            # 方向没变就不动；变了就先停旧步态、再启动新步态
            if desired_keys != active_keys:
                if gait_stop is not None:
                    gait_stop.set()
                if gait_thread is not None and gait_thread.is_alive():
                    gait_thread.join(timeout=5)

                if desired is None:
                    active_keys = None
                    gait_thread = None
                    gait_stop = None
                else:
                    label, keys, target, fargs = desired
                    gait_stop = threading.Event()

                    def should_stop(ev: threading.Event = gait_stop,
                                    ks: tuple[str, ...] = tuple(keys)) -> bool:
                        return ev.is_set() or not all(
                            keyboard.is_pressed(k) for k in ks)

                    gait_thread = threading.Thread(
                        target=target,
                        args=fargs,
                        kwargs={
                            "should_stop": should_stop,
                            "stand_first": not robot_standing,
                        },
                        daemon=True,
                    )
                    active_keys = tuple(keys)
                    robot_standing = True  # 本次步态结束时必定回到自然站立
                    print(f"\n开始{label}。")
                    gait_thread.start()

            if keyboard.is_pressed("esc"):
                break

            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print()
    finally:
        if gait_stop is not None:
            gait_stop.set()
        if gait_thread is not None and gait_thread.is_alive():
            gait_thread.join(timeout=5)
        try:
            servo_driver.unload_servos(ser, servo_driver.SERVO_IDS)
            time.sleep(0.1)
            print("\n程序退出，18 个舵机已自动失能。")
        except OSError as exc:
            print(f"\n程序退出时自动失能失败: {exc}")
        finally:
            ser.close()
            print("串口已关闭，程序退出。")


if __name__ == "__main__":
    main()
