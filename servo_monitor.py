#!/usr/bin/env python3
"""六足机器人舵机位置监视器（只读）。

通过 USB 转总线模块直连 LX-15D/LX-16A 串行总线舵机（默认 115200 8N1），
在控制台持续打印各舵机的位置值（0-1000）。

本程序只发送「ID 查询」(0x0E) 和「位置读取」(0x1C) 两种只读指令，
不会发送移动、加载/卸载等任何写指令，因此运行期间舵机不会转动。
按 Ctrl+C 退出。

用法：
  python3 servo_monitor.py [--port /dev/ttyUSB0] [--baud 115200]
                           [--interval 0] [--ids 1-18]

--ids 支持逗号分隔或区间，例如 1-18、13,14,15、1-6,13-18。
"""

import argparse
import sys
import time

import serial

from servo_protocol import (BAUD, CMD_SERVO_ID_READ, CMD_SERVO_POS_READ,
                            PORT, open_serial, send_command)

DEFAULT_IDS = "1-18"


def parse_ids(text: str) -> list[int]:
    """把 --ids 参数解析成舵机 ID 列表（去重、升序）。"""
    ids: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            if not low.isdigit() or not high.isdigit():
                raise argparse.ArgumentTypeError(f"无效的 ID 区间: {part!r}")
            ids.extend(range(int(low), int(high) + 1))
        elif part.isdigit():
            ids.append(int(part))
        else:
            raise argparse.ArgumentTypeError(f"无效的 ID: {part!r}")
    result = sorted(set(ids))
    if not result:
        raise argparse.ArgumentTypeError("--ids 至少需要一个 ID")
    return result


def ping(ser: serial.Serial, servo_id: int) -> bool:
    """查询舵机是否在线。"""
    frame = send_command(ser, servo_id, CMD_SERVO_ID_READ)
    return frame is not None


def read_position(ser: serial.Serial, servo_id: int) -> int | None:
    """读取单个舵机位置，无应答返回 None。"""
    frame = send_command(ser, servo_id, CMD_SERVO_POS_READ)
    if frame is None or len(frame) < 8:
        return None
    return frame[5] | (frame[6] << 8)


def discover(ser: serial.Serial, ids: list[int]) -> list[int]:
    """扫描 ids，返回在线舵机 ID 列表。"""
    online = [servo_id for servo_id in ids if ping(ser, servo_id)]
    time.sleep(0.02)
    return online


def main():
    parser = argparse.ArgumentParser(description="持续打印串行总线舵机位置（只读）")
    parser.add_argument("--port", default=PORT, help=f"串口设备 (默认 {PORT})")
    parser.add_argument("--baud", type=int, default=BAUD,
                        help=f"波特率 (默认 {BAUD})")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="刷新间隔秒，0 为最快 (默认 0)")
    parser.add_argument("--ids", type=parse_ids, default=DEFAULT_IDS,
                        help=f"要监视的舵机 ID (默认 {DEFAULT_IDS})")
    args = parser.parse_args()

    if args.baud <= 0:
        parser.error("--baud 必须大于 0")
    if not 0 <= args.interval <= 10:
        parser.error("--interval 必须在 0-10 之间")

    try:
        ser = open_serial(args.port, args.baud)
    except serial.SerialException as exc:
        print(f"无法打开串口 {args.port}: {exc}")
        print("请检查 USB 总线模块是否连接，或使用 --port 指定正确设备。")
        sys.exit(1)
    time.sleep(0.2)

    print(f"监视 ID: {args.ids}  @ {args.baud} baud（只读，Ctrl+C 退出）")
    print("正在扫描在线舵机...")
    online = discover(ser, args.ids)
    if online:
        print(f"在线: {online}；离线 ID: "
              f"{[sid for sid in args.ids if sid not in online]}")
    else:
        print("未发现在线舵机，将每 2 秒重新扫描一次。")

    offline_count = 0
    # 刷新率统计：EMA 平滑实际打印间隔，避免逐帧抖动
    ema_cycle = None
    last_t = None
    try:
        while True:
            if not online:
                time.sleep(2.0)
                online = discover(ser, args.ids)
                if online:
                    print(f"\r\x1b[2K发现在线舵机: {online}", flush=True)
                last_t = None  # 重新计时，避免空档拉低 EMA
                ema_cycle = None
                continue

            positions: dict[int, int] = {}
            for servo_id in online:
                pos = read_position(ser, servo_id)
                if pos is not None:
                    positions[servo_id] = pos

            if positions:
                offline_count = 0
            else:
                offline_count += 1

            parts = [
                f"{sid}:{positions[sid]}" if sid in positions else f"{sid}:-"
                for sid in online
            ]
            now = time.time()
            if last_t is not None:
                dt = now - last_t
                ema_cycle = dt if ema_cycle is None else 0.8 * ema_cycle + 0.2 * dt
            last_t = now
            hz = 1.0 / ema_cycle if ema_cycle else 0.0
            line = (f"[{time.strftime('%H:%M:%S')}] {hz:4.1f}Hz  "
                    + " ".join(parts))
            print(f"\r\x1b[2K{line}", end="", flush=True)

            # 连续多次完全无应答时重新扫描，自动适应总线上的舵机变化
            if offline_count >= 5:
                print("\r\x1b[2K舵机全部失联，重新扫描在线舵机...", flush=True)
                online = discover(ser, args.ids)
                offline_count = 0
                last_t = None
                ema_cycle = None

            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\r\x1b[2K已停止监视。")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
