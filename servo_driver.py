#!/usr/bin/env python3
"""Spiderbot 六足机器人舵机驱动程序（统一交互控制台）。

把「读取位置」「站立」「展平」三个功能整合到一个程序里。
启动后在控制台输入命令：

  read     持续读取 1-18 号舵机位置并覆盖打印，无响应自动重试（Ctrl+C 停止）
  stand    让机器人站立（内置站立姿态）
  flatten  让机器人展平（内置展平姿态）
  walk     以三角步态行走，可选速度 mm/s 与步幅 mm（默认 30 / 60，正数前进、负数后退）
  relax    卸载所有舵机（失能，可手动转动）
  help     显示本帮助
  quit     退出程序（退出前自动失能全部舵机）

站立/展平移动完成后舵机会保持通电并维持姿态；如果想让舵机失能、
方便手动摆位，输入 relax。

协议（Lobot 舵机控制板，9600 8N1）：
  帧头 0x55 0x55 | 数据长度 | 指令 | 参数
  0x15 = 多舵机位置读取
  0x03 = 多舵机移动
  0x14 = 多舵机卸载
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import serial

from hexapod_ik import HexapodIK
from tripod_gait import TripodGait

PORT = "/dev/ttyUSB0"
BAUD = 9600

FRAME_HEADER = b"\x55\x55"
CMD_MULT_SERVO_POS_READ = 0x15
CMD_SERVO_MOVE = 0x03
CMD_MULT_SERVO_UNLOAD = 0x14

SERVO_IDS = list(range(1, 19))
CONFIG_PATH = Path(__file__).resolve().with_name("physical_config.json")

# ---------------------------------------------------------------- 步态参数

WALK_SPEED_DEFAULT = 30.0      # mm/s，正数前进、负数后退
WALK_SPEED_LIMIT = 90.0        # mm/s
WALK_STRIDE_DEFAULT = 60.0     # mm，足端在每个支撑/摆动相内扫过的距离（前后各摆 stride/2）
WALK_FRAME_PERIOD = 0.10       # s，帧周期
WALK_MOVE_TIME_MS = 90         # 每帧舵机移动时间，适配 9600 串口带宽
WALK_STAND_MOVE_TIME_MS = 800  # 启动/停止时回到自然站立的移动时间

# ---------------------------------------------------------------- 内置数据

def natural_stand_pose() -> dict[int, int]:
    """自然站立姿态：由 IK 按 physical_config.json 生成。

    与 cmd_walk 使用同一套中立足端（含 5° 前偏），保证 stand -> walk
    -> 停止之间姿态连续、无跳变。
    """
    return {
        int(sid): int(round(pos))
        for sid, pos in TripodGait(ik=HexapodIK()).stand_pose().items()
    }


FLATTEN_POSE = {
    # leg1 = 左后
    1: 500, 2: 500, 3: 500,
    # leg2 = 左中
    4: 500, 5: 500, 6: 500,
    # leg3 = 左前
    7: 500, 8: 500, 9: 500,
    # leg4 = 右后
    10: 500, 11: 500, 12: 500,
    # leg5 = 右中
    13: 500, 14: 500, 15: 500,
    # leg6 = 右前
    16: 500, 17: 500, 18: 500,
}

# ---------------------------------------------------------------- 串口协议

def send_frame(ser: serial.Serial, cmd: int, params: list[int],
               wait_s: float = 0.35, timeout_s: float = 0.02) -> bytes:
    """发送指令并读取应答，在数据流中找一条完整有效应答帧后返回。"""
    frame = bytes([0x55, 0x55, len(params) + 2, cmd]) + bytes(params)
    n_servos = params[0] if cmd == CMD_MULT_SERVO_POS_READ and params else None

    def valid_frame(data: bytes):
        i = 0
        while i + 3 <= len(data):
            if data[i : i + 2] == FRAME_HEADER:
                length = data[i + 2]
                end = i + 2 + length
                if end <= len(data) and data[i + 3] == cmd:
                    if cmd == CMD_MULT_SERVO_POS_READ:
                        if length == n_servos * 3 + 3 and data[i + 4] == n_servos:
                            return data[:end]
                    elif cmd == CMD_MULT_SERVO_UNLOAD and length == len(params) + 3:
                        return data[:end]
            i += 1
        return None

    old_timeout = ser.timeout
    ser.timeout = timeout_s
    try:
        ser.reset_input_buffer()
        ser.write(frame)
        ser.flush()
        data = b""
        deadline = time.time() + wait_s
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                data += chunk
                found = valid_frame(data)
                if found is not None:
                    return found
        return data
    finally:
        ser.timeout = old_timeout


def drain_until_quiet(ser: serial.Serial, quiet_s: float = 0.02,
                      max_wait_s: float = 0.25):
    """把串口里仍在到达的数据完整读走，避免残留帧造成错位。"""
    old_timeout = ser.timeout
    ser.timeout = 0.02
    try:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                continue
            time.sleep(quiet_s)
            if not ser.in_waiting:
                break
    finally:
        ser.timeout = old_timeout


def find_frames(data: bytes):
    """按 0x55 0x55 帧头切出完整帧。"""
    frames = []
    i = 0
    while i + 3 < len(data):
        if data[i : i + 2] == FRAME_HEADER:
            length = data[i + 2]
            end = i + 2 + length
            if end <= len(data):
                frames.append(data[i:end])
                i = end
                continue
        i += 1
    return frames


def read_servos(ser: serial.Serial, ids: list[int],
                wait_s: float = 0.35) -> list[tuple[int, int]]:
    """读取多个舵机的位置，返回 [(舵机ID, 位置)]。"""
    params = [len(ids)] + ids
    data = send_frame(ser, CMD_MULT_SERVO_POS_READ, params, wait_s=wait_s)
    result = []
    for frame in find_frames(data):
        if frame[3] != CMD_MULT_SERVO_POS_READ or frame[4] != len(ids):
            continue
        i = 5
        for _ in range(frame[4]):
            if i + 3 > len(frame):
                break
            result.append((frame[i], frame[i + 1] | (frame[i + 2] << 8)))
            i += 3
    return result


RETRY_WAITS = (0.15, 0.35)


def read_with_retry(ser: serial.Serial, ids: list[int],
                    waits: tuple[float, ...] = RETRY_WAITS) -> list[tuple[int, int]]:
    """读取舵机位置，无响应时清空串口并用更长等待自动重试一次。"""
    result = read_servos(ser, ids, wait_s=waits[0])
    if result:
        return result
    drain_until_quiet(ser)  # 清掉迟到/残留的半帧，避免污染重试结果
    return read_servos(ser, ids, wait_s=waits[-1])


def build_move_frame(servos: list[tuple[int, int]], time_ms: int) -> bytes:
    """构造 0x03 多舵机移动指令帧。"""
    n = len(servos)
    frame = bytearray([0x55, 0x55, n * 3 + 5, CMD_SERVO_MOVE, n])
    frame += bytes([time_ms & 0xFF, (time_ms >> 8) & 0xFF])
    for servo_id, pos in servos:
        frame += bytes([servo_id, pos & 0xFF, (pos >> 8) & 0xFF])
    return bytes(frame)


def send_move(ser: serial.Serial, servos: list[tuple[int, int]], time_ms: int):
    frame = build_move_frame(servos, time_ms)
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()


def unload_servos(ser: serial.Serial, ids: list[int]):
    frame = bytes([0x55, 0x55, len(ids) + 3,
                   CMD_MULT_SERVO_UNLOAD, len(ids)]) + bytes(ids)
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()


# ---------------------------------------------------------------- 姿态数据

def load_pose(data: dict) -> list[tuple[int, int]]:
    """校验姿态数据，返回按 ID 排序的 [(舵机ID, 位置)]。"""
    servos = []
    for key, pos in data.items():
        servo_id = int(key)
        pos = int(pos)
        if not 1 <= servo_id <= len(SERVO_IDS):
            raise ValueError(f"舵机 ID 超出范围: {servo_id}")
        if not 0 <= pos <= 1000:
            raise ValueError(f"舵机 {servo_id} 的位置超出范围: {pos}")
        servos.append((servo_id, pos))
    return sorted(servos)


def load_leg_labels() -> dict[int, str]:
    """从 physical_config.json 读取腿名映射，读不到时退回 legN。"""
    labels = {sid: f"leg{(sid - 1) // 3 + 1}" for sid in SERVO_IDS}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            mapping = json.load(fh).get("leg_position_mapping", {})
    except (OSError, ValueError):
        mapping = {}
    roles = ("hip", "femur", "tibia")
    for leg, name in mapping.items():
        try:
            base = (int(leg) - 1) * 3
        except ValueError:
            continue
        for offset, role in enumerate(roles):
            labels[base + offset + 1] = f"{name}.{role}"
    return labels


# ---------------------------------------------------------------- 交互命令

def monitor_servos(ser: serial.Serial, ids: list[int], interval: float = 0.1):
    """持续读取舵机位置，无响应自动重试，覆盖刷新同一行，Ctrl+C 停止。"""
    print(f"持续读取 {ids[0]}-{ids[-1]} 号舵机，目标间隔 {interval:g} 秒"
          f"（按 Ctrl+C 停止）")
    drain_until_quiet(ser)  # 只在进入时清空一次，循环内不再清空
    ema_cycle = None
    try:
        while True:
            start = time.time()
            result = read_with_retry(ser, ids)
            if result:
                positions = dict(result)
                line = " ".join(f"{sid}:{positions[sid]}" for sid in ids
                                if sid in positions)
            else:
                line = "(无响应)"
            elapsed = time.time() - start
            ema_cycle = elapsed if ema_cycle is None else \
                0.8 * ema_cycle + 0.2 * elapsed
            hz = 1.0 / ema_cycle if ema_cycle > 0 else 0.0
            print(f"\r\x1b[2K[{time.strftime('%H:%M:%S')}] {hz:4.1f}Hz  {line}",
                  end="", flush=True)
            if interval > 0:
                time.sleep(max(0.0, interval - (time.time() - start)))
    except KeyboardInterrupt:
        print("\r\x1b[2K已停止读取，回到命令输入。")


def cmd_pose(ser: serial.Serial, name: str, pose: dict, time_ms: int):
    servos = load_pose(pose)
    print(f"{name}姿态: " + " ".join(f"{sid}:{pos}" for sid, pos in servos))
    send_move(ser, servos, time_ms)
    print(f"已发送移动指令（{len(servos)} 个舵机，{time_ms}ms 内到位）")
    wait = max(0.2, time_ms / 1000 + 0.5)
    try:
        time.sleep(wait)
    except KeyboardInterrupt:
        print("等待被中断；移动指令已发送，舵机仍会完成到位。")
        return
    print("移动完成，舵机保持通电维持姿态。")


def cmd_relax(ser: serial.Serial):
    unload_servos(ser, SERVO_IDS)
    print("已发送卸载指令，18 个舵机失能，可手动转动。")


def quantize_pose(pose: dict[int, float]) -> list[tuple[int, int]]:
    """把 IK 输出的浮点舵机位置四舍五入成可序列化的整数位置。"""
    return [(int(servo_id), int(round(pos))) for servo_id, pos in pose.items()]


def cmd_walk(ser: serial.Serial, speed_mm_s: float, stride_mm: float | None = None):
    """以三角步态行走，Ctrl+C 停止并回到自然站立姿态。

    speed_mm_s 为正值时前进、负值时后退，范围钳制到 ±WALK_SPEED_LIMIT。
    stride_mm 为足端在每个支撑/摆动相内扫过的距离（脚前后各摆 stride/2），
    默认取 WALK_STRIDE_DEFAULT，不做钳制；超出腿部工作空间时由 IK 抛
    UnreachableFootError，不静默钳位。
    周期 T = 2*stride/speed：支撑相内身体前进 stride、足端相对身体后退
    stride，保证足端在地面不打滑。启动时先发送自然站立姿态，随后步幅在
    约 1.5 个周期内从 0 平滑爬升。
    """
    speed = float(speed_mm_s)
    if not math.isfinite(speed):
        print(f"无效的速度: {speed_mm_s!r}（必须为有限数值）")
        return
    speed = max(-WALK_SPEED_LIMIT, min(WALK_SPEED_LIMIT, speed))

    gait = TripodGait(ik=HexapodIK(), stride=WALK_STRIDE_DEFAULT)
    if stride_mm is not None:
        try:
            gait.stride = float(stride_mm)
        except (TypeError, ValueError):
            print(f"无效的步幅: {stride_mm!r}")
            return
        if not math.isfinite(gait.stride):
            print(f"无效的步幅: {stride_mm!r}（必须为有限数值）")
            return
    stand_pose = gait.stand_pose()

    if abs(speed) < 1e-9 or gait.stride <= 0:
        send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
        print("速度为 0 或步幅非正，已发送自然站立姿态。")
        try:
            time.sleep(1.0)
        except KeyboardInterrupt:
            print()
        return

    direction = "前进" if speed > 0 else "后退"
    # 支撑/摆动相各扫过 stride，占半个周期；要让足端在地面不打滑，
    # 周期必须是 2*stride/speed（而不是 stride/speed）。
    cycle_s = 2.0 * abs(gait.stride) / abs(speed)
    ramp_s = 1.5 * cycle_s
    print(f"开始步态：速度 {speed:+.1f} mm/s（{direction}），"
          f"步幅 {gait.stride:g} mm，周期 {cycle_s:.3f} s。"
          f"按 Ctrl+C 停止。")

    # 先站到自然姿态，再进入步态循环
    send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
    try:
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("步态启动被中断，保持自然站立姿态。")
        return

    start = time.monotonic()
    try:
        while True:
            frame_start = time.monotonic()
            elapsed = frame_start - start
            phase = (elapsed / cycle_s) % 1.0

            # 约 1.5 个周期内把步幅从 0 平滑爬升到目标值
            ramp_progress = min(1.0, elapsed / ramp_s)
            smooth = ramp_progress * ramp_progress * (3 - 2 * ramp_progress)
            stride_now = math.copysign(gait.stride * smooth, speed)

            pose = gait.pose_at(phase, stride_now)
            send_move(ser, quantize_pose(pose), WALK_MOVE_TIME_MS)

            delay = frame_start + WALK_FRAME_PERIOD - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print()
    finally:
        send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
        print("步态已停止，已回到自然站立姿态。")
        try:
            time.sleep(1.0)
        except KeyboardInterrupt:
            print()


def print_help():
    print(
        "可用命令:\n"
        "  read     持续读取 1-18 号舵机位置并覆盖打印，无响应自动重试\n"
        "           可选目标间隔(秒): read 0.05，read 0 为最快\n"
        "  stand    站立（内置姿态）\n"
        "  flatten  展平（内置姿态）\n"
        "  walk     以三角步态行走（可选速度 mm/s 和步幅 mm，"
        f"默认 {WALK_SPEED_DEFAULT:g} {WALK_STRIDE_DEFAULT:g}）\n"
        "  relax    卸载所有舵机（失能，可手动转动）\n"
        "  help     显示本帮助\n"
        "  quit     退出程序（退出前自动失能全部舵机）"
    )


def run_repl(ser: serial.Serial, move_time_ms: int):
    print("Spiderbot 舵机驱动程序已启动。输入命令（help 查看帮助，quit 退出）。\n")
    while True:
        try:
            line = input("hexapod> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        try:
            if line in ("read", "r") or line.startswith(("read ", "r ")):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        interval = float(parts[1])
                    except ValueError:
                        print(f"无效的刷新间隔: {parts[1]!r}")
                        continue
                    if not math.isfinite(interval) or interval < 0:
                        print(f"无效的刷新间隔: {parts[1]!r}（必须为有限的非负数）")
                        continue
                else:
                    interval = 0.1
                monitor_servos(ser, SERVO_IDS, interval)
            elif line in ("stand", "s"):
                cmd_pose(ser, "站立", natural_stand_pose(), move_time_ms)
            elif line in ("flatten", "f"):
                cmd_pose(ser, "展平", FLATTEN_POSE, move_time_ms)
            elif line in ("walk", "w") or line.startswith(("walk ", "w ")):
                parts = line.split()
                speed = WALK_SPEED_DEFAULT
                stride = None
                if len(parts) > 1:
                    try:
                        speed = float(parts[1])
                    except ValueError:
                        print(f"无效的速度: {parts[1]!r}")
                        continue
                if len(parts) > 2:
                    try:
                        stride = float(parts[2])
                    except ValueError:
                        print(f"无效的步幅: {parts[2]!r}")
                        continue
                cmd_walk(ser, speed, stride)
            elif line == "relax":
                cmd_relax(ser)
            elif line in ("help", "h", "?"):
                print_help()
            elif line in ("quit", "exit", "q"):
                break
            else:
                print(f"未知命令: {line!r}（输入 help 查看可用命令）")
        except (OSError, ValueError) as exc:
            print(f"执行失败: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Spiderbot 六足机器人舵机驱动程序")
    parser.add_argument("--port", default=PORT, help=f"串口设备 (默认 {PORT})")
    parser.add_argument("--baud", type=int, default=BAUD, help="波特率 (默认 9600)")
    parser.add_argument("--time", type=int, default=1500,
                        help="站立/展平的舵机移动时间毫秒数 (默认 1500)")
    args = parser.parse_args()

    if args.time <= 0:
        parser.error("--time 必须大于 0")
    if args.baud <= 0:
        parser.error("--baud 必须大于 0")

    try:
        ser = serial.Serial(port=args.port, baudrate=args.baud, bytesize=8,
                            parity="N", stopbits=1, timeout=0.2, write_timeout=1)
    except serial.SerialException as exc:
        print(f"无法打开串口 {args.port}: {exc}")
        print("请检查控制板是否连接，或使用 --port 指定正确设备。")
        sys.exit(1)

    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    try:
        run_repl(ser, args.time)
    finally:
        try:
            # 退出前自动失能全部舵机，避免机器人保持僵硬的最后姿态
            unload_servos(ser, SERVO_IDS)
            time.sleep(0.1)  # 等待控制板处理完卸载帧
            print("程序退出，18 个舵机已自动失能。")
        except OSError as exc:
            print(f"程序退出时自动失能失败: {exc}")
        finally:
            ser.close()
            print("串口已关闭，程序退出。")


if __name__ == "__main__":
    main()
