#!/usr/bin/env python3
"""Spiderbot 六足机器人串行总线舵机控制程序。

本文件已把原先的 servo_protocol.py、servo_monitor.py、servo_driver.py
合并为一个模块，包含三层内容：

1. 协议层：LX-15D/LX-16A 串行总线舵机帧构造/解析与串口收发；
2. 监视层：只读持续显示舵机位置、扫描在线 ID、失联自动重扫；
3. 控制层：交互控制台 + 步态循环（站立/展平/行走/横移/旋转）。

默认运行交互控制台；加 --monitor 运行只读监视器。本模块仍导出
PORT、BAUD、open_serial、cmd_walk 等接口，供 keyboard_detect.py 导入。

协议：帧格式 0x55 0x55 | ID | 长度 | 指令 | 参数 | 校验和。
长度 = 参数个数 + 3；校验和 = ~(ID + 长度 + 指令 + 参数之和) 低 8 位。
"""

import argparse
import gc
import math
import sys
import time
from collections.abc import Callable

import serial

from kinematics import HexapodIK, TripodGait, UnreachableFootError

# ---------------------------------------------------------------- 协议常量

PORT = "/dev/ttyUSB0"
BAUD = 115200

CMD_SERVO_MOVE_TIME_WRITE = 0x01
CMD_SERVO_ID_READ = 0x0E
CMD_SERVO_POS_READ = 0x1C
CMD_SERVO_LOAD_OR_UNLOAD_WRITE = 0x1F

# ---------------------------------------------------------------- 协议层


def build_frame(servo_id: int, cmd: int,
                params: tuple[int, ...] = ()) -> bytes:
    """构造协议帧：0x55 0x55 | ID | 长度 | 指令 | 参数 | 校验和。"""
    length = len(params) + 3
    frame = bytearray([0x55, 0x55, servo_id, length, cmd])
    frame += bytes(params)
    frame.append((~sum(frame[2:])) & 0xFF)
    return bytes(frame)


def iter_frames(data: bytes):
    """从数据流中切出校验和合法的完整应答帧。"""
    i, n = 0, len(data)
    while i + 6 <= n:
        if data[i:i + 2] != b"\x55\x55":
            i += 1
            continue
        length = data[i + 3]
        total = length + 3
        if length < 3 or i + total > n:
            i += 1
            continue
        frame = data[i:i + total]
        if (~sum(frame[2:-1]) & 0xFF) == frame[-1]:
            yield frame
        i += 1


def open_serial(port: str, baud: int) -> serial.Serial:
    """打开串口（8N1），关闭 DTR/RTS 并等待 USB 总线模块稳定。"""
    ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N",
                        stopbits=1, timeout=0.2, write_timeout=1)
    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    return ser


def send_command(ser: serial.Serial, servo_id: int, cmd: int,
                 params: tuple[int, ...] = (), wait_s: float = 0.08,
                 timeout_s: float = 0.02) -> bytes | None:
    """发送指令并等待 ID/指令都匹配的合法应答，超时返回 None。

    使用 ser.read(1) 精确等待首个应答字节，避免 pyserial read(4096)
    在超时模式下为了凑满字节而阻塞整个 timeout_s。
    """
    old_timeout = ser.timeout
    ser.timeout = timeout_s
    try:
        ser.reset_input_buffer()
        ser.write(build_frame(servo_id, cmd, params))
        ser.flush()
        data = b""
        deadline = time.time() + wait_s
        while time.time() < deadline:
            first = ser.read(1)
            if not first:
                continue
            data += first + ser.read(ser.in_waiting)
            for frame in iter_frames(data):
                if frame[2] == servo_id and frame[4] == cmd:
                    return frame
        return None
    finally:
        ser.timeout = old_timeout


def write_commands(ser: serial.Serial,
                   commands: list[tuple[int, int, tuple[int, ...]]],
                   flush: bool = True):
    """把多条写指令拼成一次写入；flush=False 用于 50Hz 步态热路径。"""
    block = bytearray()
    for servo_id, cmd, params in commands:
        block += build_frame(servo_id, cmd, tuple(params))
    if block:
        ser.write(bytes(block))
        if flush:
            ser.flush()


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


# ---------------------------------------------------------------- 全局配置

SERVO_IDS = list(range(1, 19))
DEFAULT_IDS = "1-18"
RETRY_WAITS = (0.08, 0.20)

WALK_SPEED_DEFAULT = 350.0      # mm/s，正数前进、负数后退
WALK_SPEED_LIMIT = 500.0        # mm/s
WALK_STRIDE_DEFAULT = 100.0     # mm
STRAFE_SPEED_DEFAULT = WALK_SPEED_DEFAULT
STRAFE_STRIDE_DEFAULT = 100.0   # mm
TURN_SPEED_DEFAULT = 80.0       # deg/s，正数俯视逆时针（左转）
TURN_SPEED_LIMIT = 100.0        # deg/s
TURN_STEP_DEFAULT = 20.0        # deg/周期
WALK_FRAME_PERIOD = 0.02        # s，50Hz
WALK_MOVE_TIME_MS = 20          # 每帧舵机移动时间
WALK_STAND_MOVE_TIME_MS = 100   # 站立/回中移动时间
WALK_SCHEDULE_AHEAD_S = 0.0015  # 提前醒来后用忙等补齐帧间隔

assert WALK_MOVE_TIME_MS == round(WALK_FRAME_PERIOD * 1000), (
    "WALK_MOVE_TIME_MS 必须等于 WALK_FRAME_PERIOD 的毫秒数"
)

# stand/walk/strafe/turn 共享同一 IK 与步态实例，避免重复解析配置文件。
IK = HexapodIK()
GAIT = TripodGait(ik=IK)

FLATTEN_POSE = {
    sid: int(IK.config.get("default_flatten_position", 512)) for sid in SERVO_IDS
}


def natural_stand_pose() -> dict[int, int]:
    """由 IK 按 physical_config.json 生成自然站立姿态。"""
    return {int(sid): int(round(pos)) for sid, pos in GAIT.stand_pose().items()}


# ---------------------------------------------------------------- 舵机读写


def read_position(ser: serial.Serial, servo_id: int,
                  wait_s: float = 0.08) -> int | None:
    """读取单个舵机位置，无应答返回 None。"""
    frame = send_command(ser, servo_id, CMD_SERVO_POS_READ, wait_s=wait_s)
    if frame is None or len(frame) < 8:
        return None
    return frame[5] | (frame[6] << 8)


def read_servos(ser: serial.Serial, ids: list[int],
                wait_s: float = 0.08) -> list[tuple[int, int]]:
    """逐个读取舵机位置，返回应答成功的 [(ID, 位置)]。"""
    result = []
    for servo_id in ids:
        pos = read_position(ser, servo_id, wait_s)
        if pos is not None:
            result.append((servo_id, pos))
    return result


def read_with_retry(ser: serial.Serial, ids: list[int],
                    waits: tuple[float, ...] = RETRY_WAITS
                    ) -> list[tuple[int, int]]:
    """读取位置；未应答的舵机清空串口后用更长等待重试一次。"""
    result = read_servos(ser, ids, wait_s=waits[0])
    answered = {servo_id for servo_id, _ in result}
    missing = [servo_id for servo_id in ids if servo_id not in answered]
    if not missing:
        return result
    drain_until_quiet(ser)
    return result + read_servos(ser, missing, wait_s=waits[-1])


def _set_torque(ser: serial.Serial, ids: list[int], enabled: bool):
    """发送所有舵机的加载/卸载指令（0x1F，参数 0/1）。"""
    param = 0x01 if enabled else 0x00
    write_commands(ser, [(sid, CMD_SERVO_LOAD_OR_UNLOAD_WRITE, (param,))
                         for sid in ids])


def load_servos(ser: serial.Serial, ids: list[int]):
    """加载（使能）舵机扭矩。"""
    _set_torque(ser, ids, True)


def unload_servos(ser: serial.Serial, ids: list[int]):
    """卸载（失能）舵机扭矩，可手动转动。"""
    _set_torque(ser, ids, False)


def send_move(ser: serial.Serial, servos: list[tuple[int, int]], time_ms: int):
    """让多个舵机在 time_ms 内移动到各自位置（0x01，无应答）。

    把所有单舵机移动帧拼成一次写入；步态热路径不 flush，交给内核缓冲
    按 115200 线速率发出，避免 18 帧约 15.6ms 阻塞 20ms 控制周期。
    """
    time_ms = max(0, min(int(time_ms), 30000))
    commands = []
    for servo_id, pos in servos:
        pos = max(0, min(int(pos), 1000))
        commands.append((servo_id, CMD_SERVO_MOVE_TIME_WRITE,
                         (pos & 0xFF, (pos >> 8) & 0xFF,
                          time_ms & 0xFF, (time_ms >> 8) & 0xFF)))
    write_commands(ser, commands, flush=False)


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


def quantize_pose(pose: dict[int, float]) -> list[tuple[int, int]]:
    """把 IK 输出的浮点位置四舍五入为整数位置。"""
    return [(int(servo_id), int(round(pos))) for servo_id, pos in pose.items()]


# ---------------------------------------------------------------- 只读监视


def parse_ids(text: str) -> list[int]:
    """解析 --ids（逗号分隔或区间，如 1-6,13-18），去重并升序。"""
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
    return send_command(ser, servo_id, CMD_SERVO_ID_READ) is not None


def discover(ser: serial.Serial, ids: list[int]) -> list[int]:
    """扫描 ids，返回在线舵机 ID 列表。"""
    online = [servo_id for servo_id in ids if ping(ser, servo_id)]
    time.sleep(0.02)
    return online


def monitor_servos(ser: serial.Serial, ids: list[int], interval: float = 0.0):
    """只读持续打印舵机位置；全部失联后自动重新扫描在线 ID。"""
    print(f"监视 ID: {ids}  @ {ser.baudrate} baud（只读，Ctrl+C 退出）")
    print("正在扫描在线舵机...")
    online = discover(ser, ids)
    if online:
        print(f"在线: {online}；离线 ID: "
              f"{[sid for sid in ids if sid not in online]}")
    else:
        print("未发现在线舵机，将每 2 秒重新扫描一次。")

    offline_count = 0
    ema_cycle = None
    last_t = None
    try:
        while True:
            if not online:
                time.sleep(2.0)
                online = discover(ser, ids)
                if online:
                    print(f"\r\x1b[2K发现在线舵机: {online}", flush=True)
                last_t = None
                ema_cycle = None
                continue

            positions = {sid: pos for sid in online
                         if (pos := read_position(ser, sid)) is not None}
            offline_count = 0 if positions else offline_count + 1

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
            print(f"\r\x1b[2K[{time.strftime('%H:%M:%S')}] {hz:4.1f}Hz  "
                  + " ".join(parts), end="", flush=True)

            if offline_count >= 5:
                print("\r\x1b[2K舵机全部失联，重新扫描在线舵机...", flush=True)
                online = discover(ser, ids)
                offline_count = 0
                last_t = None
                ema_cycle = None
            elif interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\r\x1b[2K已停止监视。")


# ---------------------------------------------------------------- 姿态命令


def cmd_pose(ser: serial.Serial, name: str, pose: dict, time_ms: int):
    """加载扭矩、发送姿态并等待到位。"""
    servos = load_pose(pose)
    print(f"{name}姿态: " + " ".join(f"{sid}:{pos}" for sid, pos in servos))
    load_servos(ser, [sid for sid, _ in servos])
    send_move(ser, servos, time_ms)
    print(f"已发送移动指令（{len(servos)} 个舵机，{time_ms}ms 内到位）")
    try:
        time.sleep(max(0.2, time_ms / 1000 + 0.5))
    except KeyboardInterrupt:
        print("等待被中断；移动指令已发送，舵机仍会完成到位。")
        return
    print("移动完成，舵机保持通电维持姿态。")


def cmd_relax(ser: serial.Serial):
    """卸载全部 18 个舵机。"""
    unload_servos(ser, SERVO_IDS)
    print("已发送卸载指令，18 个舵机失能，可手动转动。")


def _direction_label(vx: float, vy: float) -> str:
    """把身体坐标速度向量翻译成中文方向名（vx 前正、vy 左正）。"""
    angle = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
    labels = ("前进", "左前方", "左移", "左后方",
              "后退", "右后方", "右移", "右前方")
    return labels[round(angle / 45.0) % 8]


# ---------------------------------------------------------------- 步态循环


def _smoothstep(t: float) -> float:
    """t 在 [0, 1] 上从 0 平滑过渡到 1。"""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _hold_stand(ser: serial.Serial, stand_pose: dict[int, float],
                reason: str):
    """速度为 0 等无需步态时，发送自然站立姿态并短暂等待。"""
    load_servos(ser, SERVO_IDS)
    send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
    print(reason)
    try:
        time.sleep(1.0)
    except KeyboardInterrupt:
        print()


def _wait_stand_complete(should_stop: Callable[[], bool] | None,
                         wait_s: float, name: str) -> bool:
    """等待站立到位；返回 False 表示应中止步态。"""
    try:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                print(f"\n{name}启动被取消，保持自然站立姿态。")
                return False
            time.sleep(0.02)
    except KeyboardInterrupt:
        print(f"{name}启动被中断，保持自然站立姿态。")
        return False
    return True


def _stand_up(ser: serial.Serial, stand_pose: dict[int, float],
              name: str, should_stop: Callable[[], bool] | None) -> bool:
    """加载扭矩并站到自然姿态；返回 False 表示应中止步态。"""
    load_servos(ser, SERVO_IDS)
    send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
    return _wait_stand_complete(
        should_stop, WALK_STAND_MOVE_TIME_MS / 1000 + 0.1, name)


def _run_gait_loop(ser: serial.Serial, pose_fn: Callable[..., dict[int, float]],
                   target_amps: tuple[float, ...],
                   stand_pose: dict[int, float],
                   cycle_s: float, ramp_s: float, name: str,
                   should_stop: Callable[[], bool] | None):
    """按 50Hz 帧周期运行步态，安全停止并回到自然站立姿态。

    启动后幅值在 ramp_s 内从 0 平滑爬升；停止时走完当前半步并把幅值
    平滑降到 0，最后回到自然站立。循环期间禁用 GC 并统计滑帧。
    """
    start = time.monotonic()
    stopping = False
    stop_phase = 0.0
    amps_at_stop = target_amps
    finish_span = 0.5
    slip_limit_s = WALK_FRAME_PERIOD * 1.25
    frame_slips = 0
    max_gap_s = 0.0
    prev_frame_start = start
    gc.disable()
    try:
        ser.reset_input_buffer()
        while True:
            frame_start = time.monotonic()
            gap = frame_start - prev_frame_start
            prev_frame_start = frame_start
            if gap > slip_limit_s:
                frame_slips += 1
                max_gap_s = max(max_gap_s, gap)

            elapsed = frame_start - start
            phase = (elapsed / cycle_s) % 1.0

            if stopping:
                progress = ((phase - stop_phase) % 1.0) / finish_span
                amps = tuple(a * (1.0 - _smoothstep(progress))
                             for a in amps_at_stop)
            elif should_stop is not None and should_stop():
                print("\n收到停止信号，走完当前一步后停下。")
                stopping = True
                stop_phase = phase
                amps_at_stop = tuple(a * _smoothstep(elapsed / ramp_s)
                                     for a in target_amps)
                finish_span = math.ceil(phase * 2.0) / 2.0 - phase
                if finish_span < 0.05:
                    finish_span += 0.5
                amps = amps_at_stop
            else:
                amps = tuple(a * _smoothstep(elapsed / ramp_s)
                             for a in target_amps)

            pose = pose_fn(phase, *amps)

            if stopping and ((phase - stop_phase) % 1.0) >= finish_span:
                break
            send_move(ser, quantize_pose(pose), WALK_MOVE_TIME_MS)

            deadline = frame_start + WALK_FRAME_PERIOD
            delay = deadline - time.monotonic()
            if delay > 0:
                if delay > WALK_SCHEDULE_AHEAD_S:
                    time.sleep(delay - WALK_SCHEDULE_AHEAD_S)
                while time.monotonic() < deadline:
                    pass
    except KeyboardInterrupt:
        print()
    except UnreachableFootError as exc:
        print(f"\n目标轨迹超出腿部工作空间，{name}提前停止: {exc}")
    finally:
        gc.enable()
        send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
        print(f"{name}已停止，已回到自然站立姿态。")
        print(f"帧间隔统计：滑帧 {frame_slips} 次，最大间隔 {max_gap_s*1000:.1f} ms"
              f"（阈值 {slip_limit_s*1000:.1f} ms）")
        try:
            time.sleep(WALK_STAND_MOVE_TIME_MS / 1000 + 0.1)
        except KeyboardInterrupt:
            print()


# ---------------------------------------------------------------- 步态命令


def cmd_move(ser: serial.Serial, vx_mm_s: float, vy_mm_s: float,
             stride_mm: float | None = None,
             *, should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """按身体坐标速度 (vx, vy) 平移；vx 前正、vy 左正。"""
    vx = float(vx_mm_s)
    vy = float(vy_mm_s)
    if not math.isfinite(vx) or not math.isfinite(vy):
        print(f"无效的速度: ({vx_mm_s!r}, {vy_mm_s!r})（必须为有限数值）")
        return
    speed = math.hypot(vx, vy)
    if speed > WALK_SPEED_LIMIT:
        scale = WALK_SPEED_LIMIT / speed
        vx *= scale
        vy *= scale
        speed = WALK_SPEED_LIMIT

    if stride_mm is None:
        stride = WALK_STRIDE_DEFAULT
    else:
        try:
            stride = float(stride_mm)
        except (TypeError, ValueError):
            print(f"无效的步幅: {stride_mm!r}")
            return
        if not math.isfinite(stride):
            print(f"无效的步幅: {stride_mm!r}（必须为有限数值）")
            return

    stand_pose = GAIT.stand_pose()
    if speed < 1e-9 or stride <= 0:
        _hold_stand(ser, stand_pose, "速度为 0 或步幅非正，已发送自然站立姿态。")
        return

    ux, uy = vx / speed, vy / speed
    stride_target_x = stride * ux
    stride_target_y = stride * uy
    cycle_s = 2.0 * stride / speed
    ramp_s = 1.5 * cycle_s
    print(f"开始步态：速度 {speed:.1f} mm/s（{_direction_label(vx, vy)}，"
          f"vx {vx:+.1f}，vy {vy:+.1f}），步幅 {stride:g} mm，"
          f"周期 {cycle_s:.3f} s。按 Ctrl+C 停止。")

    if stand_first and not _stand_up(ser, stand_pose, "步态", should_stop):
        return

    _run_gait_loop(ser, GAIT.pose_at, (stride_target_x, stride_target_y),
                   stand_pose, cycle_s, ramp_s, "步态", should_stop)


def cmd_gait(ser: serial.Serial, speed_mm_s: float,
             stride_mm: float | None, *, lateral: bool,
             should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """前后行走或左右平移；lateral=True 时正数为左移。"""
    if stride_mm is None:
        stride = STRAFE_STRIDE_DEFAULT if lateral else WALK_STRIDE_DEFAULT
    else:
        stride = stride_mm
    vx = 0.0 if lateral else float(speed_mm_s)
    vy = float(speed_mm_s) if lateral else 0.0
    cmd_move(ser, vx, vy, stride,
             should_stop=should_stop, stand_first=stand_first)


def cmd_walk(ser: serial.Serial, speed_mm_s: float,
             stride_mm: float | None = None,
             *, should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """三角步态前进/后退（正数前进）。"""
    cmd_gait(ser, speed_mm_s, stride_mm, lateral=False,
             should_stop=should_stop, stand_first=stand_first)


def cmd_strafe(ser: serial.Serial, speed_mm_s: float,
               stride_mm: float | None = None,
               *, should_stop: Callable[[], bool] | None = None,
               stand_first: bool = True):
    """三角步态左右平移（正数左移）。"""
    cmd_gait(ser, speed_mm_s, stride_mm, lateral=True,
             should_stop=should_stop, stand_first=stand_first)


def cmd_turn(ser: serial.Serial, speed_deg_s: float,
             step_deg: float | None = None,
             *, should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """三角步态原地旋转（正数俯视逆时针/左转）。"""
    speed = float(speed_deg_s)
    if not math.isfinite(speed):
        print(f"无效的旋转速度: {speed_deg_s!r}（必须为有限数值）")
        return
    speed = max(-TURN_SPEED_LIMIT, min(TURN_SPEED_LIMIT, speed))

    if step_deg is None:
        step = TURN_STEP_DEFAULT
    else:
        try:
            step = float(step_deg)
        except (TypeError, ValueError):
            print(f"无效的单周期转角: {step_deg!r}")
            return
        if not math.isfinite(step):
            print(f"无效的单周期转角: {step_deg!r}（必须为有限数值）")
            return

    stand_pose = GAIT.stand_pose()
    if abs(speed) < 1e-9 or step <= 0:
        _hold_stand(ser, stand_pose,
                    "旋转速度为 0 或单周期转角非正，已发送自然站立姿态。")
        return

    direction = "左转（俯视逆时针）" if speed > 0 else "右转（俯视顺时针）"
    cycle_s = 2.0 * abs(step) / abs(speed)
    ramp_s = 1.5 * cycle_s
    print(f"开始原地旋转：角速度 {speed:+.1f} deg/s（{direction}），"
          f"单周期转角 {step:g}°，周期 {cycle_s:.3f} s。按 Ctrl+C 停止。")

    if stand_first and not _stand_up(ser, stand_pose, "旋转", should_stop):
        return

    step_target = math.copysign(step, speed)
    _run_gait_loop(ser,
                   lambda phase, s: GAIT.turn_pose_at(phase, math.radians(s)),
                   (step_target,), stand_pose, cycle_s, ramp_s, "旋转",
                   should_stop)


# ---------------------------------------------------------------- 交互控制台


def print_help():
    print(
        "可用命令:\n"
        "  read     扫描在线舵机并持续打印位置，全部失联自动重扫\n"
        "           可选目标间隔(秒): read 0.05，read 0 为最快\n"
        "  stand    站立（内置姿态）\n"
        "  flatten  展平（内置姿态）\n"
        "  walk     行走（可选速度 mm/s 和步幅 mm，"
        f"默认 {WALK_SPEED_DEFAULT:g} {WALK_STRIDE_DEFAULT:g}）\n"
        "  strafe   左右平移（可选速度 mm/s 和步幅 mm，"
        f"默认 {STRAFE_SPEED_DEFAULT:g} {STRAFE_STRIDE_DEFAULT:g}，正数左移）\n"
        "  turn     原地旋转（可选角速度 deg/s 和单周期转角 deg，"
        f"默认 {TURN_SPEED_DEFAULT:g} {TURN_STEP_DEFAULT:g}，正数左转）\n"
        "  move     按速度向量平移（可选 vx vy，mm/s，默认 "
        f"{WALK_SPEED_DEFAULT:g} 0；vx 前正、vy 左正）\n"
        "  relax    卸载所有舵机（失能，可手动转动）\n"
        "  help     显示本帮助\n"
        "  quit     退出程序（退出前自动失能全部舵机）"
    )


def _parse_float_arg(tokens: list[str], index: int,
                     name: str) -> tuple[float | None, str | None]:
    """把 tokens[index] 解析为 float，失败返回 (None, 错误提示)。"""
    raw = tokens[index]
    try:
        return float(raw), None
    except ValueError:
        return None, f"无效的{name}: {raw!r}"


def _parse_optional_float(tokens: list[str], index: int, name: str,
                          default: float | None) -> tuple[float | None, str | None]:
    """解析可选 float 参数；缺省返回 default，非法返回错误提示。"""
    if len(tokens) <= index:
        return default, None
    return _parse_float_arg(tokens, index, name)


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

        tokens = line.split()
        command = tokens[0]
        try:
            if command in ("read", "r"):
                interval, error = _parse_optional_float(tokens, 1, "刷新间隔", 0.1)
                if error:
                    print(error)
                elif not math.isfinite(interval) or interval < 0:
                    print(f"无效的刷新间隔: {tokens[1]!r}（必须为有限的非负数）")
                else:
                    monitor_servos(ser, SERVO_IDS, interval)
            elif command in ("stand", "s") and len(tokens) == 1:
                cmd_pose(ser, "站立", natural_stand_pose(), move_time_ms)
            elif command in ("flatten", "f") and len(tokens) == 1:
                cmd_pose(ser, "展平", FLATTEN_POSE, move_time_ms)
            elif command in ("walk", "w"):
                speed, error = _parse_optional_float(tokens, 1, "速度",
                                                     WALK_SPEED_DEFAULT)
                stride, error2 = _parse_optional_float(tokens, 2, "步幅", None)
                if error:
                    print(error)
                elif error2:
                    print(error2)
                else:
                    cmd_walk(ser, speed, stride)
            elif command in ("strafe", "slide", "lr"):
                speed, error = _parse_optional_float(tokens, 1, "速度",
                                                     STRAFE_SPEED_DEFAULT)
                stride, error2 = _parse_optional_float(tokens, 2, "步幅", None)
                if error:
                    print(error)
                elif error2:
                    print(error2)
                else:
                    cmd_strafe(ser, speed, stride)
            elif command in ("move", "mv"):
                vx, error = _parse_optional_float(tokens, 1, "前向速度",
                                                  WALK_SPEED_DEFAULT)
                vy, error2 = _parse_optional_float(tokens, 2, "横向速度", 0.0)
                if error:
                    print(error)
                elif error2:
                    print(error2)
                else:
                    cmd_move(ser, vx, vy)
            elif command in ("turn", "rotate", "rot"):
                speed, error = _parse_optional_float(tokens, 1, "旋转速度",
                                                     TURN_SPEED_DEFAULT)
                step, error2 = _parse_optional_float(tokens, 2, "单周期转角", None)
                if error:
                    print(error)
                elif error2:
                    print(error2)
                else:
                    cmd_turn(ser, speed, step)
            elif command == "relax" and len(tokens) == 1:
                cmd_relax(ser)
            elif command in ("help", "h", "?") and len(tokens) == 1:
                print_help()
            elif command in ("quit", "exit", "q") and len(tokens) == 1:
                break
            else:
                print(f"未知命令: {line!r}（输入 help 查看可用命令）")
        except (OSError, ValueError) as exc:
            print(f"执行失败: {exc}")


# ---------------------------------------------------------------- 入口


def _open_serial_or_exit(port: str, baud: int) -> serial.Serial:
    try:
        return open_serial(port, baud)
    except serial.SerialException as exc:
        print(f"无法打开串口 {port}: {exc}")
        print("请检查 USB 总线模块是否连接，或使用 --port 指定正确设备。")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Spiderbot 六足机器人串行舵机控制/监视程序")
    parser.add_argument("--port", default=PORT, help=f"串口设备 (默认 {PORT})")
    parser.add_argument("--baud", type=int, default=BAUD,
                        help=f"波特率 (默认 {BAUD})")
    parser.add_argument("--time", type=int, default=1500,
                        help="站立/展平的舵机移动时间毫秒数 (默认 1500)")
    parser.add_argument("--monitor", action="store_true",
                        help="运行只读位置监视器（不进入交互控制台）")
    parser.add_argument("--ids", type=parse_ids, default=DEFAULT_IDS,
                        help=f"监视器要监视的舵机 ID (默认 {DEFAULT_IDS})")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="监视器刷新间隔秒，0 为最快 (默认 0)")
    args = parser.parse_args()

    if args.baud <= 0:
        parser.error("--baud 必须大于 0")
    if args.time <= 0:
        parser.error("--time 必须大于 0")
    if args.monitor and not 0 <= args.interval <= 10:
        parser.error("--interval 必须在 0-10 之间")

    ser = _open_serial_or_exit(args.port, args.baud)

    if args.monitor:
        time.sleep(0.2)
        try:
            monitor_servos(ser, args.ids, args.interval)
        finally:
            ser.close()
            print("串口已关闭，程序退出。")
        return

    try:
        run_repl(ser, args.time)
    finally:
        try:
            # 退出前自动失能全部舵机，避免机器人保持僵硬的最后姿态
            unload_servos(ser, SERVO_IDS)
            time.sleep(0.1)
            print("程序退出，18 个舵机已自动失能。")
        except OSError as exc:
            print(f"程序退出时自动失能失败: {exc}")
        finally:
            ser.close()
            print("串口已关闭，程序退出。")


if __name__ == "__main__":
    main()
