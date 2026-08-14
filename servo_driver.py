#!/usr/bin/env python3
"""Spiderbot 六足机器人舵机驱动程序（统一交互控制台）。

把「读取位置」「站立」「展平」「步态行走/横移」等功能整合到一个程序里。
启动后在控制台输入命令：

  read     持续读取 1-18 号舵机位置并覆盖打印，无响应自动重试（Ctrl+C 停止）
  stand    让机器人站立（内置站立姿态）
  flatten  让机器人展平（内置展平姿态）
  walk     以三角步态行走，可选速度 mm/s 与步幅 mm（默认 350 / 100，正数前进、负数后退）
  strafe   以三角步态横向移动，可选速度 mm/s 与步幅 mm（默认 350 / 100，正数左移、负数右移）
  move     以身体坐标系速度向量平移，可选 vx vy（mm/s，默认 350 0；vx 前正、vy 左正）
  turn     原地旋转，可选角速度 deg/s 与单周期转角 deg（默认 80 / 20，正数左转、负数右转）
  relax    卸载所有舵机（失能，可手动转动）
  help     显示本帮助
  quit     退出程序（退出前自动失能全部舵机）

站立/展平移动完成后舵机会保持通电并维持姿态；如果想让舵机失能、
方便手动摆位，输入 relax。

协议（LX-15D/LX-16A 串行总线舵机，USB 转总线模块直连，115200 8N1）：
  帧格式 0x55 0x55 | ID | 长度 | 指令 | 参数 | 校验和
  长度 = 参数个数 + 3；校验和 = ~(ID + 长度 + 指令 + 参数之和) 低 8 位
  0x01 = 单舵机移动（含时间）
  0x0E = 舵机 ID 查询
  0x1C = 舵机位置读取
  0x1F = 舵机加载/卸载（扭矩开关）
"""

import argparse
import math
import sys
import time
from collections.abc import Callable

import serial

from kinematics import HexapodIK, TripodGait, UnreachableFootError

PORT = "/dev/ttyUSB0"
BAUD = 115200

CMD_SERVO_MOVE_TIME_WRITE = 0x01
CMD_SERVO_ID_READ = 0x0E
CMD_SERVO_POS_READ = 0x1C
CMD_SERVO_LOAD_OR_UNLOAD_WRITE = 0x1F

SERVO_IDS = list(range(1, 19))

# ---------------------------------------------------------------- 步态参数

WALK_SPEED_DEFAULT = 350.0      # mm/s，正数前进、负数后退
WALK_SPEED_LIMIT = 500.0        # mm/s
WALK_STRIDE_DEFAULT = 100.0     # mm，足端在每个支撑/摆动相内扫过的距离（前后各摆 stride/2）
STRAFE_SPEED_DEFAULT = WALK_SPEED_DEFAULT
STRAFE_STRIDE_DEFAULT = 100.0   # mm，横向移动时足端在每个支撑/摆动相内扫过的左右距离
TURN_SPEED_DEFAULT = 80.0      # deg/s，正数俯视逆时针（左转）、负数右转
TURN_SPEED_LIMIT = 100.0        # deg/s
TURN_STEP_DEFAULT = 20.0       # deg，每个步态周期身体旋转的角度（左右各转 step/2）
WALK_FRAME_PERIOD = 0.10       # s，帧周期
WALK_MOVE_TIME_MS = 90         # 每帧舵机移动时间，适配直连串口带宽（18 帧 @115200）
WALK_STAND_MOVE_TIME_MS = 100  # 启动/停止时回到自然站立的移动时间

# ---------------------------------------------------------------- 内置数据

# stand、walk、strafe 与 turn 共享同一 IK/步态实例：避免每次命令都重新解析
# physical_config.json，也保证两处使用的物理参数完全一致。
IK = HexapodIK()
GAIT = TripodGait(ik=IK)


def natural_stand_pose() -> dict[int, int]:
    """自然站立姿态：由 IK 按 physical_config.json 生成（含 5° 前偏）。"""
    return {int(sid): int(round(pos)) for sid, pos in GAIT.stand_pose().items()}


# 展平姿态：所有舵机回到配置中的默认展平位置
FLATTEN_POSE = {
    sid: int(IK.config.get("default_flatten_position", 512)) for sid in SERVO_IDS
}

# ---------------------------------------------------------------- 串口协议

def open_serial(port: str, baud: int) -> serial.Serial:
    """打开串口（8N1），关闭 DTR/RTS 并等待 USB 总线模块稳定。

    打开失败抛 serial.SerialException，由调用方决定如何提示与退出。
    """
    ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N",
                        stopbits=1, timeout=0.2, write_timeout=1)
    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    return ser


def build_frame(servo_id: int, cmd: int,
                params: tuple[int, ...] = ()) -> bytes:
    """构造直连协议帧：0x55 0x55 | ID | 长度 | 指令 | 参数 | 校验和。

    长度 = 参数个数 + 3；校验和 = ~(ID + 长度 + 指令 + 参数之和) 低 8 位。
    """
    length = len(params) + 3
    frame = bytearray([0x55, 0x55, servo_id, length, cmd])
    frame += bytes(params)
    frame.append((~sum(frame[2:])) & 0xFF)
    return bytes(frame)


def iter_frames(data: bytes):
    """从数据流中切出校验和合法的完整应答帧。"""
    i = 0
    n = len(data)
    while i + 6 <= n:
        if data[i : i + 2] != b"\x55\x55":
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


def send_command(ser: serial.Serial, servo_id: int, cmd: int,
                 params: tuple[int, ...] = (), wait_s: float = 0.08,
                 timeout_s: float = 0.02) -> bytes | None:
    """发送指令并读取该舵机的应答，超时返回 None。

    只返回 ID 与指令都匹配、且校验和合法的完整应答帧；移动/卸载等
    写指令通常无应答，应使用 write_command。
    """
    frame = build_frame(servo_id, cmd, params)
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
                for candidate in iter_frames(data):
                    if candidate[2] == servo_id and candidate[4] == cmd:
                        return candidate
        return None
    finally:
        ser.timeout = old_timeout


def write_command(ser: serial.Serial, servo_id: int, cmd: int,
                  params: tuple[int, ...] = ()):
    """发送写指令（无应答），不读取串口。"""
    ser.write(build_frame(servo_id, cmd, params))
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


def read_servos(ser: serial.Serial, ids: list[int],
                wait_s: float = 0.08) -> list[tuple[int, int]]:
    """逐个读取舵机位置，返回应答成功的 [(舵机ID, 位置)]。

    直连协议没有多舵机批量读取指令，只能按 ID 逐个查询；不在总线上
    的舵机会等到各自的超时时间，因此 18 个舵机全离线时一次全量读取
    约需 wait_s * 18 秒。
    """
    result = []
    for servo_id in ids:
        frame = send_command(ser, servo_id, CMD_SERVO_POS_READ, wait_s=wait_s)
        if frame is None or len(frame) < 8:
            continue
        pos = frame[5] | (frame[6] << 8)
        result.append((servo_id, pos))
    return result


RETRY_WAITS = (0.08, 0.20)


def read_with_retry(ser: serial.Serial, ids: list[int],
                    waits: tuple[float, ...] = RETRY_WAITS) -> list[tuple[int, int]]:
    """读取舵机位置，未应答的舵机用更长等待自动重试一次。"""
    result = read_servos(ser, ids, wait_s=waits[0])
    answered = {servo_id for servo_id, _ in result}
    missing = [servo_id for servo_id in ids if servo_id not in answered]
    if not missing:
        return result
    drain_until_quiet(ser)  # 清掉迟到/残留的半帧，避免污染重试结果
    return result + read_servos(ser, missing, wait_s=waits[-1])


def send_move(ser: serial.Serial, servos: list[tuple[int, int]], time_ms: int):
    """让多个舵机在 time_ms 毫秒内移动到各自位置（0x01，无应答）。

    直连协议没有多舵机移动指令，这里把所有单舵机移动帧拼成一次写入，
    保证 18 个舵机收到几乎相同的起始时间。时间参数范围 0-30000ms。
    """
    time_ms = max(0, min(int(time_ms), 30000))
    block = bytearray()
    for servo_id, pos in servos:
        pos = max(0, min(int(pos), 1000))
        block += build_frame(
            servo_id, CMD_SERVO_MOVE_TIME_WRITE,
            (pos & 0xFF, (pos >> 8) & 0xFF,
             time_ms & 0xFF, (time_ms >> 8) & 0xFF))
    if block:
        ser.write(bytes(block))
        ser.flush()


def unload_servos(ser: serial.Serial, ids: list[int]):
    """卸载（失能）所有舵机，参数 0 = 关闭扭矩。"""
    for servo_id in ids:
        write_command(ser, servo_id, CMD_SERVO_LOAD_OR_UNLOAD_WRITE, (0x00,))


def load_servos(ser: serial.Serial, ids: list[int]):
    """加载（使能）所有舵机，参数 1 = 打开扭矩。"""
    for servo_id in ids:
        write_command(ser, servo_id, CMD_SERVO_LOAD_OR_UNLOAD_WRITE, (0x01,))


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
    load_servos(ser, [servo_id for servo_id, _ in servos])
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


def _direction_label(vx: float, vy: float) -> str:
    """把身体坐标系速度向量翻译成中文方向名（vx 前正、vy 左正）。"""
    angle = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
    labels = ("前进", "左前方", "左移", "左后方",
              "后退", "右后方", "右移", "右前方")
    return labels[round(angle / 45.0) % 8]


# ---------------------------------------------------------------- 步态循环

def _smoothstep(t: float) -> float:
    """平滑阶跃：t 在 [0, 1] 上从 0 平滑过渡到 1，两端一阶导为 0。"""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _hold_stand(ser: serial.Serial, stand_pose: dict[int, float],
                reason: str):
    """发送自然站立姿态并短暂等待（速度/步幅为 0 等不启动步态的情况）。"""
    load_servos(ser, SERVO_IDS)
    send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
    print(reason)
    try:
        time.sleep(1.0)
    except KeyboardInterrupt:
        print()


def _wait_stand_complete(should_stop: Callable[[], bool] | None,
                         wait_s: float, name: str) -> bool:
    """等待站立到位，期间响应停止信号或 Ctrl+C；返回 False 表示应中止步态。"""
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


def _run_gait_loop(ser: serial.Serial, pose_fn: Callable[..., dict[int, float]],
                   target_amps: tuple[float, ...],
                   stand_pose: dict[int, float],
                   cycle_s: float, ramp_s: float, name: str,
                   should_stop: Callable[[], bool] | None):
    """运行步态帧循环并负责安全停止，供平移与旋转两类步态共用。

    pose_fn(phase, *amps) 返回当前幅值下的 18 个舵机位置；target_amps 是
    爬升结束后的目标幅值（平移为 (x 步幅, y 步幅)，旋转为 (单周期转角,)）。
    启动后幅值在 ramp_s（约 1.5 个周期）内从 0 平滑爬升；收到停止信号后
    不立即回中，而是走完当前半步、把幅值平滑降到 0，等当前摆动组落地后
    停在自然站立姿态，避免中断步态造成的“往回拉”。任何退出路径（Ctrl+C、
    IK 越界、异常）都会回到自然站立姿态。
    """
    start = time.monotonic()
    stopping = False
    stop_phase = 0.0
    amps_at_stop = target_amps
    finish_span = 0.5
    try:
        while True:
            frame_start = time.monotonic()
            elapsed = frame_start - start
            phase = (elapsed / cycle_s) % 1.0

            if stopping:
                # 幅值按平滑阶跃从停止点衰减到 0
                progress = ((phase - stop_phase) % 1.0) / finish_span
                amps = tuple(a * (1.0 - _smoothstep(progress))
                             for a in amps_at_stop)
            elif should_stop is not None and should_stop():
                print("\n收到停止信号，走完当前一步后停下。")
                stopping = True
                stop_phase = phase
                amps_at_stop = tuple(a * _smoothstep(elapsed / ramp_s)
                                     for a in target_amps)
                # 目标为下一个半步边界（0.5 相位），即当前摆动组落地瞬间
                finish_span = math.ceil(phase * 2.0) / 2.0 - phase
                if finish_span < 0.05:
                    finish_span += 0.5
                amps = amps_at_stop
            else:
                # 幅值在约 1.5 个周期内从 0 平滑爬升到目标值
                amps = tuple(a * _smoothstep(elapsed / ramp_s)
                             for a in target_amps)

            pose = pose_fn(phase, *amps)

            if stopping and ((phase - stop_phase) % 1.0) >= finish_span:
                # 幅值已降到 0，摆动组已落地：当前姿态就是自然站立，
                # 直接交给 finally 稳住即可
                break
            send_move(ser, quantize_pose(pose), WALK_MOVE_TIME_MS)

            delay = frame_start + WALK_FRAME_PERIOD - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print()
    except UnreachableFootError as exc:
        print(f"\n目标轨迹超出腿部工作空间，{name}提前停止: {exc}")
    finally:
        send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
        print(f"{name}已停止，已回到自然站立姿态。")
        try:
            time.sleep(WALK_STAND_MOVE_TIME_MS / 1000 + 0.1)
        except KeyboardInterrupt:
            print()


def cmd_move(ser: serial.Serial, vx_mm_s: float, vy_mm_s: float,
             stride_mm: float | None = None,
             *, should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """按身体坐标系速度 (vx, vy) 平移，Ctrl+C 停止并回到自然站立姿态。

    vx 正值前进、负值后退；vy 正值左移、负值右移；两者同时给出时合成
    任意水平方向（等大同号即 45° 斜向）。合速度大小钳制到 WALK_SPEED_LIMIT
    （方向不变）。stride_mm 是足端在每个支撑/摆动相内沿合速度方向扫过的
    距离，默认 WALK_STRIDE_DEFAULT；周期 T = 2*stride/speed，支撑相内足端
    相对身体反向移动，保证足端在地面不打滑。启动时先发送自然站立姿态，
    随后步幅在约 1.5 个周期内从 0 平滑爬升。should_stop 与 stand_first
    的含义同 cmd_gait。
    """
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

    # 合速度方向的单位向量：把总步幅按方向分解到身体 x/y 两个轴。
    ux, uy = vx / speed, vy / speed
    stride_target_x = stride * ux
    stride_target_y = stride * uy
    # 支撑/摆动相各扫过 stride，占半个周期；要让足端在地面不打滑，
    # 周期必须是 2*stride/speed（而不是 stride/speed）。
    cycle_s = 2.0 * stride / speed
    ramp_s = 1.5 * cycle_s
    print(f"开始步态：速度 {speed:.1f} mm/s（{_direction_label(vx, vy)}，"
          f"vx {vx:+.1f}，vy {vy:+.1f}），步幅 {stride:g} mm，"
          f"周期 {cycle_s:.3f} s。按 Ctrl+C 停止。")

    if stand_first:
        # 先站到自然姿态，再进入步态循环
        load_servos(ser, SERVO_IDS)
        send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
        if not _wait_stand_complete(should_stop,
                                    WALK_STAND_MOVE_TIME_MS / 1000 + 0.1,
                                    "步态"):
            return

    _run_gait_loop(ser, GAIT.pose_at, (stride_target_x, stride_target_y),
                   stand_pose, cycle_s, ramp_s, "步态", should_stop)


def cmd_gait(ser: serial.Serial, speed_mm_s: float,
             stride_mm: float | None, *, lateral: bool,
             should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """以三角步态前后行走或左右平移，Ctrl+C 停止并回到自然站立姿态。

    lateral=False 时走前后方向：speed_mm_s 正值前进、负值后退；
    lateral=True 时横向移动：speed_mm_s 正值左移、负值右移。
    本函数把单轴速度换算成身体坐标速度向量 (vx, vy)，其余参数与行为
    见 cmd_move。
    """
    if stride_mm is None:
        stride_mm = STRAFE_STRIDE_DEFAULT if lateral else WALK_STRIDE_DEFAULT
    vx = 0.0 if lateral else float(speed_mm_s)
    vy = float(speed_mm_s) if lateral else 0.0
    cmd_move(ser, vx, vy, stride_mm,
             should_stop=should_stop, stand_first=stand_first)


def cmd_walk(ser: serial.Serial, speed_mm_s: float,
             stride_mm: float | None = None,
             *, should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """以三角步态前后行走；参数含义见 cmd_gait。"""
    cmd_gait(ser, speed_mm_s, stride_mm, lateral=False,
             should_stop=should_stop, stand_first=stand_first)


def cmd_strafe(ser: serial.Serial, speed_mm_s: float,
               stride_mm: float | None = None,
               *, should_stop: Callable[[], bool] | None = None,
               stand_first: bool = True):
    """以三角步态左右平移（正数左移、负数右移）；参数含义见 cmd_gait。"""
    cmd_gait(ser, speed_mm_s, stride_mm, lateral=True,
             should_stop=should_stop, stand_first=stand_first)


def cmd_turn(ser: serial.Serial, speed_deg_s: float,
             step_deg: float | None = None,
             *, should_stop: Callable[[], bool] | None = None,
             stand_first: bool = True):
    """以三角步态原地旋转，Ctrl+C 停止并回到自然站立姿态。

    speed_deg_s 为正值时俯视逆时针（左转）、负值时顺时针（右转），
    范围钳制到 ±TURN_SPEED_LIMIT；step_deg 是每个完整步态周期身体旋转的
    角度，默认取 TURN_STEP_DEFAULT，不做钳制；超出腿部工作空间时打印提示
    并安全回到自然站立姿态。
    周期 T = 2*step/speed：支撑相内身体旋转 step、足端相对身体反向旋转
    step，保证足端在地面不打滑。启动时先发送自然站立姿态，随后转角在
    约 1.5 个周期内从 0 平滑爬升。should_stop 为可选回调，每个步态帧
    检查一次，返回 True 时走完当前半步、把转角平滑降到 0，等当前摆动组
    落地后停在自然站立姿态。stand_first=False 时跳过启动前的站立流程
    直接进入旋转循环，适用于机器人已经处于自然站立姿态的场景。
    """
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
    # 支撑/摆动相各扫过 step，占半个周期；要让足端在地面不打滑，
    # 周期必须是 2*step/speed（而不是 step/speed）。
    cycle_s = 2.0 * abs(step) / abs(speed)
    ramp_s = 1.5 * cycle_s
    print(f"开始原地旋转：角速度 {speed:+.1f} deg/s（{direction}），"
          f"单周期转角 {step:g}°，周期 {cycle_s:.3f} s。"
          f"按 Ctrl+C 停止。")

    if stand_first:
        # 先站到自然姿态，再进入旋转步态循环
        load_servos(ser, SERVO_IDS)
        send_move(ser, quantize_pose(stand_pose), WALK_STAND_MOVE_TIME_MS)
        if not _wait_stand_complete(should_stop,
                                    WALK_STAND_MOVE_TIME_MS / 1000 + 0.1,
                                    "旋转"):
            return

    step_target = math.copysign(step, speed)
    _run_gait_loop(ser,
                   lambda phase, s: GAIT.turn_pose_at(phase, math.radians(s)),
                   (step_target,), stand_pose, cycle_s, ramp_s, "旋转",
                   should_stop)


def print_help():
    print(
        "可用命令:\n"
        "  read     持续读取 1-18 号舵机位置并覆盖打印，无响应自动重试\n"
        "           可选目标间隔(秒): read 0.05，read 0 为最快\n"
        "  stand    站立（内置姿态）\n"
        "  flatten  展平（内置姿态）\n"
        "  walk     以三角步态行走（可选速度 mm/s 和步幅 mm，"
        f"默认 {WALK_SPEED_DEFAULT:g} {WALK_STRIDE_DEFAULT:g}）\n"
        "  strafe   以三角步态左右平移（可选速度 mm/s 和步幅 mm，"
        f"默认 {STRAFE_SPEED_DEFAULT:g} {STRAFE_STRIDE_DEFAULT:g}，"
        "正数左移、负数右移）\n"
        "  turn     原地旋转（可选角速度 deg/s 和单周期转角 deg，"
        f"默认 {TURN_SPEED_DEFAULT:g} {TURN_STEP_DEFAULT:g}，"
        "正数左转、负数右转）\n"
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
                interval = 0.1
                if len(tokens) > 1:
                    interval, error = _parse_float_arg(tokens, 1, "刷新间隔")
                    if error:
                        print(error)
                        continue
                    if not math.isfinite(interval) or interval < 0:
                        print(f"无效的刷新间隔: {tokens[1]!r}（必须为有限的非负数）")
                        continue
                monitor_servos(ser, SERVO_IDS, interval)
            elif command in ("stand", "s") and len(tokens) == 1:
                cmd_pose(ser, "站立", natural_stand_pose(), move_time_ms)
            elif command in ("flatten", "f") and len(tokens) == 1:
                cmd_pose(ser, "展平", FLATTEN_POSE, move_time_ms)
            elif command in ("walk", "w"):
                speed = WALK_SPEED_DEFAULT
                stride = None
                if len(tokens) > 1:
                    speed, error = _parse_float_arg(tokens, 1, "速度")
                    if error:
                        print(error)
                        continue
                if len(tokens) > 2:
                    stride, error = _parse_float_arg(tokens, 2, "步幅")
                    if error:
                        print(error)
                        continue
                cmd_walk(ser, speed, stride)
            elif command in ("strafe", "slide", "lr"):
                speed = STRAFE_SPEED_DEFAULT
                stride = None
                if len(tokens) > 1:
                    speed, error = _parse_float_arg(tokens, 1, "速度")
                    if error:
                        print(error)
                        continue
                if len(tokens) > 2:
                    stride, error = _parse_float_arg(tokens, 2, "步幅")
                    if error:
                        print(error)
                        continue
                cmd_strafe(ser, speed, stride)
            elif command in ("move", "mv"):
                vx = WALK_SPEED_DEFAULT
                vy = 0.0
                if len(tokens) > 1:
                    vx, error = _parse_float_arg(tokens, 1, "前向速度")
                    if error:
                        print(error)
                        continue
                if len(tokens) > 2:
                    vy, error = _parse_float_arg(tokens, 2, "横向速度")
                    if error:
                        print(error)
                        continue
                cmd_move(ser, vx, vy)
            elif command in ("turn", "rotate", "rot"):
                speed = TURN_SPEED_DEFAULT
                step = None
                if len(tokens) > 1:
                    speed, error = _parse_float_arg(tokens, 1, "旋转速度")
                    if error:
                        print(error)
                        continue
                if len(tokens) > 2:
                    step, error = _parse_float_arg(tokens, 2, "单周期转角")
                    if error:
                        print(error)
                        continue
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


def main():
    parser = argparse.ArgumentParser(description="Spiderbot 六足机器人舵机驱动程序")
    parser.add_argument("--port", default=PORT, help=f"串口设备 (默认 {PORT})")
    parser.add_argument("--baud", type=int, default=BAUD,
                        help=f"波特率 (默认 {BAUD})")
    parser.add_argument("--time", type=int, default=1500,
                        help="站立/展平的舵机移动时间毫秒数 (默认 1500)")
    args = parser.parse_args()

    if args.time <= 0:
        parser.error("--time 必须大于 0")
    if args.baud <= 0:
        parser.error("--baud 必须大于 0")

    try:
        ser = open_serial(args.port, args.baud)
    except serial.SerialException as exc:
        print(f"无法打开串口 {args.port}: {exc}")
        print("请检查 USB 总线模块是否连接，或使用 --port 指定正确设备。")
        sys.exit(1)

    try:
        run_repl(ser, args.time)
    finally:
        try:
            # 退出前自动失能全部舵机，避免机器人保持僵硬的最后姿态
            unload_servos(ser, SERVO_IDS)
            time.sleep(0.1)  # 等待总线模块发送完卸载帧
            print("程序退出，18 个舵机已自动失能。")
        except OSError as exc:
            print(f"程序退出时自动失能失败: {exc}")
        finally:
            ser.close()
            print("串口已关闭，程序退出。")


if __name__ == "__main__":
    main()
