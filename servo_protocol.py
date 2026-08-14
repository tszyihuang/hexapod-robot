#!/usr/bin/env python3
"""LX-15D/LX-16A 串行总线舵机直连协议层（USB 转总线模块，115200 8N1）。

帧格式：0x55 0x55 | ID | 长度 | 指令 | 参数 | 校验和
  - 长度 = 参数个数 + 3
  - 校验和 = ~(ID + 长度 + 指令 + 参数之和) 的低 8 位

本模块只负责帧的构造、解析与串口收发，不包含任何机器人姿态、
步态或交互逻辑；servo_driver 与 servo_monitor 共用这一层。
"""

import time

import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200

CMD_SERVO_MOVE_TIME_WRITE = 0x01
CMD_SERVO_ID_READ = 0x0E
CMD_SERVO_POS_READ = 0x1C
CMD_SERVO_LOAD_OR_UNLOAD_WRITE = 0x1F


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
    """发送指令并等待该舵机的合法应答，超时返回 None。

    只返回 ID 与指令都匹配、且校验和合法的完整应答帧；写指令（移动、
    加载/卸载）通常无应答，应使用 write_commands。

    用 ser.read(1) 精确等待首个应答字节，而非 ser.read(4096)：pyserial 的
    read(size) 在有超时时会一直等到凑满 size 字节或超时才返回，读 4096
    字节会白白阻塞整个 timeout_s（20ms），把 18 舵机扫描拖到约 2.7Hz。
    read(1) 由内核 select 唤醒，没有 sleep 轮询的粒度损失。
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
            # 精确阻塞到首个应答字节或 timeout_s；随后读走当前已到的其余字节
            first = ser.read(1)
            if not first:
                continue  # read(1) 超时，尚未等到数据
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
    """把多条写指令拼成一次写入（无应答，不读取串口）。

    flush=True 时等待整块数据全部送出（tcdrain）；对 18 条移动帧而言
    这会阻塞约 15.6ms。50Hz 控制环的每帧周期只有 20ms，因此热路径
    （send_move）传 flush=False，把写入交给内核缓冲：50Hz 时的生产速率
    9000 B/s 低于 115200 的线速率 11520 B/s，不会积压。
    """
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
