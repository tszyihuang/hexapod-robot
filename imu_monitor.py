#!/usr/bin/env python3
"""IMU 模块监视器（只读，覆盖式打印）——亚博（Yahboom）IMU-Sensor 模块。

通过 USB 串口直连亚博 IMU 姿态传感器（CH340，默认 115200 8N1，6/9/10 轴，
内置 32 位处理器 + 卡尔曼滤波），在控制台以「覆盖式」方式持续刷新
姿态角（Roll/Pitch/Yaw）、四元数、加速度、角速度与磁场，不滚动刷屏。
按 Ctrl+C 退出。

本程序只读取串口、不发送任何指令，运行期间不会影响 IMU 或机器人。

用法：
  python3 imu_monitor.py [--port /dev/ttyUSB1] [--baud 115200]
                         [--interval 0] [--plain]

未指定 --port 时会自动探测：枚举 /dev/ttyUSB*、/dev/ttyACM* 及
/dev/serial/by-id/*，挑选第一个能解出本协议合法帧的设备。

协议（已对照亚博官方 STM32 教程源码核实）：
  帧结构（无帧尾，帧长由「长度」字段给出）
      | 0x7E | 0x23 | 长度 | 功能 | 数据… | 校验和 |
  - 0x7E 0x23：帧头（官方 FRAME_HEAD1 / FRAME_HEAD2）
  - 长度：整帧字节数（从 0x7E 到校验和，含首尾）
  - 校验和 = (0x7E + 0x23 + 长度 + 功能 + 各数据字节之和) 低 8 位
  - 功能 0x04：九轴原始数据，9 个 int16 小端 —— ax ay az gx gy gz mx my mz
  - 功能 0x16：四元数，4 个 float32 小端 —— q0(w) q1(x) q2(y) q3(z)
  - 功能 0x26：欧拉角，3 个 float32 小端 —— roll pitch yaw（弧度）
  - （10 轴另有气压/版本/回传状态帧，9 轴不输出气压）
  换算（官方比例）：
  - 加速度 g    = raw × 16 / 32767
  - 角速度 °/s  = raw × 2000 / 32767（±2000°/s 量程）
  - 磁场   µT   = raw × 800 / 32767

  参考：
  - https://github.com/YahboomTechnology/IMU-Sensor
  - 官方 STM32 串口读取教程《2. Serial communication / 1.STM32.pdf》
"""

import argparse
import glob
import math
import os
import shutil
import struct
import sys
import time
import unicodedata

import serial

DEFAULT_PORT = None  # None = 自动探测
DEFAULT_BAUD = 115200

FRAME_HEAD1 = 0x7E   # 帧头第 1 字节（官方 FRAME_HEAD1）
FRAME_HEAD2 = 0x23   # 帧头第 2 字节（官方 FRAME_HEAD2）
HEAD_BYTES = bytes([FRAME_HEAD1, FRAME_HEAD2])   # b"\x7e\x23"

FUNC_RAW = 0x04      # 九轴原始数据（加速度+角速度+磁场，int16 × 9）
FUNC_QUAT = 0x16     # 四元数（float32 × 4）
FUNC_EULER = 0x26    # 欧拉角（float32 × 3，弧度）

ACCEL_RATIO = 16.0 / 32767.0     # 加速度 raw → g（±16g 量程）
GYRO_RATIO = 2000.0 / 32767.0    # 角速度 raw → °/s（±2000°/s 量程）
MAG_RATIO = 800.0 / 32767.0      # 磁场 raw → µT（±800µT 量程）


def candidate_ports() -> list[str]:
    """按优先级返回候选串口设备：显式 by-id 链接 → ttyACM → ttyUSB。"""
    ports = []
    ports += sorted(glob.glob("/dev/serial/by-id/*"))
    ports += sorted(glob.glob("/dev/ttyACM*"))
    ports += sorted(glob.glob("/dev/ttyUSB*"))
    seen = set()
    result = []
    for p in ports:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def looks_like_imu(ser: serial.Serial, probe_s: float = 0.6) -> bool:
    """短时间读取，判断该串口是否在流式输出本协议（0x7E 0x23 …）的帧。"""
    ser.reset_input_buffer()
    deadline = time.time() + probe_s
    data = b""
    while time.time() < deadline:
        chunk = ser.read(256)
        if not chunk:
            continue
        data += chunk
        if HEAD_BYTES in data:
            return True
    return HEAD_BYTES in data


def find_imu_port(baud: int, ports: list[str] | None = None) -> str | None:
    """在候选端口中找到 IMU，找不到返回 None。"""
    for port in (ports or candidate_ports()):
        try:
            ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N",
                                stopbits=1, timeout=0.05)
        except (serial.SerialException, OSError):
            continue
        try:
            ser.dtr = False
            ser.rts = False
            if looks_like_imu(ser):
                return port
        except (serial.SerialException, OSError):
            continue
        finally:
            try:
                ser.close()
            except Exception:
                pass
    return None


def open_serial(port: str, baud: int) -> serial.Serial:
    """打开串口（8N1），关闭 DTR/RTS，等模块稳定。"""
    ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N",
                        stopbits=1, timeout=0.1, write_timeout=1)
    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    return ser


def parse_frame(frame: bytes) -> tuple[str, tuple] | None:
    """解析一帧完整数据，返回 (类型, 数值)；非法帧返回 None。

    frame 必须是完整的整帧（0x7E 0x23 长度 功能 数据 校验和），长度字段
    已由 parse_stream 保证。"""
    if len(frame) < 5 or frame[0] != FRAME_HEAD1 or frame[1] != FRAME_HEAD2:
        return None
    if len(frame) != frame[2]:            # 长度字段须等于整帧字节数
        return None
    if (sum(frame[:-1]) & 0xFF) != frame[-1]:   # 累加和校验
        return None
    func = frame[3]
    data = frame[4:-1]
    try:
        if func == FUNC_QUAT and len(data) == 16:
            return "quat", struct.unpack("<4f", data)
        if func == FUNC_EULER and len(data) == 12:
            return "euler", struct.unpack("<3f", data)
        if func == FUNC_RAW and len(data) == 18:
            return "raw", struct.unpack("<9h", data)
    except struct.error:
        return None
    return None


def parse_stream(buf: bytearray) -> list[tuple[str, tuple]]:
    """从字节流中按长度字段切出完整帧并解析，返回 [(类型, 数值), ...]。

    与「按 0x7E 定界切帧」不同，这里依据官方的「长度」字段精确切帧，
    数据区里即使出现 0x7E 字节也不会造成错位。会就地消费 buf 中已解析的
    字节，只把尚未收齐的半帧留在 buf 里。
    """
    out = []
    while True:
        start = buf.find(FRAME_HEAD1)
        if start < 0:
            buf.clear()
            return out
        if start > 0:
            del buf[:start]               # 丢弃帧头前的残留字节
        if len(buf) < 4:                  # 头2 + 长度 + 功能
            return out
        if buf[1] != FRAME_HEAD2:         # 头2 不匹配，跳过这个 0x7E 再找
            del buf[:1]
            continue
        length = buf[2]
        if length < 5 or length > 64:     # 非法长度，丢弃头字节再找
            del buf[:1]
            continue
        if len(buf) < length:             # 帧未收齐，等待更多数据
            return out
        frame = bytes(buf[:length])
        del buf[:length]
        parsed = parse_frame(frame)
        if parsed is not None:
            out.append(parsed)
    return out


def disp_width(text: str) -> int:
    """字符串在终端占用的显示列数（CJK 等全角字符按 2 列计）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in text)


def fit_width(text: str, width: int) -> str:
    """把文本截断到不超过 width 显示列，避免在终端里折行。"""
    if disp_width(text) <= width:
        return text
    out = []
    used = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + cw > width:
            break
        out.append(ch)
        used += cw
    return "".join(out)


class Screen:
    """覆盖式多行显示：每次重写上一次打印的若干行，避免滚动刷屏。

    所有输出行都会按终端宽度截断，保证每行只占一个物理行，从而让光标
    上移（\\x1b[N A）重写的行数计算始终正确，不会因折行而刷屏。
    """

    def __init__(self, plain: bool = False):
        self.plain = plain or not sys.stdout.isatty()
        try:
            cols = shutil.get_terminal_size().columns
        except Exception:
            cols = 80
        # 留 1 列余量，避免写到最后一列时触发终端折行
        self.width = max(20, min(cols, 300)) - 1
        self.nlines = 0

    def render(self, lines: list[str]):
        if self.plain:
            # 单行覆盖模式：合并为一行，用 \r 覆盖（不截断，保留完整数据）
            sys.stdout.write("\r\x1b[2K" + " | ".join(lines))
            sys.stdout.flush()
            return
        # 多行覆盖：光标上移 nlines 行后逐行清屏重写
        lines = [fit_width(ln, self.width) for ln in lines]
        if self.nlines:
            sys.stdout.write(f"\x1b[{self.nlines}A")
        buf = []
        for i, line in enumerate(lines):
            buf.append("\r\x1b[2K" + line)
            if i != len(lines) - 1:
                buf.append("\n")
        # 若新行数比旧行数少，清掉多余的旧行
        if self.nlines > len(lines):
            for _ in range(self.nlines - len(lines)):
                buf.append("\n\r\x1b[2K")
        sys.stdout.write("".join(buf))
        sys.stdout.flush()
        self.nlines = len(lines)


def build_lines(state: dict, hz: float, port_short: str, baud: int) -> list[str]:
    """根据最新数据生成固定行数的显示内容（尽量紧凑，避免折行）。"""
    euler = state.get("euler")
    quat = state.get("quat")
    raw = state.get("raw")

    ts = time.strftime("%H:%M:%S")
    head = f"[{ts}] {hz:5.1f}Hz  {port_short}@{baud}  (只读, Ctrl+C 退出)"

    def euler_line():
        if euler is None:
            return "姿态角 Roll/Pitch/Yaw(°) : 等待数据…"
        r, p, y = [v * 180.0 / math.pi for v in euler]
        return ("姿态角 Roll/Pitch/Yaw(°) : "
                + "  ".join(f"{v:8.2f}" for v in (r, p, y)))

    def quat_line():
        if quat is None:
            return "四元数 w/x/y/z            : 等待数据…"
        w, x, y, z = quat
        return ("四元数 w/x/y/z            : "
                + "  ".join(f"{v:9.4f}" for v in (w, x, y, z)))

    def accel_line():
        if raw is None:
            return "加速度 ax/ay/az (g)       : 等待数据…"
        ax, ay, az = raw[0:3]
        return ("加速度 ax/ay/az (g)       : "
                + "  ".join(f"{v * ACCEL_RATIO:8.3f}" for v in (ax, ay, az)))

    def gyro_line():
        if raw is None:
            return "角速度 gx/gy/gz (°/s)     : 等待数据…"
        gx, gy, gz = raw[3:6]
        return ("角速度 gx/gy/gz (°/s)     : "
                + "  ".join(f"{v * GYRO_RATIO:8.2f}" for v in (gx, gy, gz)))

    def mag_line():
        if raw is None:
            return "磁场 mx/my/mz (µT)        : 等待数据…"
        mx, my, mz = raw[6:9]
        return ("磁场 mx/my/mz (µT)        : "
                + "  ".join(f"{v * MAG_RATIO:8.1f}" for v in (mx, my, mz)))

    return [head, euler_line(), quat_line(), accel_line(), gyro_line(), mag_line()]


def main():
    parser = argparse.ArgumentParser(description="覆盖式打印亚博 IMU 数据（只读）")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help="串口设备（默认自动探测 /dev/ttyUSB* 等）")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"波特率（默认 {DEFAULT_BAUD}）")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="刷新间隔秒，0 为数据到达即刷（默认 0）")
    parser.add_argument("--plain", action="store_true",
                        help="单行覆盖模式（非交互终端或 Windows 旧终端用）")
    args = parser.parse_args()

    if args.baud <= 0:
        parser.error("--baud 必须大于 0")
    if not 0 <= args.interval <= 10:
        parser.error("--interval 必须在 0-10 之间")

    # 确定端口：显式指定或自动探测
    port = args.port
    if port is None:
        print("正在探测 IMU 串口（115200 8N1）…")
        port = find_imu_port(args.baud)
        if port is None:
            print("未找到 IMU 设备。请确认模块已连接，或用 --port 指定。")
            sys.exit(1)
        print(f"探测到 IMU 设备: {port}")

    try:
        ser = open_serial(port, args.baud)
    except (serial.SerialException, OSError) as exc:
        print(f"无法打开串口 {port}: {exc}")
        sys.exit(1)
    ser.reset_input_buffer()

    # 显示用的短设备名（by-id 符号链接解析成 ttyUSBx）
    try:
        port_short = os.path.basename(os.path.realpath(port)) or port
    except Exception:
        port_short = os.path.basename(port) or port

    screen = Screen(plain=args.plain)
    state: dict = {}
    buf = bytearray()
    ema_cycle = None
    last_t = None
    last_frame_t = time.time()

    print(f"正在监视 {port}（Ctrl+C 退出）")
    try:
        while True:
            # 读走当前已到数据；无数据时阻塞最多约一个 timeout，避免空转
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buf += chunk
                for kind, value in parse_stream(buf):
                    state[kind] = value
                    last_frame_t = time.time()

            # 数据流中断超过 2 秒：提示并等待恢复
            if time.time() - last_frame_t > 2.0:
                now = time.time()
                if last_t is not None:
                    dt = now - last_t
                    ema_cycle = dt if ema_cycle is None else 0.8 * ema_cycle + 0.2 * dt
                last_t = now
                hz = 1.0 / ema_cycle if ema_cycle else 0.0
                lines = build_lines(state, hz, port_short, args.baud)
                screen.render(lines + ["  （无数据，正在等待 IMU 数据流…）"])
                time.sleep(0.2)
                continue

            now = time.time()
            if last_t is not None:
                dt = now - last_t
                ema_cycle = dt if ema_cycle is None else 0.8 * ema_cycle + 0.2 * dt
            last_t = now
            hz = 1.0 / ema_cycle if ema_cycle else 0.0
            lines = build_lines(state, hz, port_short, args.baud)
            screen.render(lines)

            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止监视。")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
