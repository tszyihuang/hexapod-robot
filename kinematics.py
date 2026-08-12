#!/usr/bin/env python3
"""六足机器人运动学模块：逆/正运动学求解 + 三角步态生成（身体坐标系）。

所有足端坐标使用身体坐标系：
  - 原点 = 身体中心
  - x 向前为正，y 向左为正，z 向上为正

`physical_config.json` 是本模块唯一且权威的物理参数来源。几何尺寸、
安装位置、腿-舵机映射、左右腿方向规则、关节限位都从这里读取。

每条腿为平面三连杆机构（俯视）：
  - 髋(hip)：绕竖直轴旋转，方向角 theta，0° = 指向身体正前方(+x)，
    俯视向左（逆时针）为正。各腿的髋舵机安装朝向不同：位置 512 时，
    前腿指向 ±45°、中腿指向 ±90°、后腿指向 ±135°（见配置的
    hip_anchor_direction_deg），旋转方向规则六条腿相同。
  - coxa：髋关节之后、沿 theta 方向的水平偏置段（43 mm）。
  - 大腿(femur)：在含 theta 方向的竖直平面内，相对水平面上抬为正（phi）。
  - 小腿(tibia)：相对大腿轴线向下偏为正（kappa），0° = 与大腿同向伸直。

IK 求解返回的是未量化的浮点舵机位置（0-1000 有效量程），由上层在
串口发送前四舍五入，避免取整误差累积后超出回环测试的 0.5 mm 精度要求。

三角步态（TripodGait）：
  - A 组：腿 1（左后）、腿 3（左前）、腿 5（右中），相位偏移 0
  - B 组：腿 2（左中）、腿 4（右后）、腿 6（右前），相位偏移 0.5

每条腿的自然足端由安装点、该腿的自然方向 theta0、水平伸出量 reach 和
身体高度 body_height 生成：
  foot = mount + (coxa + reach) * (cos(theta0), sin(theta0)), z = -H

theta0 按腿分别取值：以各腿髋舵机的安装朝向（位置 512 时的指向）为基准，
前/中/后腿分别为 ±45° / ±90° / ±135°（右腿取负），再向身体前方偏约 5°。

相位约定：全局相位 phase 随时间线性前进，u = (phase - 偏移) mod 1。
  - 支撑相 0 <= u < 0.5：足端从 natural + stride/2 线性后退到
    natural - stride/2，z 恒为 -H。
  - 摆动相 0.5 <= u < 1：足端从 natural - stride/2 前进到
    natural + stride/2，z = -H + step_height*sin(pi*v) 抬起。
  - stride 为身体 x 向（前后）的带符号步幅：正数前进、负数后退；
    lateral_stride 为身体 y 向（左右）的带符号步幅：正数向左、负数向右。
    两者可以同时给出，用来合成任意水平方向的平移；step_height 始终为
    抬起的幅值。
  - 时间映射由调用方决定：周期 T = 2*|步幅|/speed。这样支撑相内身体前进
    stride、足端相对身体后退 stride，足端在地面不打滑。

原地旋转（turn_feet_at/turn_pose_at）：
  - turn_step 为每个完整步态周期身体绕 z 轴旋转的角度（弧度），正数表示
    俯视逆时针（向左），负数表示顺时针（向右）。
  - 足端在支撑相内相对身体从 +turn_step/2 反向旋转到 -turn_step/2；
    摆动相抬脚并沿身体旋转方向转回 +turn_step/2。

本模块只负责生成足端轨迹和舵机姿态，不负责串口与步态时间调度；
启动时的自然站立姿态和约 1.5 个周期的步幅平滑爬升由 servo_driver 的
walk / strafe / turn 命令完成。
"""

from __future__ import annotations

import json
import math
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("physical_config.json")

# 舵机实际有效量程 0-1000（对应 0-240°），控制板标称 10 位量程 0-1023；
# 这里统一按有效量程收进 1000，避免生成硬件无法正确执行或可能被钳制的位置。
SERVO_POS_MAX = 1000

# 数值比较容差
_EPS = 1e-9
_LIMIT_EPS = 1e-6


class UnreachableFootError(ValueError):
    """目标足端位置超出工作空间，或所需的舵机位置超出关节限位。"""


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _rotate_xy(x: float, y: float, angle: float) -> tuple[float, float]:
    """把平面点 (x, y) 绕原点旋转 angle 弧度（俯视逆时针为正）。"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


class HexapodIK:
    """读取 physical_config.json 并求解 6 条腿的逆/正运动学。"""

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        with path.open("r", encoding="utf-8") as fh:
            self.config = json.load(fh)

        self.scale = float(self.config["angle_scale_deg_per_unit"])

        geometry = self.config["leg_geometry_mm"]
        self.coxa = float(geometry["coxa_length_mm"])
        self.femur_len = float(geometry["femur_length_mm"])
        self.tibia_len = float(geometry["tibia_length_mm"])

        # 腿 id -> {role: 舵机 id}
        self.legs: dict[int, dict[str, int]] = {}
        for entry in self.config["legs"]:
            leg_id = int(entry["id"])
            self.legs[leg_id] = {
                role: int(servo_id)
                for role, servo_id in entry["servos"].items()
            }

        # 腿 id -> 左右侧
        self.sides: dict[int, str] = {}
        for side, leg_ids in self.config["leg_side_by_id"].items():
            for leg_id in leg_ids:
                self.sides[int(leg_id)] = "left" if side == "left" else "right"

        # 腿 id -> 髋舵机在 512 位置时腿的指向（身体坐标，度）
        # 每条腿的髋舵机安装朝向不同，缺省时退回旧的左右 ±45° 约定
        anchors = self.config.get("hip_anchor_direction_deg", {})
        self.hip_anchors: dict[int, float] = {}
        for leg_id in self.legs:
            default = 45.0 if self.sides[leg_id] == "left" else -45.0
            self.hip_anchors[leg_id] = float(anchors.get(str(leg_id), default))

        # 腿 id -> 髋关节安装点 (x, y)，髋平面 z = 0
        self.mounts: dict[int, tuple[float, float]] = {}
        mapping = self.config["leg_position_mapping"]
        mounts = self.config["mounting_positions_mm"]
        for leg_str, mount_name in mapping.items():
            pos = mounts[mount_name]
            self.mounts[int(leg_str)] = (float(pos["x"]), float(pos["y"]))

        # role -> side -> (下界, 上界) 的舵机位置限位
        self.limits: dict[str, dict[str, tuple[float, float]]] = {}
        for role in ("hip", "femur", "tibia"):
            by_side = self.config["joint_limits"][role]["position_range"]
            self.limits[role] = {
                "left": (
                    float(by_side["left_legs"][0]),
                    min(float(by_side["left_legs"][1]), SERVO_POS_MAX),
                ),
                "right": (
                    float(by_side["right_legs"][0]),
                    min(float(by_side["right_legs"][1]), SERVO_POS_MAX),
                ),
            }

    # ------------------------------------------------------------- 基础查询

    def side_of(self, leg_id: int) -> str:
        return self.sides[int(leg_id)]

    def mount_of(self, leg_id: int) -> tuple[float, float]:
        return self.mounts[int(leg_id)]

    def _resolve_leg_positions(
        self, leg_id: int, positions
    ) -> tuple[float, float, float]:
        """把 dict(舵机 id -> 位置) 或 (髋, 大腿, 小腿) 序列归一成三角元组。"""
        leg_id = int(leg_id)
        if isinstance(positions, dict):
            servos = self.legs[leg_id]
            return (
                float(positions[servos["hip"]]),
                float(positions[servos["femur"]]),
                float(positions[servos["tibia"]]),
            )
        hip, femur, tibia = positions
        return float(hip), float(femur), float(tibia)

    # ------------------------------------------------------------- 角度换算

    def angles_from_positions(
        self, leg_id: int, positions
    ) -> tuple[float, float, float]:
        """舵机位置 -> (theta, phi, kappa)，角度为弧度。

        换算规则与 physical_config.json 的 position_to_angle_formula 一致：
        512 对应左腿 theta=+45° / 右腿 theta=-45°，phi=0°，kappa=+45°。
        """
        hip, femur, tibia = self._resolve_leg_positions(leg_id, positions)
        scale = self.scale
        anchor = self.hip_anchors[int(leg_id)]
        theta = math.radians(anchor + (hip - 512) * scale)
        if self.side_of(leg_id) == "left":
            phi = math.radians(-(femur - 512) * scale)
            kappa = math.radians(45 - (tibia - 512) * scale)
        else:
            phi = math.radians((femur - 512) * scale)
            kappa = math.radians(45 + (tibia - 512) * scale)
        return theta, phi, kappa

    def _positions_from_angles(
        self, leg_id: int, theta: float, phi: float, kappa: float
    ) -> dict[int, float]:
        """(theta, phi, kappa) 弧度 -> 三个舵机位置（不做限位检查）。"""
        scale = self.scale
        anchor = self.hip_anchors[int(leg_id)]
        # 把 theta 归一化到 [anchor-180, anchor+180) 区间：solve_leg 里的
        # atan2 会把“越过正后方”的角度折回 ±180° 另一侧，直接代入会误判
        # 后腿的一小段物理可达扇区为不可达。
        theta_deg = math.degrees(theta)
        while theta_deg - anchor > 180.0:
            theta_deg -= 360.0
        while theta_deg - anchor < -180.0:
            theta_deg += 360.0
        hip = 512 + (theta_deg - anchor) / scale
        if self.side_of(leg_id) == "left":
            femur = 512 - math.degrees(phi) / scale
            tibia = 512 + (45 - math.degrees(kappa)) / scale
        else:
            femur = 512 + math.degrees(phi) / scale
            tibia = 512 + (math.degrees(kappa) - 45) / scale
        servos = self.legs[int(leg_id)]
        return {
            servos["hip"]: hip,
            servos["femur"]: femur,
            servos["tibia"]: tibia,
        }

    # ------------------------------------------------------------- 正运动学

    def forward_kinematics(
        self, leg_id: int, positions
    ) -> tuple[float, float, float]:
        """给定一条腿的三个舵机位置，返回身体坐标足端 (x, y, z)。"""
        leg_id = int(leg_id)
        theta, phi, kappa = self.angles_from_positions(leg_id, positions)
        mount_x, mount_y = self.mounts[leg_id]
        horizontal_reach = (
            self.coxa
            + self.femur_len * math.cos(phi)
            + self.tibia_len * math.cos(phi - kappa)
        )
        x = mount_x + horizontal_reach * math.cos(theta)
        y = mount_y + horizontal_reach * math.sin(theta)
        z = (
            self.femur_len * math.sin(phi)
            + self.tibia_len * math.sin(phi - kappa)
        )
        return x, y, z

    # ------------------------------------------------------------- 逆运动学

    def solve_leg(
        self, leg_id: int, foot_xyz: tuple[float, float, float]
    ) -> dict[int, float]:
        """给定身体坐标足端，返回该腿三个舵机位置（未量化浮点值）。

        足端不可达或所需位置超出关节限位时抛 UnreachableFootError，
        不做静默钳位。
        """
        leg_id = int(leg_id)
        foot_x, foot_y, foot_z = (float(c) for c in foot_xyz)
        mount_x, mount_y = self.mounts[leg_id]

        dx = foot_x - mount_x
        dy = foot_y - mount_y
        dz = foot_z - 0.0  # 髋关节平面 z = 0

        theta = math.atan2(dy, dx)
        r = math.hypot(dx, dy) - self.coxa

        # 足端水平投影落在髋关节的 coxa 半径内时，真实腿无法折回，
        # 但平面两连杆模型仍会给出“折叠”假解，必须显式拒绝。
        if r <= _EPS:
            raise UnreachableFootError(
                f"腿 {leg_id} 足端 {foot_xyz} 距离髋关节过近"
                f"（水平距离 {math.hypot(dx, dy):.3f}mm ≤ coxa {self.coxa:g}mm），"
                "超出工作空间"
            )

        # 两连杆（大腿 + 小腿）在工作平面 (r, dz) 内的余弦定理
        cos_k = (
            r * r
            + dz * dz
            - self.femur_len * self.femur_len
            - self.tibia_len * self.tibia_len
        ) / (2 * self.femur_len * self.tibia_len)
        if not (-1 - _EPS <= cos_k <= 1 + _EPS):
            raise UnreachableFootError(
                f"腿 {leg_id} 足端 {foot_xyz} 超出工作空间"
                f"（膝关节余弦 {cos_k:g} 超出 [-1, 1]）"
            )
        cos_k = _clamp(cos_k, -1.0, 1.0)

        # 小腿相对大腿的向下偏角
        kappa = math.acos(cos_k)
        # 大腿相对水平面的上抬角；"肘朝上"解，膝盖位于髋-足连线之上
        phi = math.atan2(dz, r) + math.atan2(
            self.tibia_len * math.sin(kappa),
            self.femur_len + self.tibia_len * math.cos(kappa),
        )

        positions = self._positions_from_angles(leg_id, theta, phi, kappa)
        self._check_limits(leg_id, positions)
        return positions

    def solve_all(self, feet_by_leg: dict[int, tuple[float, float, float]]
                  ) -> dict[int, float]:
        """求解所有给出足端的腿，返回按舵机 ID 排序的 18 个位置。"""
        result: dict[int, float] = {}
        for leg_id, foot in feet_by_leg.items():
            result.update(self.solve_leg(leg_id, foot))
        return {servo_id: result[servo_id] for servo_id in sorted(result)}

    def _check_limits(self, leg_id: int, positions: dict[int, float]):
        side = self.side_of(leg_id)
        for role in ("hip", "femur", "tibia"):
            low, high = self.limits[role][side]
            pos = positions[self.legs[leg_id][role]]
            if pos < low - _LIMIT_EPS or pos > high + _LIMIT_EPS:
                raise UnreachableFootError(
                    f"腿 {leg_id} 的 {role} 位置 {pos:.3f} 超出限位 [{low:g}, {high:g}]"
                )


# ---------------------------------------------------------------- 三角步态

TRIPOD_A = (1, 3, 5)  # 左后、左前、右中
TRIPOD_B = (2, 4, 6)  # 左中、右后、右前


class TripodGait:
    """根据全局相位生成 6 条腿的足端位置或 18 个舵机位置。"""

    def __init__(
        self,
        ik: HexapodIK | None = None,
        body_height: float = 70.0,
        forward_bias_deg: float = 5.0,
        reach: float = 90.0,
        stride: float = 60.0,
        step_height: float = 30.0,
    ):
        self.ik = ik if ik is not None else HexapodIK()
        self.body_height = float(body_height)
        self.reach = float(reach)
        self.stride = float(stride)
        self.step_height = float(step_height)

        self.leg_ids = [1, 2, 3, 4, 5, 6]
        self.offsets: dict[int, float] = {}
        for leg_id in TRIPOD_A:
            self.offsets[leg_id] = 0.0
        for leg_id in TRIPOD_B:
            self.offsets[leg_id] = 0.5

        # 每条腿的自然方向（弧度）：以髋舵机安装朝向为基准，向前偏 5°
        self.natural_direction: dict[int, float] = {}
        for leg_id in self.leg_ids:
            anchor = self.ik.hip_anchors[leg_id]
            if self.ik.sides[leg_id] == "left":
                direction = anchor - float(forward_bias_deg)
            else:
                direction = anchor + float(forward_bias_deg)
            self.natural_direction[leg_id] = math.radians(direction)

    # ------------------------------------------------------------- 自然姿态

    def neutral_foot(self, leg_id: int) -> tuple[float, float, float]:
        """自然站立时该腿的足端（身体坐标，z = -body_height）。"""
        leg_id = int(leg_id)
        mount_x, mount_y = self.ik.mounts[leg_id]
        theta0 = self.natural_direction[leg_id]
        arm = self.ik.coxa + self.reach
        x = mount_x + arm * math.cos(theta0)
        y = mount_y + arm * math.sin(theta0)
        return x, y, -self.body_height

    def stand_feet(self) -> dict[int, tuple[float, float, float]]:
        """全部 6 条腿的自然站立足端。"""
        return {leg_id: self.neutral_foot(leg_id) for leg_id in self.leg_ids}

    def stand_pose(self) -> dict[int, float]:
        """自然站立姿态的 18 个舵机位置（按舵机 ID 排序）。"""
        return self.ik.solve_all(self.stand_feet())

    # ------------------------------------------------------------- 步态轨迹

    def _leg_phase(self, leg_id: int, phase: float) -> float:
        """返回给定全局相位下该腿自身的归一化相位 u。"""
        return (phase - self.offsets[int(leg_id)]) % 1.0

    def feet_at(
        self,
        phase: float,
        stride: float | None = None,
        lateral_stride: float = 0.0,
    ) -> dict[int, tuple[float, float, float]]:
        """给定全局相位（周期归一化到 [0, 1)），返回 6 条腿的足端。

        stride 沿身体 x 轴：正数前进、负数后退；lateral_stride 沿身体 y 轴：
        正数向左、负数向右。两者可同时使用，合成任意水平方向的平移。
        """
        stride_now = self.stride if stride is None else float(stride)
        lateral_now = float(lateral_stride)
        phase = phase % 1.0
        feet: dict[int, tuple[float, float, float]] = {}
        for leg_id in self.leg_ids:
            u = self._leg_phase(leg_id, phase)
            neutral_x, neutral_y, neutral_z = self.neutral_foot(leg_id)
            if u < 0.5:
                # 支撑相：足端相对身体从 +stride/2 线性后退到 -stride/2
                progress = u / 0.5
                x = neutral_x + stride_now * (0.5 - progress)
                y = neutral_y + lateral_now * (0.5 - progress)
                z = neutral_z
            else:
                # 摆动相：足端前进，并按正弦弧线抬起
                v = (u - 0.5) / 0.5
                x = neutral_x - stride_now / 2 + stride_now * v
                y = neutral_y - lateral_now / 2 + lateral_now * v
                z = -self.body_height + self.step_height * math.sin(math.pi * v)
            feet[leg_id] = (x, y, z)
        return feet

    def pose_at(
        self,
        phase: float,
        stride: float | None = None,
        lateral_stride: float = 0.0,
    ) -> dict[int, float]:
        """给定相位返回 18 个舵机位置（按舵机 ID 排序）。

        stride 与 lateral_stride 分别控制身体前后/左右平移，含义与
        feet_at 相同。
        """
        return self.ik.solve_all(
            self.feet_at(phase, stride, lateral_stride)
        )

    def turn_feet_at(
        self, phase: float, turn_step: float
    ) -> dict[int, tuple[float, float, float]]:
        """给定全局相位与每周期旋转角，返回原地旋转时的 6 条腿足端。

        turn_step 为每个完整步态周期内身体绕 z 轴旋转的角度（弧度）：
        正数表示俯视逆时针（向左）旋转，负数表示顺时针（向右）。
        足端相位规则与平移步态相同：支撑相内足端相对身体从 +turn_step/2
        反向旋转到 -turn_step/2，摆动相抬脚并转回 +turn_step/2，保证足端
        相对地面不打滑。
        """
        phase = phase % 1.0
        feet: dict[int, tuple[float, float, float]] = {}
        for leg_id in self.leg_ids:
            u = self._leg_phase(leg_id, phase)
            neutral_x, neutral_y, neutral_z = self.neutral_foot(leg_id)
            if u < 0.5:
                # 支撑相：足端相对身体从 +turn_step/2 转到 -turn_step/2
                progress = u / 0.5
                foot_angle = turn_step * (0.5 - progress)
                z = neutral_z
            else:
                # 摆动相：足端沿身体旋转方向转回 +turn_step/2，并抬起
                v = (u - 0.5) / 0.5
                foot_angle = -turn_step / 2 + turn_step * v
                z = -self.body_height + self.step_height * math.sin(math.pi * v)
            x, y = _rotate_xy(neutral_x, neutral_y, foot_angle)
            feet[leg_id] = (x, y, z)
        return feet

    def turn_pose_at(
        self, phase: float, turn_step: float
    ) -> dict[int, float]:
        """给定相位与每周期旋转角，返回原地旋转的 18 个舵机位置。"""
        return self.ik.solve_all(self.turn_feet_at(phase, turn_step))
