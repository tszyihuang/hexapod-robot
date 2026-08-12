#!/usr/bin/env python3
"""六足机器人三角步态生成器（前进/后退）。

三角组固定为：
  - A 组：腿 1（左后）、腿 3（左前）、腿 5（右中），相位偏移 0
  - B 组：腿 2（左中）、腿 4（右后）、腿 6（右前），相位偏移 0.5

每条腿的自然足端由安装点、该腿的自然方向 theta0、水平伸出量 reach 和
身体高度 body_height 生成：
  foot = mount + (coxa + reach) * (cos(theta0), sin(theta0)), z = -H

theta0 按腿分别取值：以各腿髋舵机的安装朝向（位置 512 时的指向）为基准，
前/中/后腿分别为 ±45° / ±90° / ±135°（右腿取负），再向身体前方偏约 5°。
各腿安装朝向不同，因此前、侧、后腿的摆动方向不再共用同一套角度逻辑。

相位的约定：全局相位 phase 随时间线性前进，u = (phase - 偏移) mod 1。
  - 支撑相 0 <= u < 0.5：足端从 natural + stride/2 线性后退到
    natural - stride/2，z 恒为 -H。
  - 摆动相 0.5 <= u < 1：足端从 natural - stride/2 前进到
    natural + stride/2，z = -H + step_height*sin(pi*v) 抬起。
  - stride 带符号：正数前进、负数后退；step_height 始终为抬起的幅值。
    时间映射由调用方决定：周期 T = 2*stride/speed。这样支撑相内身体前进
    stride、足端相对身体后退 stride，足端在地面不打滑。

本模块只负责生成足端轨迹和舵机姿态，不负责串口与步态时间调度；
启动时的自然站立姿态和约 1.5 个周期的步幅平滑爬升由 servo_driver 的
walk 命令完成。
"""

from __future__ import annotations

import math

from hexapod_ik import HexapodIK


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

    def feet_at(
        self, phase: float, stride: float | None = None
    ) -> dict[int, tuple[float, float, float]]:
        """给定全局相位（周期归一化到 [0, 1)），返回 6 条腿的足端。"""
        stride_now = self.stride if stride is None else float(stride)
        phase = phase % 1.0
        feet: dict[int, tuple[float, float, float]] = {}
        for leg_id in self.leg_ids:
            u = (phase - self.offsets[leg_id]) % 1.0
            neutral_x, neutral_y, neutral_z = self.neutral_foot(leg_id)
            if u < 0.5:
                # 支撑相：足端相对身体从 +stride/2 线性后退到 -stride/2
                progress = u / 0.5
                x = neutral_x + stride_now * (0.5 - progress)
                z = neutral_z
            else:
                # 摆动相：足端前进，并按正弦弧线抬起
                v = (u - 0.5) / 0.5
                x = neutral_x - stride_now / 2 + stride_now * v
                z = -self.body_height + self.step_height * math.sin(math.pi * v)
            feet[leg_id] = (x, neutral_y, z)
        return feet

    def pose_at(
        self, phase: float, stride: float | None = None
    ) -> dict[int, float]:
        """给定相位返回 18 个舵机位置（按舵机 ID 排序）。"""
        return self.ik.solve_all(self.feet_at(phase, stride))
