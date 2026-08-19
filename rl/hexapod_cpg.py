"""向量化 CPG（中枢模式发生器）—— 把 6 个步态参数变成 18 个关节目标角。

这是"CPG + RL"架构的核心：优雅的步态由数学保证，而不是靠奖励函数打补丁。

参数 -> 足端轨迹 -> 逆运动学 -> 关节角，全流程张量化（一次算 N 个平行世界 ×
6 条腿），与 src/kinematics.py 里的 TripodGait / HexapodIK 完全同源：

    步态模式：三角步态（腿 1/3/5 与 2/4/6 相位差 0.5）
      - 支撑相 0 <= u < 0.5：足端相对身体从 +stride/2 匀速后退到 -stride/2，贴地
      - 摆动相 0.5 <= u < 1：足端前进，并按正弦弧线抬起 step_height
      - turn 让足端绕身体旋转，合成原地转向；lateral 合成横移
    逆运动学：两连杆闭式解（余弦定理），肘朝上解，与 solve_leg 一致
    关节角约定：q_hip = theta - anchor、q_femur = phi(上抬为正)、q_tibia = kappa(下弯为正)，
    直接输出 Isaac Lab 需要的 URDF 关节角（弧度）。

CPG 参数（策略输出 6 个 [-1,1] 数字，经中心/幅度换算）：
    [0] stride      步幅（m/周期）：决定前后速度  v ≈ 2·stride·freq
    [1] lateral     横移步幅（m/周期）
    [2] turn        每周期转角（rad，正=左转）
    [3] step_height 抬脚高度（m）
    [4] freq        步频（Hz）
    [5] body_height 身体高度（m）
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _THIS_DIR.parent / "src" / "physical_config.json"

# 参数下标（动作空间顺序，改这里要同步改对称模块与部署代码）
IDX_STRIDE, IDX_LATERAL, IDX_TURN, IDX_STEP_H, IDX_FREQ, IDX_H = range(6)


class HexapodCPG:
    """三角步态 CPG + 向量化 IK：params (N,6) + phase (N,) -> 关节目标角 (N,18)。"""

    def __init__(self, device: str = "cuda:0"):
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)

        # 腿 1-6：髋安装朝向（度）、安装点（m）
        anchors_deg = [cfg["hip_anchor_direction_deg"][str(i)] for i in range(1, 7)]
        self.anchors = torch.tensor([math.radians(a) for a in anchors_deg], device=device)  # (6,)
        mounts = cfg["mounting_positions_mm"]
        names = [cfg["leg_position_mapping"][str(i)] for i in range(1, 7)]  # 腿1..6 -> 位置名
        self.mounts = torch.tensor(
            [[mounts[n]["x"], mounts[n]["y"]] for n in names], dtype=torch.float32, device=device
        ) * 1e-3  # (6,2)

        # 几何（m）
        geo = cfg["leg_geometry_mm"]
        self.coxa = float(geo["coxa_length_mm"]) * 1e-3
        self.l_femur = float(geo["femur_length_mm"]) * 1e-3
        self.l_tibia = float(geo["tibia_length_mm"]) * 1e-3
        self.reach = 0.09  # 站姿水平伸出量（与实机 TripodGait(reach=90) 一致）

        # 三角步态相位偏移：腿 1/3/5 = 0，腿 2/4/6 = 0.5
        self.offsets = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, 0.5], device=device)  # (6,)

        # 关节限位（URDF 弧度）：髋相同；大腿/小腿左右略有差异
        lim = cfg["joint_limits"]
        hip_deg = 48.0  # ±200 单位 × 0.24°/单位，绕 512 对称
        self.q_hip_limit = math.radians(hip_deg)
        # 大腿：左 [100,900] -> q=[-1.625, 1.726]；右对称
        self.q_femur_lim = torch.tensor(
            [[-1.625251, 1.725782]] * 3 + [[-1.725782, 1.625251]] * 3, device=device
        )  # (6,2)
        # 小腿：左 [-1.259, 2.930]；右 [-1.359, 2.830]
        self.q_tibia_lim = torch.tensor(
            [[-1.258731, 2.930059]] * 3 + [[-1.359262, 2.829528]] * 3, device=device
        )  # (6,2)

        self.device = device

    # ------------------------------------------------------------- 足端轨迹
    def feet(self, params: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """params (N,6), phase (N,) -> 足端目标 (N,6,3)，身体坐标系，单位 m。"""
        stride = params[:, IDX_STRIDE]      # (N,)
        lateral = params[:, IDX_LATERAL]
        turn = params[:, IDX_TURN]
        step_h = params[:, IDX_STEP_H]
        body_h = params[:, IDX_H]

        n = phase.shape[0]
        # 每条腿的相位 u ∈ [0,1)
        u = (phase[:, None] - self.offsets[None, :]) % 1.0  # (N,6)

        # 自然站立足端
        arm = self.coxa + self.reach
        nx = self.mounts[:, 0] + arm * torch.cos(self.anchors)   # (6,)
        ny = self.mounts[:, 1] + arm * torch.sin(self.anchors)
        nz = -body_h                                              # (N,)

        stance = u < 0.5  # (N,6)
        p = torch.where(stance, u / 0.5, (u - 0.5) / 0.5)         # (N,6)

        # 支撑相：从 +/2 后退到 -/2，贴地；摆动相：从 -/2 前进到 +/2，抬脚
        x = torch.where(
            stance,
            nx[None, :] + stride[:, None] * (0.5 - p),
            nx[None, :] - stride[:, None] / 2 + stride[:, None] * p,
        )
        y = torch.where(
            stance,
            ny[None, :] + lateral[:, None] * (0.5 - p),
            ny[None, :] - lateral[:, None] / 2 + lateral[:, None] * p,
        )
        z = torch.where(
            stance,
            nz[:, None].expand(n, 6),
            nz[:, None] + step_h[:, None] * torch.sin(math.pi * p),
        )
        # 转向：足端绕身体原点旋转
        ang = torch.where(stance, turn[:, None] * (0.5 - p), -turn[:, None] / 2 + turn[:, None] * p)
        cos_a, sin_a = torch.cos(ang), torch.sin(ang)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        return torch.stack([x_rot, y_rot, z], dim=-1)  # (N,6,3)

    # ------------------------------------------------------------- 逆运动学
    def ik(self, feet: torch.Tensor) -> torch.Tensor:
        """足端 (N,6,3) -> 18 个 URDF 关节角 (N,18)（髋1-6, 大腿1-6, 小腿1-6），全部钳制在限位内。"""
        dx = feet[..., 0] - self.mounts[None, :, 0]
        dy = feet[..., 1] - self.mounts[None, :, 1]
        dz = feet[..., 2]  # 髋平面 z=0（身体坐标系）

        # 髋：指向足端的方向
        theta = torch.atan2(dy, dx)                       # (N,6)
        q_hip = theta - self.anchors[None, :]             # 相对安装朝向
        q_hip = q_hip.clamp(-self.q_hip_limit, self.q_hip_limit)

        # 两连杆（大腿+小腿）闭式解
        r = torch.hypot(dx, dy) - self.coxa
        r = r.clamp(min=1e-3)                             # 防止足端落在 coxa 半径内的退化
        cos_k = (r * r + dz * dz - self.l_femur ** 2 - self.l_tibia ** 2) / (2 * self.l_femur * self.l_tibia)
        cos_k = cos_k.clamp(-1.0, 1.0)
        kappa = torch.acos(cos_k)                         # 小腿相对大腿的下弯角 (0, π)
        phi = torch.atan2(dz, r) + torch.atan2(
            self.l_tibia * torch.sin(kappa),
            self.l_femur + self.l_tibia * torch.cos(kappa),
        )
        q_femur = phi.clamp(self.q_femur_lim[None, :, 0], self.q_femur_lim[None, :, 1])
        q_tibia = kappa.clamp(self.q_tibia_lim[None, :, 0], self.q_tibia_lim[None, :, 1])

        return torch.cat([q_hip, q_femur, q_tibia], dim=-1)  # (N,18) 髋、大腿、小腿

    # ------------------------------------------------------------- 一步到位
    def forward(self, params: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """params (N,6), phase (N,) -> 18 关节目标角 (N,18)。"""
        return self.ik(self.feet(params, phase))


# ---------------------------------------------------------------- 部署换算
def servo_positions_from_joint_angles(q: torch.Tensor) -> torch.Tensor:
    """URDF 关节角 (…, 18) -> 舵机位置 0-1000（实机部署用，与 physical_config 一致）。"""
    scale = 0.24  # 度/单位
    q_hip, q_femur, q_tibia = q[..., :6], q[..., 6:12], q[..., 12:18]
    deg = torch.rad2deg
    pos_hip = 512 + deg(q_hip) / scale
    left = torch.tensor([True, True, True, False, False, False], device=q.device)
    pos_femur = torch.where(left[None, :], 512 - deg(q_femur) / scale, 512 + deg(q_femur) / scale)
    pos_tibia = torch.where(left[None, :], 512 + (45.0 - deg(q_tibia)) / scale, 512 + (deg(q_tibia) - 45.0) / scale)
    pos = torch.cat([pos_hip, pos_femur, pos_tibia], dim=-1)
    return pos.clamp(0.0, 1000.0)
