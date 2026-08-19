"""六足机器人 RL 环境（CPG 架构）：策略输出步态参数 -> CPG 生成 18 舵机目标。

架构（CPG + RL，六足文献主流做法）：
    策略输出 6 个 [-1,1] 参数 -> 换算成 [步幅, 横移, 转角, 抬脚高, 步频, 身高]
    -> HexapodCPG 按三角步态相位生成足端轨迹 -> 向量化 IK 解出 18 个关节目标角
    -> 下发给 PD 控制器（与实机 LX-15D 位置指令一致）。

为什么这样设计：步态的"优雅"由 CPG 的数学保证（正弦轨迹 = 平滑、相位差 0.5
= 三角步态节律、左右镜像 = 对称），策略只需要学"用什么参数达到目标速度"，
奖励函数因此可以保持极简（8 个标准项，零补丁）。

关键换算（来自 src/physical_config.json 的 position_to_angle_formula）：
    舵机位置 pos(0-1000) 与关节角的换算 deg = (pos - 512) * 0.24
    站姿（实机 TripodGait(body_height=70, reach=90) 的站立位置）：
        髋 512、大腿 281(左)/743(右)、小腿 176(左)/848(右)
        -> URDF 关节角：髋 0.000、大腿 0.967、小腿 2.192 rad（六腿相同）
"""

import math
import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hexapod_cpg import IDX_FREQ, HexapodCPG

# 机器人模型文件：注意必须用 models/ 下的 hexapod.usd（完整舞台，包含关节链根节点），
# 而不是 models/configuration/hexapod_robot.usd（那只是一个引用外壳，单独加载没有实体）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD = os.path.join(_THIS_DIR, "..", "models", "hexapod.usd")

# ---------------------------------------------------------------- 站立默认姿态
# 实机自然站立姿态对应的 18 个 URDF 关节角（弧度）。
# 用 src/kinematics.py 可以复现这个结果：
#     from kinematics import HexapodIK, TripodGait
#     stand = TripodGait(ik=HexapodIK(), body_height=70.0, reach=90.0).stand_pose()
# 然后按上面的换算公式转成 URDF 关节角。
STAND_POSE = {
    # 髋：512 位置 -> 0 rad，腿指向各自安装朝向（前 ±45°、中 ±90°、后 ±135°）
    "leg1_hip_joint": 0.0, "leg2_hip_joint": 0.0, "leg3_hip_joint": 0.0,
    "leg4_hip_joint": 0.0, "leg5_hip_joint": 0.0, "leg6_hip_joint": 0.0,
    # 大腿：抬起 55.4°（左腿舵机 281 / 右腿舵机 743）
    "leg1_femur_joint": 0.9668, "leg2_femur_joint": 0.9668, "leg3_femur_joint": 0.9668,
    "leg4_femur_joint": 0.9668, "leg5_femur_joint": 0.9668, "leg6_femur_joint": 0.9668,
    # 小腿：相对大腿下弯 125.6°（左腿舵机 176 / 右腿舵机 848）
    "leg1_tibia_joint": 2.1922, "leg2_tibia_joint": 2.1922, "leg3_tibia_joint": 2.1922,
    "leg4_tibia_joint": 2.1922, "leg5_tibia_joint": 2.1922, "leg6_tibia_joint": 2.1922,
}

# 身体中心离地高度（米）：实机站姿 body_height=70mm，足端在髋平面下方 70mm
BODY_HEIGHT_M = 0.07


@configclass
class HexapodSceneCfg(InteractiveSceneCfg):
    """场景：每个"平行世界"（env）里放 1 台机器人 + 1 块地面。"""

    # 地面：prim_path 里没有 {ENV_REGEX_NS}，表示所有平行世界共用同一块大地板
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            # 摩擦系数对齐实机：足端是橡胶材质，实机行走不打滑
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
        ),
    )

    # 机器人：{ENV_REGEX_NS} 会被自动替换成每个平行世界的路径前缀
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/hexapod",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,  # 给所有实体加 ContactReporter API（接触传感器必需）
        ),
        # 出生位置：身体中心在 7cm 高（比站姿略高 5mm，让它轻轻落稳）
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BODY_HEIGHT_M + 0.005),
            # 出生关节姿态 = 站立姿态（也是 CPG 相位 0 时的目标姿态）
            joint_pos=STAND_POSE,
        ),
        actuators={
            # "servo" 组名必须与 USD 文件里舵机组的名字一致
            "servo": IdealPDActuatorCfg(
                joint_names_expr=[".*"],  # 正则表达式，匹配全部 18 个关节
                stiffness=30.0,           # PD 的 P 增益：越大关节越"硬"，跟踪越紧
                damping=1.0,              # PD 的 D 增益：抑制振荡，让动作更稳
                effort_limit=1.5,         # 最大力矩 1.5 N·m（接近 LX-15D 舵机的堵转力矩）
                velocity_limit=6.5,       # 最大速度 6.5 rad/s
            ),
        },
    )

    # 接触传感器：判断"身体是否蹭地"（undesired_contacts 惩罚 / base_contact 终止）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hexapod/.*",
        history_length=3,
        track_air_time=True,   # 追踪滞空时间（诊断/评估用）
        update_period=0.0,    # 每个物理步都更新
    )


# ---------------------------------------------------------------- CPG 动作项
class CPGAction(ActionTerm):
    """把 6 个 CPG 参数变成 18 个关节目标位置并下发给 PD 控制器。"""

    def __init__(self, cfg: "CPGActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cpg = HexapodCPG(device=env.device)
        self._joint_ids, _ = self._asset.find_joints([".*"])  # 全部 18 关节（articulation 顺序）
        self._center = torch.tensor(cfg.param_center, dtype=torch.float32, device=env.device)
        self._scale = torch.tensor(cfg.param_scale, dtype=torch.float32, device=env.device)
        # 每个平行世界的 CPG 相位（0~1 圈）
        self._phase = torch.zeros(env.num_envs, device=env.device)
        # 当前关节目标角（初始 = 站姿）
        self._targets = self._asset.data.default_joint_pos.clone()
        self._raw_actions = torch.zeros(env.num_envs, 6, device=env.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets

    @property
    def phase(self) -> torch.Tensor:
        """CPG 相位（0~1 圈），观测项 cpg_phase_obs 用它。"""
        return self._phase

    def process_actions(self, actions: torch.Tensor):
        """每步（50Hz）执行一次：换算参数 -> 推进相位 -> CPG+IK 解出 18 关节目标。"""
        self._raw_actions = actions
        params = self._center + actions * self._scale
        self._phase = (self._phase + params[:, IDX_FREQ] * self._env.step_dt) % 1.0
        self._targets = self.cpg.forward(params, self._phase)

    def apply_actions(self):
        """每个物理步执行：把当前目标角写入 PD 控制器。"""
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        """回合重置：相位归零、目标回到站姿。"""
        if env_ids is None:
            env_ids = slice(None)
        self._phase[env_ids] = 0.0
        self._targets[env_ids] = self._asset.data.default_joint_pos[env_ids]


@configclass
class CPGActionCfg(ActionTermCfg):
    """CPG 动作配置：动作 = 6 个 [-1,1] 参数，param = center + action × scale。

    参数顺序（与 hexapod_cpg.py 一致）：
        [0] stride 步幅   [1] lateral 横移   [2] turn 转角
        [3] step_height   [4] freq 步频     [5] body_height 身高
    """

    class_type: type = CPGAction

    param_center: tuple = (0.045, 0.0, 0.0, 0.03, 1.2, 0.07)
    """参数中心（动作=0 时的值）：步幅 4.5cm、步频 1.2Hz、抬脚 3cm、身高 7cm"""
    param_scale: tuple = (0.045, 0.04, 0.25, 0.015, 0.9, 0.015)
    """参数幅度（动作 ±1 的变化量）：步幅 0~9cm、步频 0.3~2.1Hz、身高 5.5~8.5cm"""


@configclass
class ActionsCfg:
    """动作空间：6 个 CPG 参数（见 CPGActionCfg 说明）。"""

    cpg = CPGActionCfg(asset_name="robot")


# ---------------------------------------------------------------- 自定义观测项
def cpg_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """CPG 实时相位（sin/cos，2 维）——策略的"节拍器"。

    与旧版的固定 2Hz 时钟不同，这里的相位来自 CPG 自身：步频是策略的输出，
    相位随之快慢变化，策略才能知道"现在六条腿各在周期的什么位置"。
    """
    phase = env.action_manager.get_term("cpg").phase.unsqueeze(-1)  # (N, 1)
    return torch.cat(
        [torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)], dim=-1
    )


def yaw_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """世界系偏航角（sin/cos，2 维）——策略的"指南针"。

    没有它策略根本看不见自己的朝向：heading_deviation 惩罚想让它直行，但它
    无从得知当前偏了多少、该往哪边打 turn 修正。加上后才能主动抵消步态的
    净偏航漂移（旧策略 10 秒漂 80°+ 的根因之一）。
    """
    quat = env.scene["robot"].data.root_quat_w   # (N, 4) wxyz
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


@configclass
class ObservationsCfg:
    """观测空间：58 维向量。

    拼接顺序（类内定义顺序即向量顺序，训练后不能乱改）：
        索引 [ 0: 3)  身体线速度   base_lin_vel      m/s（身体坐标系）
        索引 [ 3: 6)  身体角速度   base_ang_vel      rad/s
        索引 [ 6: 9)  重力投影     projected_gravity 单位向量
        索引 [ 9:27)  关节位置     joint_pos_rel     rad（相对站姿）
        索引 [27:45)  关节速度     joint_vel_rel     rad/s
        索引 [45:48)  速度指令     velocity_commands (vx, vy, wz)
        索引 [48:54)  上一帧动作   last_action       6 个 CPG 参数 [-1,1]
        索引 [54:56)  CPG 相位     sin(2π·phase), cos(2π·phase)
        索引 [56:58)  偏航角       sin(yaw), cos(yaw)（世界系指南针）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ---- 本体运动状态（3 + 3 + 3 = 9 维）----
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # ---- 关节状态（18 + 18 = 36 维）----
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # ---- 任务指令（3 维）----
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # ---- 上一帧动作（6 维 CPG 参数）----
        actions = ObsTerm(func=mdp.last_action)
        # ---- CPG 相位（2 维）----
        cpg_phase = ObsTerm(func=cpg_phase_obs)
        # ---- 偏航角（2 维）----
        yaw = ObsTerm(func=yaw_obs)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """任务指令：训练时环境周期性给机器人下达"速度指令"。

    先只练"直线前进"：速度 0.1~0.3 m/s 随机（每 5 秒换一次），10% 的环境指令
    为 0（练站立）。CPG 架构天然支持横移/转向，后续如需横向与转向指令，
    放开下方 ranges 的 lin_vel_y / ang_vel_z 即可。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.3),  # 前进速度（CPG 最大可达 ~0.38 m/s）
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------- 自定义惩罚项
# 足尖在胫节(tibia) link 坐标系中的位置：URDF 中 foot_tip 球心位于 (0.14, 0, 0)
FOOT_TIP_LOCAL = (0.14, 0.0, 0.0)
# 六条腿的胫节刚体（足端），按正则匹配
FOOT_BODY_RE = ".*tibia"


def feet_slide_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """足端打滑惩罚：支撑相中段、着地稳定的足端，其水平滑动速度越大惩罚越重。

    正常步态支撑相足尖相对地面静止（身体从足上"走过"）；打滑时足尖在地面横向
    搓动。注意胫节 link 帧原点在膝关节——支撑相膝关节以体速前进，必须用
    v_tip = v_link + ω × r_tip 换算到足尖点才是真实的打滑速度。

    只在 CPG 相位 u ∈ (0.1, 0.4) 的"支撑相中段"生效：摆动相结束落地瞬间足端
    轨迹水平速度高达 ~3× 体速（落地前的自然滑动），若按接触掩码全部计入，
    会把正常行走误判成打滑，导致策略缩步幅原地踏步。
    """
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    cpg_term = env.action_manager.get_term("cpg")

    # 支撑相中段掩码：每条腿相位 u = (phase - offset) % 1，u ∈ (0.1, 0.4) 为稳定支撑
    phase = cpg_term.phase                                        # (N,)
    offsets = cpg_term.cpg.offsets                                # (6,)
    u = (phase.unsqueeze(-1) - offsets.unsqueeze(0)) % 1.0        # (N,6)
    solid_stance = (u > 0.1) & (u < 0.4)

    foot_ids, _ = robot.find_bodies(FOOT_BODY_RE)      # 足端在 articulation 中的索引
    sensor_ids, _ = sensor.find_bodies(FOOT_BODY_RE)   # 足端在接触传感器中的索引

    # 足尖点速度（世界系）= link 线速度 + 角速度 × 足尖偏移
    lin_vel = robot.data.body_link_vel_w[:, foot_ids, :3]   # (N, 6, 3)
    ang_vel = robot.data.body_link_vel_w[:, foot_ids, 3:]   # (N, 6, 3)
    quat = robot.data.body_link_quat_w[:, foot_ids]         # (N, 6, 4)
    r_tip = quat_apply(
        quat,
        torch.tensor(FOOT_TIP_LOCAL, dtype=torch.float32, device=env.device).expand_as(lin_vel),
    )
    tip_vel = lin_vel + torch.linalg.cross(ang_vel, r_tip, dim=-1)

    # 接触掩码：法向力模长超过 1N 视为支撑相
    foot_force = sensor.data.net_forces_w[:, sensor_ids]
    in_contact = torch.linalg.vector_norm(foot_force, dim=-1) > 1.0

    # 只惩罚"支撑相中段 + 接触中"的足的水平滑动速度平方
    mask = solid_stance & in_contact
    slip = torch.sum(torch.square(tip_vel[:, :, :2]), dim=-1)
    return torch.sum(slip * mask, dim=-1)


def lin_vel_y_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """横向偏移惩罚：身体坐标系下的侧向速度绝对值（直行时应当 ≈ 0）。"""
    return torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 1])


def ang_vel_z_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """偏航角速度惩罚：抑制原地转向/来回摆头（角速度惩罚压不住缓慢的净漂移）。"""
    return torch.square(env.scene["robot"].data.root_ang_vel_b[:, 2])


def heading_deviation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """偏航角（净朝向）惩罚：直行时机器人世界系偏航角应保持 ≈ 出生朝向。

    角速度惩罚只能压振荡，压不住每周期累积一点的净偏航漂移（旧策略 10 秒
    偏 81°）；直接罚偏航角本身，策略会自动用 turn 参数抵消步态的自然偏航。
    """
    quat = env.scene["robot"].data.root_quat_w   # (N, 4) wxyz
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.square(yaw)


@configclass
class RewardsCfg:
    """奖励函数：12 项 = 8 个标准项 + 打滑/横向/偏航角速度/偏航角 4 个直行质量惩罚。

    CPG 结构保证了步态平滑、对称、有节律，因此这里只需要"走对方向、走稳、
    别摔、别打滑、别跑偏"的基础评分。速度追踪沿用 Isaac Lab 官方四足任务的惯例。
    """

    # -- 主任务：追踪速度指令 --
    # std=0.2（比官方四足默认 0.5 更紧）：本机器人速度只有 0.1~0.3 m/s，
    # 用 0.5 时 0.2 m/s 的速度误差奖励仍有 0.92（饱和），策略没动力提速
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    # -- 姿态质量 --
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)                # 惩罚上下乱跳
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)             # 惩罚摇晃
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)  # 保持身体水平
    # -- 动作质量 --
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)       # 省力（能耗）
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)           # 参数平滑变化
    # -- 直行质量：打滑 + 偏移 --
    feet_slide = RewTerm(func=feet_slide_penalty, weight=-1.0)    # 足端打滑（支撑相中段的水平滑动）
    lin_vel_y = RewTerm(func=lin_vel_y_penalty, weight=-1.0)      # 横向偏移（侧向速度）
    ang_vel_z = RewTerm(func=ang_vel_z_penalty, weight=-0.5)      # 偏航角速度（压摆动，别太狠以免牺牲步态）
    heading_deviation = RewTerm(func=heading_deviation_penalty, weight=-1.5)  # 偏航角（压净漂移）
    # -- 安全 --
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    """终止条件：超时（不算失败）/ 身体触地（失败）。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """总配置：场景 + CPG 动作 + 观测 + 指令 + 奖励 + 终止。"""

    scene: HexapodSceneCfg = HexapodSceneCfg(num_envs=16, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """仿真与控制的时间参数。"""
        # 物理仿真步长：5 ms
        self.sim.dt = 0.005
        # 渲染间隔：与 decimation 一致，消除多余渲染
        self.sim.render_interval = 4
        # decimation = 每多少个物理步下发一次新动作
        # 4 × 5ms = 20ms -> 控制频率 50 Hz（和实机串口舵机刷新率一致！）
        self.decimation = 4
        # 一个回合持续 10 秒
        self.episode_length_s = 10.0
        # 接触传感器每个控制步（50Hz）更新一次：undesired_contacts 奖励与 base_contact
        # 终止都在控制步读取接触力，200Hz 更新纯属浪费（接触报告是本机每步固定开销大头之一）
        self.scene.contact_forces.update_period = self.decimation * self.sim.dt
