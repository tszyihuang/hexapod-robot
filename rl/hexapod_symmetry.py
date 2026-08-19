"""左右镜像对称数据增强（CPG 架构版）。

背景：策略不知道"左边和右边是镜像关系"。每次收集到一批经验 (观测, 动作) 时，
额外生成"左右镜像版"一起训练，强制策略满足：左侧状态下的决策 = 右侧镜像状态
下的决策镜像——左右腿天然平衡。

观测向量布局（58 维，与 hexapod_env_cfg.py 一致）：
    [ 0: 3)  线速度 (x, y, z)    ->  (x, -y, z)
    [ 3: 6)  角速度 (x, y, z)    ->  (-x, y, -z)   （角速度是伪矢量）
    [ 6: 9)  重力投影            ->  (x, -y, z)
    [ 9:27)  关节位置 ×18        ->  腿1↔4, 腿2↔5, 腿3↔6 互换
    [27:45)  关节速度 ×18        ->  同上
    [45:48)  速度指令 (vx,vy,wz) ->  (vx, -vy, -wz)
    [48:54)  上一帧动作（6 个 CPG 参数）-> (+, -, -, +, +, +)
                 步幅不变、横移取反、转角取反、抬脚高/步频/身高不变
    [54:56)  CPG 相位            ->  不变
    [56:58)  偏航角 (sin,cos)    ->  (-sin, +cos)（镜像世界偏航取反）

动作（6 个 CPG 参数）：(+, -, -, +, +, +)，只取反号，不换位。

关节顺序（Isaac Lab articulation 顺序）：髋 1-6, 大腿 1-6, 小腿 1-6。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 每个 6 关节块内：腿1↔4（0↔3）、腿2↔5（1↔4）、腿3↔6（2↔5）
_JOINT_SWAP = [3, 4, 5, 0, 1, 2]
# CPG 参数镜像：步幅+, 横移-, 转角-, 抬脚高+, 步频+, 身高+
_PARAM_SIGN = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, 1.0])


@torch.no_grad()
def compute_mirrored_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """生成 (原始, 左右镜像) 两倍批量的观测与动作。"""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_actions(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """把观测变换成"左右镜像世界"里的等价观测。"""
    obs = obs.clone()
    device = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)     # 线速度
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)    # 角速度
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)     # 重力
    obs[:, 45:48] = obs[:, 45:48] * torch.tensor([1, -1, -1], device=device)  # 指令
    obs[:, 48:54] = obs[:, 48:54] * _PARAM_SIGN.to(device)                  # CPG 参数
    obs[:, 56:58] = obs[:, 56:58] * torch.tensor([-1, 1], device=device)    # 偏航角 sin/cos
    # 关节位置(9:27) / 关节速度(27:45)：左右腿互换
    for start in (9, 27):
        for k in range(3):  # 髋、大腿、小腿三个 6 关节块
            blk = obs[:, start + k * 6 : start + k * 6 + 6]
            obs[:, start + k * 6 : start + k * 6 + 6] = blk[:, _JOINT_SWAP]
    return obs


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """CPG 参数镜像：横移、转角取反，其余不变。"""
    return actions.clone() * _PARAM_SIGN.to(actions.device)
