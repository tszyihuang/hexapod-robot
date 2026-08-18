"""PPO 训练器配置（训练 train.py 与评估 play_trained.py 共用）。

第 7 课调参时主要改这里。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from hexapod_symmetry import compute_mirrored_states


@configclass
class HexapodPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO 训练器超参数。

    这些是 Isaac Lab 官方四足机器人的标准起步参数，六足同样适用：
    """

    num_steps_per_env = 24          # 每轮迭代每个平行世界收集 24 步经验
    save_interval = 50              # 每 50 轮保存一次策略
    experiment_name = "hexapod"     # 实验名（决定日志目录）
    run_name = ""                   # 本次运行的名字（留空用时间戳）
    seed = 42
    device = "cuda:0"
    # 观测分组：策略(actor)和评价(critic)都用 policy 组观测
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    # 神经网络结构：观测 68 维 -> 128 -> 128 -> 128 -> 动作 18 维
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,              # 初始探索噪声（越大越爱乱试）
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    # PPO 算法参数（标准值，先不动）
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,                  # PPO 的"信任区域"裁剪系数
        entropy_coef=0.005,              # 探索欲望系数
        num_learning_epochs=5,           # 每轮收集的经验学 5 遍
        num_mini_batches=4,              # 分成 4 个小批学
        learning_rate=1.0e-3,
        schedule="adaptive",             # 学习率自适应调整
        gamma=0.99,                      # 折扣因子：多看重远期收益
        lam=0.95,                        # GAE 的 lambda
        desired_kl=0.01,                 # 每轮学习允许的最大策略变化
        max_grad_norm=1.0,
        # 左右镜像对称数据增强：强制策略公平使用左右腿（见 hexapod_symmetry.py）
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_mirrored_states,
        ),
    )
