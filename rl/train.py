"""PPO 强化学习训练脚本 —— 让六足机器人学会走路！

训练流程（每一轮迭代 = 一个"收集经验 -> 学习"的循环）：
    1. 收集经验：num_envs 个平行世界各跑 24 步（当前策略 + 探索噪声），记录
       每一步的 (观测, 动作, 奖励, 下一个观测)
    2. 学习：PPO 算法把这些经验"消化" 5 遍（5 个学习轮次），更新神经网络
       参数——让"拿高分"的动作更可能被选中
    3. 重复 max_iterations 次，机器人从随机乱动逐渐学会走路

运行方式（在终端里）：
    source /home/tszyi/miniforge3/envs/env_isaaclab/bin/activate   # 激活 Isaac Lab 环境
    cd rl
    python train.py --headless                          # 完整训练（默认 1000 轮）
    python train.py --headless --max_iterations 50      # 先小跑 50 轮试试水温

训练过程打印：
    mean_reward：所有平行世界当前策略的平均每步奖励（最重要，应持续上升）
    mean_episode_length：平均每回合走多久（越接近 10 秒说明越少摔倒）
    learning_rate / value_loss 等：网络训练内部指标

训练产物：
    logs/rsl_rl/hexapod/<时间戳>/model_<迭代>.pt    # 定期保存的策略权重
    logs/rsl_rl/hexapod/<时间戳>/events...          # TensorBoard 曲线（可选看）
    选出最佳检查点后复制到 checkpoints/ 保留（该目录入库）。
"""

import argparse
import os

from isaaclab.app import AppLauncher

# 命令行参数：训练规模相关
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024,
                    help="平行世界数量（本机 RTX 5060 建议 512-1024）")
parser.add_argument("--max_iterations", type=int, default=1000,
                    help="训练轮数（400 轮会走、1000 轮走得更好）")
parser.add_argument("--save_interval", type=int, default=100,
                    help="每多少轮保存一次策略")
parser.add_argument("--log_dir", type=str, default=None,
                    help="日志目录（默认 logs/）")
# 注意：--headless 参数由 AppLauncher 提供，不要自己重复定义
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hexapod_agent_cfg import HexapodPPORunnerCfg  # noqa: E402
from hexapod_env_cfg import HexapodEnvCfg  # noqa: E402


def main():
    # 让 print 立即输出
    import sys

    sys.stdout.reconfigure(line_buffering=True)

    # 1. 创建训练环境
    env_cfg = HexapodEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 2. 包装成 rsl_rl 认识的接口
    env = RslRlVecEnvWrapper(env)

    # 3. 组装 PPO 训练器
    agent_cfg = HexapodPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    log_dir = args_cli.log_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    print("[训练] 平行世界", args_cli.num_envs, "| 日志目录:", os.path.join(log_dir, agent_cfg.experiment_name))
    print("[训练] 开始训练：", args_cli.max_iterations, "轮 ×", args_cli.num_envs,
          "个平行世界 ×", agent_cfg.num_steps_per_env, "步")

    # 4. 开始训练！rsl_rl 每 10 轮打印一次进度
    runner.learn(num_learning_iterations=args_cli.max_iterations)

    # 5. 收尾
    env.close()
    # 注意：无头模式下 simulation_app.close() 可能直接结束进程，属正常现象
    simulation_app.close()


if __name__ == "__main__":
    main()
