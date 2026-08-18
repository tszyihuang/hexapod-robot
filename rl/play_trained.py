"""第 6 课评估脚本：加载训练好的策略，看机器人走路！

运行方式（在终端里）：
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    cd ~/hexapod-robot/rl
    python play_trained.py                          # 弹窗口看机器人走路
    python play_trained.py --checkpoint <路径>       # 指定某个策略文件
    python play_trained.py --headless --checkpoint <路径>  # 无窗口 + 只看数字

不指定 --checkpoint 时自动使用 logs/rsl_rl/hexapod/ 下最新的策略文件。
"""

import argparse
import glob
import os
import sys

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="策略权重文件路径（.pt）")
parser.add_argument("--num_envs", type=int, default=4, help="评估用平行世界数量")
parser.add_argument("--steps", type=int, default=500, help="评估步数（500 步 = 10 秒）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hexapod_agent_cfg import HexapodPPORunnerCfg  # noqa: E402
from hexapod_env_cfg import HexapodEnvCfg  # noqa: E402


def find_latest_checkpoint() -> str:
    """在日志目录里找最新的策略文件（按训练轮数排序，而不是文件名字符串排序）。"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    files = glob.glob(os.path.join(log_dir, "model_*.pt"))

    def iter_num(path: str) -> int:
        return int(os.path.basename(path).split("_")[1].split(".")[0])

    if not files:
        raise FileNotFoundError(f"没有找到策略文件，请先运行 train.py（查找目录：{log_dir}）")
    return max(files, key=iter_num)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    checkpoint = args_cli.checkpoint or find_latest_checkpoint()
    print("使用策略文件:", checkpoint)

    # 1. 创建环境 + 包装 + 加载策略
    env_cfg = HexapodEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = HexapodPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="logs_eval", device=agent_cfg.device)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()

    # 2. 用训练好的策略跑一个回合，统计前进距离
    obs = env.get_observations()
    start_x = env.unwrapped.scene["robot"].data.root_pos_w[:, 0].clone()
    best_dist = 0.0  # 全程最大平均前进距离（回合结束时环境会重置，用峰值更准确）
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            obs, rew, dones, infos = env.step(actions)
            root = env.unwrapped.scene["robot"].data.root_pos_w
            dist = (root[:, 0] - start_x).mean().item()
            best_dist = max(best_dist, dist)
            if step % 50 == 0:
                cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].tolist()
                print(
                    f"step {step:3d} | 平均前进距离 {dist:+.3f} m"
                    f" | 当前指令 (vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f})"
                )

    # 3. 最终成绩（用全程峰值，避免回合结束自动重置导致的距离清零）
    fallen = (env.unwrapped.scene["robot"].data.root_pos_w[:, 2] < 0.03).sum().item()
    print(
        f"=== 评估完成：10 秒内平均前进 {best_dist:+.3f} m"
        f"（平均速度 {best_dist/10:+.3f} m/s）| 摔倒 {fallen}/{args_cli.num_envs} ==="
    )

    env.close()
    # 注意：无头模式下 simulation_app.close() 可能直接结束进程，属正常现象
    simulation_app.close()


if __name__ == "__main__":
    main()
