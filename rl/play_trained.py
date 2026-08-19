"""演示脚本：加载训练好的策略，让机器人一直走下去！

运行方式（在终端里）：
    source /home/tszyi/miniforge3/envs/env_isaaclab/bin/activate   # 激活 Isaac Lab 环境
    cd rl
    python play_trained.py                          # 一直走，Ctrl+C 停止
    python play_trained.py --checkpoint <路径>       # 指定某个策略文件
    python play_trained.py --headless --checkpoint <路径>  # 无窗口 + 只看数字
    python play_trained.py --steps 500              # 只跑 500 步（= 10 秒）后退出

默认无限运行：演示模式关掉了 10 秒的回合超时，机器人会连续行走，
只在摔倒时重置；按 Ctrl+C 停止并打印累计统计（前进距离/平均速度/摔倒次数）。
不指定 --checkpoint 时自动搜索 checkpoints/ 与 logs/ 下最新的策略文件。
"""

import argparse
import glob
import os
import sys
import time

import torch
# Isaac Sim GUI 渲染会挂起 GPU（内核日志 Xid 119），驱动重置恢复期间 CUDA 可见
# 设备数会从 1 掉到 0，torch 惰性初始化若拖到那时才执行会崩 "device=0, num_gpus=0"。
# 这里提前初始化并检查设备数；异常时给出明确提示而不是晦涩的堆栈。
try:
    torch.cuda.init()
    n_visible = torch.cuda.device_count()
    if n_visible < 1:
        print(f"[提示] CUDA 可见设备为 0（本机应为 1 个，RTX 5060）：有 GPU 正在驱动"
              "重置恢复中（Xid 119）。建议稍等几分钟后重试；若反复出现请重启机器。", flush=True)
except Exception as e:
    print(f"[错误] CUDA 初始化失败（GPU 可能正在驱动恢复中，稍等重试或重启机器）：{e}", flush=True)
    sys.exit(1)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="策略权重文件路径（.pt）")
parser.add_argument("--num_envs", type=int, default=4, help="平行世界数量")
parser.add_argument("--steps", type=int, default=0,
                    help="最多运行步数；0 = 一直运行直到 Ctrl+C（默认 0）")
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
    """在 checkpoints/ 与 logs/ 里递归找最新的策略文件（按训练轮数排序，而不是文件名字符串排序）。"""
    rl_dir = os.path.dirname(os.path.abspath(__file__))
    files = []
    for sub in ("checkpoints", "logs"):
        files += glob.glob(os.path.join(rl_dir, sub, "**", "model_*.pt"), recursive=True)

    def iter_num(path: str) -> int:
        return int(os.path.basename(path).split("_")[1].split(".")[0])

    if not files:
        raise FileNotFoundError(
            f"没有找到策略文件，请先运行 train.py（查找目录：checkpoints/ 与 logs/）"
        )
    return max(files, key=iter_num)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """从四元数 (w, x, y, z) 提取偏航角（rad，范围 [-π, π]）。"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    sys.stdout.reconfigure(line_buffering=True)

    checkpoint = args_cli.checkpoint or find_latest_checkpoint()
    print("使用策略文件:", checkpoint)

    # 1. 创建环境 + 包装 + 加载策略
    env_cfg = HexapodEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    # 演示模式：关掉 10 秒回合超时，让机器人连续行走（只在摔倒时重置）
    env_cfg.episode_length_s = 1e6
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = HexapodPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="logs_eval", device=agent_cfg.device)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()

    # 2. 一直走：跨回合累计前进距离 + 直行质量统计，Ctrl+C 停止
    obs = env.get_observations()
    root = env.unwrapped.scene["robot"].data
    device = root.root_pos_w.device

    # 每个平行世界"当前回合"的起点（回合结束重置时更新）
    ref_x = root.root_pos_w[:, 0].clone()
    ref_y = root.root_pos_w[:, 1].clone()
    ref_yaw = yaw_from_quat(root.root_quat_w).clone()
    prev_x = ref_x.clone()

    cum_dist = torch.zeros(args_cli.num_envs, device=device)  # 累计前进距离（跨回合累加）
    falls = 0           # 摔倒次数（回合结束且非超时）
    best_lateral = 0.0  # 全程峰值横向偏移
    best_yaw = 0.0      # 全程峰值偏航偏移

    t_start = time.monotonic()
    step = 0
    print(f"开始演示（{args_cli.num_envs} 个平行世界）。按 Ctrl+C 停止。")
    try:
        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs, rew, dones, infos = env.step(actions)

                x = root.root_pos_w[:, 0]
                y = root.root_pos_w[:, 1]
                yaw = yaw_from_quat(root.root_quat_w)

                # 本步位移只计入"未重置"的平行世界（重置瞬间位置跳回起点）
                alive = (~dones).float()
                cum_dist += (x - prev_x) * alive
                prev_x = x.clone()

                # 回合结束（摔倒）：累计该回合成绩，并把起点参考更新为新回合起点
                done_mask = dones.bool()
                falls += int(done_mask.sum().item())
                ref_x = torch.where(done_mask, x, ref_x)
                ref_y = torch.where(done_mask, y, ref_y)
                ref_yaw = torch.where(done_mask, yaw, ref_yaw)

                # 直行质量：本回合内相对起点的峰值偏移
                best_lateral = max(best_lateral, (y - ref_y).abs().max().item())
                best_yaw = max(best_yaw, (yaw - ref_yaw).abs().max().item())

                step += 1
                if step % 50 == 0:
                    elapsed = time.monotonic() - t_start
                    mean_dist = cum_dist.mean().item()
                    cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].tolist()
                    print(
                        f"t={elapsed:5.1f}s | 累计前进 {mean_dist:+.3f} m"
                        f"（平均 {mean_dist / elapsed:+.3f} m/s）"
                        f" | 摔倒 {falls} 次"
                        f" | 指令 (vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f})",
                        flush=True,
                    )
                if args_cli.steps and step >= args_cli.steps:
                    break
    except KeyboardInterrupt:
        print("\n已停止演示。")

    # 3. 汇总成绩
    elapsed = time.monotonic() - t_start
    mean_dist = cum_dist.mean().item()
    print(
        f"=== 演示结束：运行 {elapsed:.1f} s，累计前进 {mean_dist:+.3f} m"
        f"（平均 {mean_dist / elapsed:+.3f} m/s）| 摔倒 {falls} 次 ==="
    )
    print(
        f"=== 直行质量（全程峰值）：横向偏移 {best_lateral:.3f} m |"
        f" 偏航偏移 {best_yaw:.2f} rad（{best_yaw * 57.3:.1f}°）==="
    )

    env.close()
    # 注意：无头模式下 simulation_app.close() 可能直接结束进程，属正常现象
    simulation_app.close()


if __name__ == "__main__":
    main()
