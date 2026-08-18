"""CPG 手工验证：直接给参数，看机器人走出优雅三角步态（不经过强化学习）。

两种模式：
    python play_cpg.py            演示模式：站立 -> 前进 -> 转向（参数温和，慢慢看）
    python play_cpg.py --realistic 实机模式：完全复刻 servo_driver.py 的默认参数
                                   （步幅 100mm、300mm/s、转向 60°/s、20°/周期、
                                   1.5 周期的 smoothstep 幅值爬升、50Hz）

参数换算：param = center + action × scale（见 hexapod_env_cfg.CPGActionCfg）
    步幅 stride = 0.045 + 0.045·a   （演示模式最大 9cm；实机模式 100mm 会超过动作域，
                                     手动下发不受限）
    横移 lateral = 0.04·a
    转角 turn = 0.25·a              （rad/周期，正=左转）
    抬脚高 step_height = 0.03 + 0.015·a
    步频 freq = 1.2 + 0.9·a
    身高 body_height = 0.07 + 0.015·a

实机代码对照（hexapod-robot/servo_driver.py）：
    步态数学  TripodGait.feet_at/turn_pose_at  <->  hexapod_cpg.HexapodCPG.feet（同源，0.00mm 误差）
    相位推进  实机 phase = 时间/周期(2·步幅/速度)  <->  仿真 phase += 步频×dt（数学等价）
    幅值爬升  实机 smoothstep(1.5 周期)           <->  本脚本相同实现
    帧率      实机 50Hz 串口帧                    <->  仿真 50Hz（decimation=4 × 5ms）

运行方式（在终端里）：
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    cd ~/hexapod-robot/rl
    python play_cpg.py              # 弹窗看机器人行走
    python play_cpg.py --headless   # 只看数字
"""

import math
import sys

import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless="--headless" in sys.argv)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from hexapod_env_cfg import HexapodEnvCfg  # noqa: E402


def smoothstep(t: float) -> float:
    """与 servo_driver.py 的 _smoothstep 完全一致：t∈[0,1] 从 0 平滑过渡到 1。"""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def action_for(stride=0.0, lateral=0.0, turn=0.0, step_h=0.03, freq=1.2, h=0.07) -> torch.Tensor:
    """把物理参数换算成 [-1,1] 动作向量（与 CPGActionCfg 的中心/幅度一致）。"""
    return torch.tensor(
        [
            [
                (stride - 0.045) / 0.045, lateral / 0.04, turn / 0.25,
                (step_h - 0.03) / 0.015, (freq - 1.2) / 0.9, (h - 0.07) / 0.015,
            ]
        ],
        device="cuda:0",
    )


def build_plan(realistic: bool):
    """生成演示时间轴：[(时长秒, 参数函数 t->action)]。"""
    if realistic:
        # ---- 实机 servo_driver.py 的默认参数 ----
        speed_mm_s, stride_mm = 300.0, 100.0        # cmd_walk 默认：300mm/s、步幅 100mm
        walk_cycle_s = 2.0 * stride_mm / speed_mm_s  # = 0.667s -> 步频 1.5Hz
        walk_ramp_s = 1.5 * walk_cycle_s             # 实机：1.5 个周期爬升

        turn_speed_dps, turn_step_deg = 60.0, 20.0   # cmd_turn 默认：60°/s、20°/周期
        turn_cycle_s = 2.0 * turn_step_deg / turn_speed_dps  # = 0.667s
        turn_ramp_s = 1.5 * turn_cycle_s

        def stand_fn(t):
            return action_for(stride=0.0, freq=1.5)

        def walk_fn(t):
            amp = smoothstep(t / walk_ramp_s)
            return action_for(stride=0.10 * amp, freq=1.5, step_h=0.03, h=0.07)

        def turn_fn(t):
            amp = smoothstep(t / turn_ramp_s)
            return action_for(stride=0.02 * amp, turn=math.radians(20.0) * amp, freq=1.5)

        return [(1.0, stand_fn), (6.0, walk_fn), (3.0, turn_fn), (1.0, stand_fn)], "实机参数模式"
    else:
        # ---- 温和演示参数 ----
        return [
            (2.0, lambda t: action_for(stride=0.0, freq=1.2)),                       # 站立
            (6.0, lambda t: action_for(stride=0.03, freq=1.2, step_h=0.03)),          # 前进 0.072m/s
            (2.0, lambda t: action_for(stride=0.02, turn=0.15, freq=1.0)),            # 左转
        ], "演示模式"


def main():
    sys.stdout.reconfigure(line_buffering=True)

    realistic = "--realistic" in sys.argv
    plan, mode_name = build_plan(realistic)
    print(f"模式：{mode_name}")

    env_cfg = HexapodEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()
    print("观测形状:", tuple(obs["policy"].shape), "（应为 (1, 56)）| 动作维度:", env.action_manager.total_action_dim)

    start_x = env.scene["robot"].data.root_pos_w[:, 0].clone()
    start_y = env.scene["robot"].data.root_pos_w[:, 1].clone()
    step = 0
    with torch.inference_mode():
        for seg_dur, seg_fn in plan:
            seg_start = step * 0.02
            while step * 0.02 < seg_start + seg_dur:
                t = step * 0.02 - seg_start
                actions = seg_fn(t).repeat(env.num_envs, 1)
                obs, rew, terminated, truncated, info = env.step(actions)
                step += 1
                if step % 50 == 0:
                    root = env.scene["robot"].data.root_pos_w[0]
                    phase = env.action_manager.get_term("cpg").phase[0].item()
                    print(
                        f"t={step*0.02:5.1f}s | 前进 {root[0].item()-start_x[0].item():+.3f} m"
                        f" | 横移 {root[1].item()-start_y[0].item():+.3f} m"
                        f" | 高度 {root[2].item():.3f} m | CPG 相位 {phase:.2f} 圈"
                    )

    env.close()
    # 注意：无头模式下 simulation_app.close() 可能直接结束进程，属正常现象
    simulation_app.close()
    print(f"=== CPG 验证完成（{mode_name}）===")


if __name__ == "__main__":
    main()
