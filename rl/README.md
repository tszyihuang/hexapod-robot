# 六足机器人强化学习（CPG + RL）

Spiderbot 六足机器人（18 × Hiwonder LX-15D 舵机）行走训练，Isaac Lab + rsl_rl PPO。

## 架构

策略输出 6 个步态参数 → CPG 振荡器（三角步态 + 向量化 IK）→ 18 个舵机目标。
步态的平滑、节律、对称由 CPG 数学结构性保证，奖励函数只有 8 个标准项。

```
策略(神经网络) ──> [步幅, 横移, 转角, 抬脚高, 步频, 身高] ──> HexapodCPG ──> 18 舵机目标 ──> 实机
   ▲                                                                  │
   └──────────── 56 维观测（含 CPG 实时相位） ◄────────────────────────┘
```

## 文件

| 文件 | 职责 |
|---|---|
| `hexapod_cpg.py` | CPG 核心：参数 → 足端轨迹 → 向量化 IK → 关节角（与 `../kinematics.py` 的 TripodGait 同源，足端轨迹 0.00mm 误差） |
| `hexapod_env_cfg.py` | Isaac Lab 环境：CPG 动作项、56 维观测、8 项标准奖励 |
| `hexapod_symmetry.py` | 左右镜像对称数据增强 |
| `hexapod_agent_cfg.py` | PPO 超参数（含对称增强） |
| `train.py` | 训练入口 |
| `play_cpg.py` | 展示：手工参数驱动 CPG（`--realistic` = 实机 servo_driver 同款参数） |
| `play_trained.py` | 展示：训练好的策略走路（自动加载 `logs/` 下最新策略） |
| `logs/model_499.pt` | 当前最佳策略（500 轮训练：节律 0.73、六腿均衡、零摔倒） |

## 运行

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/hexapod-robot/rl

python play_cpg.py --realistic   # 实机同款参数的三角步态演示
python play_trained.py           # 训练好的策略
python train.py --headless       # 重新训练（默认 1000 轮 ≈ 20 分钟）
```

## 关键设计

- **位置控制 + CPG**：策略输出步态参数，CPG 生成舵机位置指令，与实机 LX-15D
  行为一致；步态天然平滑、三角步态节律、左右对称。
- **站姿参考**：实机验证过的自然站立姿态（`TripodGait(body_height=70, reach=90)`）。
- **控制频率 50 Hz**：`decimation=4 × dt=5ms = 20ms`，与实机串口舵机刷新率一致。
- **地面摩擦对齐实机**：橡胶足端不打滑（static 0.9 / dynamic 0.7）。
- **舵机 PD**：stiffness=30、damping=1、effort_limit=1.5 N·m（接近 LX-15D 堵转力矩）。

## 已知怪癖

- 无头模式下 `simulation_app.close()` 可能直接结束进程、不打印最后一行输出，属正常。
- 必须加载仓库根目录的 `hexapod.usd`（完整舞台），`configuration/hexapod_robot.usd`
  单独加载只是空壳。
