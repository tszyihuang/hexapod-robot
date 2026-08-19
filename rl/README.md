# 六足机器人强化学习（CPG + RL）

Spiderbot 六足机器人（18 × Hiwonder LX-15D 舵机）行走训练，Isaac Lab + rsl_rl PPO。

## 架构

策略输出 6 个步态参数 → CPG 振荡器（三角步态 + 向量化 IK）→ 18 个舵机目标。
步态的平滑、节律、对称由 CPG 数学结构性保证，奖励 = 8 个标准项 + 打滑/偏移惩罚。

```
策略(神经网络) ──> [步幅, 横移, 转角, 抬脚高, 步频, 身高] ──> HexapodCPG ──> 18 舵机目标 ──> 实机
   ▲                                                                  │
   └──────────── 58 维观测（含 CPG 实时相位 + 偏航角） ◄────────────────────────┘
```

## 文件

| 文件 | 职责 |
|---|---|
| `hexapod_cpg.py` | CPG 核心：参数 → 足端轨迹 → 向量化 IK → 关节角（与 `../src/kinematics.py` 的 TripodGait 同源，足端轨迹 0.00mm 误差） |
| `hexapod_env_cfg.py` | Isaac Lab 环境：CPG 动作项、58 维观测、12 项奖励（8 标准 + 打滑/横向/偏航角速度/偏航角） |
| `hexapod_symmetry.py` | 左右镜像对称数据增强 |
| `hexapod_agent_cfg.py` | PPO 超参数（含对称增强） |
| `train.py` | 训练入口（单卡） |
| `play_cpg.py` | 展示：手工参数驱动 CPG（`--realistic` = 实机 servo_driver 同款参数） |
| `play_trained.py` | 展示：训练好的策略走路（默认一直运行直到 Ctrl+C，自动加载 `checkpoints/` 或 `logs/` 下最新策略） |
| `checkpoints/`（入库） | 保留的最佳策略 `model_499.pt`，`play_trained.py` 默认加载 |
| `logs/`（不入库） | 训练产物：`rsl_rl/hexapod/<时间戳>/model_*.pt` 检查点与 TensorBoard 日志，已被 `.gitignore` 忽略 |

## 运行

```bash
source /home/tszyi/miniforge3/envs/env_isaaclab/bin/activate   # 激活 Isaac Lab 环境
cd rl

python play_cpg.py --realistic   # 实机同款参数的三角步态演示
python play_trained.py           # 训练好的策略（一直走，Ctrl+C 停止）

python train.py --headless               # 单卡训练（默认 1000 轮）
```

### 单卡训练（本机 RTX 5060）

默认 `--num_envs 1024`：1024 个平行世界只吃约 3GB 显存，5060（16GB）完全够用。
本机瓶颈不是 GPU 算力，而是 Isaac Sim 每步的固定开销（Python 调度 + CPU/GPU 同步），
因此单卡下每轮约需 2 秒，1000 轮约 35 分钟：

```bash
python train.py --headless --num_envs 512 --max_iterations 10   # 先小跑验证
python train.py --headless --max_iterations 1000                # 正式训练
```

检查点定期保存到 `logs/rsl_rl/hexapod/<时间戳>/model_*.pt`；选出最佳策略后复制到
`checkpoints/` 保留（该目录入库，不会被 `.gitignore` 忽略）。

## 关键设计

- **位置控制 + CPG**：策略输出步态参数，CPG 生成舵机位置指令，与实机 LX-15D
  行为一致；步态天然平滑、三角步态节律、左右对称。
- **打滑惩罚**：`feet_slide` 用 CPG 相位掩码（支撑相中段 u∈0.1~0.4）+ 接触力掩码 +
  足尖点速度（胫节 link 速度经 `v_tip = v_link + ω × r_tip` 旋转修正，足尖偏移来自
  URDF 的 0.14m）惩罚足端在地面的水平滑动。落地瞬间足端轨迹水平速度高达 3× 体速，
  属于正常步态，故只在支撑相中段惩罚。
- **偏移惩罚**：`lin_vel_y`（横向速度 L1）+ `ang_vel_z`（偏航角速度 L2，压摆动）+
  `heading_deviation`（偏航角 L2，压净漂移——角速度惩罚压不住每周期累积一点的
  偏航，旧版策略 10 秒内偏航 81°、横移 0.9m）。观测里加入偏航角 `sin/cos`
  （"指南针"），否则策略看不见自己的朝向，无法主动修正漂移。
- **站姿参考**：实机验证过的自然站立姿态（`TripodGait(body_height=70, reach=90)`）。
- **控制频率 50 Hz**：`decimation=4 × dt=5ms = 20ms`，与实机串口舵机刷新率一致。
- **接触传感器 50 Hz**：`update_period = decimation × dt`，与奖励/终止读取频率一致
  （原来 200Hz 更新纯属浪费，接触报告是每步固定开销的大头之一，已按本机适配下调）。
- **地面摩擦对齐实机**：橡胶足端不打滑（static 0.9 / dynamic 0.7）。
- **舵机 PD**：stiffness=30、damping=1、effort_limit=1.5 N·m（接近 LX-15D 堵转力矩）。

## 训练效果（`play_trained.py` 10 秒评估参考值）

| 指标 | 旧策略（无惩罚） | 当前策略（打滑+偏移惩罚+偏航观测） |
|---|---|---|
| 前进距离 | 0.79 m | **1.39 m**（+76%） |
| 横向偏移峰值 | 0.90 m | **0.20 m** |
| 偏航偏移峰值 | 81° | **6.9°** |
| 摔倒 | 0/4 | 0/4 |

> 上表为旧 3×3090 环境训练所得策略的评估值；单卡 5060 重新训练后请以实际评估为准。

## 已知怪癖

- 无头模式下 `simulation_app.close()` 可能直接结束进程、不打印最后一行输出，属正常。
- 必须加载 `models/hexapod.usd`（完整舞台），`models/configuration/hexapod_robot.usd`
  单独加载只是空壳。
