# Spiderbot 六足机器人控制程序

> 基于 Hiwonder LX-15D 串行总线舵机的六足机器人运动学与步态控制程序。
> 机体为亚博（Yahboom）Spiderbot 六足原型机，从物理配置到逆运动学、
> 三角步态、串口驱动，再到键盘遥控，一套纯 Python 实现。
>
> *A Python hexapod robot control stack: kinematics, tripod gait, serial-bus
> servo driver and keyboard teleoperation — config-driven, zero hardcoded magic
> numbers.*

## 项目简介

本项目为亚博（Yahboom）Spiderbot 六足原型机（18 舵机，6 腿 × 3 关节）
提供完整的底层控制软件：以 `physical_config.json` 为唯一物理参数来源，通过解析
逆运动学求解每条腿的舵机角度，生成三角步态（tripod gait）足端轨迹，
经 USB 转总线模块以 115200 8N1 串口实时驱动 Hiwonder LX-15D 舵机。

支持三种控制方式：

- **交互控制台（REPL）**：站立、展平、行走、横移、原地旋转、卸力；
- **键盘实时遥控**：W/S/A/D 平移、Q/E 旋转、斜向组合，松开即停；
- **只读监视器**：舵机位置持续监视与 IMU 姿态数据实时显示。

## 特性

- 🕷️ **纯 Python 运动学**：正/逆运动学与步态生成完全独立于硬件 I/O，
  可在无机器人环境下离线测试（`kinematics.py` 仅依赖标准库）；
- 📐 **配置驱动**：腿几何、舵机映射、关节限位、角度约定、安装坐标全部
  来自 `physical_config.json`，改硬件参数无需改代码；
- 🦵 **三角步态**：支撑相/摆动相相位差 0.5，支撑相足端相对身体等速后退
  保证不打滑，支持任意水平速度向量合成（平移 + 横移 + 旋转）；
- 🔌 **串行总线舵机协议**：LX-15D/LX-16A 帧的构造、校验与解析，
  含超时与应答匹配、失联自动重扫；
- 🎮 **实时键盘控制**：全局按键监听，多键斜向组合，松开自动回自然站立；
- 📡 **IMU 监视**：亚博（Yahboom）IMU-Sensor 九轴数据的只读解析与
  覆盖式实时显示（欧拉角 / 四元数 / 加速度 / 角速度 / 磁场）；
- 🤖 **URDF 模型**：由配置自动生成的机器人模型，可直接用于 RViz /
  PyBullet 等工具做可视化与仿真。

## 硬件清单

| 部件 | 规格 |
| --- | --- |
| 机体 | 亚博（Yahboom）Spiderbot 六足原型机（6 腿 × 3 关节） |
| 舵机 | Hiwonder LX-15D × 18，串行总线舵机 |
| 通信 | USB 转总线模块直连，串口 115200 8N1 |
| 主机 | Linux（Python 3.10+），需串口权限 |
| 传感器（可选） | 亚博 Yahboom IMU-Sensor（CH340 USB 串口） |

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `kinematics.py` | 纯数学层：正/逆运动学、三角步态轨迹生成。无 I/O、无副作用，可独立测试 |
| `servo_driver.py` | 串口协议 + 只读位置监视 + 步态控制循环 + 交互控制台（REPL）。原 `servo_protocol.py` / `servo_monitor.py` 已合并到本文件 |
| `keyboard_detect.py` | 键盘实时控制（W/S/A/D 平移、Q/E 旋转、斜向组合，松开即停） |
| `physical_config.json` | **唯一的物理参数来源**：腿几何、舵机映射、关节限位、角度约定、安装坐标 |
| `hexapod.urdf` | 机器人 URDF 模型（由配置自动生成，勿手改） |
| `imu_monitor.py` | 亚博 IMU 姿态传感器监视器（只读，覆盖式打印） |

依赖方向单向：`keyboard_detect` → `servo_driver` → `kinematics`；
`kinematics` 只依赖 Python 标准库。

## 安装

需要 Python 3.10+（代码使用了 `X | None` 联合类型语法）。

```bash
pip install -r requirements.txt
```

Linux 下访问串口需要加入 dialout 组（改完重新登录生效）：

```bash
sudo usermod -aG dialout $USER
```

`keyboard_detect.py` 依赖全局键盘事件监听，Linux 下需要 root 权限运行。

## 使用

### 交互控制台

```bash
python3 servo_driver.py [--port /dev/ttyUSB0] [--baud 115200] [--time 1500]
```

| 命令 | 说明 |
| --- | --- |
| `read [间隔秒]` | 持续读取 18 个舵机位置并覆盖打印（Ctrl+C 停止） |
| `stand` | 自然站立（由 IK 按配置生成） |
| `flatten` | 展平（所有舵机回到默认展平位置 512） |
| `walk [速度] [步幅]` | 三角步态前进/后退（正数前进，默认 350 mm/s / 100 mm） |
| `strafe [速度] [步幅]` | 三角步态左右平移（正数左移） |
| `move [vx] [vy]` | 按身体坐标系速度向量平移（vx 前正、vy 左正） |
| `turn [角速度] [单周期转角]` | 原地旋转（正数左转，默认 80 deg/s / 20°） |
| `relax` | 卸载所有舵机（失能，可手动转动） |
| `help` / `quit` | 显示帮助 / 退出（退出前自动失能全部舵机） |

### 只读位置监视

```bash
python3 servo_driver.py --monitor [--port /dev/ttyUSB0] [--baud 115200] [--ids 1-18] [--interval 0]
```

`--ids` 支持逗号分隔或区间，例如 `1-18`、`13,14,15`、`1-6,13-18`。

### 键盘控制

```bash
sudo python3 keyboard_detect.py [--port ...] [--speed ...] [--turn-speed ...]
```

W/S/A/D 前后左右，Q/E 逆/顺时针旋转；W+A、W+D、S+A、S+D 斜向组合
（斜向分量自动取 √2/2，合速度与单方向一致）；松开即停，ESC 退出。

### IMU 监视器

```bash
python3 imu_monitor.py [--port /dev/ttyUSB1] [--baud 115200] [--interval 0] [--plain]
```

未指定 `--port` 时自动探测 `/dev/ttyUSB*`、`/dev/ttyACM*` 与
`/dev/serial/by-id/*`。只读运行，不影响机器人与 IMU。

## 物理参数

所有几何与舵机参数以 `physical_config.json` 为准，代码不另设硬编码副本。
修改配置（如腿长度、安装点、关节限位）后无需改代码即可生效；
`hexapod.urdf` 需重新生成（见文件头部注释）。
`kinematics.py` 模块 docstring 包含坐标系、角度符号与步态相位约定的完整说明。

## 串口协议

USB 总线模块直连舵机，帧格式
`0x55 0x55 | ID | 长度 | 指令 | 参数 | 校验和`：

- 长度 = 参数个数 + 3
- 校验和 = `~(ID + 长度 + 指令 + 参数之和)` 的低 8 位
- `0x01` 单舵机移动（含时间）
- `0x0E` 舵机 ID 查询
- `0x1C` 舵机位置读取
- `0x1F` 舵机加载/卸载（扭矩开关）

## 许可

暂无 License 文件；如需引用或复用代码，请先联系作者。
