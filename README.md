# Spiderbot 六足机器人控制程序

基于 Hiwonder LX-15D × 18 串行总线舵机（USB 转总线模块直连，串口
115200 8N1）的六足机器人运动学与步态控制程序，支持交互控制台与键盘
实时控制。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `kinematics.py` | 纯数学层：正/逆运动学、三角步态轨迹生成。无 I/O、无副作用，可独立测试 |
| `servo_driver.py` | 串口协议 + 步态控制循环 + 交互控制台（REPL） |
| `keyboard_detect.py` | 键盘实时控制（W/S/A/D 平移、Q/E 旋转、斜向组合，松开即停） |
| `physical_config.json` | **唯一的物理参数来源**：腿几何、舵机映射、关节限位、角度约定、安装坐标 |

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
| `stand` | 自然站立（由 IK 按配置生成，含 5° 前偏） |
| `flatten` | 展平（所有舵机回到默认展平位置 512） |
| `walk [速度] [步幅]` | 三角步态前进/后退（正数前进，默认 350 mm/s / 100 mm） |
| `strafe [速度] [步幅]` | 三角步态左右平移（正数左移） |
| `move [vx] [vy]` | 按身体坐标系速度向量平移（vx 前正、vy 左正） |
| `turn [角速度] [单周期转角]` | 原地旋转（正数左转，默认 80 deg/s / 20°） |
| `relax` | 卸载所有舵机（失能，可手动转动） |
| `help` / `quit` | 显示帮助 / 退出（退出前自动失能全部舵机） |

### 键盘控制

```bash
sudo python3 keyboard_detect.py [--port ...] [--speed ...] [--turn-speed ...]
```

W/S/A/D 前后左右，Q/E 逆/顺时针旋转；W+A、W+D、S+A、S+D 斜向组合
（斜向分量自动取 √2/2，合速度与单方向一致）；松开即停，ESC 退出。

## 物理参数

所有几何与舵机参数以 `physical_config.json` 为准，代码不另设硬编码副本。
修改配置（如腿长度、安装点、关节限位）后无需改代码即可生效。
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
