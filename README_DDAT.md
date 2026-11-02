# DDAT UUV 训练

当前打包目录不含下文提到的 `animate_mpc_tracking.py`；动画命令是原项目的历史说明。本目录的现行结构与实验配置见 `README.md`。

`DDAT_train.py` 沿用 `train_residual_ppo_batch.py` 的路径 CSV 格式、批量 UUV 动力学、
PPO 训练和评估流程。它使用下列控制链：

1. 将未来 12 个路径点旋转到艇体坐标系，连同相对航向、俯仰和有效位组成 12×6 预览；
   策略内的前馈 MLP 将其编码为 64 维特征，随 PPO 联合训练。
2. 拼接当前艇体状态、机体系跟踪误差、目标速度预览、五维动力学扰动估计与
   三维海流估计。训练时为每艘艇每回合采样一股固定的水平 NED 海流；真实值
   只进入仿真，不进入策略观测。
3. 扰动估计由实测艇体速度变化减去标称模型预测，并用一阶低通滤波更新。
   这是 L1 风格的在线估计器，不是严格 L1 自适应控制律。
4. 策略输出相对名义 MPC 指令的推进力修正及俯仰、偏航角速度修正。
   低层控制器用标称动力学逆解舵角，并用扰动估计修正推进力和角加速度目标。
   执行器延迟、响应和失配艇体仍由批量环境仿真。
5. 海流估计由实测位移减去艇体相对水速度预测的位移，并经低通滤波；名义
   MPC 用该估计修正未来参考位置和速度。这里假设艇体系速度是相对水速度，
   海流只造成惯性系位置漂移。
6. 路径 CSV 显式给出 `target_pitch_rad`、`target_yaw_rad`。DDAT 的 MPC 跟踪输出为
   `[north, east, depth, pitch, yaw, north_velocity, east_velocity, depth_velocity]`；
   默认俯仰与偏航权重分别为 100 和 30，可用 `--attitude-pitch-weight`、
   `--attitude-yaw-weight` 调整。旧路径未提供姿态列时仍由几何切线推导。
   这两个权重改善姿态目标的表达，但不保证瞬态可行或避免俯仰越界；仍需按
   目标速度、执行器延迟和随机海流验证每条轨迹。

运行示例：

```bash
python3 DDAT_train.py --device cpu --backend admm --envs 8 \
  --current-max-mps 1.0 --current-filter-time 1.0 \
  --output outputs/DDAT/example
python3 DDAT_train.py --device cpu \
  --eval-actor outputs/DDAT/example/final_actor.pt
```

默认训练路径为 `DDAT_trajectories/train`；评估使用 `DDAT_trajectories/val`。
轨迹设计和再生成方法见 `DDAT_trajectories/README.md`。输出包含训练配置、逐步奖励、每回合奖励、历史指标和
`final_actor.pt`。`--current-max-mps` 默认 1.0，设为 0 可做无海流对照；
训练和验证使用同一上限，验证的采样种子独立。新 checkpoint 的 `architecture`
为 `DDAT_UUV_current`，观测增加三维，旧 DDAT checkpoint 不能直接加载；
请使用本脚本的 `--eval-actor`。

## 单条路径的 MPC 跟踪动画

`animate_mpc_tracking.py` 接受任意兼容的路径 CSV，以零策略动作运行 DDAT 环境中的
MPC 和低层控制器。在 Matplotlib 的 3D 窗口中播放路径跟踪，艇体上的红、绿、蓝
三根轴分别表示前、右、下方向，随艇体位置与俯仰、偏航一起运动；视窗跟随艇体。
仿真到成功、失败或超时后结束。先在脚本顶部设置 `PATH`、`BACKEND` 等全局变量；
`OUTPUT = None` 时弹出窗口，设为 `.gif` 或 `.mp4` 路径时保存文件。

```bash
python3 animate_mpc_tracking.py
```

修改顶部的 `SEED`、`HORIZON`、`CURRENT_MAX_MPS`、`RANDOMIZE_PLANT` 和姿态权重
可重现或调整工况；`FOLLOW_RADIUS` 控制视野，`FRAME_STRIDE` 控制抽帧间隔。
图标题会显示位置误差、MPC 是否回退及最终状态。

## 验证与绘图

`validate_ddat.py` 在相同路径、种子、海流和艇体参数下，分别运行 DDAT 控制器与
零策略动作的 MPC 对照。默认读取 `outputs/DDAT` 或 `outputs/experiments` 中最新的 `final_actor.pt`，
按 checkpoint 的配置选择验证路径、控制器参数、预览模式、奖励和艇体随机化设置；
每条路径运行一个配对种子。自动设备选择会为 OSQP 和 CPU RTI 使用 CPU。

```bash
python3 validate_ddat.py
python3 validate_ddat.py --checkpoint outputs/DDAT/example/final_actor.pt \
  --path DDAT_trajectories/val/s_reversal_01.csv --seed 101026 \
  --device cpu --no-show
python3 validate_ddat.py --split train --seeds 3
```

可用 `--eval-episodes` 限制路径数，用 `--randomize-plant` / `--no-randomize-plant`
及 `--feasible-only` / `--no-feasible-only` 覆盖 checkpoint 设置。
`--output-dir` 指定结果目录；单路径单种子时，`--show` 可打开 3D 窗口。
输出目录包含汇总 CSV、JSON，以及按路径和种子分目录保存的逐步 CSV 和图像。
对照组仍使用 DDAT 环境的低层逆动力学与海流、扰动估计，只将策略动作置零。
