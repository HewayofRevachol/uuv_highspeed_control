# RL 增强 MPC 研究进展与续跑手册

日期：2026-11-02。续跑工作目录：`/home/prufrock/uuv_workspace`。原始实验在 `/home/prufrock/uuv_simulator` 完成，已将脚本、轨迹、checkpoint 和结果复制到本工作区。**本文件是下次继续工作的入口**；完整论文清单见 [`../academic_search_report/RL_MPC_top_tier_20261101.md`](../academic_search_report/RL_MPC_top_tier_20261101.md)，详细实验假设与历次记录见 [`RL_MPC_innovation_experiments_20261102.md`](RL_MPC_innovation_experiments_20261102.md)。

## 一页结论

1. **当前最值得推进的是“双侧可行速度包络 + MPC”。** 低速时需要足够航速维持舵效，高速急弯又需要局部减速以缓解推力/舵角饱和；单纯“弯道一律减速”不适用于这套 UUV。用户已确认任务采用统一允许航速范围，上限 **18 m/s**，下限尚未确定。实验中的 9 m/s 下限是探针值，不能当任务约束或物理定律。
2. **强确定性基线已优于当前 RL 原型。** 在 4 条留出高速压力任务、每条 2 个扰动种子的 18 m/s 仿真中，固定参考完成 **7/8**，前方曲率限速 `curv6` 完成 **8/8**。双方共同成功的 7 回合中，位置 RMSE 均值 **1.996→1.462 m**，平均用时从 **603.6→615.1 个控制步**（约 +1.9%）。固定 15 m/s 上限也完成 8/8，但平均用时更长、速度偏差更大。`curv6` 的横向加速度阈值尚未标定，**没有形式安全保证**。
3. **两版直接 PPO 调参考没有超过强基线。** 第一版在简单留出任务几乎不调速；第二版在高速压力任务完成 7/8，位置误差略好于固定18，但低于固定17的精度，未修复唯一的失败回合。第三版“曲率规则＋受限 RL 残差”已训练完成；留出评估按用户本次要求暂停，现有 **2/8** 的部分数据不能作结论。
4. 这些都是仿真、少量种子的机制实验。当前优先级应是**让 RL 学规则无法覆盖的海流/参数不确定度和长期权衡**，并以曲率规则、固定限速和更强 MPC 为对照；不能把“RL 接入 MPC”本身写成创新结果。

## 我手动的补充

发现结合diffusion model或许可以作为一个创新方向，昨天看了一个论文，它是使用传统的趋势/残差检测模型+条件diffusion模型进行扰动预测的，毕竟UUV/AUV的这个方向很少，可以调查一下文献，看看有没有机会进行结合

我又调查了一下，发现去年有人发过AUV+Diffusion了，不过依然可以试一试能不能针对特定模块和问题来引入它。

-----------------------

另外就是，当前完成的实验中，我怀疑有些过于死磕目前的路径了，但是这些路径合不合适我并不确定，后续应该考虑一下这个问题。还有就是，我预期的新工作区位于~/uuv_workspace 我希望进行迁移




## 值得借鉴的顶刊/顶会方法

| 来源 | 核心方法 | 对本仓库的迁移判断 |
| --- | --- | --- |
| [Actor–Critic MPC，IEEE T-RO](https://arxiv.org/html/2306.09852) | PPO actor 输出可微 MPC 二次代价参数 `Q,p`，MPC 是 actor 最后一层；critic 学长期价值，预测轨迹还用于价值扩展。 | 后续可学习低维代价或进度参数，再研究可微 QP。论文实现主要约束输入，不能把它直接当状态安全保证。 |
| [TD-MPC，ICML 2022](https://proceedings.mlr.press/v162/hansen22a.html) | 任务导向潜在动力学、短窗规划和 TD 终端价值联合训练；[作者项目](https://td-mpc.github.io/)说明推理用 MPPI。 | UUV 状态维度低，先在物理 MPC 上试受限终端价值，不必照搬大型潜在模型。 |
| [RL-guided robust MPC，IEEE/CAA JAS 2024](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.124350) | RL 规划高层参考，鲁棒 MPC 负责跟踪。 | 与本文的一维参考航速接口最接近；仅调参考不是充分新意。该文是 3 页 Letter。 |
| [Event-triggered RL-MPC，IEEE T-IV 2024](https://jchen2020.net/pubs/j_tiv2024-e.pdf) | RL 学何时重求解 MPC；比较 DDQN、PPO、SAC，并使用 PER/LSTM。 | 本仓库 OSQP 失败回合的回退率为 0，先解决参考与模型问题，计算调度暂列次要。 |
| [MPCC++，RSS 2024](https://www.roboticsproceedings.org/rss20/p109.html)；[HyperMPC，CoRL 2025](https://proceedings.mlr.press/v305/wegrzynowski25a.html) | 前者采用路径/终端安全约束和动力学残差；后者学习预测窗内时变模型参数。 | 两者是必须考虑的**非 RL 强 MPC 基线**，不能误写成 RL 先例。 |

顶刊分区和正式收录信息有版本差异，完整清单已标明使用的分区口径。当前未下载论文 PDF；上述结论来自出版方页面、作者公开 HTML 或可访问的全文。若下次要精确复现 T-IV 的 Q 网络终端代价，建议由用户将 DOI `10.1109/TIV.2024.3363239` 的全文放到 `/home/prufrock/uuv_workspace/Papers/Qi2024_TIV_terminal_value.pdf`，再逐式核对。

## 已完成的对照实验

统一设置：`/bin/python3`（Torch 2.14，当前无 CUDA）、FSP 开启、MPC 预测 25 步，控制周期 0.1 s。评估用 CPU OSQP、随机艇体参数、最大 1 m/s 海流、匹配路径与扰动种子；训练用 CPU 批量 ADMM。位置 RMSE 只在**共同安全完成**的回合间比较，失败回合先看完成率和失败类型。原始压力轨迹被重新赋值为 18 m/s，此任务只用于仿真方法筛选，不代表实艇可行性。

| 任务与方法 | 安全完成 | 共同成功 7 回合位置 RMSE | 共同成功 7 回合平均控制步 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 高速压力集，固定 18 m/s | 7/8 | 1.996 m | 603.6 | 强零动作 FSP＋OSQP MPC 基线。 |
| 高速压力集，固定上限 17 m/s | 7/8 | 1.554 m | 627.3 | 减小误差，仍未修复困难回合。 |
| 高速压力集，固定上限 15 m/s | 8/8 | 1.296 m | 701.7 | 完成更多，但明显牺牲任务航速。 |
| 高速压力集，前方曲率限速 `curv6` | **8/8** | **1.462 m** | **615.1** | 与固定18相比，在速度和位置精度间取得较好权衡。 |
| 高速压力集，第二版直接 PPO 参考 | 7/8 | 1.707 m | 607.7 | 比固定18略好，未超过固定17或 `curv6`。 |

补充的低速探针：6 m/s、5 条验证路径 × 2 种子，固定参考 **0/10**，参考下限 8 m/s **3/10**，参考下限 9 m/s **8/10**。9 m/s 成功回合的任务航速 RMSE 均值仍约 **3.402 m/s**，说明“完成路径”与“维持 6 m/s”是两个不同指标。18 m/s 常规验证路径上，固定、17 m/s 上限、15 m/s 上限都 **10/10**，故常规集本身不足以证明 RL 的安全收益。

### 三次 RL 原型

| 版本 | 训练任务与步骤 | 留出结果 | 判定 |
| --- | --- | --- | --- |
| V1：直接调参考 | 原 20 条训练路径，12/18 m/s，60,000 环境步。 | 5 条验证路径 × 2 种子：策略与固定 MPC 均 10/10，位置 RMSE 均 0.449 m；策略平均参考 17.998 m/s。 | 策略实质不调速，负结果。 |
| V2：困难任务直接调参考 | 24 条程序生成的半径/入弯/深度坡度组合，18 m/s，80,000 步，加饱和惩罚。 | 高速压力集 7/8，成功回合位置 RMSE 1.707 m；困难入弯种子仍失败。 | 有小幅调速收益，未超过强规则。 |
| V3：曲率规则＋RL 残差 | 同一程序生成任务，60,000 步；RL 只改参考约 ±2 m/s，带残差与饱和惩罚。 | 留出评估暂停，现只有 2/8 部分回合。 | **下次首先完成评估。** |

程序生成训练集的半径为 36/44/52/60 m、入弯长度为 10/20/30 m、深度坡度为 0/0.04；原始压力轨迹中的半径 40 m、15 m 入弯未用于 V2/V3 训练。V3 零动作环境已与独立 `curv6` 基线做逐回合核对：同一回合均为 370 步、位置 RMSE 2.9838327276 m。

## 代码、结果和 tmux

| 用途 | 文件/目录 |
| --- | --- |
| 论文清单 | `academic_search_report/RL_MPC_top_tier_20261101.md` |
| 详细方案与阶段记录 | `high_speed_experiments/RL_MPC_innovation_experiments_20261102.md` |
| 固定/曲率参考探针 | `high_speed_experiments/reference_schedule_probe.py` |
| 生成训练任务 | `high_speed_experiments/generate_reference_curriculum.py`、`high_speed_experiments/generated/reference_curriculum_1102/` |
| 三版 RL 训练代码 | `high_speed_experiments/train_reference_policy.py` |
| RL 配对评估代码 | `high_speed_experiments/evaluate_reference_policy.py` |
| 最关键的静态基线数据 | `high_speed_experiments/outputs/reference_schedule_stress18_1102.jsonl`、`reference_curvature_stress18_1102.jsonl` |
| V1/V2/V3 模型与配置 | `high_speed_experiments/outputs/reference_policy_1102/`、`reference_policy_curriculum_1102/`、`reference_policy_curvature_1102/` |
| V2 留出评估 | `high_speed_experiments/outputs/reference_policy_curriculum_eval_stress18_1102.jsonl` |
| V3 部分留出评估 | `high_speed_experiments/outputs/reference_policy_curvature_eval_stress18_1102.jsonl` |

所有训练均用 `/bin/python3` 放在 tmux；socket 为 `/tmp/uuv_ref_probe_1102.sock`。近期会话包括 `uuv_rl_ref_train`、`uuv_rl_ref_curriculum_v2`、`uuv_rl_curvature_residual`，各输出日志在 `high_speed_experiments/outputs/` 的同名前缀 `*_train.log`。用户提出先写报告后，已停止刚启动的 V3 留出评估；**没有仍需等待的训练**。模型和已有 JSONL 均保留。原目录中的新增文件当时均未提交；迁移后的文件为独立副本。V1 checkpoint 内的训练路径仍是原目录绝对路径，仅作为历史记录，评估不依赖这些路径。



## 下次从这里继续

1. **先跑完 V3 留出评估。** 使用 `final_actor.pt`，在四条原压力轨迹、种子 `101026 101027`、18 m/s 上重跑 `evaluate_reference_policy.py --modes policy --backend osqp`；输出文件会被脚本以写入模式覆盖，当前 2/8 部分结果只作进度记录。比较 `curv6` 的 8/8、共同成功回合 RMSE 及速度/饱和/步数。不要用训练完成率选模型。
2. 若 V3 仍不优于 `curv6`，把**双侧动力学可行速度包络**作为主线：用舵效/扰动估计求局部最低可控速度，用推力、曲率、深度变化和延迟求局部最高可行速度；无可行区间时要求任务重规划。RL 只在经过投影的区间内优化长期速度/能耗/精度权衡。与仅设上限、仅设下限、固定限速和确定性曲率规则配对比较。
3. 若要研究论文中更强的结构化 RL，优先在强 MPC 上试 **TD 终端价值**，或将 actor 输出改成可微 MPC 的受限代价参数；先验证价值校准/多步模型误差，再谈安全或泛化。事件触发求解暂不优先。
4. 正式结论至少增加独立训练种子、更多未见几何家族和配对扰动种子；分开报告成功率、失败类别、成功回合位置/航速 RMSE、饱和、总时间和控制周期耗时。当前 8 或 10 个回合的结果只能用于选方向。

下次重跑 V3 的核心命令（在 `/home/prufrock/uuv_workspace` 执行；若是训练，继续放进 tmux）：

```bash
/bin/python3 high_speed_experiments/evaluate_reference_policy.py \
  --checkpoint high_speed_experiments/outputs/reference_policy_curvature_1102/final_actor.pt \
  --paths trajectories/stress_15mps/cold_helix_r49_15p9mps.csv \
          trajectories/stress_15mps/coupled_turn_depth_16mps.csv \
          trajectories/stress_15mps/ramped_helix_r40_15mps.csv \
          trajectories/stress_15mps/rapid_depth_wave_15mps.csv \
  --speeds 18 --seeds 101026 101027 --modes policy --backend osqp \
  --output high_speed_experiments/outputs/reference_policy_curvature_eval_stress18_1102.jsonl
```
