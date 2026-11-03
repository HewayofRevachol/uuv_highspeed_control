# UUV 控制故障与路径现实性实跑审查

日期：2026-11-02。审查对象：[推荐路线](RECOMMENDED_ROUTE_ACTUATOR_AWARE_RL_MPC_20261102.md)与[首次证据核验](VERIFY_ACTUATOR_AWARE_RL_MPC_EVIDENCE_20261102.md)涉及的低速失效、执行器机制、18 m/s 压力任务及路径意义。原项目代码、检查点和轨迹来自 `~/uuv_simulator`。本次直接运行代码；新产物集中保存在[审查运行目录](../outputs/reports/uuv_mechanism_audit_20261102/)，源码版本、输入文件、种子及每组配置见[运行清单](../outputs/reports/uuv_mechanism_audit_20261102/run.json)。

## 一、实跑设置和结果

使用[审查脚本](../high_speed_experiments/audit_uuv_mechanisms.py)调用原项目 `ExperimentDDATPathEnv`、`PathBank`、`LoopPath` 及检查点 `outputs/DDAT/run_20261029_114239/final_actor.pt`。零策略动作，CPU OSQP，种子 101027；除特别标注外，固定 `MISMATCHED_PLANT`、零海流、零过程噪声，保留检查点的 MPC 代价与 FSP。每组保存逐周期 CSV 和 JSON 摘要。这里的“完成/失败”沿用环境终止规则；独立列出位置、速度、走廊和俯仰指标。

| 路径及条件 | 预测窗 | 结果 | 关键量 |
| --- | ---: | --- | --- |
| `constant_turn_01`，6 m/s，固定失配植物，静水无噪声 | 2.5 s | **失败**，OSQP 回退 0 | 位置 RMSE 45.825 m；最大横向偏差 44.087 m；速度 RMSE 5.968 m/s；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_constant_turn_01_v6_h25_osqp_fixed_still_quiet_yaw30.json)、[逐时刻轨迹](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_constant_turn_01_v6_h25_osqp_fixed_still_quiet_yaw30.csv)。 |
| 同上，只把预测窗延到 5 s | 5 s | **完成**，OSQP 回退 0 | 位置 RMSE 0.415 m；最大横向偏差 0.599 m；速度 RMSE 0.274 m/s；97.1% 周期处于目标 ±1 m/s；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_constant_turn_01_v6_h50_osqp_fixed_still_quiet_yaw30.json)、[轨迹](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_constant_turn_01_v6_h50_osqp_fixed_still_quiet_yaw30.csv)。 |
| 6 m/s，去掉艇体参数失配、全部执行器延迟/惯性及 FSP，静水无噪声 | 2.5 s | **仍失败**，OSQP 回退 0 | 位置 RMSE 44.471 m；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_constant_turn_01_v6_h25_osqp_ideal_still_quiet_yaw30.json)。这是理想化模型消融，表明原例的短窗失败不需要执行器迟滞或参数失配。 |
| 6 m/s，原固定失配植物，仅把航向参考权重 30 降为 0 | 2.5 s | **仍失败**，OSQP 回退 0 | 位置 RMSE 46.001 m；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_constant_turn_01_v6_h25_osqp_fixed_still_quiet_yaw0.json)。航向权重冲突不是唯一原因。 |
| `speed_envelope_01` **直线**改为 3 m/s，固定失配植物，静水无噪声 | 2.5 s | **完成** | 位置 RMSE 0.232 m，速度 RMSE 0.109 m/s，回退 0；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_speed_envelope_01_v3_h25_osqp_fixed_still_quiet_yaw30.json)。 |
| 同一 3 m/s 直线，保留代码原有过程噪声 | 2.5 s | **失败**，俯仰超过 70° | 最大俯仰 70.056°，回退 0；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_speed_envelope_01_v3_h25_osqp_fixed_still_noise_yaw30.json)、[轨迹](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_speed_envelope_01_v3_h25_osqp_fixed_still_noise_yaw30.csv)。 |
| 同一 3 m/s 直线和过程噪声 | 12.5 s | **完成**，OSQP 回退 0 | 位置 RMSE 0.371 m；最大俯仰 9.856°，速度 RMSE 0.182 m/s；总计算墙钟 112.8 s / 1266 周期，平均约 89 ms/周期；[摘要](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_speed_envelope_01_v3_h125_osqp_fixed_still_noise_yaw30.json)、[轨迹](../outputs/reports/uuv_mechanism_audit_20261102/runs/control_speed_envelope_01_v3_h125_osqp_fixed_still_noise_yaw30.csv)。未记录 95/99 分位耗时，不能判定满足 100 ms 最坏周期。 |

6 m/s 短窗回合在约 18.5 s 首次超过 1 m 横向偏差，21.7 s 超过 5 m；20 s 左右航向约 −6.3°时方向舵命令已反转至 +45°，之后持续偏离。5 s 回合在相近时刻保持向弯道方向转向，横向偏差全程小于 0.6 m。因两组均无求解回退，差异确实存在于此代码的有限时域决策；但延长窗口同时改变了 QP 大小和优化累积代价，尚不能把收益专门归给“窗外路径几何”。

## 二、动力学机制：代码内真实，实艇外推未证实

[物理与路径审查输出](../outputs/reports/uuv_mechanism_audit_20261102/summary/physics_path_audit.json)直接积分 [`simulator/model.py`](../../uuv_simulator/simulator/model.py) 中的 10 状态模型。在固定失配植物下，从 3 m/s 直航配平出发持续施加 +1° 方向舵，东向位移在 2.5 s 为 −0.00449 m，10 s 为 +0.02086 m；6 m/s 的 2.5 s 为 −0.00980 m，5 s 已变为 +0.02086 m。**短时反向、长时正向的响应在代码中确实存在**。把有效质量矩阵临时取对称平均做纯敏感性试算后，3 m/s 的 2.5/10 s 位移仍分别约 −0.00414/+0.02698 m，说明该符号反转并非仅由质量矩阵非对称造成；对称平均不是经过辨识的替代模型。

但模型的外部有效性有明显缺口：

- `MASS=214,225 kg`；`M_EFF`、阻尼、舵力和 1.5 MN 推力上限在原仓库中未附可追溯的具体艇型、试验辨识或不确定度。`MISMATCHED_PLANT` 在代码里明确写为“illustrative fixed mismatch”。不能把这套数值当成某艘真实 UUV 的已核验参数。18 m/s 直航配平的 `推力×速度` 代理量约 25.6 MW，但代码没有推进效率与电机功率模型，不能将此数值直接解释成实艇耗电功率。
- 有效质量矩阵最大交叉项不对称量为 **347,342**（矩阵相对 Frobenius 非对称度 1.39%）。标准 [Fossen 船舶动力学模型](https://www.fossen.biz/html/marineCraftModel.html)要求物理惯性矩阵对称正定；本实现没有给出非对称项的辨识或变换推导。仅凭此不能断言所有动力学结论错误，但必须先做方程和参数审计。
- 现模型没有静水恢复力/力矩项 `g(η)`、横摇自由度、推进器转速/功率曲线或艇体尺寸，舵效直接按 `u²` 缩放；过程扰动标准差不随速度变。这些简化可能放大低速由噪声触发的俯仰问题，也无法验证真实低速航行与深度控制。Fossen 标准模型包含恢复项；是否可以忽略应由目标艇的浮心、重心与试验数据证明。
- 原项目的 0.05 s 延迟、推力 0.10 s / 舵面 0.08 s 时间常数也只是示例值。6 m/s 理想化消融仍失败，故不能将该案例主要归因于这些延迟常数。

**当前能确认的是“此仿真器的短窗控制问题”，不是“真实 UUV 上会以相同速度、时间尺度和舵角出现的问题”。** 某些尾舵艇在转向初期有反向横移并不意外；本模型预测的反向持续时间及失控阈值仍需水池、实艇或独立公开辨识模型验证。

## 三、路径与速度任务是否有实际意义

路径由原项目 [`DDAT_trajectories/generate.py`](../../uuv_simulator/DDAT_trajectories/generate.py) 按五次平滑曲率、深度和速度剖面**合成**，未对应海图、任务航路、障碍、可用航道或目标艇型。原验证集初始目标速度约 15.3–15.6 m/s，峰值至 17.4–19.2 m/s；3/6 m/s 扫描通过代码改写目标速度、初始航速和时限，几何未重新设计。它们有**压力测试**意义，尚无真实任务代表性证明。

关键几何和需求来自重新读取 CSV 与运行原项目 [`screen_dynamic_feasibility.py`](../../uuv_simulator/high_speed_experiments/screen_dynamic_feasibility.py)：

| 场景 | 几何和所需机动 | 在本模型内的判断 |
| --- | --- | --- |
| `constant_turn_01`，6 m/s | 最小半径约 48 m；所需向心加速度约 0.75 m/s²、偏航率约 7.16°/s。稳态转弯配平约需 0.199 MN 推力、9.86° 方向舵、约 −2.10 m/s 横荡速度（约 20.5° 侧滑）。 | 低于 1.5 MN/45° 命令上限，**稳态段未被筛掉**；所以不能把短窗失败简单说成推力/舵角不够。如此大的配平侧滑也提示必须用目标艇数据审查路径。闭环严格可行性仍未由稳态配平证明。 |
| 同一弯道，18 m/s | 向心加速度约 6.75 m/s²（0.69 g）、偏航率约 21.5°/s；稳态配平需约 **1.788 MN** 推力，超过上限约 19%。 | 在约 111–227 m 的持续弯段被[稳态筛查](../outputs/reports/uuv_mechanism_audit_20261102/summary/steady_screen_val.json)判为超限。若要求严格 18 m/s 跟随，当前参数与路径不自洽；降速完成仍可能被环境记为“安全完成”。 |
| `s_reversal_01`，18 m/s | 最小半径约 45 m，峰值向心加速度约 7.20 m/s²（0.73 g）；两段稳态所需推力约 1.84/1.77 MN。 | 原验证路径和改为固定 18 m/s 后均有持续超限段。 |
| `coupled_axes_01`，18 m/s | 最小半径约 52 m、最大路径坡角约 23.5°；若在该坡角处保持 18 m/s，垂向速度约 7.17 m/s。 | 固定 18 m/s 版本的一处约 27 m 稳态段需约 1.63 MN 推力；更多深度变化段因非稳态被筛查器排除，**未通过筛查不等于可行**。 |
| `speed_envelope_01`，原目标峰值 19.2 m/s | 在本模型中，固定失配植物的直航最大稳态速度仅约 **18.47 m/s**；19.2 m/s 配平需约 1.621 MN。 | 原验证路径两段长直线目标超出推力上限。 |

在原验证集 5 条路径中，稳态筛查直接标出 `constant_turn_01`、`s_reversal_01`、`speed_envelope_01` 的持续超限段；另两条存在大量过渡段，筛查器无法证明它们可行。结果见[完整 JSON](../outputs/reports/uuv_mechanism_audit_20261102/summary/steady_screen_val.json)。6 m/s 低速弯道的几何可作为**该模型内的控制判别例**；18 m/s 原路径更适合被标为“超限压力场景”，不能用宽松完成率证明严格速度跟踪。

现实参照也显示目标平台必须先确定：HII [REMUS 系列官方规格](https://www.hii.com/products/remus-uuvs)列出 REMUS 620 速度上限 8 节（约 4.1 m/s）、REMUS 6000 上限 4.2 节（约 2.2 m/s）；对 REMUS 130/300 只写“大于 4 节”，未给上限。当前路径约 280–302 m 深、6–19.2 m/s，且模型质量 214 吨，**不能直接代表这些 REMUS 航行器**。这不排除某些特殊高速无人潜航器，但项目必须提供与 214 吨、1.5 MN、约 18 m/s 对应的真实艇型和动力数据；否则应明确称为抽象高速潜航器仿真。

环境的“安全完成”没有目标速度容差或沿程走廊要求；例如原 18 m/s 扫描中，`constant_turn_01` 被记为成功而回合平均艇体速度仅 16.93 m/s，`speed_envelope_01` 被记为成功而平均速度 17.24 m/s，见[原始逐回合数据](../../uuv_simulator/outputs/DDAT/mpc_speed_sweep/default_val_5paths_6speeds_1seed/episodes.csv)。因此若论文问题是严格 18 m/s 跟踪，这些成功回合不构成正例。

## 四、对研究路线的裁决

1. **6 m/s：仿真故障成立，主要证据指向有限时域。** 零扰、无求解回退、无执行器失配时短窗仍失败；确定性 5 s MPC 已修复同一路径与种子。应先在更多路径/种子下比较同算力长窗、非均匀网格和确定性终端项。现有结果不给 RL 终端价值独立必要性。
2. **3 m/s：噪声触发的短窗俯仰故障成立，但实艇意义未定。** 直线、静水、固定参数下只改变过程噪声即由完成变失败；保持噪声并把预测窗增至 12.5 s 则完成。模型缺少静水恢复力和经辨识低速舵效，必须先用可信模型重验；长窗平均约 89 ms/周期，尚无实时尾部耗时保证。
3. **18 m/s：部分任务在模型内已与严格目标速度冲突。** 把超限路径作为鲁棒性压力测试可以，但须预先允许降速，并同时报告航速偏差。若任务坚持严格 18 m/s，应重新设计满足推力、舵角、俯仰及推进功率边界的路径。
4. **实艇结论目前不可下。** 下一项真正有判别力的工作是选定目标艇或公开辨识模型，核对质量矩阵、阻尼、浮力恢复、舵效/舵机、推力功率曲线和水动力试验；按该艇的速度、深度、转弯半径重新生成任务，再复跑短/长窗与噪声对照。

复现实验脚本：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 high_speed_experiments/audit_uuv_mechanisms.py physics`，或将 `physics` 改为 `control --speed 6 --horizon 25/50 --backend osqp`。脚本参数 `--ideal-plant` 消除模型失配和执行器迟滞；`--path speed_envelope_01 --speed 3 --noise` 复跑直线噪声例。稳态筛查调用原项目脚本并将输出定向到本工作区。所有新运行结果仅证明所用代码、参数和随机种子的行为；它们不是水池或实艇试验。
