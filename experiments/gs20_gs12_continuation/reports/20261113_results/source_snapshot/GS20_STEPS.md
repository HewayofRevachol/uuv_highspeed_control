# 详细实验步骤

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-11-12
- Verification Status: UNVERIFIED
- Version Label: gs12_continuation_v2_code
- Repro Lock: null
- Repro Lock Reason: 代码与执行顺序已提供，仅做最小检查；本协议的仿真/训练尚未执行，未形成效果结论。

## 0. 固定框架并检查入口

工作目录为 `uuv_workspace`，下文所有路径均相对该目录。本目录简称 `experiments/gs20_gs12_continuation`，结果统一写入新的 `outputs/gs20_continuation/` 子目录。

固定水动力、原执行器研究植物、控制周期0.1秒、RK4子步0.02秒、N35、原跟踪/控制/控制变化代价、命令盒约束、一次RTI及原求解精度。β仅改变预测舵效 `1−0.15β` 和舵时间常数 `1+β`，不改真实对象或推进参数。

保留两个结构：`first`为原22状态一阶预测器；`hybrid`为二阶推进＋一阶双舵。`first:0`、`first:1`、`hybrid:1`分别连接原22、保守22及原混合候选；混合结构中的推力速率使用原两样本因果观测器，不读取真实执行器速率。

`legacy/repo/aav_mpc/condensed_mpc.py`从原提交 `926b7ac`提取，私有加载，确保当前工作区新增的权重接口不改变复现对象。其他结构辅助文件取自GS-12原交付ZIP；只有混合模块的同目录导入改为相对导入。来源与哈希见 `legacy_sources.json`。生产文件不被覆盖。

先做最小检查：

```bash
python3 -m compileall -q experiments/gs20_gs12_continuation
python3 -m experiments.gs20_gs12_continuation.check
```

检查仅涉及导入、任务ID、位置/速度解析一致性、合成RMSE、β维数与源码哈希，不构造植物、不求解QP、不更新策略。这里也不编写大规模测试框架、自动重试或后端回退。

**与上轮文字协议的一个修正：原七频信号从t=0直接按原公式生成；16秒是丢弃的过渡观察段，不是渐入包络。** 新增单频/切换路径才使用16秒平滑渐入，速度方向同时使用对应的解析积分。

## 1. 写入固定任务清单

```bash
python3 -m experiments.gs20_gs12_continuation.prepare --out experiments/gs20_gs12_continuation/inputs_v1
```

此命令只生成输入，不运行实验。目录须为新目录，避免覆盖旧清单。文件内容如下：

| 清单 | 条数 | 内容 |
|---|---:|---|
| `bridge.jsonl` | 8 | 匹配/组合植物，各含零激励、东向、深度、北向速度七频参考，15m/s、144秒 |
| `maneuvers.jsonl` | 36 | 12/15/18m/s × 匹配/组合/组合tau×1.2 × 原直线、水平、深度、耦合24秒参考 |
| `static_dev.jsonl` | 70 | 15m/s；三单轴＋耦合同相/π/2；七频；两开发相位 |
| `static_test.jsonl` | 70 | 同一单频族的新相位，仅作冻结后静态泛化评价 |
| `range_dev.jsonl` | 60 | 12/18m/s、两幅值、五方向配置、k=2/8/24 |
| `half_dev.jsonl` | 35 | 15m/s、半幅0.003、五方向配置、七频 |
| `mismatch_dev.jsonl` | 30 | 舵效下降/单独二阶/组合tau变化，五方向配置、k=8/24 |
| `switch_dev.jsonl` | 12 | 东向/深度/耦合，两个阶段顺序，两开发相位 |
| `switch_train.jsonl` | 48 | 三方向、两个顺序、四训练相位、两幅值 |
| `switch_test.jsonl` | 72 | 三方向、两个顺序、两留出相位、两幅值、三航速 |

所有新任务已在生成器中固定，运行前一起生成，不能看到结果后只留下成功路径。`manifest.json`记录配置、源码/模型哈希和实际依赖版本。旧数据属于复现/开发，不能重新标为留出。

原七频为k=`[2,4,6,8,12,16,24]`，`ω=2πk/64`，周期32至2.67秒。新增路径幅值以归一化0.006为主、0.003为半幅：东向4.8/2.4厘米、深度3/1.5厘米、北向速度0.012/0.006m/s。耦合路径两个分量均使用各自原尺度；相位差单列。

新增切换路径采用k=8/24、相同位置幅值、144秒总时长；在48、96秒附近各用4秒平滑混合，速度参考包含混合权重的导数。顺序为低—高—低或高—低—高。没有采用近期东向实验的大幅值和频率平方幅值匹配。

## 2. 阶段A：连接原频响与原机动

以下命令会实际运行，**本次交付未执行**。首次可用 `--limit 1` 只运行一个任务的三种方法，测量成本；其余任务用 `--start 1`、新的输出目录分段运行，分析时用 `--runs`合并互不重复的分段。不要把短时调试轨迹与正式144秒任务混用。

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_v1/bridge.jsonl --methods first:0 first:1 hybrid:1 --initial endpoint --out outputs/gs20_continuation/A_bridge
python3 -m experiments.gs20_gs12_continuation.frequency --runs outputs/gs20_continuation/A_bridge --out outputs/gs20_continuation/A_frf
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/A_bridge --out outputs/gs20_continuation/A_rmse
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_v1/maneuvers.jsonl --methods first:0 first:1 hybrid:1 --initial endpoint --out outputs/gs20_continuation/A_maneuvers
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/A_maneuvers --out outputs/gs20_continuation/A_maneuver_rmse
```

`--initial endpoint`让固定方法从自己的原参数开始，复现原实验；阶段B以后统一从β=1开始，每周期最多移动0.02。两种初始化协议应分开报告。

频响输出包括两个窗口的复矩阵、七频全部奇异值、最大方向幅相、第一/第二奇异值相对间隔、周期差异和约束状态。零激励按同方法、同植物扣除。原22状态的匹配/组合矩阵与旧ZIP中的矩阵直接比较，逐频相对Frobenius差小于10⁻³记为连接通过；保守/混合的谱再与原报告七频表逐项比较。

15m/s组合植物的已公布最高频σmax分别约0.121988、0.104493、0.104463，混合方案的纯速度列范数约0.007050。应同时看前六点代价，不能只看最高点。

连接不通过时，定位参考、初态、QP、依赖版本或观测器差异，记录差异后再继续，不通过修改幅值/权重追求对齐。未运行之前不宣称原数值已复现。

## 3. 阶段B：固定结构内做β—频率—方向对照

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_v1/static_dev.jsonl --methods first:0 first:0.25 first:0.5 first:0.75 first:1 hybrid:0 hybrid:0.25 hybrid:0.5 hybrid:0.75 hybrid:1 --out outputs/gs20_continuation/B_static
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/B_static --out outputs/gs20_continuation/B_analysis --freeze outputs/gs20_continuation/static_selection.json
```

每个任务使用相同真实对象、参考、初态、QP代价和限速。分析 `rmse.csv`中东向/深度/速度的逐频变化；耦合同相/π/2分开查看。用R²选择固定基线时按任务等权，不把不同物理单位的RMSE直接相加。只考虑所有任务完整、无QP失败或饱和的固定候选；如果没有合格候选，则没有可冻结的基线，应如实报告并检查任务范围。

`static_selection.json`对每个结构冻结一个总体固定β，以及各方向在k=8/24时的规则β。规则参数只从本阶段开发任务选择，不预设深度与东向具有相同趋势，也不硬编码旧的0.75/0.25。

从已保存的结果绘制β—频率—深度RMSE图：

```bash
python3 -m experiments.gs20_gs12_continuation.plot --table outputs/gs20_continuation/B_analysis/rmse.csv --axis depth --metric D_m --out outputs/gs20_continuation/B_analysis/depth_rmse.png
```

东向、速度分别使用 `--axis east --metric E_m`、`--axis speed --metric VN_mps`；耦合使用 `--axis coupled`并用 `--depth-phase-offset 0`或`1.5707963267948966`分开绘制。图中只对同方向、同幅值、同航速、同相位差的合格开发相位取均值。

## 4. 阶段C：幅值、航速及失配范围

分别将阶段B的运行命令中的任务清单改为 `half_dev.jsonl`、`range_dev.jsonl`、`mismatch_dev.jsonl`，输出分别使用 `C_half`、`C_range`、`C_mismatch`；每组仍运行同一个十方法网格。分别调用 `analyze`写入对应的新分析目录，**不覆盖阶段B冻结文件**。

逐项回答：半幅下方向与取舍是否一致；航速变化是否改变最优参数；舵效/阶次/tau变化是否改变趋势。记录18m/s的推力饱和，不把它归因于β不足。这里都是开发性适用范围证据，不能用于调参后又声称是留出验证。

## 5. 阶段D：验证真正的时间调度增量

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_v1/switch_dev.jsonl --methods first:0 first:0.25 first:0.5 first:0.75 first:1 first:rule hybrid:0 hybrid:0.25 hybrid:0.5 hybrid:0.75 hybrid:1 hybrid:rule --selection outputs/gs20_continuation/static_selection.json --out outputs/gs20_continuation/D_switch
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/D_switch --out outputs/gs20_continuation/D_analysis --freeze outputs/gs20_continuation/switch_selection.json --rules-from outputs/gs20_continuation/static_selection.json
```

规则使用已知参考阶段，提前一个8秒决策间隔切换目标，实际β仍遵守每0.1秒变化≤0.02。它是有预览的机制参照，不是RL学会在线辨识的证明。RL得到同样8秒长度的参考预览，但不读取阶段标签或真实失配参数。

**重新从切换开发路径的完整固定网格选出本阶段最佳固定β**，写入 `switch_selection.json`；规则沿用阶段B，不能据切换结果反复改规则。`D_analysis/summary.json`的 `paired`会比较规则与这个更强的同结构固定基线。

主要分数用全程R²与物理RMSE；分段输出包括16–46、50–94、98–144秒，过渡段46–50/94–98秒单列，启动仍计入全程。若分段偏好不同但动态全程没有改善，记录为调度尚无优势，不能因此选择性删除过渡误差。

若存在完整有效、跨开发路径一致的正收益，再进入RL试跑；若没有，先总结在哪些方向/结构下取舍不可利用。禁止以某个任务的最佳固定参数拼成“动态最优上界”。

## 6. 阶段E：训练RL，先试跑再固定预算

```bash
python3 -m experiments.gs20_gs12_continuation.train --tasks experiments/gs20_gs12_continuation/inputs_v1/switch_train.jsonl --structure first --seed 41001 --episodes 64 --batch 8 --out outputs/gs20_continuation/E_first_pilot
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_v1/switch_dev.jsonl --methods first:best first:rule first:rl --selection outputs/gs20_continuation/switch_selection.json --policy outputs/gs20_continuation/E_first_pilot/final.pt --out outputs/gs20_continuation/E_first_pilot_dev
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/E_first_pilot_dev --out outputs/gs20_continuation/E_first_pilot_analysis
```

19维观测依次为：8个归一化当前输出误差、当前β、上次目标β、3个归一化实际执行器位置，以及未来4秒/8秒的三个参考变化量。没有真实执行器速率、真实gain/tau、失配标签、任务ID或阶段标签。执行器位置与当前状态属于原框架可用观测。

策略为19→64→64的tanh网络，高斯原始动作经sigmoid映射到0–1。每8秒给出目标β，MPC仍每0.1秒求解。训练采用CPU原QP；不改用GPU近似后端。

逐步奖励是三输出归一化误差平方的负和，按请求总子步数归一化并乘10⁶；gamma=1，因此完整无失败轨迹的总跟踪奖励严格对应 `−10⁶ R²`。诊断越界或QP失败另罚100；没有额外权重调节奖励或控制变化惩罚。PPO采用lr=3×10⁻⁴、clip=0.2、GAE λ=0.95、4轮更新、熵系数0.01。梯度裁剪1沿用常见PPO更新，未添加自适应异常恢复。

64episode只验证训练过程，不作为最终效果结论。试跑后在打开留出任务前记录正式预算；建议固定256episode、batch8，对41001/41002/41003三个种子分别运行。将 `--structure`改为 `hybrid`单独训练、单独评价，不跨结构加载模型。正式策略采用固定预算的最终检查点，不从测试结果挑最好episode/种子。

训练输出协议、逐episode完成/R²/QP情况、更新历史和检查点；完整状态/命令/β轨迹由后续 `run`评价统一保存。若试跑需修改奖励/观测，修改后重新命名协议与输出，留出集保持未打开。

## 7. 冻结后一次性评价留出路径

以first结构、seed41001为例，正式训练写入 `E_first_s41001`后执行：

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_v1/switch_test.jsonl --methods first:best first:rule first:rl --selection outputs/gs20_continuation/switch_selection.json --policy outputs/gs20_continuation/E_first_s41001/final.pt --out outputs/gs20_continuation/F_first_s41001
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/F_first_s41001 --out outputs/gs20_continuation/F_first_s41001_analysis
```

对其他种子与hybrid结构分别重复，使用各自目录并报告全部种子；不要把多个种子的同名任务/方法记录混成同一个任务样本。`static_test.jsonl`可另作新相位单频泛化，但规则只适用于切换任务，静态评价用 `first:best first:rl`（或hybrid相应方法）。

冻结文件不在test上重排固定β。结果须同时报告所有路径物理RMSE、R/R²、三维位置RMSE、失败/越界/饱和、逐任务配对变化及各种子均值/离散程度。少量确定性路径不支持失效概率或统计显著性结论。

## 8. 结果文件与解释

每个运行目录有 `protocol.json`、`episodes.jsonl`、`traces/*.npz`；轨迹保留每0.02秒的状态/参考/输出/实际执行器，以及每0.1秒的命令/β/QP信息。`analyze`输出物理RMSE表 `rmse.csv`与 `summary.json`。逐窗口分数保留在每条episode的 `scores`字段，不丢弃启动或切换代价。

`R = sqrt(mean((eE/8)²+(eD/5)²+(eVN/2)²))` 无量纲，`R²`用于等权任务选参，与同窗口旧三量代价一致。物理RMSE不能用σmax直接替代，σmax也不是RMSE改善率。N位置误差、姿态及惯性速度三个分量均单列，不强制额外的十状态零侧滑/攻角目标。

提前越界/失败记录没有完整R²；表中的共同前缀误差只作诊断，不参与完整任务排名。总体完整轨迹均值注明 `complete_only`，同时必须展示分母、失败数和资格情况；配对摘要给出有效配对覆盖率，覆盖不足不能称为整体获胜。

本次交付仅准备上述代码与步骤。后续执行应按阶段在前台收集结果，单次输出目录不自动覆盖、失败不自动重试、不后台驻留。没有修改其他实验或启动其训练。
