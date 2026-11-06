# GS-13 SR-PPO 实验步骤

本目录实现的是 GS-12 混合 MPC 的安全残差 PPO 实验。策略只输出四个门控通道
`[theta, u, vertical, lateral]` 的小残差；MPC、QP、执行器物理边界和植物积分仍由
`experiments/gs12_gate` 负责。代码中的 `run.py smoke` 和 `train.py` 都实际调用
GS-12 `PlantHarness`，不是合成环境。实验结果要区分训练闭环结果与 smoke 结果。

## 1. 固定接口和安全逻辑

- 每个控制周期为 0.1 s，植物积分为 0.02 s，每五个周期更新一次门控；每周期只执行一次 MPC 求解。
- 观测是 GS-12 的 200 维因果历史特征，包含最近十帧、实测执行器、已发命令、估计器状态、QP 状态和余量。植物真实失配参数不会进入策略输入。
- actor/critic 是两层 128 单元 Tanh MLP。actor 在未裁剪的 Gaussian `raw_z` 上记录 `log_prob`，再以教师门控 `[1,1,1,0]` 为中心生成 `0.25*tanh(raw_z)` 残差。
- `safety.project_gate` 执行 `[0,1]` 限制、每次 0.25 slew、边界余量、QP 状态、估计器回退和 residual clip 检查。QP 失败、非有限观测或非有限动作使用零门控紧急回退，并写入 `override_reason`；不会增加 MPC 次数。
- `costs.py` 保留 tracking、boundary barrier、actuator/margin、solver/deadline、fallback 和 gate slew 的逐项正代价。physical failure 只加入一次 `20+100*remaining_s` 吸收代价，time-limit truncation 只 bootstrap、不计失败代价。
- PPO 对四列约束代价分别计算正成本 GAE，优化 `A_tracking - lambda @ A_safety`。`LagrangeState`、checkpoint 和更新日志保存四个 multiplier、KL、entropy、clip fraction、value loss、explained variance、梯度范数及 shield override。

## 2. 场景和数据隔离

`sampler.build_cases` 使用固定速度 12/15/18 m/s、十类植物失配和五类 maneuver 的平衡周期。train 使用真实 trim 能力检查并显式拒绝 capacity-limited case；dev/test 保留这些 case，并在报告中单独分层。三个 split 使用不重叠 seed namespace、初态和参考相位；同一个 `cases.jsonl` 被所有方法复用。teacher-grid 只读取 train case，禁止在 dev/test 生成标签。

默认配置 `config.json` 是正式协议（train/dev/test=300/24/120、24 s），首次接入建议用小 run 验证流程。当前归一化使用冻结的 GS-12 engineering scale，manifest 会写明来源；没有把 dev/test 统计偷偷用于拟合。

## 3. 可执行步骤

在仓库根目录运行，`python3` 可替换为已安装依赖的解释器：

```bash
# T0：固定 manifest 和场景；每次改代码/配置都使用新 run 目录
python3 -m experiments.gs13_srppo.run prepare \
  --out outputs/gs13_srppo/r0 \
  --config experiments/gs13_srppo/config.json
python3 -m experiments.gs13_srppo.run smoke --run outputs/gs13_srppo/r0
python3 -m pytest -q tests/unit/test_gs13_srppo.py tests/unit/test_gs13_ppo.py \
  tests/unit/test_gs13_sampler.py tests/test_gs13_report.py

# T1：在 train split 上运行六个可达门控候选，生成 teacher/dataset.npz
python3 -m experiments.gs13_srppo.run teacher-grid \
  --run outputs/gs13_srppo/r0 --limit 8

# T2：BC warm-start 后进行真实闭环 SR-PPO；默认 300/24/120 场景规模
python3 -m experiments.gs13_srppo.train --run outputs/gs13_srppo/r0 \
  --seed 41002 --decisions 8192 --update-every 512
python3 -m experiments.gs13_srppo.run evaluate \
  --run outputs/gs13_srppo/r0 \
  --checkpoint outputs/gs13_srppo/r0/training/41002/checkpoint_008192.pt \
  --split dev --methods hybrid_zero,fixed_best,rule_best,srppo
python3 -m experiments.gs13_srppo.run select --run outputs/gs13_srppo/r0

# T3：只有 select 完成后才能在完整 test 运行一次
python3 -m experiments.gs13_srppo.run evaluate \
  --run outputs/gs13_srppo/r0 \
  --checkpoint outputs/gs13_srppo/r0/training/41002/checkpoint_008192.pt \
  --split test --methods hybrid_zero,fixed_best,rule_best,srppo

# T4：固定整数频点的经验频域诊断，不拼接成统一 LTI 结论
python3 -m experiments.gs13_srppo.frequency --run outputs/gs13_srppo/r0 \
  --checkpoint outputs/gs13_srppo/r0/training/41002/checkpoint_008192.pt \
  --mode pilot --methods fixed_best,rule_best,srppo
```

若只做接口接入，可将 `prepare` 的 case 数量和时长分别设为
`--train-cases 1 --dev-cases 1 --test-cases 1 --duration 1`；这只验证链路，不产生算法结论。
训练决策数必须是 `update-every` 的整数倍，checkpoint 为中点和终点，并绑定 manifest、teacher
dataset 和 policy architecture hash。完整确认使用五个独立训练 seed（42001–42005），每个 seed
先按同一 dev 规则选择，再把冻结 checkpoint 评估到 test。

## 4. 评估、晋级和停止

`report.write_report` 要求每个 method 有完全相同的 scene 集；重复、缺失、非有限值、split 或
capacity 元数据不一致都会直接失败。报告同时给出 all、feasible 和 capacity-limited 三层，保留
失败和截断计数。截断 pair 不参与 checkpoint 选择；common-success gain 只是条件诊断，失败仍在
全 pair 中计入。

T2 只有同时满足以下条件才能进入 T3：可行场景 physical failure 不增加；相对 `fixed_best` 和
`rule_best` 中更优者，完整配对的 tracking/cost 至少改善 5%；QP failure、deadline、饱和和非激励
频域能量没有明显回归。任一 NaN、QP 失败激增、余量回退失控或 p99 延迟异常都立即回滚到
`fixed_best`，保留完整日志。未达到门槛时只允许新 run 只修改一个预先声明的因素（奖励权重、残差
上限或场景配额），不能覆盖原 run，也不能依据 test 选 seed 或 checkpoint。

原始 `control.npz`、`decisions.npz`、`substeps.npz`、checkpoint 和频域数组通常被忽略规则排除；
需保存时应把对应 run 目录连同 `run_manifest.json`、`cases.jsonl`、`obs_schema.json`、
`training_summary.json`、`paired_metrics.csv`、`report.md` 和 SHA256 清单一并归档。报告必须明确
标注这是仿真闭环结果，不等同于实艇安全保证。
