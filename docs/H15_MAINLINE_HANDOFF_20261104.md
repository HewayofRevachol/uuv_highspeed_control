# H15 主线实验接手说明

最新停点已移至 [Gate2 PPO 暂停接手](H15_GATE2_PPO_HANDOFF_20261104.md)：
四回合训练与初始 checkpoint 六回合 dev 已完成；用户要求在本轮后停止，
训练后 checkpoint 的 dev 评估及 Gate2 总项仍待续接。以下早期记录保留作历史。

更新时间：2026-11-04。目标仍是按
`H15_RECOMMENDED_MAINLINE_EXPERIMENT_PLAN_QUICKSTART_20261104.md` 的 Gate 顺序，
检验历史响应估计、确定性速度治理及单一速度意图 RL 是否改善 H15 控制表现。
文献分析见 `H15_RL_MPC_LITERATURE_DIRECTIONS_20261104.md`。

**当前停点：**Gate1 三类新版 smoke 已完整运行并审计，Gate1 通过范围仅为
链路可运行。Gate2 的 E1 与确定性治理器已在同一 `coupled` 回合完成闭环配对；
两方法仍主要依赖短窗备份，严格主线失败，未见实质性能收益。阶段证据见
`outputs/reports/h15_gate2_fullplan_estimator_20261104.md`。完整近期 MPC 计划历史
已接入；下一步是实现五动作掩码 PPO 与三种子训练。

## 当前用户优先级与阶段状态

用户明确表示 **5 ms 筛选预算不重要，关心算法本身能否改变性能**。
因此继续做离线、同回合配对的算法机制实验；记录真实耗时，但不以 5 ms
阻断离线评估，也不声称当前方法在线可用。性能至少看完成、横向/深度/实际
切向对水速度误差、路径进度、物理违约和备份/空候选，而非只看 reward。

Gate0、Gate1 已通过各自的阶段条件；Gate1 的通过不代表 `strict_mainline`
成功或 B2 有性能增益。`B0_shared` 三类新版 smoke 和 `coupled` B2 完整到终点，
但后者主要用备份。
Gate2 的已核验子项已逐项勾选；Gate2 总项及 Gate3–Gate5 保持未勾选。全局实时预算停止规则已勾选，含义是
现有超时结果只能按离线机制报告；它不阻断本轮离线配对。

三回合是 `run_smoke.py` 中的 `simple`（train 广弯，31001）、`coupled`
（dev 转弯/深度耦合，32001）和 `mismatch`（dev 失配、海流、噪声，32002）。
test manifest 与测试路径只读，尚未用于方法选择或评估；不要用 test 调参。

## 本轮已得到的算法结果

`experiments/h15_mainline/run_offline_governor_smoke.py` 接入了候选专属
N35 QP、五场景非线性重放、固定 15 m/s 选择器或 B2 短窗代价选择器、
直接执行所选 MPC 命令的真实环境闭环积分、逐周期候选/执行器/队列/QP/计时
和 0.02 s 子步日志。
Gate1 暂用固定 E0 `.85/2.0` 回退及已冻结的 dev 残差半径；完整 E1 与 RL
仍属后续 Gate。候选按 `physical_accepted` 选取，5 ms 的
`online_accepted` 只用于记录工程状态。候选为空、固定 15 被拒绝或 QP
失败时显式中止并记为失败，不能隐式续跑。

简单路径完整配对已完成，原始结果在：

- `outputs/experiments/gov_uuv_h15_smoke_s31001/full_direct_b0_simple_20261104/`
- `outputs/experiments/gov_uuv_h15_smoke_s31001/full_direct_b2_simple_20261104/`

两者均用 199 周期完成，未发生子步物理违约、空候选或备份。
`B2_gov` 全程选 15 m/s，逐周期状态与 `B0_shared` 完全相同；
因此简单路径没有观察到治理器增益。两条运行各有 995 条子步样本；
发出命令与下一周期队列首项一致，源码哈希已核验。具体数值见
`outputs/reports/h15_offline_governor_simple_pair_direct_20261104.md`。
这不能外推到耦合/失配回合。`strict_mainline=False` 是冻结合同对超时的判定，
不影响离线物理完成和算法指标的单独报告。

后续配对补充：`mismatch` B0/B2 均 199 周期完成且逐状态相同；`coupled`
B0 在 14 周期因固定意图被拒绝中止，B2 在 16 周期因候选全空中止，
无已执行子步物理违约。六条运行审计通过。冻结 35 步俯仰半径过大，加
终端余量后只留 4.4263°，是 `coupled` 筛选提前失去候选的明确机制。
Gate1 总项仍未勾选，不能进入正式 Gate2 或宣称治理器成功。

后续机制诊断：已修正候选筛选对周期末日志副作用的依赖，并在 condensed
QP 增加可开关的命令变化率约束。原点模型 35 步俯仰残差与五场景包络叠加
存在重复收紧；使用 dev 包络外残差及 QP 变化率的**诊断** B0/B2 均延至
51.0 m 后因未来终端俯仰/速度预测失去候选，B2 全程仍选 15 m/s。
该诊断不替代旧冻结方法的正式结果，不得据此勾选 Gate1。

## Gate1 配对完成后的下一动作

该节以下四条命令与旧版中止记录只作历史复现。最新版 Gate1 通过证据在
`outputs/reports/h15_gate1_backup_smoke_20261104.md`；Gate2 已有 E1 配对证据在
`outputs/reports/h15_gate2_e1_coupled_pair_20261104.md`。当前应按本文开头所述
补齐 RL 训练协议，不覆盖已有运行目录。完整计划历史接线见本文开头的阶段记录。

原先四条命令均已执行，输出目录已存在，**不要覆盖或原样重跑**。
先在 train/dev 上处理 `coupled` 的状态受限计划或共同备份，形成版本化配置；
核验后用新输出目录重跑三类配对。当前已冻结的残差半径与终端余量不能仅为
通过 Gate1 而无证据放宽。下面保留原始命令作复现记录。

从仓库根目录运行，固定单线程与 `/bin/python3`。下面四条命令各写入独立的
`outputs/experiments/<algorithm>_<model>_<scenario>_s<seed>/<run>/`；已有输出
不可覆盖。每条完整运行可能需要数分钟，应等待进程结束并检查 `summary.json`、
`cycle_log.jsonl`、`substep_samples.json`、`run.json` 和 `console.log`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_offline_governor_smoke \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --case coupled --method B0_shared \
  --out outputs/experiments/gov_uuv_h15_smoke_s32001/full_b0_coupled_20261104

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_offline_governor_smoke \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --case coupled --method B2_gov \
  --out outputs/experiments/gov_uuv_h15_smoke_s32001/full_b2_coupled_20261104

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_offline_governor_smoke \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --case mismatch --method B0_shared \
  --out outputs/experiments/gov_uuv_h15_smoke_s32002/full_b0_mismatch_20261104

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /bin/python3 -m experiments.h15_mainline.run_offline_governor_smoke \
  --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
  --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104/checkpoint.json \
  --config experiments/h15_mainline/candidate_filter_config.json \
  --case mismatch --method B2_gov \
  --out outputs/experiments/gov_uuv_h15_smoke_s32002/full_b2_mismatch_20261104
```

成对比较完成后，核验每个回合的候选选择、实际参考斜坡首步 ≤0.05 m/s、
命令队列与实际执行器时序、状态与子步样本有限、首次违约、终点与固定最大
时域。若 B2 仍全程选 15 m/s，应如实报告“当前治理器未改变意图/性能”，
再审查代价权重及可行集，而非以 reward 宣称增益。若回合中止或出现违约，
先查明原因；Gate1 总项保持未勾选。Gate1 证据充分后才进入 Gate2 的完整
E1、治理器及 RL 接线。

## 复现与注意事项

- 先读 `experiments/h15_mainline/README.md`、quickstart checklist 和本说明。
- `candidate_filter_config.json` 的推力命令变化上限已从错误的 0.4 改为
  train 轨迹支持的 1.0 倍量程；旧版初态预算输出仅作历史诊断。
- 当前 `run_offline_governor_smoke.py` 的 mismatch 目标域舵效系数为 .9，
  与参考 smoke 默认设置一致。旧版 simple 运行目录保留
  `runner_source_at_run.txt`，其 SHA 与旧版 `run.json` 完全一致；
  **旧版运行器调用了上层低层控制器，实际命令与核验命令不一致，
  旧版完整配对不能作为性能证据。**有效结论只取 `full_direct_*` 目录。
- 参考链路 smoke 与离线 Gate1 运行不可当成在线 `strict_mainline` 成功。
  5 ms 超时仍须记录，算法性能另按同场景、同模型/信息/约束比较。
- 不要勾选尚未满足完整条目要求的后续 Gate。测试时只用 `/bin/python3`；
  改动后至少运行 H15 单元测试与 `git diff --check`。
