# 精简版 B–F 实验计划

## 目标与边界

阶段 A 已完成并复现了 GS-12 的组合失配频响规律。本计划只使用这批结果作基线，不重复桥接和 36 条机动实验。新的 B–F 只回答三个问题：

1. β 从 0 增至 1 时，低/高两个代表频带上的东向、深度、速度和耦合误差如何变化？
2. 开发路径选出的参数能否迁移到两个航速及频率切换路径？
3. 若规则调度在切换开发集上有一致收益，一个小型 PPO 试跑能否达到固定参数/规则基线？

精简后只保留 β=`0, 0.5, 1`、k=`8, 24`、匹配/组合失配对象，以及一个耦合相位差（π/2）。不再独立扫描全部幅值、失配类型、相位和三种航速。18 m/s 等高负荷风险已由阶段 A 机动结果覆盖；新速度检查用 12/18 m/s 的深度与耦合路径。

## 冻结任务和运行规模

生成器为 `prepare_light.py`，清单写入新目录；共 40 条任务：

| 阶段 | 任务设计 | 闭环运行数 |
|---|---|---:|
| B β筛选 | 2 个植物 × 4 方向 × 2 频带，共16条64秒路径；两种预测结构各测 β=0/0.5/1 | 96 |
| C 迁移检查 | 12/18 m/s × 深度/耦合，共4条64秒组合失配路径；各结构比较 β=0 与 B 选出的固定β | 16 |
| D 调度比较 | 东向/深度 × 低高低/高低高，共4条144秒切换路径；比较两结构的固定β与按阶段规则 | 16 |
| E PPO | 单一结构、单随机种子、16回合；从8条训练路径抽样。另在D开发路径上比较固定/规则/PPO | 28 |
| F 留出 | 两方向 × 两顺序 × 12/18 m/s，共8条144秒新相位、半幅路径；比较所选结构的固定/规则/PPO | 24 |

E 的28次包括16次训练回合、12次开发路径评价。D 若规则在4条路径上没有至少3条胜过固定基线，或完整配对不足4条，则按预先约定停止，不进入 PPO；此时交付结论为“当前证据不支持继续训练”。

144秒记录在 A 中的实际耗时为约41–123秒，中位数约55秒；64秒任务按时长折算约18–55秒。按串行执行估算，完整计划约1.5–4小时，加上分析约半天内可完成。这个估算来自现有轨迹耗时，不是硬性时限保证。若实际耗时超过4小时，先完成 B、D 的配对结论，再决定是否继续 PPO；不得删掉失败路径后宣称全局获胜。

## 执行命令

从仓库根目录运行。输出目录必须是新的目录；若需重跑，使用新名字。

### 0. 固定清单

```bash
python3 -m experiments.gs20_gs12_continuation.prepare_light --out experiments/gs20_gs12_continuation/inputs_light_v1
```

预期计数：`beta_dev=16, speed_check=4, switch_dev=4, switch_train=8, switch_test=8`。在打开 `switch_test` 结果前冻结 β、结构、规则、训练种子和 PPO 回合数。

### B. β频带筛选

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_light_v1/beta_dev.jsonl --methods first:0 first:0.5 first:1 hybrid:0 hybrid:0.5 hybrid:1 --out outputs/gs20_continuation/LB_beta
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/LB_beta --out outputs/gs20_continuation/LB_analysis --freeze outputs/gs20_continuation/LB_selection.json
```

用全程物理 RMSE（m、m/s）和无量纲 R²报告每一任务。固定 β 按任务等权平均 R²选择，并要求该候选在16条任务上完整、无QP失败或饱和。保留三轴分项 RMSE；不把单位不同的误差直接相加。

### C. 速度与耦合迁移

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_light_v1/speed_check.jsonl --methods first:0 first:best hybrid:0 hybrid:best --selection outputs/gs20_continuation/LB_selection.json --out outputs/gs20_continuation/LC_speed
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/LC_speed --out outputs/gs20_continuation/LC_analysis --selection outputs/gs20_continuation/LB_selection.json
```

只检验 B 选出的 β 在12/18 m/s深度和耦合路径上的迁移，不据4条任务重新调参。

### D. 固定参数与规则调度

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_light_v1/switch_dev.jsonl --methods first:best first:rule hybrid:best hybrid:rule --selection outputs/gs20_continuation/LB_selection.json --out outputs/gs20_continuation/LD_switch
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/LD_switch --out outputs/gs20_continuation/LD_analysis --selection outputs/gs20_continuation/LB_selection.json
```

记录全程、切换前后及过渡段 RMSE、完成率、QP失败和饱和。4条任务都需完整且无饱和，规则至少在3条上降低 R² 才继续 E；否则停止并报告没有支持学习调度的证据。

### E. 一结构、单种子的 PPO 可行性试跑

按 D 的配对结果，选择规则收益更一致的结构；若两者都不满足门槛，跳过 E/F 的策略评价并以D为终点。以下以 `first` 为例：

```bash
python3 -m experiments.gs20_gs12_continuation.train --tasks experiments/gs20_gs12_continuation/inputs_light_v1/switch_train.jsonl --structure first --seed 41001 --episodes 16 --batch 4 --out outputs/gs20_continuation/LE_first_41001
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_light_v1/switch_dev.jsonl --methods first:best first:rule first:rl --selection outputs/gs20_continuation/LB_selection.json --policy outputs/gs20_continuation/LE_first_41001/final.pt --out outputs/gs20_continuation/LE_dev
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/LE_dev --out outputs/gs20_continuation/LE_analysis --selection outputs/gs20_continuation/LB_selection.json
```

16回合和单种子只用于判断训练链路及是否出现初步收益，不足以声称 RL 稳定优于基线。

### F. 一次冻结的留出评价

```bash
python3 -m experiments.gs20_gs12_continuation.run --tasks experiments/gs20_gs12_continuation/inputs_light_v1/switch_test.jsonl --methods first:best first:rule first:rl --selection outputs/gs20_continuation/LB_selection.json --policy outputs/gs20_continuation/LE_first_41001/final.pt --out outputs/gs20_continuation/LF_test
python3 -m experiments.gs20_gs12_continuation.analyze --runs outputs/gs20_continuation/LF_test --out outputs/gs20_continuation/LF_analysis --selection outputs/gs20_continuation/LB_selection.json
```

若 D/E 实际选出 hybrid，则命令中的 `first` 改为 `hybrid` 并使用对应训练策略。留出集只运行一次，不以测试集挑 β、规则、结构或 PPO 检查点。

## 可支持的结论

这套精简实验可判断 β 频带取舍在三轴和耦合路径上是否重现、在少量航速变化下是否保持、规则调度是否值得做 PPO 试跑。单种子 PPO 不能支持统计显著性或可靠性结论；若需要论文级 RL 结论，后续再扩展种子和留出路径。
