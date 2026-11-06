# GS-12 四通道 RL 门控实验代码

入口与 `docs/GS-12_RL实验执行清单.md` 对应。所有程序从仓库根目录以 `python3 -m` 运行。`prepare` 会冻结源码、配置、场景和观测 schema 哈希；修改代码后要创建新的运行目录。训练、调参和评估各自是前台命令，程序不会自动在后台运行。

```bash
python3 -m experiments.gs12_gate.run prepare --out outputs/gs12_gate
python3 -m experiments.gs12_gate.run smoke --run outputs/gs12_gate
python3 -m experiments.gs12_gate.run collect --run outputs/gs12_gate --split train_calibration --episodes 32
python3 -m experiments.gs12_gate.frequency --run outputs/gs12_gate --mode freeze-direction
python3 -m experiments.gs12_gate.train --run outputs/gs12_gate --seed 41001 --decisions 8192 --update-every 512
python3 -m experiments.gs12_gate.run tune --run outputs/gs12_gate --split dev_screen
python3 -m experiments.gs12_gate.run evaluate --run outputs/gs12_gate --split dev_quick --checkpoint outputs/gs12_gate/training/41001/checkpoint_008192.pt
python3 -m experiments.gs12_gate.frequency --run outputs/gs12_gate --mode pilot --direction-source outputs/gs12_gate/frequency/dev_direction/direction_source.json --checkpoint outputs/gs12_gate/training/41001/checkpoint_008192.pt
python3 -m experiments.gs12_gate.report --run outputs/gs12_gate --stage pilot
```

`train.py` 在训练结束后用 `dev_quick` 选择中点或终点 checkpoint；评估时应传入 `selected_checkpoint.json` 指向的文件。中点预览可用 `run evaluate --tag midpoint`，其日志与最终评估分开保存。

有开发集收益信号后，确认阶段可运行：

```bash
python3 -m experiments.gs12_gate.train --run outputs/gs12_gate --seeds 42001,42002,42003,42004,42005 --decisions 32768 --update-every 512
python3 -m experiments.gs12_gate.run evaluate --run outputs/gs12_gate --split test --checkpoints PATH1,PATH2,PATH3,PATH4,PATH5
python3 -m experiments.gs12_gate.frequency --run outputs/gs12_gate --mode confirm --checkpoints PATH1,PATH2,PATH3,PATH4,PATH5
python3 -m experiments.gs12_gate.frequency --run outputs/gs12_gate --mode locked --checkpoints PATH1,PATH2,PATH3,PATH4,PATH5
python3 -m experiments.gs12_gate.run evaluate --run outputs/gs12_gate --ablation freeze4,no_theta --checkpoints PATH1,PATH2,PATH3,PATH4,PATH5
python3 -m experiments.gs12_gate.report --run outputs/gs12_gate --stage final
```

只有在需要论文级模块归因时，另运行两套同预算重新训练消融：

```bash
python3 -m experiments.gs12_gate.train --run outputs/gs12_gate --ablation tied_residual --seeds 42001,42002,42003,42004,42005 --decisions 32768
python3 -m experiments.gs12_gate.train --run outputs/gs12_gate --ablation no_history --seeds 42001,42002,42003,42004,42005 --decisions 32768
python3 -m experiments.gs12_gate.run evaluate --run outputs/gs12_gate --split ablation_holdout --checkpoints PATHS_FOR_STANDARD_AND_ABLATIONS
python3 -m experiments.gs12_gate.report --run outputs/gs12_gate --stage final --split ablation_holdout
```

`PATHS_FOR_STANDARD_AND_ABLATIONS` 是用逗号连接的已选 checkpoint 路径。`prepare` 不开展频域仿真；若没有现成开发集方向源，先完成训练集校准，再用 `freeze-direction` 生成冻结开发方向。训练前会把待定的 SVD 方向填入 `train_cases_resolved.jsonl` 并保存哈希。

`PATH1` 等为每个训练 seed 的已选 checkpoint。确认频域使用七个精确整数频点；逐轴拼接的动态响应只用于诊断。锁定模式记录同一适应快照，再分别运行三轴与零激励。

本目录重建了 25 维混合控制器和植物接入。执行清单提及的仓库外 `gs12_followup` 源文件未提供；`smoke` 检查自身传播、Jacobian、边界、因果时序和已发命令，无法声称与那份独立原型逐位一致。默认环境是理想的 .1 秒艇体/实际执行器测量，物理参数只进入植物与评估记录。
