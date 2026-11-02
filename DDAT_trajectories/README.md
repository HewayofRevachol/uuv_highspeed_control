# DDAT 轨迹集

`generate.py` 独立生成 20 条训练轨迹和 5 条验证轨迹；不读取、复制或变换
`trajectories/` 中的任何文件。每类机动有四组训练参数和一组未见过的验证参数。
CSV 提供 NED 位置（北、东、深度，单位 m）、`target_pitch_rad`、`target_yaw_rad` 和正的目标
切线速度（m/s），按约 0.8 m 的弧长间距采样。俯仰角与偏航角由路径切线计算，
其中偏航角连续展开后保存；`LoopPath` 会优先读取这两列。`manifest.json` 列出
几何和速度范围。

| 类别 | 参考机动 | 主要挑战 |
| --- | --- | --- |
| `speed_envelope` | 长直线中从 15.3 m/s 升至 17.8–19.4 m/s，减速后再升速 | 推力余量、速度跟踪和推进器迟滞 |
| `constant_turn` | 左右稳曲率弯，曲率平滑接入和退出，转弯中加减速 | 高速舵效、侧滑和转弯失配 |
| `s_reversal` | 相邻的反向曲率平台及短末端修正，弯前加速 | 换舷的舵角反转和执行器延迟 |
| `depth_return` | 高速下潜或上浮、改平、反向返回原深度 | 纵向惯性与深度超调 |
| `coupled_axes` | 水平反向转弯同时升沉、两次变速 | 多轴耦合与控制余量竞争 |

所有轨迹从至少 15.3 m/s 的目标速度开始，先给一段直线/平深预热区。转弯
曲率、深度和速度变化均使用五次平滑过渡，避免折线或瞬时阶跃。弯道最大
曲率对应约 42–88 m 的名义半径；具体范围见 `manifest.json`。这些设置意在
使名义 MPC 在高速、执行器延迟、模型失配和随机海流下有明显跟踪挑战。
在 CPU、`horizon=30`、`admm_iterations=400`、随机海流上限 1 m/s、固定失配
艇体的零残差试算中，种子 102600 完成 1/5 条，种子 102601 完成 0/5 条。
这里的“零残差 MPC”与 `DDAT_train.py` 的 `mpc_only` 定义一致，仍包含其低层
逆动力学和海流/扰动估计。试算 **并不证明 DDAT 能完成失败轨迹**；训练后应使用脚本的
`--eval-actor` 对验证集的 `policy` 与 `mpc_only` 指标做同种子的实际比较。

`PathBank` 重置时把艇首方向设为轨迹首点切线，因此 CSV 不能表达图中建议的
“初始航向偏差”；海流和艇体/执行器参数仍由 `DDATPathEnv` 按回合采样。

再生成：

```bash
python3 DDAT_trajectories/generate.py
python3 -m DDAT_trajectories.benchmark_mpc --seed 102600
```

默认训练直接读取此目录。评估时 `DDAT_train.py --eval-actor <checkpoint>`
读取验证集，并报告策略与零残差 MPC 的成功率、位置 RMSE 和回报。
