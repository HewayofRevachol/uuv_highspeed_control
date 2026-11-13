# GS-23 功能检验记录

日期：2026-11-13。**未运行训练入口、S0/S1/S2/S3 或完整闭环轨迹；未更新网络参数。**

环境：Python 3.10、PyTorch 2.14.0+cu130、CasADi 3.6.5、RTX 4060 Laptop（8 GB）。使用工作空间已有固定版 CuSADi 源码，首次功能检查编译了本地 CUDA 内核缓存。

执行：

```bash
python3 -m compileall -q experiments/gs23_frequency_rl
CUSADI_ROOT=/home/prufrock/uuv_workspace/.third_party/cusadi \
  python3 -m pytest experiments/gs23_frequency_rl/test_contract.py -q
python3 -m experiments.gs23_frequency_rl.train --help
```

最终结果：**9 passed**，语法编译和 CLI 帮助正常。

验证范围：

- 双锚点最小二乘投影可区分在 4/8 秒抽样上混叠的频率，历史相位旋转后与当前参考系一致。
- first 预测器与仓库 CasADi 符号模型在 β=0/1、12/18 m/s 样本上一致，容差 2e-12。
- CPU/CUDA box ADMM 与 OSQP 对受限 QP 的解一致，绝对容差 1e-7，含修改输入后的 CUDA Graph 重放。
- 冻结任务数量为开发 4、训练 16、留出 8；参考位置的数值导数与发布参考速度一致。
- 初始策略均值为 β=0.5，γ=1 的 GAE 正确截断已终止 lane，失败补偿比剩余最坏阶段代价更差。
- S0 门槛的通过/失败条件及 PNG/CSV 导出可用；测试曲线使用合成数值，仅在 pytest 临时目录生成，不是训练结果。
- 强制一个 lane QP 失败时只罚一次，另一个 lane 继续积分并计分。
- GPU 集成仅在两个 lane 上各推进两次控制周期（0.2 秒），覆盖 CuSADi QP 组装、ADMM、对象积分三类 CUDA Graph；QP 成功，β 从 0.5 到 0.54，观测为有限的 78 维张量。改变隐藏执行器速率和对象延迟队列不改变观测。历史已填满分支仅用合成缓冲状态检查，不额外推进仿真。

检查中修正了 CUDA Graph 捕获时常量从 CPU 拷入 GPU 的问题，将该常量改为设备端标量运算。保留的非阻塞环境提示：OSQP 默认参数的未来变更提示，以及 Matplotlib 的 Axes3D 不可用提示；本实验输出二维奖励曲线。

边界：没有实测训练吞吐、GPU 利用率、完整轨迹 QP 收敛率或 RL 收益。float64 与 1e-5 残差门槛仅经上述小样本功能检查，正式机会检查是否通过仍待使用者执行。没有以合成曲线或初始网络冒充训练产物。
