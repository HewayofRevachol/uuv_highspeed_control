# CuSADi/cuDSS 非线性 RTI-SQP

当前打包目录不含下文示例使用的 `train_residual_ppo_batch.py`、`benchmark_rti_cuda.py` 与 `tests/test_rti_mpc.py`；这些命令保留作原项目的历史参考。此处可通过 `DDAT_train.py --backend rti_cpu` 或 `--backend rti_cuda` 使用对应后端。

新增的 `rti_cpu` 与 `rti_cuda` 后端使用与现有批量环境相同的 10 状态、3 实际执行器模型。每个控制周期只解一个 SQP 子问题：先按上次控制序列滚动标称轨迹，CasADi/CuSADi 计算 RK4 离散动力学及解析雅可比，构造多重打靶 QP，再把第一步控制交给残差 PPO 环境。QP 的动力学等式由 cuDSS 分解的对称不定 KKT 精确处理，输入上下界由 ADMM 处理。所有环境共享 KKT 稀疏结构与符号分析；每周期重新进行批量数值分解。

## 安装

需要 CUDA 13 工具链、`nvcc`、可运行 CUDA 的 PyTorch 解释器。本机已用 `/bin/python3`、RTX 4060、CUDA 13 验证。运行：

```bash
bash setup_rti_cuda.sh /bin/python3
```

脚本通过 `/bin/python3 -m pip install --user` 将 `requirements-rti.txt` 安装到该解释器可见的用户级包目录，并将 CuSADi 源码固定在提交 `bffea25849a901277a8f8dc4db16748e629f9053`。首次运行 GPU 后端会在 `.rti_build` 生成并用 `nvcc` 编译 CUDA 内核。可用 `CUSADI_ROOT`、`UUV_RTI_BUILD`、`NVCC` 更改默认路径。

## 运行

```bash
/bin/python3 train_residual_ppo_batch.py \
  --device cuda --backend rti_cuda --envs 128 --horizon 20 \
  --admm-iterations 400
```

CPU 正确性基线使用 `--device cpu --backend rti_cpu`。原有 `admm` 与 `osqp` 后端仍可用。`mpc_ok` 为假时保留上一条安全控制命令；训练指标会记录回退比例。`CudssEqualityBoxQP.last_primal` 和 `last_dual` 分别记录无穷范数的可行性与对偶残差。

## 验证与范围

```bash
/bin/python3 -m pytest -q tests/test_rti_mpc.py
RUN_RTI_CUDA_TESTS=1 /bin/python3 -m pytest -q tests/test_rti_mpc.py
/bin/python3 benchmark_rti_cuda.py --device cuda --envs 128 \
  --horizon 20 --steps 20
```

GPU 与 CPU 使用同一个 QP。验证包括离散模型与已有 RK4 对照、解析雅可比差分、QP 动力学等式与输入边界、CuSADi 生成值与 CPU CasADi 对照，以及 cuDSS 解与 OSQP 对照。

这版仍是模型的 5 自由度/10 状态版本；原始 `train_residual_sac.py` 未改动。当前 KKT 非零结构稀疏，但 QP 拼装暂用 PyTorch 稠密张量，后续若要提升大时域吞吐量，应将 QP 拼装改为按稀疏块写入。
