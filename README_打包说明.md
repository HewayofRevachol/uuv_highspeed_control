# DDAT 训练与验证包

本包围绕 `DDAT_train.py` 和 `validate_ddat.py` 收集其实际导入的项目脚本、默认轨迹、
一份可直接验证的 checkpoint。解压后在本目录运行命令。已有输出日志、其他实验脚本和大型
`trajectories/path_corpus` 未收录；DDAT 训练使用包内 `DDAT_trajectories/train`。
现已增加研究模块、实验配置启动器和预留的测试、输出目录，使用方法见 `README.md`。

## 环境

建议 Python 3.10+。先安装与你的 CPU/CUDA 环境匹配的 PyTorch，再运行：

```bash
python3 -m pip install -r requirements-ddat.txt
```

默认 `admm` 后端不需要 OSQP 或 CasADi。`osqp` 后端需要 `osqp`；`rti_cpu`
需要 `casadi` 和 `osqp`。`rti_cuda` 还需要 CUDA、cuDSS 和 CuSADi 源码，
参见 `README_RTI_CUDA.md` 与 `setup_rti_cuda.sh`，CuSADi 源码不在包内。

## 训练

```bash
python3 DDAT_train.py --list-training-paths --device cpu
python3 DDAT_train.py --device cpu --backend admm --envs 8 --total-steps 4096 \
  --output outputs/DDAT/example
```

第二条只是快速试运行示例；正式训练应按设备性能调整规模与训练步数。
默认设备可能是 CUDA，只有 CPU 时请显式加 `--device cpu`。

## 验证

已附带 `outputs/DDAT/run_20261029_114239/final_actor.pt`。默认验证脚本读取包内
`trajectories/stress_15mps/cold_helix_r49_15p9mps.csv`，结果写到 checkpoint
目录下的 `evaluation/`。用 `--no-show` 在无图形界面的环境运行：

```bash
python3 validate_ddat.py --device cpu --no-show
python3 validate_ddat.py --path DDAT_trajectories/val/s_reversal_01.csv \
  --device cpu --no-show
```

如需验证完整验证集，把 `validate_ddat.py` 顶部的 `VALIDATION_PATH` 改为 `None`，
再运行 `python3 validate_ddat.py --split val --no-show`。
checkpoint 的原训练配置记录了原机器的绝对轨迹目录；本包的验证脚本优先使用同名的包内目录。
`--split train` 也会把训练清单映射到包内训练轨迹。

目录中的 `README_DDAT.md` 和 `DDAT_trajectories/README.md` 说明算法与轨迹格式。
