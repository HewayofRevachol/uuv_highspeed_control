# 测试说明

使用 Python 3.10+，从仓库根目录执行。先单独安装适配本机 CPU/CUDA 的 PyTorch，再安装测试依赖：

```bash
python3 -m pip install torch
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q
```

第一条使用当前平台的默认 PyTorch 构建；需要指定 CUDA 版本时，改用对应 PyTorch 安装命令。CPU 测试不要求 CUDA RTI 工具链；测试依赖与后端覆盖以 `requirements-test.txt` 和测试代码为准。

| 位置 | 用途 |
| --- | --- |
| `unit/` | 网络、控制器、动力学、配置与接口的快速检查 |
| `integration/` | 训练、验证及组件组合的短程检查 |
| `benchmarks/` | 性能与资源开销实验，按需添加脚本 |

`pytest` 同时收集函数式测试和 `unittest` 用例。仅使用 `unittest discover` 不会覆盖已有函数式测试。CPU CI 的定义见 `.github/workflows/tests.yml`；本说明只列运行方法，不代表某次 CI 已通过。

命令行入口和配置也可单独检查：

```bash
python3 -m training.ddat --help
python3 -m evaluation.ddat --help
python3 experiments/run.py experiments/configs/ddat_cpu_smoke.json --dry-run
```

烟雾测试验证导入、参数传递、组件构造或短程仿真；它不证明收敛或真实艇体上的控制安全性。需要 RTI CUDA 性能和数值验证时，先安装对应工具链，并以当前仓库实际存在的测试/脚本为准；历史 README 中列出的部分旧脚本未收录。
