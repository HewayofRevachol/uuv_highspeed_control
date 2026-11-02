# 测试目录

- `unit/`：网络、控制器组件、轨迹处理的快速测试。
- `integration/`：训练、验证和仿真链路的短程测试。
- `benchmarks/`：速度、显存、求解器耗时测试脚本。

从项目根目录运行：`python3 -m unittest discover -s tests -p 'test_*.py'`。
