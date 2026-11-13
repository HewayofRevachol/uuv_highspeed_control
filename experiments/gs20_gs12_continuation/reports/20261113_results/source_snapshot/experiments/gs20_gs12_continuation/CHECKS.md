# 本次最小检查

日期：2026-11-12。工作目录：`uuv_workspace`。

以下命令均退出0：

```bash
python3 -m compileall -q experiments/gs20_gs12_continuation
python3 -m experiments.gs20_gs12_continuation.check
```

检查覆盖模块导入、任务ID唯一性、参考位置/速度解析一致性、合成数据RMSE、策略输出维数与范围、原QP及三个公共模型文件的源码哈希。

内存中生成的任务数依次为：bridge 8、maneuvers 36、static_dev 70、static_test 70、range_dev 60、half_dev 35、mismatch_dev 30、switch_dev 12、switch_train 48、switch_test 72。

没有构造植物、求解QP、推进动力学、更新PPO或运行训练；没有生成实验输入文件或效果数据。闭环运行和性能结论仍待按STEPS.md执行后检验。
