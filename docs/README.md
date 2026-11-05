# 文档总索引

当前使用方式以根目录 [README](../README.md) 和本目录指南为准。历史报告保留原有实验条件、命名和结论；其中的原机器绝对路径、会话名及未随仓库收录的输出，不表示当前机器具备相同环境。

## 开发与复现

| 文档 | 内容 |
| --- | --- |
| [架构说明](architecture.md) | 代码职责、训练与验证调用链、组件边界 |
| [扩展指南](extending.md) | 新增控制器、策略网络、模型与配置示例 |
| [命名与数据生命周期](conventions.md) | 实验标识、数据划分、输出归档及复现记录 |
| [迁移映射](migration.md) | 原入口与新模块对应关系、历史路径保留原则 |
| [数据索引](../data/README.md) | CSV、模型、研究结果和报告的集中清单 |
| [测试说明](../tests/README.md) | 依赖准备、单元和集成测试命令 |
| [输出约定](../outputs/README.md) | 每次运行的目录结构与归档要求 |

## 算法与历史材料

| 文档 | 用途及适用范围 |
| --- | --- |
| [H15 连续 RL-MPC 验证分析（GS-11）](GS-11_H15_RL_MPC_VALIDATION_ANALYSIS_20261105.md) | 补齐 CAT-MPC P0/P1/P2 原始结果后的复算、接口审计和下一轮实验建议 |
| [H15 结果分析与下一主线（GS-10）](GS-10_H15_RL_MPC_RESULT_ANALYSIS_AND_NEXT_MAINLINE_20261104.md) | Gate3 文档证据与代码核查；反馈一致的可行性管理、速度决策空间和下一轮实验设计；详细原始输出未收录 |
| [DDAT 算法说明](../README_DDAT.md) | 观测、残差控制、在线估计和配对验证；部分动画脚本未收录 |
| [DDAT 轨迹设计](../DDAT_trajectories/README.md) | 五类轨迹、训练/验证划分和再生成命令 |
| [RTI CUDA 说明](../README_RTI_CUDA.md) | RTI 求解与 GPU 工具链；其中部分旧测试脚本未收录 |
| [原打包说明](../README_打包说明.md) | 历史打包范围；当前批量验证用 `--all-paths`，无需修改源码 |
| [基础阶段报告](../report1030.md) | 历史 DDAT 实验记录 |
| [高速 RL–MPC 续跑报告](../high_speed_experiments/REPORT_RL_MPC_20261102.md) | 参考控制策略、轨迹与模型实验 |
| [高速实验细节](../high_speed_experiments/RL_MPC_innovation_experiments_20261102.md) | 研究假设、对照和历次实验 |
| [高速研究方向](../high_speed_experiments/RESEARCH_DIRECTION_20261102.md) | 后续方案与研究边界 |
| [AAV-MPC 复现指南](../aav_mpc/README.md) | 分阶段入口、任务配置和数据位置 |
| [AAV-MPC 报告](../aav_mpc/REPORT_20261102.md) | 执行器感知与离线价值路线结果 |
| [顶会文献扫描](../academic_search_report/RL_MPC_top_tier_20261101.md) | RL–MPC 先例整理 |
| [选题综述](../academic_search_report/UUV_RL_MPC_INNOVATION_REVIEW_20261102.md) | 候选路线和实验边界 |
| [推荐路线](../academic_search_report/RECOMMENDED_ROUTE_ACTUATOR_AWARE_RL_MPC_20261102.md) | 执行器状态与长期价值增强 MPC 假设 |
| [证据核验](../academic_search_report/VERIFY_ACTUATOR_AWARE_RL_MPC_EVIDENCE_20261102.md) | 原始回合证据与待验证机制 |
| [仿真现实性审查](../academic_search_report/AUDIT_UUV_MECHANISM_REALISM_20261102.md) | 动力学响应与路径可行性分析 |

报告中提及的 `outputs/reports/uuv_mechanism_audit_20261102/` 未随当前仓库提供；现有审查报告和复现脚本仍可查阅，不能把该缺失目录当作已交付运行证据。
