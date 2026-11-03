# 高速 UUV 的 RL＋MPC：优质期刊与顶会文献线索

检索日期：2026-11-01（北京时间）。本次优先选 **2025 年中科院升级版大类 1 区**期刊和 ICML、ICLR、CoRL、RSS 等顶会。分区有版本差异：例如 IEEE T-IV 在 2025 版为 1 区、2026 新锐版为 2 区；IEEE T-RO 为 2025 版 1 区。以下分区判断参考[公开分区查询：T-IV](https://www.sci365.cn/journals/ieee-transactions-on-intelligent-vehicles-23798858/)、[T-RO](https://topj.lib.whu.edu.cn/show.asp?cat=zk&id=3227)、[JAS](https://www.sci365.cn/journals/ieee-caa-journal-of-automatica-sinica-23299266/)和[TIE](https://www.sci365.cn/journals/ieee-transactions-on-industrial-electronics-02780046/)；若用于考核，应以本校采用的正式分区版本复核。论文方法根据出版方摘要或官方论文页整理；“借鉴”栏是针对本仓库的研究假设，不是原文在 UUV 上的实验结论。

检索词覆盖 `reinforcement learning model predictive control`、`actor critic MPC quadrotor`、`RL terminal cost MPC autonomous driving`、`event triggered MPC reinforcement learning`、`AUV RL MPC`、`speed adaptive model predictive contouring control`。筛选排除了仅在摘要里将 MPC 作为对照、未真正结合的方法的绝大多数结果，也排除了检索时尚未到正式发表日期的论文。引用数未取得可追溯的同一平台记录，故均记“未核验”；不以搜索片段或第三方页面的数字代填。PDF 链接只表示来源提供入口，未做本次文件下载及字节、论文身份验证；未列出的代码链接表示未从论文/作者来源核实。

## 最值得先读：RL 确实进入 MPC 结构

| 优先级 | 论文、作者、出处 | RL 与 MPC 如何结合 | 对本仓库的具体借鉴 | DOI/标识、PDF、代码 |
| --- | --- | --- | --- | --- |
| 1 | [Actor–Critic Model Predictive Control: Differentiable Optimization Meets Reinforcement Learning for Agile Flight](https://ieeexplore.ieee.org/document/11301631/)，Angel Romero、Elie Aljalbout、Yunlong Song、Davide Scaramuzza，*IEEE Transactions on Robotics* 42，2026（在线发表于 2025-12-16） | 将可微 MPC 放在 actor 的输出端，actor-critic 的长期学习与 MPC 的短期在线优化共同训练；四旋翼竞速有仿真和实飞验证，论文摘要报告速度至 21 m/s。 | 用策略生成**代价图、参考进度或有限维 MPC 参数**，控制命令仍由约束优化产生；考察跨速度、参数失配和执行器延迟的泛化。 | DOI [10.1109/TRO.2025.3644945](https://doi.org/10.1109/TRO.2025.3644945)；[作者 PDF](https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf)；[作者代码](https://github.com/uzh-rpg/acmpc_public)。 |
| 2 | [Learning-Based MPC for Autonomous Motion Planning at Freeway Off-Ramp Diverging](https://ieeexplore.ieee.org/document/10423843/)，Xudong Qi、Libin Zhang、Ping Wang、Jingwen Yang、Tao Zou、Wanquan Liu，*IEEE Transactions on Intelligent Vehicles* 9(10)，2024 | Q 网络的长期价值作为 MPC 终端代价，并随场景调整预测窗。摘要主要报告汽车仿真。 | 在 2.5–3 s 短窗后端加入 `V(s_N, 未来曲率, 舵机状态, 控制余量)`，专门处理窗外急弯/深度反转。先验证价值函数误差和 OOD 退化。 | DOI [10.1109/TIV.2024.3363239](https://doi.org/10.1109/TIV.2024.3363239)；PDF/代码未核验。 |
| 3 | [Event-Triggered Model Predictive Control With Deep Reinforcement Learning for Autonomous Driving](https://ieeexplore.ieee.org/document/10306336/)，Fengying Dang、Dong Chen、Jun Chen、Zhaojian Li，*IEEE Transactions on Intelligent Vehicles* 9(1)，2024 | RL 决定**何时**重求解 MPC，论文比较 DDQN、PPO、SAC 与阈值触发策略，目标兼顾控制质量和计算频率。 | 让策略选择计算资源：弯道/预测残差/求解器异常时高频求解或调用更强求解器，直线段低频。须同时报告最坏周期耗时和失败率。 | DOI [10.1109/TIV.2023.3329785](https://doi.org/10.1109/TIV.2023.3329785)；[作者 PDF](https://jchen2020.net/pubs/j_tiv2024-e.pdf)；[作者代码](https://github.com/DangFengying/RL-based-event-triggered-MPC)。 |
| 4 | [Level Curve Tracking via Robust RL-Guided Model Predictive Control](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.124350)，Zhuo Li、Yunlong Guo、Gang Wang、Wei Chen，*IEEE/CAA Journal of Automatica Sinica* 11(12)，2024；**3 页 Letter** | RL 规划高层参考轨迹，鲁棒 MPC 执行跟踪；原问题是未知标量场等值线跟踪。 | 把 RL 动作改成局部路径进度率或可行速度参考，MPC 保留低层约束。此文篇幅短，需与更完整的 T-RO/ICML 方法一起读。 | DOI [10.1109/JAS.2024.124350](https://doi.org/10.1109/JAS.2024.124350)；[出版方 PDF](https://www.sciengine.com/cfs/files/pdfs/view/2329-9266/C0D75389EC504DCA8CF911E71BBCF3C0.pdf)；代码未核验。 |
| 5 | [Computationally Efficient Long-Horizon Predictive Control for Power Converter: A Reinforcement Learning Approach](https://ieeexplore.ieee.org/document/10957823/)，Yihao Wan、Yang Zhang、Qianwen Xu，*IEEE Transactions on Industrial Electronics* 72(10)，2025 | 对长预测窗有限控制集 MPC 的组合爆炸，用分层动作空间与 RL 学开关策略，并做仿真和硬件实验。 | 借鉴“分层缩小决策空间”：让 RL 只在有限个预测窗/模型精度/求解预算模板间选择；其电力电子离散开关结果不能直接推断连续舵角控制收益。 | DOI [10.1109/TIE.2025.3555033](https://doi.org/10.1109/TIE.2025.3555033)；PDF/代码未核验。 |
| 6 | [Temporal Difference Learning for Model Predictive Control](https://proceedings.mlr.press/v162/hansen22a.html)，Nicklas A. Hansen、Hao Su、Xiaolong Wang，**ICML 2022** | 学习任务相关潜在动力学和终端价值，用短窗轨迹优化代替长窗开环规划；二者由 TD 学习联合训练。 | 可只借“短窗＋长期价值”，不必照搬高维潜在模型；本仓库状态仅 10 维，先与物理模型＋小价值网络比较。 | PMLR 162 / `hansen22a`；[PDF](https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf)；[作者代码](https://github.com/nicklashansen/tdmpc)。 |
| 7 | [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU)，Nicklas Hansen、Hao Su、Xiaolong Wang，**ICLR 2024** | 扩展 TD-MPC 的世界模型与多任务训练；论文报告跨 104 项连续控制任务的结果。 | “同一策略跨速度段”可借多任务表示，但大型世界模型对 10 维 UUV 状态可能计算冗余；重点学其训练和消融设计。 | arXiv [2310.16828](https://arxiv.org/abs/2310.16828)；[会议 PDF](https://openreview.net/pdf?id=Oxh5CstDJU)；[作者代码](https://github.com/nicklashansen/tdmpc2)。 |
| 8 | [Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination](https://proceedings.mlr.press/v306/spieler26a.html)，Jonathan Spieler、Sven Behnke，**ICML 2026** | 用策略展开少量候选轨迹，在学习的世界模型中作梯度优化，使用不确定性正则并复用上一轮优化。 | 用策略提出参考/初值，MPC 作最后的可行性优化；若学习模型多步预测误差大，应限制其影响范围。 | PMLR 306 / `spieler26a`；[PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/spieler26a/spieler26a.pdf)；[项目页](https://dream-mpc.github.io/)，代码仓库未核验。 |

## 强 MPC 基线：它们不等同于 RL＋MPC，但应作为论文对照

| 论文、作者、出处 | 方法与应检验的问题 | 标识、PDF/代码 |
| --- | --- | --- |
| [Meta-Learning Enhanced Model Predictive Contouring Control for Agile and Precise Quadrotor Flight](https://ieeexplore.ieee.org/document/10989530/)，Mingxin Wei、Lanxiang Zheng、Ying Wu、Ruidong Mei、Hui Cheng，*IEEE Transactions on Robotics* 41，2025 | 针对不同速度的气动阻力学习模型参数，并在线适配 MPCC；非常贴合本项目跨速度水动力失配。它用**元学习**，不应写成 RL。 | DOI [10.1109/TRO.2025.3567491](https://doi.org/10.1109/TRO.2025.3567491)；PDF/代码未核验。 |
| [Model Predictive Contouring Control for Time-Optimal Quadrotor Flight](https://arxiv.org/abs/2108.13205)，Angel Romero、Sihao Sun、Philipp Foehn、Davide Scaramuzza，*IEEE Transactions on Robotics* 38(6)，2022 | 把路径轮廓误差与路径进度直接纳入优化，是将固定时间参考改为 MPCC 的强基线。 | [作者 PDF](https://rpg.ifi.uzh.ch/docs/Arxiv21_MPCC_Romero.pdf)；arXiv `2108.13205`。 |
| [MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints](https://www.roboticsproceedings.org/rss20/p109.html)，Maria Krinner、Angel Romero、Leonard Bauersfeld、Melanie Zeilinger、Andrea Carron、Davide Scaramuzza，**RSS 2024** | 路径安全约束、终端安全集、数据驱动气动残差与贝叶斯调参共同组成强模型基线；调参是 Bayesian optimization，**不是 RL**。 | [RSS PDF](https://www.roboticsproceedings.org/rss20/p109.pdf)；arXiv `2403.17551`。 |
| [Beyond Constant Parameters: Hyper Prediction Models and HyperMPC](https://proceedings.mlr.press/v305/wegrzynowski25a.html)，Jan Węgrzynowski、Piotr Kicki、Grzegorz Czechmanowski、Maciej Piotr Krupka、Krzysztof Walas，**CoRL 2025** | 沿预测窗学习时变动力学参数；F1TENTH 实车验证。适合验证本仓库“单次线性化后长窗模型误差”问题。此方法是学习模型增强 MPC，**不是 RL 控制策略**。 | PMLR 305 / `wegrzynowski25a`；[PDF](https://raw.githubusercontent.com/mlresearch/v305/main/assets/wegrzynowski25a/wegrzynowski25a.pdf)。 |

## 领域边界：AUV 直接相关但不进入主清单

- [Neural Network Model-Based Reinforcement Learning Control for AUV 3-D Path Following](https://ieeexplore.ieee.org/document/10143677/)，Dongfang Ma、Xi Chen、Weihao Ma、Huarong Zheng、Fengzhong Qu，*IEEE Transactions on Intelligent Vehicles* 9(1)，2024，DOI [10.1109/TIV.2023.3282681](https://doi.org/10.1109/TIV.2023.3282681)。它以神经网络模型辅助 PPO 控制，**MPC 是比较对象**，适合借 AUV 模型训练与基线设计，不能当作 RL 增强 MPC 的直接先例。PDF/代码未核验。
- [A RL-Based MPC Algorithm for AUV Trajectory Tracking](https://ieeexplore.ieee.org/document/11104376/)，*OCEANS 2025 Brest*，DOI [10.1109/OCEANS58557.2025.11104376](https://doi.org/10.1109/OCEANS58557.2025.11104376)。RL 调 MPC/CBF 权重，属于直接先例；按用户的顶刊/顶会筛选，放在边界清单，不能将“RL 自适应权重＋AUV”本身当新意。作者、PDF、代码与被引数未核验。
- [VG-MPCC: A virtual-guide model predictive contouring control formulation for path following of autonomous underwater vehicles](https://www.sciencedirect.com/science/article/pii/S0029801826031781)，Liang Zhao、Fang Wang，*Ocean Engineering* 365，2026，DOI [10.1016/j.oceaneng.2026.127344](https://doi.org/10.1016/j.oceaneng.2026.127344)。3 自由度水平面 AUV 的虚拟路径参数基线；2025 中科院大类分区不满足本次主清单筛选，[作者代码](https://github.com/LiangZhao13/VGMPCC)。PDF 未核验。

## 针对当前仓库的三个研究切口

仓库的现有 PPO 在 `simulator/ddat_env.py` 中输出推进力和俯仰/偏航角速度修正；`_reference()` 按当前目标速度推进整段预测参考。`simulator/batch_mpc.py` 每轮只在起点线性化一次；FSP 修正执行时刻错位，但 MPC 窗内尚未完整包含舵机状态/命令队列。原项目的[15 m/s 路线决策报告](../../uuv_simulator/high_speed_experiments/REPORT_mpc_rl_route_decision.md)显示原低层残差在 18 组配对中平均位置 RMSE 从 0.113 m 变为 0.181 m；[近期报告](../report1030.md)又记录 3/6 m/s 整组失败与 18 m/s 个别失稳。它们来自不同实验设置，不能直接合并统计。

### A. 学“预测窗外价值”，而不是小幅修舵

先做执行器感知的强 MPC/MPCC，再用 RL 学终端价值 `V(s_N, κ_future, u, q, r, δ_act, current_est, uncertainty)`。训练目标是未来安全完成率、时间/能耗和轨迹误差；部署时价值仅在**验证过的状态区域**影响 MPC 目标，保留硬约束。对照：零终端价值、人工二次终端代价、长窗强 MPC、TD-MPC 式价值、现有低层 PPO。关键风险是价值高估使优化器把末端状态推到训练分布外。

### B. 学“可行参考进度”，同时显式刻画低速舵效

将策略输出限制为路径进度率/目标速度/预瞄长度等少数高层量，投影到由推力、舵效、曲率、俯仰界和执行器延迟计算的可行域；MPC 负责控制执行。低速舵效随 `u²` 下降，故策略不能简单对所有弯道减速，还需有最低可控速度或路径重规划。对照：固定速度、确定性曲率限速、同信息量的非 RL 网格/贝叶斯调参、高层 RL。新意应落在**跨速度的可行域与高层学习协同**，不是“RL 调权重”。

### C. 学“模型何时可信、求解何时升级”

让轻量策略依据多步预测残差、估计海流、舵机状态和 ADMM 残差选择：普通 QP、增强模型/重线性化、OSQP 校验或保守参考。必须设置确定性回退和耗时上限。对照固定求解模式、阈值触发与 RL 触发，报告 95/99 分位控制周期、QP 回退率、约束违反率及配对完成率。这条路线更接近 [事件触发 RL-MPC](https://ieeexplore.ieee.org/document/10306336/)；若低速失败在高精度求解下仍存在，就不能把它归因于求解器。

## 最小验证协议与阅读顺序

1. **先读** T-RO Actor–Critic MPC、T-IV 终端价值、T-IV 事件触发；它们分别回答“学什么”“学多远”“何时求解”。再读 T-RO 元学习 MPCC、RSS MPCC++，确认强模型基线。
2. 以路径几何、速度段（3/6/9/12/15/18 m/s）、海流、艇体参数失配和执行器延迟分层；先筛出动态可行但固定 MPC 有失误的挑战集。训练路径家族与测试路径家族隔离；全部方法使用相同路径和扰动种子。
3. 固定至少 3 个独立训练种子、每条测试轨迹至少 10 个配对扰动种子。主指标是安全完成率与越界率；成功回合再比较横向/深度 RMSE、95 分位误差、能耗、饱和率和控制周期耗时。失败回合不以短 RMSE 排名。
4. 先验证机制再谈创新：多步模型误差、终端价值校准、可行速度覆盖率、求解器回退率分别做消融。任何安全保证只在所用模型、误差界和约束假设成立时适用；当前仿真结果不等同实艇验证。
