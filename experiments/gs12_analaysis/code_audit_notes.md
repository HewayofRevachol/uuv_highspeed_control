本审阅针对检出提交 `d001f998331157ff6d45d11e2e663508a9e8cf44`，只读生产源码；诊断脚本和结果均在 `gs12_analysis/`。下列行号相对于仓库 `uuv_highspeed_control`。小型数值探针用本工作区 `.venv-gs12/bin/python`，NumPy/Torch/CasADi CPU；未做训练、全实验矩阵或实艇验证。

## 1. 已确认的控制架构

基础状态为 `x=[north,east,depth,pitch,yaw,u,v,w,q,r]`，控制为 `[T,δe,δr]`，无横摇自由度。艇体模型位于 `simulator/batch_mpc.py:33`；固定参数位于 `simulator/model.py:94`。物理输入范围为推力 `[0,1.5 MN]`、两舵 `±45°`（`simulator/batch_mpc.py:15`、`:27`）。

H15 CAT-MPC 是：路径几何/流速估计→参考→22 状态 condensed RTI-QP→DDAT 低层扰动补偿→发出命令→延迟队列/真实执行器→真实艇体积分→观测更新。22 状态是艇体10、实际执行器3、最新三个待执行命令9；不含二阶执行器速率（`simulator/rti_symbolic.py:16`，`aav_mpc/controller.py:21`）。实际环境保留五个队列槽（`simulator/batch_path_env.py:152`）。

控制周期0.1s，植物每周期5×0.02s。H15 runner 固定目标15m/s、N=35即3.5s预测窗（`experiments/h15_cat_mpc/run_episode.py:297`）。`CondensedActuatorMPC` 默认单次 SQP/RTI，基于非线性预滚动、局部雅可比和OSQP解75/105等数量的控制增量；H15 N35为105决策变量（`aav_mpc/condensed_mpc.py:111`）。输入盒约束默认开启；slew、场景预测约束为可选，CAT runner未开启它们（`:39`、`:151`、`:154`、`:160`）。物理越界由子步日志检测并终止回合，不是被预先证明不变的安全集合（`run_episode.py:143`、`:202`）。

连续门控仅修改预测模型和加速度预报：

`θ_applied=θ0+αθ(θ_hat−θ0)`，`d_applied[k]=αd*d_hat[k]`。

这里θ是共同舵效与共同舵时间常数倍率，θ0=(.85,2)（`adapter.py:8`）；门控每5个控制周期即0.5s更新，MPC每0.1s更新（`run_episode.py:388`）。`CondensedActuatorMPC.solve` 的参数混合与残差乘法在`:225`和`:238`。估计器只使用已完成周期，估计的是短窗等效响应，不是真实植物参数（`experiments/h15_mainline/response_estimator.py:1`、`:145`）。

## 2. 名义预测与仿真真实模型的具体差异

不要混淆三种“nominal”：基础10状态模型缺省全参数1；H15增强预测固定使用 `MISMATCHED_PLANT=(1.04,1.05,.98,.95,.95)`；现H15控制基线进一步把两舵预测增益乘.85、舵τ乘2（`simulator/model.py:43`，`rti_symbolic.py:97`，`run_episode.py:307`）。

令ν=[u,v,w,q,r]，p=(pm,pd,gT,ge,gr)，代码的加速度为：

`νdot=(pm M0)^−1{[gT*T,0,0,0,0]+u²(ge*E*δe+gr*R*δr)−pm*m*[-rv+qw,ru,−qu,0,0]−pd*u*D*ν}+d`。

位置为 `pdot=R(θ,ψ)[u,v,w]+current_NED`，姿态为 `θdot=q, ψdot=r/cosθ`（`simulator/batch_mpc.py:43`至`:62`）。流仅加到位置导数，未通过水相对速度改变阻尼、舵力或科氏项；如果u,v,w解释为相对水速度，这是一个有限的运动学流模型，而非完整非均匀海流水动力。

增强预测固定上述“失配”参数；固定3个20ms延迟槽（0.05s向上取整成0.06s）；一阶精确离散执行器 `a+=1−exp(−.02/τ)`；5次RK4；不直接读取随机真参数、真实delay_steps、流或未来扰动（`rti_symbolic.py:79`、`:108`、`:129`）。

真实target在同一艇体方程上加入：

- 每回合参数随机化：质量1.04×[.97,1.03]，阻尼1.05×[.93,1.07]，推力.98×[.95,1.05]，两舵各.95×[.90,1.10]；两舵参数独立采样（`batch_path_env.py:187`至`:198`）。
- 延迟各从.02–.08s采样并取整；实际通常2–4子步；推力τ .06–.14s、舵τ .05–.12s。
- target额外舵效×.9，真实舵范围因而约[.7695,.9405]；此缩放需reset hook实际执行（`aav_mpc/h15_target_domain.py:62`）。
- 二阶临界阻尼执行器 `a_ddot=ω²(target−a)−2ω*a_dot`，用半隐式欧拉更新；注意实现对**推力和两舵三个通道**都二阶化，仅τ倍率限两舵（`:28`至`:35`）。所以不能把该消融严格称为“只把舵机改二阶”。
- 艇体改成每0.02s两个显式中点子步，仍调用同一 `model.derivative`；这只改变数值积分器，不是独立物理模型（`:10`至`:19`）。
- 有色体加速度扰动，相关时间2s，标准差[.012,.012,.012,.0008,.0008]；谐波水平流频率.18/.55Hz，幅度受current_max约束（`simulator/model.py:85`，`batch_path_env.py:332`，`ddat_env.py:60`、`:91`）。

因此当前target是有结构差异的sim-to-sim鲁棒性压力环境，仍不能证明对独立高保真水动力或实艇有效。基础参数源码自己将固定失配标为“illustrative…not…identified from a vehicle”（`model.py:44`）。无浮力/重力恢复项、无滚转、无有限攻角舵力/失速、无空化/推进器转速功率模型；有效质量矩阵不对称（`model.py:95`），其来源/坐标推导需要核实。上述是适用范围限制，不能仅凭这些简化断言模型必然错误。

## 3. 已数值复现的两项缺陷及一个覆盖缺口

可复现脚本：`gs12_analysis/control_audit_probe.py`；机器可读结果：`control_audit_probe.json`，含相关源码SHA256。

### 3.1 residual标签的舵τ被重复乘θ

`run_episode.py:59`先调用`nominal_delay_tau(fin_tau_scale=θ[1])`；`residual_predictor.py:16`已经返回`.08*θ[1]`。随后`run_episode.py:127`再次调用`nominal_period_rollout(...fin_tau_scale=θ[1])`；`residual_predictor.py:36`至`:38`在二阶模式下再次乘θ。因此标签实际τ=.08*θ²；θ0=2时为.32s，而`matched_nominal`植物τ=.16s（`run_episode.py:329`），同基线MPC一阶τ也=.16s。离线训练同样重复缩放（`train_predictor.py:36`、`:44`）。

独立真实执行器调用+真实midpoint积分，与生产label函数对照，初始u=15m/s、零流零扰动，单周期命令[800000N,.1rad,−.1rad]：

| 量 | 实际匹配植物 | 生产label rollout |
| --- | ---: | ---: |
| 舵τ | .16s | .32s |
| 周期末升降舵 | .0042724609375rad | .00112152099609375rad |
| 周期末方向舵 | −.0042724609375rad | −.00112152099609375rad |

无外扰仍生成加速度残差约`[−2.13e−9,.00214917,−.00205665,−.000326434,−.000334689]`（前三维m/s²、后两维rad/s²）。只在诊断参考调用中取消第二次τ倍率，真实/预测状态及舵误差均严格0。此探针证实标签污染；未测其对全回合完成率的量化影响。

### 3.2 旧diagnose入口在reset之后注册hook，漏掉target增益/延迟

`aav_mpc/diagnose.py:157`先reset，`:172`才`install_target_domain`。新installer仅在未来reset hook中应用fin_gain和fin_delay（`h15_target_domain.py:62`、`:89`）。`diagnose.run`后续不reset，因此本回合不会发生这些修改；二阶执行器/积分器则安装时立即改变，形成部分target生效。

真实调用`diagnose.run`，N3、1周期、无随机/流/噪声，仅启用fin组件，明确请求gain×.5、delay×.5：安装前、安装后、运行后均为舵增益`.95/.95`、delay`3/3`，且无reset snapshot；再次reset后才变`.475/.475`、delay`2/2`。该结论针对当前代码的入口调用顺序，不追溯否定历史旧源码产生的实验。

### 3.3 已有E0零残差未覆盖舵激励

原残差单测直接计算`predicted−predicted`，无独立植物对照（`tests/unit/test_h15_residual_predictor.py:25`），必然为0。现E0的step/reversal仅改推力action[0]（`run_episode.py:436`），固定train_late_entry前40周期尚无转向。

重新执行现有A0/matched_nominal/off/step/seed71001/40周期得到：最大舵命令0、最大实际舵0、最大残差0；完整输出`gs12_analysis/matched_nominal_e0_existing/episode.json`。这解释了文档`GS-11_H15_POST_REPAIR_IMPLEMENTATION_20261105.md`的0残差与上述τ缺陷可同时成立；此处只实跑了这一个E0组合，未把其它11个组合表述成重新验证过。

## 4. 其余已确认的接口差异与解释边界

- 标签rollout采用二阶执行器和midpoint；被残差修正的MPC采用一阶执行器和RK4。标签可用于相对于指定二阶模型的差值，但不能直接称为“该MPC的完整预测误差”或纯外部扰动（`run_episode.py:127`、`rti_symbolic.py:129`）。即便修复τ双乘，两者结构仍不同。
- 标签使用完整θ_hat，MPC用门控后θ_applied；应明确参数误差和残差支路的定义，尤其当两个α不同。当前label名字`theta_used`易让人误以为它就是MPC实际θ（`run_episode.py:65`、`:115`、`:174`）。
- 低层扰动补偿始终是控制链的一部分，零RL动作不等于纯MPC。其参数基础是全1模型、用测得实际舵每子步积分提取innovation，而不是上述22状态预测（`ddat_env.py:195`、`:244`，`batch_path_env.py:350`）；补偿开/关必须独立配对，避免同一误差重复补偿。
- 当前issued_v2日志确实记录实际发出的饱和后命令，hook位于队列更新之前（`batch_path_env.py:308`，`run_episode.py:238`）；这是已有修复，不能重复报告为未修复。
- 速度合规指标是`dot(Rν,tangent)−target_speed`，不含current_NED，即沿路径的对水速度；参考又扣除了流估计以修正对地路径（`batch_path_env.py:265`，`reference_schedule_probe.py:62`）。论文需要明确15m/s的速度语义。
- `diagnose.predict_accepted_step`的说明声称使用issued命令，实际run在step之前传base_command，step后才读取info[command]（`diagnose.py:193`、`:204`、`:222`）。补偿非零时该旧入口的一步预测还混有命令定义差异；本次未为此单独数值量化。

## 5. 高速机制与重新计算的阶跃结果

使用仓库原函数`aav_mpc.h15_step_response.evaluate`，12/15/18m/s，5°舵阶跃，30s，无流无噪声、固定MISMATCHED_PLANT、一阶执行器。结果在`gs12_analysis/step_response.json`。这不是target二阶执行器测试。

| 初始速度 | 直航配平推力MN | 瞬时偏航加速度增益(rad/s²)/rad | 角速度90%时间s | 该时刻路程m | 实际舵90%时间s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | .633252 | .307307 | 13.80 | 164.246 | .26 |
| 15 | .989456 | .480167 | 11.42 | 169.833 | .26 |
| 18 | 1.424817 | .691441 | 9.64 | 172.013 | .26 |

代码定义的90%是达到末2s平均偏航角速度的90%，并非无限时稳态极限，也不是正式辨识的时间常数（`h15_step_response.py:32`）。推力是初始直航配平后保持固定；转弯过程中速度未被强制固定。

关键机制：舵加速度增益∝u²，15→18增加44%；同半径路径的向心需求也∝u²，不能据此简单声称高速必然更难或更易。直航阻力∝u²，使18m/s直航已耗1.425/1.5MN推力上限，曲率/侧滑可继续增加需求。固定.06s输入延迟在15m/s对应约.9m行进距离；动力学相位误差、输入饱和和有限时域共同影响闭环。现有3.5s预测窗与15m/s阶跃约11.4s的90%响应时间不相等，但不能仅靠这一比较证明窗口必然不足；历史实验已显示更长窗未必更优。

合理但尚未独立证明的解释：二阶临界阻尼`1/(τs+1)²`低频相位延迟约2τ，一阶预测把τ乘2可匹配其低频延迟，可能解释固定保守先验的部分收益。两者高频动态、推力通道和随机延迟仍不同；这只是可检验机制假设。

## 6. 历史实验可引用内容及限度

`aav_mpc/H15_BASELINE_FINAL_20261103.md:98`报告源域29/40、仅舵效×.9为10/40、仅二阶执行器11/40、两者合并4/40；midpoint单改29/40。该历史结果支持执行器/舵效敏感性，当前新hook已导致旧runner语义漂移，重新复现实验前须修正或冻结历史源码，不能用现入口悄悄重跑。

同文`:164`报告冻结保守预测(.85,2)留出14/20，原condensed N35为0/20；仍6个失败，非耦合回归10/10。`:140`的离线无扰可行命令只证明特定模型/路径存在命令；搜索失败不是不可行性证明。

本审阅重新计算了阶跃数字、两个接口缺陷和一项E0覆盖，不重新证明上述完成率、泛化、实时性或真实物理有效性。只有源码和本次小型probe是本次直接证据；历史报告结果必须附其原始运行版本与artifact来源。
