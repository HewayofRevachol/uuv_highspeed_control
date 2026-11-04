# H15 Gate2 PPO 暂停接手

更新时间：2026-11-04。用户要求完成当前一轮、写接手文档并停止。本文件记录停点；
总目标和 Gate 顺序仍以 H15_MAINLINE_HANDOFF_20261104.md 与
H15_RECOMMENDED_MAINLINE_EXPERIMENT_PLAN_QUICKSTART_20261104.md 为准。
所有 Python 命令使用 /bin/python3。用户优先检验离线算法性能；5 ms 筛选预算只限制
在线可用性主张，不阻断离线配对。不得用 test 选择策略或调参。

## 当前结论和勾选状态

- Gate0、Gate1 已通过。Gate2 的 E1、确定性治理、五动作 PPO 接口、冻结三种子奖励、
  train 期 B0/B2 同步记录子项已勾选。
- Gate2 2.3 的“完整评估尝试”和“只按 dev 选择 checkpoint”尚未勾选；
  Gate2 总项、Gate3 及以后尚未通过。不要把下述初始策略评分当作最终策略选择。
- 三个策略种子 41001、41002、41003 均完成冻结的四个 train 回合。
  4 回合 × 3 种子 × 3 方法共 36 条 train 轨迹已核对：全部到终点、
  无已记录物理违约；12 条 B3 的动作掩码审计文件位于各正式运行目录。
- 当前“最近一轮”是三个种子的初始 checkpoint 在 dev manifest 上各 6 个回合。
  18 条 B3 dev 轨迹均已到终点，无已记录物理违约；审计检查了运行身份、
  冻结输入/奖励哈希、策略 checkpoint、动作掩码、候选、备份、状态/执行器/队列
  连续性、参考斜坡及 0.02 s 子步数量。各种子的 dev 分数重新从六条 summary
  计算并与 dev_scores.json 的 initial 项相同：
  [0, 6, -540, 1790.0030402675754, -2.032394920670706, -5.759795823938358]。
  分数依次为严格完成数、物理完成数、负备份周期、总进度、各回合最大横向误差
  之和的负数、各回合最大实际速度误差之和的负数；18 条合计 1620 个备份周期。
  三个种子的分数完全相同，
  不能据此推断训练后策略的效果。
- 耦合 dev 三回合每种子分别有 171、174、180 个空候选周期，备份为
  171、174、184 周期；反转 dev 三回合每种子合计 11 个备份周期。
  严格主线完成数为零，不能将备份完成称为在线成功。

## 运行目录与停止状态

正式运行目录分别为：

- outputs/experiments/rl_uuv_h15_train_s41001/full_obs60_20261104/
- outputs/experiments/rl_uuv_h15_train_s41002/full_obs60_20261104/
- outputs/experiments/rl_uuv_h15_train_s41003/full_obs60_20261104/

每个目录有 train_history.json、policy_initial.pt、policy_after_0.pt 至
policy_after_3.pt、audit_train_0_b3.json 至 audit_train_3_b3.json、
六个 dev_initial_*_B3_RL/ 和 dev_scores.json。dev_scores.json 目前只有
initial 项。没有 dev_trained_* 目录，也没有 selection.json 或全局策略选择。

原训练会话 33480、17185、11553 在写出各自 dev initial 六回合评分后，
按用户停止要求由 Ctrl-C 结束，退出码均为 130；这不是实验失败。
三组会话均已关闭。不要重启原 train_masked_ppo 命令覆盖现有目录，
也不要将旧 34 维诊断目录 full_train_dev_20261104 用于选择。

## 下一个智能体的续接顺序

1. 先确认用户已解除停止要求，再核查上述三个目录、dev_scores.json 仅有
   initial 项以及没有 dev_trained_* 和 selection.json。若已出现新输出，
   先审计它们，不要盲目重跑。
2. 只补跑三个 policy_after_3.pt 的 dev 六回合，共 18 条。
   保持 train 脚本原有 manifest 顺序：dev_reversal、dev_coupled；
   各路径的植物/扰动种子为 32001、32002、32003。使用同一冻结 E1、
   候选配置、奖励、共同备份与 deterministic 策略。
   可按下面模板逐一执行；输出路径存在时先核验，不可覆盖。

~~~bash
for policy_seed in 41001 41002 41003; do
  run="outputs/experiments/rl_uuv_h15_train_s$policy_seed/full_obs60_20261104"
  for case_id in dev_reversal dev_coupled; do
    for plant_seed in 32001 32002 32003; do
      out="$run/dev_trained_${case_id}_s${plant_seed}_B3_RL"
      if [ -e "$out" ]; then
        printf 'Existing output requires audit: %s\n' "$out"
        exit 1
      fi
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        /bin/python3 -m experiments.h15_mainline.run_offline_governor_smoke \
        --freeze outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json \
        --checkpoint outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_fullplan_20261104/checkpoint.json \
        --config experiments/h15_mainline/candidate_filter_fullplan_config.json \
        --split dev --case-id "$case_id" --plant-seed "$plant_seed" \
        --method B3_RL --backup-mode one_step_verified --estimator-mode E1 \
        --policy-checkpoint "$run/policy_after_3.pt" \
        --reward-config experiments/h15_mainline/reward_v1.json \
        --policy-seed 0 --out "$out"
    done
  done
done
~~~

3. 用 experiments/h15_mainline/audit_gate2_training.py 的 check_summary
   逐条审计新轨迹。按照 train_masked_ppo.py 中 dev_score()、main() 末段的
   固定算法，保留 dev_scores.json 的 initial 项，增加 trained 项，
   每种子只在 initial/trained 间按六回合 dev 分数选 checkpoint，
   写 selection.json。不得使用 test。建议写一个可恢复的小脚本完成这一
   机械续接，核对 checkpoint SHA 和六个回合数量；勿改动训练时哈希冻结的
   masked_speed_ppo.py、run_offline_governor_smoke.py、train_masked_ppo.py。
4. 三个种子的 selection.json 齐全后，运行
   experiments.h15_mainline.audit_gate2_training --runs（上述三个目录）
   --out（新 Gate2 审计 JSON），再运行
   experiments.h15_mainline.select_policy_checkpoint --runs（同三个目录）
   --out（新全局 dev-only 选择 JSON）。写 Gate2 报告，检查完整评估尝试、
   首次违约、最大时域、空候选和 fallback；仅在证据满足条件后勾选 Gate2
   剩余子项与总项。
5. Gate2 通过后才正式运行 Gate3。已备好
   experiments/h15_mainline/run_gate3_paired.py：输入 Gate2 审计 JSON、
   全局选择 JSON、全新输出目录；它将在同一六回合 dev manifest 下
   串行运行 B0/B2/B3 并写 dev_paired_results.csv。该脚本此前仅做语法及
   单条字段检查，尚未正式运行。若 Gate3 发现 B3 与 B2 持平、
   候选大部分为空或只靠降速改善误差，按 quickstart 停止规则报告，
   不扩大训练规模掩盖结果。Gate4 的 test 留出仍未读取。

## 核验和工作区注意

- 批量审计工具为 experiments/h15_mainline/audit_gate2_training.py；
  已用它核对全部 36 条 train 和本轮 18 条 dev initial 轨迹。
  完整 audit_seed() 需要 trained dev 和 selection.json，当前不能运行成功。
- H15 单元检查此前为 31 passed、28 warnings；新增两个审计/配对脚本
  已用 /bin/python3 -m py_compile 检查。继续前重新运行相关检查及
  git diff --check。
- 当前工作区包含此前阶段的未提交代码、数据和文档。不要重置、覆盖
  已有输出或重新运行冻结训练。实时筛选仍超 5 ms；只报告离线机制效果。

