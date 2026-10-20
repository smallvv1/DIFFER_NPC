# PHASE NPC-FP32-MOMENTUM-CONFIRMATORY — 双臂正式性能消融预注册

## 一、已有可核实证据

来源：`PHASE_NPC_MULTIBATCH_RISK_VALIDATION_REPORT.md` 与 `PHASE_NPC_RISK_SIGNAL_AUDIT_REPORT.md`（GitHub，2026-10-20）。

- Best epoch17，16批/60有效参考/4独立身份：继承SGD动量后临时/正式更新方向余弦从0.396699提升到0.943440；参考损失变化 MAE 从0.000326266降至0.000255695。
- Final epoch60，16批/70有效参考/13独立身份：方向余弦从0.825072提升到0.960336，参考MAE从0.000359500降至0.000244705。
- 原始平均风险比值存在双通道抵消，True模式Best 3条/Final 13条可靠参考关系被另一通道改善掩盖；但 any-channel 影子规则在严格1e-4代理阈值下 Final 有14次FP，不应直接替换权重。
- 最近一个完整LTCC Seed1234确认对照：B0最佳Rank-1 50.26%；N2 56.38%，同轮 mAP31.51%；原N3 50.77%，同轮 mAP28.54%。CE batch_mean 单变量N3效果未改善。
- 已在独立的Best/Final真实检查点上完成零学习率FP32-anchor校准；此前混合AMP锚点导致35%–53%有效观测被错误降权，校准后已测零步风险全部为1。
- **上述全部是机制级观察，不是新N3正式60轮性能。**

## 二、当前独立正式实验设计（仅一个训练变量）

固定同一个Git提交、LTCC、EVA02预训练、Seed1234、官方评价和60 epoch，使用两个彼此独立的新训练：

| 实验 | PREAWARE_FP32_ANCHOR | TEMP_COPY_OPT_STATE | 目的 |
|---|---|---|---|
| N3_FP32_ONLY | True | False | 精度校准后的干净基线 |
| N3_FP32_MOMENTUM | True | True | 严格检验SGD动量状态复制的边际收益 |

其余 **GMM 0.99、Memory 0.1、原始tanh权重、CE weight_sum、SGD/LR/数据/身份采样/预训练初始化/模型选择规则** 全部不变。不得合并PR #1(batch_mean)或PR #2(continuous)，不得引入NDP/GPC。两个训练均**从EVA02预训练开始**，绝不使用历史N3 best/final作正式训练初始化。旧B0、N2和N3的60epoch结果只供背景对照，不能作为新动量消融的唯一基线。

相关脚本：
- `scripts/run_npc_fp32_momentum_ablation.py`：默认仅打印命令（dry run），显式 `--execute` 才顺序训练，原子写入manifest，独立输出根目录，不覆盖旧结果，错误时停止并保留日志。
- `scripts/compare_npc_fp32_momentum_ablation.py`：拒绝不完整60epoch、错误seed、非一致配置、缺失best/final checkpoint、更新计数不一致；选best Rank1并只报告**同轮**mAP。
- `tests/test_npc_fp32_momentum_formal_ablation.py`：CPU-only测试脚本命令、对照配置、数据完整性与best同轮评估。

## 三、第二台PPU服务器执行流程（需要在服务器检查资源）

### 前置条件

1. 独立worktree检出 `experiment/npc-fp32-momentum-formal-ablation-20261020`，不得对旧工作目录覆盖。
2. 检查正在执行的PPU任务、磁盘空间、旧 `processor` 导入、EVA02预训练来源、LTCC train/query/gallery协议、caption特征数据，确认没有其它任务被干扰。
3. 先在正确的Python环境执行：
   ```bash
   PYTHONPATH=.:/home/DIFFER_NPC /usr/local/bin/python3 -m pytest -q \
      tests/test_npc.py tests/test_npc_temp_sgd_state.py \
      tests/test_npc_fp32_momentum_formal_ablation.py
   ```
   环境如需补充PPU SDK请沿用服务器先前通过验证的设置，不要修改系统驱动。
4. 从独立worktree根目录运行：
   ```bash
   /usr/local/bin/python3 scripts/run_npc_fp32_momentum_ablation.py \
      --python /usr/local/bin/python3
   ```
   这一步只打印、检查两条训练命令，不会消耗PPU算力。
5. 启动短迭代预检（会用PPU）：
   ```bash
   /usr/local/bin/python3 scripts/run_npc_fp32_momentum_ablation.py \
      --python /usr/local/bin/python3 --execute --smoke-iterations 30
   ```
   两组各跑2个epoch、每轮最多30个迭代；检查无NaN/Inf、模型与优化器不污染、CPU/PPU运行态、有效参考与权重是否实际非平凡、每轮记录和检查点隔离。短迭代**不能保证在早期形成有效参考**，因此还应沿用已完成的Best/Final真实模型机制测试，不能以空参考 smoke 代替全部动态路径覆盖。
6. 上述全部通过、且资源允许时，明确授权并在安全窗口开始正式60轮；该命令**顺序执行120 epoch**，耗时可能很长：
   ```bash
   /usr/local/bin/python3 scripts/run_npc_fp32_momentum_ablation.py \
      --python /usr/local/bin/python3 --execute
   ```
   每个训练有独立 `N3_FP32_ONLY`/`N3_FP32_MOMENTUM` 根目录、60epoch CSV、best/final checkpoint和独立日志。任何失败必须终止后续训练，不得绕过失败检查强行续训。
7. 两组均真实完成后，执行：
   ```bash
   /usr/local/bin/python3 scripts/compare_npc_fp32_momentum_ablation.py \
      --experiment-root '<上一步launcher打印的正式实验根目录>'
   ```
   生成 `PHASE_NPC_FP32_MOMENTUM_CONFIRMATORY_REPORT.md` 和 `comparison_summary.json`。
8. 保存实际代码SHA、服务器输出、关键配置/指标和中文报告至**本PR分支**。不要推送大型checkpoint到GitHub，除非服务器存储与仓库策略明确允许。

## 四、数据质量和因果解释限制

- 训练的现有 `updates` 字段是循环计数，未必等同AMP GradScaler成功执行的优化器步数；在论文中不能直接将它称为无跳步更新数。需对AMP scale下降/非有限梯度另做审计，如存在偏差先核对两臂是否一致。
- Best与Final两套机制审计的参考样本数和PID覆盖不同；不等于实际60epoch全部阶段覆盖。
- 只有当 True 优于同批次协议的 FP32_ONLY 时，才能初步支持SGD动量继承的性能价值。即便 True 超过历史N2，亦必须做第二种子和PRCC扩展才能声称稳定。
- 两组共享原始风险函数，即本实验**不检验双通道影子规则是否能改进检索性能**；影子规则有显著误报，不在此实验引入。
- 用户的NDP是另一台服务器的研究创新方向，不能混到本NPC实验或宣称已做异身份邻域保护。

## 五、当前执行状态

本PR仅提交安全启动器、配置/结果审计及测试；**尚未在用户服务器运行新测试、smoke或120epoch正式实验**。不能将本PR作为检索性能已提升的证据。
