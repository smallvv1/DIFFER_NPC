# PHASE NPC-RISK-SIGNAL-AUDIT：双通道风险抵消与误降权审计

本报告仅对已有 LTCC train-only 多批次结果进行离线分析；
没有修改模型、GMM、记忆库、动态权重、损失或 checkpoint。
所谓有害是正式更新后的可靠参考身份CE或跨服装一致性损失恶化代理，
**并不等同于异身份邻域被破坏，也不代表最终 Rank-1/mAP 性能损失**。

比较 original_mean_ratio、identity_only、consistency_only、
either_channel、both_channels 五种**只读决策规则**，
后四种只作为研究诊断，不是自动采用的新 NPC 方法。

## 来源：`/home/DIFFER_NPC_worktrees/npc-risk-signal-audit-20261020/outputs/PHASE_NPC_TEMP_OPTIMIZER_MULTIBATCH_PPU_20261020_retry/temp_optimizer_multibatch_ppu.json`

检查点：`/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth`
SHA256：`0b33bc4d02b70a1d8d83a7ecdc8dd801a479c307685fa537a39276e2188ee7db`
真实配对批次数：16

| 动量继承 | 有效观测 | 独立身份 | p/q相反方向 | 均值抵消风险 | 真正降权<0.9 |
|---|---:|---:|---:|---:|---:|
| False | 60 | 4 | 3 | 1 | 2 |
| True | 60 | 4 | 21 | 3 | 19 |

各个阈值均为预先固定的 0、1e-5、1e-4，不依据该日志选择权重超参。

### TEMP_COPY_OPT_STATE_FALSE

| 损失恶化阈值 | 风险规则 | TP/FP/FN/TN | Precision | Recall |
|---:|---|---|---:|---:|
| 0.0 | original_mean_ratio | 2/0/26/32 | 100.0% | 7.1% |
| 0.0 | identity_only | 0/0/28/32 | N/A | 0.0% |
| 0.0 | consistency_only | 3/0/25/32 | 100.0% | 10.7% |
| 0.0 | either_channel | 3/0/25/32 | 100.0% | 10.7% |
| 0.0 | both_channels | 0/0/28/32 | N/A | 0.0% |
| 1e-05 | original_mean_ratio | 2/0/25/33 | 100.0% | 7.4% |
| 1e-05 | identity_only | 0/0/27/33 | N/A | 0.0% |
| 1e-05 | consistency_only | 3/0/24/33 | 100.0% | 11.1% |
| 1e-05 | either_channel | 3/0/24/33 | 100.0% | 11.1% |
| 1e-05 | both_channels | 0/0/27/33 | N/A | 0.0% |
| 0.0001 | original_mean_ratio | 0/2/22/36 | 0.0% | 0.0% |
| 0.0001 | identity_only | 0/0/22/38 | N/A | 0.0% |
| 0.0001 | consistency_only | 1/2/21/36 | 33.3% | 4.5% |
| 0.0001 | either_channel | 1/2/21/36 | 33.3% | 4.5% |
| 0.0001 | both_channels | 0/0/22/38 | N/A | 0.0% |

注意：规则间的混淆矩阵是假设性诊断，不等于实际训练后的变化；
不同阈值对应不同代理标签，不能把某个阈值下的Precision/Recall与另一个阈值直接混为同一任务。

### TEMP_COPY_OPT_STATE_TRUE

| 损失恶化阈值 | 风险规则 | TP/FP/FN/TN | Precision | Recall |
|---:|---|---|---:|---:|
| 0.0 | original_mean_ratio | 16/3/12/29 | 84.2% | 57.1% |
| 0.0 | identity_only | 3/0/25/32 | 100.0% | 10.7% |
| 0.0 | consistency_only | 17/3/11/29 | 85.0% | 60.7% |
| 0.0 | either_channel | 19/3/9/29 | 86.4% | 67.9% |
| 0.0 | both_channels | 1/0/27/32 | 100.0% | 3.6% |
| 1e-05 | original_mean_ratio | 15/4/12/29 | 78.9% | 55.6% |
| 1e-05 | identity_only | 3/0/24/33 | 100.0% | 11.1% |
| 1e-05 | consistency_only | 16/4/11/29 | 80.0% | 59.3% |
| 1e-05 | either_channel | 18/4/9/29 | 81.8% | 66.7% |
| 1e-05 | both_channels | 1/0/26/33 | 100.0% | 3.7% |
| 0.0001 | original_mean_ratio | 12/7/10/31 | 63.2% | 54.5% |
| 0.0001 | identity_only | 3/0/19/38 | 100.0% | 13.6% |
| 0.0001 | consistency_only | 13/7/9/31 | 65.0% | 59.1% |
| 0.0001 | either_channel | 15/7/7/31 | 68.2% | 68.2% |
| 0.0001 | both_channels | 1/0/21/38 | 100.0% | 4.5% |

注意：规则间的混淆矩阵是假设性诊断，不等于实际训练后的变化；
不同阈值对应不同代理标签，不能把某个阈值下的Precision/Recall与另一个阈值直接混为同一任务。

## 来源：`/home/DIFFER_NPC_worktrees/npc-risk-signal-audit-20261020/outputs/PHASE_NPC_TEMP_OPTIMIZER_MULTIBATCH_PPU_20261020_final/temp_optimizer_multibatch_ppu.json`

检查点：`/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_final.pth`
SHA256：`e275cdcabc199988fbe5f77a17e3a70abcf9c28b4c03bb5d70d0befe5b08137b`
真实配对批次数：16

| 动量继承 | 有效观测 | 独立身份 | p/q相反方向 | 均值抵消风险 | 真正降权<0.9 |
|---|---:|---:|---:|---:|---:|
| False | 70 | 13 | 8 | 7 | 6 |
| True | 70 | 13 | 20 | 13 | 19 |

各个阈值均为预先固定的 0、1e-5、1e-4，不依据该日志选择权重超参。

### TEMP_COPY_OPT_STATE_FALSE

| 损失恶化阈值 | 风险规则 | TP/FP/FN/TN | Precision | Recall |
|---:|---|---|---:|---:|
| 0.0 | original_mean_ratio | 6/0/44/20 | 100.0% | 12.0% |
| 0.0 | identity_only | 10/1/40/19 | 90.9% | 20.0% |
| 0.0 | consistency_only | 7/0/43/20 | 100.0% | 14.0% |
| 0.0 | either_channel | 12/1/38/19 | 92.3% | 24.0% |
| 0.0 | both_channels | 5/0/45/20 | 100.0% | 10.0% |
| 1e-05 | original_mean_ratio | 6/0/42/22 | 100.0% | 12.5% |
| 1e-05 | identity_only | 9/2/39/20 | 81.8% | 18.8% |
| 1e-05 | consistency_only | 7/0/41/22 | 100.0% | 14.6% |
| 1e-05 | either_channel | 11/2/37/20 | 84.6% | 22.9% |
| 1e-05 | both_channels | 5/0/43/22 | 100.0% | 10.4% |
| 0.0001 | original_mean_ratio | 6/0/25/39 | 100.0% | 19.4% |
| 0.0001 | identity_only | 8/3/23/36 | 72.7% | 25.8% |
| 0.0001 | consistency_only | 7/0/24/39 | 100.0% | 22.6% |
| 0.0001 | either_channel | 10/3/21/36 | 76.9% | 32.3% |
| 0.0001 | both_channels | 5/0/26/39 | 100.0% | 16.1% |

注意：规则间的混淆矩阵是假设性诊断，不等于实际训练后的变化；
不同阈值对应不同代理标签，不能把某个阈值下的Precision/Recall与另一个阈值直接混为同一任务。

### TEMP_COPY_OPT_STATE_TRUE

| 损失恶化阈值 | 风险规则 | TP/FP/FN/TN | Precision | Recall |
|---:|---|---|---:|---:|
| 0.0 | original_mean_ratio | 19/0/31/20 | 100.0% | 38.0% |
| 0.0 | identity_only | 27/0/23/20 | 100.0% | 54.0% |
| 0.0 | consistency_only | 17/0/33/20 | 100.0% | 34.0% |
| 0.0 | either_channel | 32/0/18/20 | 100.0% | 64.0% |
| 0.0 | both_channels | 12/0/38/20 | 100.0% | 24.0% |
| 1e-05 | original_mean_ratio | 19/0/29/22 | 100.0% | 39.6% |
| 1e-05 | identity_only | 26/1/22/21 | 96.3% | 54.2% |
| 1e-05 | consistency_only | 16/1/32/21 | 94.1% | 33.3% |
| 1e-05 | either_channel | 30/2/18/20 | 93.8% | 62.5% |
| 1e-05 | both_channels | 12/0/36/22 | 100.0% | 25.0% |
| 0.0001 | original_mean_ratio | 12/7/19/32 | 63.2% | 38.7% |
| 0.0001 | identity_only | 17/10/14/29 | 63.0% | 54.8% |
| 0.0001 | consistency_only | 9/8/22/31 | 52.9% | 29.0% |
| 0.0001 | either_channel | 18/14/13/25 | 56.2% | 58.1% |
| 0.0001 | both_channels | 8/4/23/35 | 66.7% | 25.8% |

注意：规则间的混淆矩阵是假设性诊断，不等于实际训练后的变化；
不同阈值对应不同代理标签，不能把某个阈值下的Precision/Recall与另一个阈值直接混为同一任务。

## 下一步决策

1. 报告 p、q 预测方向相反及均值抵消风险的数量，检查是否导致真实有害参考被遗漏。
2. 检查继承动量后新增降权是否真正对准损害严重的样本，还是主要落在极小损失变化附近。
3. 本报告不允许直接把 either_channel 或新阈值引入正式 N3；若确有潜力，需另外开分支做独立预注册消融。
4. 当前证据只覆盖少量身份，不可宣称 NPC 已超过 N2，也不可当成 NDP 邻域异身份风险结论。
