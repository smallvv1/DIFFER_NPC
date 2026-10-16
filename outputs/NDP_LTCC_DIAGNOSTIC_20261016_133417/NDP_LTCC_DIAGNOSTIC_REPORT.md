# DIFFER + GPC + NDP：LTCC 局部机制短程验证报告

## 1. 结论摘要

本阶段只做了 LTCC train-only 参考库、PPU 短程更新和离线风险诊断，没有启动正式 60 epoch，也没有运行官方 query/gallery 评估。

结论是：**NDP 值得继续做受控机制开发，但当前不能进入正式训练，也不能宣称已经有效。**

- 局部结构变化信号存在：GPC-only 在 64 步时固定诊断集的平均局部间隔下降 `0.009212`，正样本局部 Rank-1 平均下降 `4.51` 个百分点，异身份侵入率平均增加 `4.51` 个百分点。
- 独立 NDP 局部间隔损失确实激活并反向传播：有效 anchor 中平均约 `60.28%` 的 hinge 为正；原始 NDP 梯度探针非零，范围约 `3.48—61.05`。
- NDP+GPC 在 64 步的平均局部间隔恶化小于 GPC-only：`-0.007414` 对 `-0.009212`；跨衣正样本相似度变化为 `+0.000497` 对 `-0.009270`；但 3 次重复中只有 2 次方向有利，1 次明显恶化，因此属于 **混合的短程机制证据**，不是稳定收益。
- 更新前局部间隔不能预测后续恶化：Spearman `-0.0033`，ROC-AUC `0.4632`，接近随机；风险引导权重没有启用。
- 首次 AMP 短程在第 3 步出现非有限梯度。随后使用同一 PPU、同一 batch、同一 GPC selector/base loss 的 FP32 技术恢复，3 次重复均完成 `64/64` optimizer update、无 NaN/Inf。

因此，本阶段不建议直接训练 NDP 正式模型；建议先修正参考更新/局部损失定义并完成更强的重复验证。

## 2. 冻结对象与范围

| 项目 | 内容 |
|---|---|
| 实际工作区 | `/home/DIFFER_NPC`；用户指定的 `/home/DIFFER` 在当前环境不存在 |
| GPC checkpoint | `/home/DIFFER_NPC/outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/model_final.pth` |
| checkpoint | E60，SHA256 `51e94a312d09d4e172afb8c04d8988e6cf9ad3b8cf4c82bbdf003562a6f6da26` |
| 参考库 | `/home/DIFFER_NPC/outputs/NDP_LTCC_DIAGNOSTIC_20261016_133417/参考库/` |
| 成功诊断输出 | `/home/DIFFER_NPC/outputs/NDP_LTCC_DIAGNOSTIC_20261016_133417/` |
| 中文日志 | `NDP_LTCC短程诊断_20261016_133417.log` |
| 训练范围 | LTCC train split only；未读取 query/gallery 特征 |
| 重复与步数 | 3 次重复；16、32、64 步；GPC 与 GPC+NDP 共 6 个短程 arm |
| 正式训练 | 未启动；历史 checkpoint 未覆盖 |

GPC-v1 的 selector、pair-budget、`L_GPC=mean(1-cos)` 和 `lambda_GPC=0.1` 没有改写。NDP 通过独立脚本在 GPC 总损失之外添加探针项。

## 3. 修改文件及代码位置

新增文件：

- [`ndp/local_margin.py`](../../ndp/local_margin.py:27)：train-only 参考库、同身份跨衣正样本、K=10 异身份困难邻居、局部几何统计、独立局部间隔保护损失。
- [`ndp/__init__.py`](../../ndp/__init__.py:1)：NDP 诊断组件导出。
- [`ndp/test_local_margin.py`](../../ndp/test_local_margin.py:1)：自引用为 0 和非零梯度的最小测试，`2/2` 通过。
- [`scripts/ndp_ltcc_short_diagnostic.py`](../../scripts/ndp_ltcc_short_diagnostic.py:292)：不改 GPC 源码的短程入口；包含 PPU 特征提取、固定 batch、匹配 arm、更新前后几何和离线风险诊断。

历史 GPC 文件没有修改。成功目录中 `代码与检查点冻结.json` 保存了 GPC 训练入口、loss、model、dataset、配置和 NDP 诊断入口的 SHA256。

## 4. train-only 局部参考库

参考库规则是预先固定的：

1. 正样本必须满足 `same PID`、`different clothes_id`、`index != anchor`；在有效候选中选择初始 checkpoint cosine 最高者。
2. 负样本必须满足 `different PID`；从同一 train-only 初始特征库中选择 cosine 最高的 K=10 个邻居。
3. 没有有效跨衣正样本的 anchor 标记为无效，不回退到自身，也不伪造监督。

| 统计量 | 结果 |
|---|---:|
| train 图像数 | 9,576 |
| train PID 数 | 77 |
| 至少有跨衣参考的 PID 数 | 46/77（59.74%） |
| 只有单一服装状态的 PID 数 | 31/77 |
| 图像级可靠正样本覆盖率 | 87.3747% |
| 完整 K=10 参考覆盖率 | 87.3747% |
| 自引用数量 | **0** |
| 初始跨衣正样本相似度（固定 96 anchor） | 0.791960 |
| 初始 hard-negative 相似度（固定 96 anchor） | 0.743482 |
| 初始局部间隔 | 0.048478 |
| 初始正样本 Rank-1 | 79.17% |
| 初始异身份侵入率 | 20.83% |

这里同时报告 image-level 和 PID-level 覆盖率。31 个单服装 PID 没有可靠跨衣正样本，属于明确无效参考，不能参与 NDP 监督。

## 5. 局部结构变化：GPC-only 对照

固定 96 个有效 anchor，在更新前后使用同一组正样本和 K=10 负邻居。`delta` 相对冻结 GPC checkpoint 的初始几何。

| arm | 步数 | Δ局部间隔均值 | 重复标准差 | Δ跨衣正相似度 | Δ正样本 Rank-1 | Δ侵入率 |
|---|---:|---:|---:|---:|---:|---:|
| GPC | 16 | -0.000036 | 0.001145 | -0.001027 | -2.08 pp | +2.08 pp |
| GPC | 32 | -0.001955 | 0.000443 | -0.003353 | -0.69 pp | +0.69 pp |
| GPC | 64 | -0.009212 | 0.005523 | -0.009270 | -4.51 pp | +4.51 pp |

16 步信号低于重复噪声；32 步平均间隔变化的绝对值约为重复标准差的 `4.41` 倍；64 步约为 `1.67` 倍。说明 32/64 步存在可观测结构变化，但 64 步的重复波动仍然不小。

## 6. NDP 局部间隔保护损失与严格匹配比较

本轮探针公式为：

```text
L_NDP = mean(relu(margin + max_k cos(anchor, negative_k) - cos(anchor, positive)))
margin = 0.05
lambda_NDP_probe = 0.1
```

参考特征来自冻结的初始 train-only 库，梯度只从当前 batch anchor 反传。这是一个独立的局部保护探针，不是完整动态 NDP 实现；没有风险预测权重，也没有在线更新参考特征。

| 量 | 结果 |
|---|---:|
| 有效 anchor 观测总数 | 2,704（GPC arm 的诊断计算） |
| hinge 激活总数 | 1,640 |
| hinge 激活率 | 60.65% |
| GPC+NDP 有效 anchor 观测 | 1,352 |
| GPC+NDP hinge 激活率 | 60.28% |
| NDP 原始梯度 L2 探针范围 | 3.48—61.05 |
| NDP 原始梯度 L2 中位数 | 8.11（GPC arm），7.81（GPC+NDP arm） |
| GPC selector active batch | 每个 64 步重复为 3—5 个，说明 GPC 本身仍是稀疏曝光 |

NDP loss 不是“数值合法但从未进入计算图”：它在有效参考上真实产生了 hinge 和非零梯度。另一方面，GPC+NDP 的平均结果如下：

| arm | 步数 | Δ局部间隔均值 | 重复标准差 | Δ跨衣正相似度 | Δ正样本 Rank-1 | Δ侵入率 |
|---|---:|---:|---:|---:|---:|---:|
| GPC+NDP | 16 | -0.000557 | 0.001209 | -0.000630 | -2.78 pp | +2.78 pp |
| GPC+NDP | 32 | -0.003420 | 0.001271 | +0.000014 | -2.08 pp | +2.08 pp |
| GPC+NDP | 64 | -0.007414 | 0.005618 | +0.000497 | -2.08 pp | +2.08 pp |

64 步相对于 GPC-only 的平均差异为：局部间隔多保留 `+0.001798`，跨衣正相似度多保留 `+0.009767`，正样本 Rank-1 少下降 `2.43` 个百分点，侵入率少增加 `2.43` 个百分点。但逐重复 64 步 Δ局部间隔为：

- repeat 1：NDP `-0.015331`，GPC `-0.006087`，NDP 更差；
- repeat 2：NDP `-0.002882`，GPC `-0.004575`，NDP 更好；
- repeat 3：NDP `-0.004030`，GPC `-0.016974`，NDP 更好。

所以当前证据只能说明“在本设置和 3 次重复中，NDP 有潜在保护信号但存在明显重复依赖”；不能宣称稳定有效或已证明因果收益。

## 7. 更新前动态风险预测

以更新前局部间隔作为风险分数，之后的局部间隔下降作为恶化标签，使用 864 个 GPC-only anchor-step 观测做离线诊断：

| 指标 | 结果 |
|---|---:|
| Spearman | -0.0033 |
| ROC-AUC | 0.4632 |
| 风险引导权重 | **未启用** |

当前风险预测没有超过随机水平，因此不能把更新前分数用于动态加权。`risk/risk_diagnostic.json` 和 `risk/pre_update_risk_rows.csv` 保存了原始记录。

## 8. PPU、AMP 与训练正确性

- 当前环境只暴露 1 张 `PPU-ZW810E`，未宣称双卡结果。
- 初次 AMP 短程输出目录：`outputs/NDP_LTCC_DIAGNOSTIC_20261016_132828/`；第 3 步梯度非有限，已保留失败日志。
- FP32 技术恢复目录：`outputs/NDP_LTCC_DIAGNOSTIC_20261016_133417/`。
- FP32 恢复中 6 个 arm 均完成 64/64 optimizer update。
- 6 个 arm 的 loss、gradient、optimizer 参数更新均在 PPU；最终短程无 NaN、无 Inf、无跳步。
- query/gallery 没有进入参考特征、NDP loss、GPC loss、optimizer 或风险分析。

当前工作区还缺少历史正式入口 import 时才需要的 `processor` 包。诊断脚本使用了不可调用的 evaluator 占位，只复用 train-only sampler、GPC selector 和 base loss；若误调用正式 evaluator 会主动报错。该兼容处理没有修改 GPC 源码。

另外，当前服务器缺少正式配置曾引用的 `train_caption_summary_biometric.json`，短程采用仓库基础配置的 `SUMMERY_TEXT=false` 读取 `train.npz`。因此本报告的 FP32 短程可用于机制/梯度诊断，但不应当被当作与历史正式 GPC 训练完全同配置的性能复现实验。

## 9. 与完整 NDP 机制的边界

本阶段验证的是“冻结 train-only 局部参考 + 独立局部间隔保护损失”的有效激活和短程结构影响。它没有验证以下内容：

1. 参考条目是否应当随临时模型同步更新；
2. NDP 原论文意义上的在线负面影响估计是否等价于本轮的冻结参考 hinge；
3. 更新前风险预测能否泛化到不同 batch、seed 或训练阶段；
4. 正式 LTCC Rank-1/mAP 是否改善。

因此不能把 NPC 历史结果、NPC 动态权重或 NPC 预感知结果直接视为 NDP 结果。本阶段没有移植 NPC 动态权重。

## 10. 下一步建议

1. 先恢复缺失的 `train_caption_summary_biometric.json` 或冻结一份与历史 GPC 完全一致的 caption 资产，再做正式 NDP 对照。
2. 对 NDP 局部损失做机制拆分：分别记录正项、hard-negative 项和总间隔项的梯度，确认 repeat 1 的恶化是否来自异身份项上升，而不是跨衣正样本不足。
3. 比较冻结参考、临时模型当前参考和双向当前参考三种计算对象；这一步只做诊断，不立即替换算法。
4. 扩大重复数至至少 5—10 次，并固定相同 batch/augmentation 序列；以“重复方向一致 + 局部间隔/侵入率同时改善”为 promotion gate。
5. 在风险 AUC 接近随机前，不启动任何风险引导权重；先增加 pre-update 特征、局部负样本变化和 batch-level context 的离线预测变量。
6. 只有局部保护信号在严格匹配短程中稳定后，才进行小规模 epoch 级训练；正式 60 epoch 前仍需恢复 AMP 或解释 AMP found_inf，并重新完成 PPU preflight。

## 11. 最终判定

**NDP：继续开发（机制证据 MIXED，暂不具备正式训练条件）。**

有效的是：train-only 参考库关系约束、无自引用、K=10 困难负样本、NDP hinge 的真实激活和 PPU 梯度传播，以及 GPC 64 步局部恶化信号的可观测性。

存在问题的是：跨衣参考只覆盖 46/77 个 PID；GPC 曝光稀疏；NDP 保护效果依赖重复，无法稳定减少局部恶化；更新前风险预测接近随机；AMP 数值稳定性尚未解决；当前冻结参考探针不能等同于完整 NDP。

本阶段不支持启动 NDP 正式 60 epoch，也不支持报告 LTCC 检索性能提升。
