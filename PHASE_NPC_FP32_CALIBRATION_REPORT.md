# PHASE NPC FP32 锚点校准报告

审计时间：2026-10-19（北京时间）
分支：`experiment/npc-fp32-anchor-preaware-20261019`
独立工作树：`/home/DIFFER_NPC_worktrees/npc-fp32-anchor-preaware-20261019`
PR 基线提交：`4e1a54a56dbc48d0ce1d52ab5f0480fa55612ee6`
PPU：PPU-ZW810E × 1；PyTorch 2.4.0

## 结论摘要

真实 LTCC train-only、Seed1234、零临时学习率的成对 PPU 审计支持：预更新锚点与临时更新后锚点在精度和 batch membership 上不一致，会让零更新被误判为负面影响。epoch17 best checkpoint 的 AMP 锚点路径有 39.47% 的有效样本满足 `r<1`，并被原始 tanh 权重函数降到 `w<0.9`；显式 `--fp32-anchor` 后，该 checkpoint 的 152/152 个有效记录均为 `p=q=r=w=1`。epoch60 final checkpoint 的短程复核也从 35% 的严格误判降至 0%。

这只是零步数值正确性结论，不代表 FP32-anchor 已提高检索性能。GMM 可靠参考身份覆盖只有 4/77（5.19%），结论不能外推到全部身份或训练过程。没有启动任何 60 epoch 训练，也没有引入数值中性区间。

## 隔离、进程和测试

- 检查开始前未发现训练、torchrun 或 NPC 诊断进程；之后四次审计均顺序运行，未打断其他任务。主分支及其预存未跟踪文件未被修改，所有工作均在独立 worktree 完成。
- 数据来自 `/home/data/LTCC_ReID` 和 `/home/data/DIFFER_TextCaption/LTCC_ReID`，诊断只读取 LTCC train split（9,576 张、77 个身份）；没有读取 query/gallery，也没有保存或覆盖检查点。
- 完整 NPC 测试：`PYTHONPATH=/home/DIFFER_NPC python3 -m pytest -q tests/test_npc.py`，结果 **12 passed**。直接在 worktree 收集测试时，隔离目录缺少共享 `data` 包导致一次导入失败；使用主工作区只读代码作为 `PYTHONPATH` 后测试完整通过。分支 `tests/` 下的 NPC 测试文件只有 `test_npc.py`。
- PPU 环境：`PPU_HOME=/usr/local/PPU_SDK`、`PPU_SDK=/usr/local/PPU_SDK`，`LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:/usr/local/PPU_SDK/sailSHMEM/lib`；没有 CPU 模型回退。

## 检查点、样本和配对协议

所有审计均使用 seed 1234、原 N3 配置、GMM 阈值 0.99、临时优化步数 1、临时学习率比例 0；唯一处理变量是默认关闭的 `MODEL.NPC.PREAWARE_FP32_ANCHOR`。每次诊断都在真实 LTCC train-only 数据上重建记忆库，并用完全相同的采样种子与 train-loader 批次。

| 检查点 | GMM epoch | SHA256 | 批次数 × 重复数 | no-flag / flag 有效记录 |
|---|---:|---|---:|---:|
| `model_best.pth`（历史 best，epoch17） | 17 | `0b33bc4d02b70a1d8d83a7ecdc8dd801a479c307685fa537a39276e2188ee7db` | 16 × 4 | 各 152 |
| `model_final.pth`（epoch60） | 60 | `e275cdcabc199988fbe5f77a17e3a70abcf9c28b4c03bb5d70d0befe5b08137b` | 8 × 2 | 各 40 |

逐行核验表明，同一检查点的 no-flag 与 `--fp32-anchor` CSV 中样本索引、PID、有效参考标记和参考索引完全一致。best 有 38 个不同的有效样本—参考对，各重复 4 次；final 有 20 个不同对，各重复 2 次。故误差率差别不是来自更换样本、参考关系或 GMM 选择。

## 零更新比率和权重

以下比例的分母均为该行有效参考记录；`FP32 输入` 指脚本内的外部锚点前向为 FP32，但配置开关仍关闭。正式比较重点是 AMP 输入下 no-flag 与显式 FP32-anchor。

| 检查点 / 配置 / 锚点输入 | n | p 均值 | q 均值 | r 均值（最小值） | r<1 | r<0.999 | w<0.9 | w<0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best / no-flag / AMP | 152 | 1.000000 | 0.999825 | 0.999913（0.994334） | 39.47% | 10.53% | 39.47% | 39.47% |
| best / no-flag / FP32 输入 | 152 | 1.000000 | 0.9999996 | 0.9999998（0.9999951） | 10.53% | 0% | 10.53% | 10.53% |
| best / `--fp32-anchor` / AMP | 152 | 1.000000 | 1.000000 | 1.000000（1.000000） | 0% | 0% | 0% | 0% |
| best / `--fp32-anchor` / FP32 输入 | 152 | 1.000000 | 1.000000 | 1.000000（1.000000） | 0% | 0% | 0% | 0% |
| final / no-flag / AMP | 40 | 1.000000 | 1.000137 | 1.000068（0.998486） | 35% | 10% | 35% | 35% |
| final / no-flag / FP32 输入 | 40 | 1.000000 | 0.9999993 | 0.9999997（0.9999951） | 15% | 0% | 15% | 15% |
| final / `--fp32-anchor` / AMP | 40 | 1.000000 | 1.000000 | 1.000000（1.000000） | 0% | 0% | 0% | 0% |
| final / `--fp32-anchor` / FP32 输入 | 40 | 1.000000 | 1.000000 | 1.000000（1.000000） | 0% | 0% | 0% | 0% |

`p_ratio` 在所有行都严格等于 1；主要误差落在 `q_ratio`（锚点—参考一致性）上。best no-flag AMP 的 q 最小值为 0.988667，final 为 0.996971；显式校准后 q 均严格等于 1。AMP 与 FP32 输出特征的最大绝对差约 0.0724，平均绝对差约 0.00365–0.00374；尽管 q 的平均相对误差较小，靠近 1 的比较仍会跨过原有严格阈值。

## 误判来源与模型状态

当前原始权重规则为：`r < 1` 时 `w=tanh(r)`，否则 `w=1`。`tanh(1)≈0.761594`，因此从 `r=1` 向下任意小的偏差都会造成约 23.84% 的单样本降权。best 在 no-flag、FP32 外部输入支路仍有 `r_min=0.9999951`，其 `r<1` 记录对应约 0.7616 权重；这说明问题不仅是 AMP 半精度，也包括旧路径将完整 batch 的锚点特征与过滤后有效子 batch 的临时更新结果比较。校准路径对相同的有效锚点子 batch 做 FP32 前向，和临时模型更新后的 batch membership、精度一致，零步结果恢复为 1。此解释由严格配对结果支持，但不等同于已证明各种模型和硬件上的普遍因果结论。

- 四次审计正式模型的参数和缓冲区均未变化：差异张量数 0，最大绝对变化 0。
- 临时模型状态核验：best 共比较 126 次、final 共比较 30 次；所有比较均无参数或缓冲区变化，最大绝对变化 0。
- 在五组不同 RNG 种子下重复训练态前向，AMP 与 FP32 的 logits 和特征差异均为 0；重复审计的有效记录逐项重合。没有观察到 NaN/Inf。
- p、q、r、w 的统计仅涵盖有效记忆参考。无效参考被排除在风险估计之外，维持默认权重 1。

## 记忆库与身份覆盖限制

best 与 final 的重拟合结果相同：GMM 选中 1,269/9,576（13.25%）；GMM/可靠参考身份覆盖 4/77（5.19%）；可靠记忆参考 1,793/9,576（18.72%）。有效参考中跨服装 1,793/1,793（100%）、自引用 0、身份错配 0；其余 7,783 条没有可靠参考。增加 best 的观察 batch 数至 16 后，有效参考仍来自 4 个身份；final 复核也未扩大覆盖。未对 GMM 做任何变更，也未用未选中身份伪造覆盖。

## 代码位置及变更

PR #3 提供并由本次测试/PPU 审计验证的实现：

- `config/defaults.py:69-71`：增加默认关闭的 `PREAWARE_FP32_ANCHOR`，历史 N3 默认行为不变。
- `npc/engine.py:368-404`：开启时，对当前有效参考对应的同一锚点子批执行 FP32、无梯度前向；用于预更新的一致性与身份 CE 对象和临时模型更新后的有效子批对齐。GMM、记忆参考选择、损失和权重公式均未改。
- `tests/test_npc.py:334-377`：模拟 AMP 锚点扰动，检查旧路径与校准路径的 q 差异，并断言模型状态不变和校准权重为 1。

本轮为比较不同 checkpoint 新增的可复现诊断参数：

- `scripts/diagnose_npc_zero_update_fp32.py:163-176`：增加 `--checkpoint`、`--checkpoint-epoch`、`--num-batches`，保留原默认值；epoch 参数用于按对应轮次重拟合 GMM。
- `scripts/diagnose_npc_zero_update_fp32.py:202-206`：仍固定 Seed1234、零临时学习率，并由 `--fp32-anchor` 明确启用校准。
- `scripts/diagnose_npc_zero_update_fp32.py:449-489`：保存检查点哈希、覆盖、逐模式比率、权重阈值比例和模型状态核验结果。

工作树本轮没有改动 GMM、记忆监督、临时优化算法、权重公式或检查点。没有加入中性阈值区间；鉴于开启 FP32-anchor 后两个 checkpoint 的零步分布均精确为 1，目前没有基于本数据另行设计阈值的依据。

## 下一步建议

FP32-anchor 通过本次零更新校准门槛，建议进入严格配对训练验证，但本报告不启动训练：

1. 建立 N3 原路径与 N3-FP32-anchor 两组，从同一 EVA02 初始化开始，Seed1234、完整 LTCC train split、60 epoch；仅将 `PREAWARE_FP32_ANCHOR` 作为两组唯一差异。
2. 固定并逐项校验优化器、学习率/调度器、采样器、数据增强、AMP、GMM、记忆库、损失和评估规则；训练输出与 checkpoint 使用全新独立目录，不续训或覆盖历史 N3。
3. 记录每轮有效锚点的 r/w、有效覆盖、GMM 身份覆盖及检索指标；先训练完两组，再按相同官方协议评估。单种子仅作初步效果验证；若有收益再以 Seed4321 复核。
4. 当前校准证据的身份覆盖只有 4/77，因此正式训练还需报告覆盖变化，并将“零步测量正确”与“检索性能收益”分开判断。

本阶段结论：**FP32 锚点校准在已测的 best/final checkpoint 上消除了零学习率错误降权；原始 tanh 的阈值不连续性确实会放大微小数值偏差。机制校准通过，性能收益尚未验证。**

## 产物

四次真实 PPU 审计的完整日志、CSV、GMM/记忆覆盖明细及 JSON 汇总保存在：

`outputs/NPC零更新FP32锚点校准_20261019_141739/`

子目录为 `best_legacy/`、`best_fp32_anchor/`、`final_legacy/`、`final_fp32_anchor/`。仅包含诊断数据与日志，不包含 checkpoint。
