# PHASE NPC-MULTIBATCH-RISK-VALIDATION — 索引纠错与风险预测有效性验证

## 先前诊断必须纠正的内容

2026-10-20 最新 PPU `PHASE_NPC_TEMP_OPTIMIZER_FIDELITY_REPORT.md` 在单批6个可靠参考样本上声称 TEMP_COPY_OPT_STATE=True：
- 临时/正式参数更新方向余弦 0.5363 → 0.9625；
- 参考损失预测 MAE 0.0003081 → 0.0001477；
- 方向一致率 83.33% → 100%。

进一步阅读 `scripts/diagnose_npc_temp_optimizer_fidelity_ppu.py::reference_post_metrics`，发现**已确认的记录映射错误**：

```python
for position, sample_index, identity_value, consistency_value in zip(
    valid_positions.tolist(),
    indices.tolist(),   # 旧代码这里错误地使用全 batch
    identity.tolist(),
    consistency.tolist(),
):
    ...
```

`valid_positions` 和参考损失均已过滤，但 `indices` 未过滤。如果有效位置不是 batch 起始连续区间，实际 post-loss 将与错误样本索引配对。**额外核验关键限定：**已读取成功 retry 的两份实际 `npc_weights.csv`，其中唯一用于最新报告的 batch size 8，恰有前6个位置 [0,1,2,3,4,5] 为有效参考，最后两个位置无效。因此该具体批次虽然走过错误代码，但全批索引的前6个恰好对应全部有效位置，**没有实际触发样本错配**；该报告的 MAE/方向一致率可保留为6样本的初步观察性证据。对于任何非前缀有效 mask，原代码确实会产生错误映射，不能扩展至其他批次而不修复。原更新方向余弦另由模型参数差值计算，未受该潜在映射错误直接影响。

## 已提交的修复

- `npc/audit_utils.py`：提供 `align_valid_reference_losses`，只使用经过 valid_positions 筛选的全局样本索引；输入长度越界、不一致和重复样本时报错，禁止静默错配。
- `scripts/diagnose_npc_temp_optimizer_fidelity_ppu.py`：调用上述工具，修复原 PPU 诊断脚本。**没有修改 NPC 核心训练代码与旧检查点**。
- `tests/test_npc_reference_alignment.py`：测试有效位置 [1,3]、长度错误、越界、重复样本。
- `scripts/diagnose_npc_temp_optimizer_multibatch_ppu.py`：新增多批次真实采样诊断；默认3步预热，16个合格比较批次，严格复用相同模型、optimizer和RNG状态；False/True 使用同一正式 *中性权重 w=1* 更新作为目标，避免权重模式自行改变被预测的 ground truth。
- `npc/risk_validation.py` + `tests/test_npc_risk_validation.py`：逐样本对和逐批统计余弦、p/q 误差、危险样本代理的 TP/FP/FN/TN，并在缺乏配对信息时拒绝输出伪结论；阈值敏感性同时使用0、1e-5和1e-4。

## 正确性边界

1. 「实际有害」仅是参考样本身份 CE 或跨衣一致性损失在正式更新后**恶化**的代理标签，尚不能代替 Rank-1/mAP 或局部身份判别风险。
2. 训练阶段真实 N3 的正式优化更新有加权 CE，新的多批次脚本故意令两种策略的正式更新都采用 `w=1`，仅为**一致的反事实预测目标**，不是在完全模拟 N3 性能训练。
3. 观测样本可能共享身份、参考图像甚至跨批重复，因此不能按条目数假装是独立统计样本；同时记录唯一 (anchor,reference) 对和唯一 PID。
4. 临时 FP32 与正式 AMP、临时可靠样本子批与正式全批、CE fallback 与 Triplet 等机制仍存在差异。该消融只验证 SGD 历史动量继承的边际作用。
5. 旧 N3 最优模型仅用于机制诊断；**不重新训练旧模型，不修改 main，不自动开展60轮训练**。

## 服务器端执行顺序（第二台 PPU）

首先保护当前训练进程并在独立 worktree 检出本分支，确认依赖与 LTCC train-only 配置一致。

```bash
PYTHONPATH=.:/home/DIFFER_NPC python3 -m pytest -q tests/test_npc.py tests/test_npc_audit_checkpoint.py tests/test_npc_temp_sgd_state.py tests/test_npc_reference_alignment.py tests/test_npc_risk_validation.py

# 首先重新运行老的单批 PPU 诊断（已修正真实参考损失索引）。
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_temp_optimizer_fidelity_ppu.py --warmup-steps 3

# 然后在同一 PPU 上执行多批次对照（不训练60轮）。
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_temp_optimizer_multibatch_ppu.py --checkpoint-epoch 17 --warmup-steps 3 --comparison-batches 16 --scan-limit 1000
```

两种模式需验证每个批次的 reference index、PID、服装状态，以及一致的正式参数更新目标。非有限损失或 AMP 跳步必须标注并排除，不得默默计为正常步。若 Best 检查点覆盖有限，可另行使用真实 Final epoch60 检查点进行同协议复核，并报告独立 PID 范围。

## 是否进入正式60轮的决策门槛

只有以下条件满足才建议单变量正式消融（且需用户允许正式算力投入）：
- 新单批 MAE/一致率的索引纠错后结果仍显示动量改进；
- 多批次下方向余弦、参考 p/q 预测 MAE、实测风险代理召回/准确率等具有一致、可重复的优势；
- 观测范围包含困难/易样本和不同身份；不能只从6个样本外推；
- 不出现正式模型/优化器污染、GMM索引偏差的新实现错误；
- 若结果方向冲突，保留机制探索状态，不擅自启动60轮实验。

**提交本报告时，新增脚本与测试尚未在用户 PPU 上执行，不得宣称改进得到新实验验证。**
