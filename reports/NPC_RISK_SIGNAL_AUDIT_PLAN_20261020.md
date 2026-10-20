# NPC 双通道风险抵消审计计划（2026-10-20）

## 本轮依据（GitHub最新多批次报告）

`PHASE_NPC_MULTIBATCH_RISK_VALIDATION_REPORT.md` 在 LTCC Seed1234 的 Best e17 / Final e60 两个检查点，各16批显示：
- `TEMP_COPY_OPT_STATE=True` 的方向余弦 Best 0.943440 / Final 0.960336；False 为 0.396699 / 0.825072。
- True 的可靠参考一致性预测 MAE Best 0.000255695 / Final 0.000244705；False 为 0.000326266 / 0.000359500。
- True 在「真实损害阈值 1e-4」的混淆矩阵均出现 7 次 FP（Best TP/FP/FN/TN=12/7/10/31；Final=12/7/19/32）；`w<0.9` 的非有害降权也均为7次。
- Best 60条可靠参考观测仅4个独立身份；Final 70条观测仅13个身份。不能据此宣称稳定的全 LTCC 风险识别性能。
- 这些损害标签是正式更新后同身份可靠参考的 CE 或跨服装一致性损失恶化，不代表检索 mAP 或局部异身份结构的真实损害。

## 为什么进一步审计 NPC 平均比值

当前 `npc/losses.py::calculate_npc_weights` 中：

```text
p_ratio = identity_loss_before / identity_loss_after
q_ratio = reference_consistency_before / reference_consistency_after
r = 0.5*(p_ratio + q_ratio)
w = tanh(r) if r<1 else 1
```

若一条参考的身份 CE 恶化（p_ratio<1），但跨服装一致性改善（q_ratio>1），平均比值可能仍 `r>=1`，于是原规则漏检某个通道的损害；反过来，由于原规则在 `r=1` 不连续，微小平均负变化可能造成约24%降权。两者均是**待验证的风险决策偏差假设**，不是已经证实的检索性能原因。

## 已提交的代码（不影响训练）

- `npc/risk_signal_audit.py`：从已有 PPU 配对结果中分解 identity/reference 两个损害通道，检验平均比值被另一通道抵消的数量；使用预定义阈值 0、1e-5、1e-4 对照原规则与仅作观察的四种影子规则 identity-only、consistency-only、either-channel、both-channels。影子规则**不能直接写回训练权重**。
- `scripts/analyze_npc_risk_signals.py`：仅加载两份已保存的多批 JSON，输出中文 Markdown 和机器可读 JSON，不需要加载网络/模型/显卡。
- `tests/test_npc_risk_signal_audit.py`：验证通道抵消检测、FP、严格阈值、非有限值、配对索引和同一正式更新标签。
- 对配对 PPU 的极小数值差异，本审计用 **False/True 两边正式实际损失增量的平均值**定义统一的 proxy ground truth。严格拒绝样本对索引不一致，记录目标最大差，避免因数值波动导致两个模式拥有不同真值。

## 在用户服务器上的执行

在独立 worktree 检出 `experiment/npc-risk-signal-audit-20261020`：

```bash
PYTHONPATH=.:/home/DIFFER_NPC /home/DIFFER_NPC_worktrees/.venv-npc-cpu/bin/python -m pytest -q tests/test_npc_risk_signal_audit.py tests/test_npc_risk_validation.py

PYTHONPATH=.:/home/DIFFER_NPC /home/DIFFER_NPC_worktrees/.venv-npc-cpu/bin/python scripts/analyze_npc_risk_signals.py \
 --input outputs/PHASE_NPC_TEMP_OPTIMIZER_MULTIBATCH_PPU_20261020_retry/temp_optimizer_multibatch_ppu.json \
 --input outputs/PHASE_NPC_TEMP_OPTIMIZER_MULTIBATCH_PPU_20261020_final/temp_optimizer_multibatch_ppu.json
```

如果这些全量 JSON 尚未同步到独立 worktree，可从原项目只读路径传入，不能复制或覆盖正式模型检查点。

## 决策门槛

- 先报告两个通道相反方向、平均值掩盖、真正预测准确/漏检、FP 严重度；区分 Batch / PID / 唯一关系数。
- 不用这两个小样本集合去「选择最优阈值」或「证明 either-channel 更优」。
- 如果证据有价值，再设计 **单变量、预先固定方案** 的影子风险决策对照；与当前 FP32 锚点、动量继承的正式消融分开进行。
- 当前不直接开启 60 epoch、不修改 GMM、记忆库、CE reduction、原损失或主分支，也不引入 NDP/GPC。

**代码已推送；本环境尚未运行新增单测或服务器侧影子审计。**
