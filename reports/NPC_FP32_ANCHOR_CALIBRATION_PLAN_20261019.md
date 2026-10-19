# NPC 预感知 FP32 锚点校准：只改变测量路径的独立实验

## 已确认的证据
2026-10-19 零学习率诊断，LTCC 训练集、旧 N3 epoch17 最优检查点、8 个固定批次/20 个不同有效样本对：
- AMP 锚点 + FP32 参考前后：严格 r<1 的不同样本对占 35%，r<0.999 占 10%；连续权重的最小值约 0.998487。
- FP32 锚点 + FP32 参考前后：严格 r<1 占 15%，r<0.999 为 0%，最小 r 约 0.999995。
- 正式与临时模型状态未变化；重复 RNG 不产生可观测差异。
- 这说明原测量链存在精度不匹配，但不能据此证明 N3 性能下降由此单独导致。

**重要：FP32 后残余约 1e-6 的 r<1 偏差，对不连续的原始 tanh(r) 权重仍可能导致显著错误降权。FP32 修正本身不一定能通过“零更新权重≈1”的门槛。不要跳过这一检查直接训练。**

## 代码变更
仅在本独立分支：
- `MODEL.NPC.PREAWARE_FP32_ANCHOR=False` 为默认，保留历史 N3 的锚点特征来源。
- 启用 True 时，在 NPC 控制器内、实际临时更新**之前**，使用 FP32 对所有有效参考的锚点做无梯度前向，并沿用训练态及有效样本子批大小。
- pre-reference 和 post-reference 均用同一精度与相同输入子批计算；不更改正式模型的 AMP 前向、正式梯度、GMM、记忆参考选择、NPC 风险公式或损失归一化。
- 新增 CPU 单元测试：用人为扰动的“AMP 特征”检验在零更新情况下，FP32 重算能恢复 r≈1，并确认主模型状态未改变。

## 审计脚本
`scripts/diagnose_npc_zero_update_fp32.py` 以原报告的零步脚本为基础，仅替换权重配置并加入选择开关：

```bash
# 首先在独立 worktree 内检查当前训练进程，再运行：
PYTHONPATH=.:/home/DIFFER_NPC python3 -m pytest -q tests/test_npc.py

# 不加 --fp32-anchor：历史风险测量路径。
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_zero_update_fp32.py

# 只有这一项切换：预更新有效锚点统一 FP32。
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_zero_update_fp32.py --fp32-anchor
```

上述脚本必须在有 PPU 和旧模型检查点的服务器上执行，输出到独立时间戳目录，不会启动训练。

## 严格正确性门槛
1. 两种模式都在相同 checkpoint、有效 train-only memory、相同随机样本对集合上测量。
2. 正式模型与临时模型零学习率状态不变；每个样本对确认 reference index、PID 和 clothes_id。
3. 先比较 q_ratio、r 的量级变化，再比较 **原始 tanh 权重 w<0.9 比例**，不要只观察连续权重的 w<0.99。
4. 扩展到更多 checkpoint 和至少 30 个 PID（如 GMM 无法覆盖，应标注不可达到并考虑训练期其它检查点），避免目前仅4个身份代表性不足。
5. 若校准后仍有明显零更新权重小于0.9，则继续停留在校准阶段；不要启动完整训练。可以另起独立研究分支，预先依据零更新噪声分位数制定风险中性区间，不准用 LTCC 验证集结果调该区间。
6. 若校准测试通过，才考虑独立训练 N3-FP32-anchor 60 epoch，唯一变化 `PREAWARE_FP32_ANCHOR=True`，其余严格继承旧 N3。
7. 对比较使用相同模型选择协议，并逐模型汇报最优 Rank-1 对应的同轮 mAP、E60，避免不同模型轮次指标拼接。

## 尚未完成
提交的代码仅是候选工程修复；在当前对话环境**尚未执行 PPU 测试、零步风险复核或60轮训练**。不得称其已经提升 LTCC/PRCC。
