# NPC 连续风险权重独立消融：实验依据与验证条件

## 证据范围
- LTCC，Seed1234，60 epochs，仓库 `outputs/NPC_CONFIRMATORY_20261014_140116`：
  - B0 best(Rank-1) e23：50.2551% / mAP 27.153%。
  - N2 best e18：56.3776% / mAP 31.511%。
  - 原 N3 best e17：50.7653% / mAP 28.541%。
- `outputs/NPC_CE_REDUCTION_ABLATION_20261016_140800`，N3-batch_mean best e17：50.7653% / mAP 27.39%，E60：45.9184% / mAP 25.4606%。
- 比较均使用该模型 Rank-1 最优轮次对应的同轮 mAP，不混合不同 checkpoint。
- 以上仅一随机种子，不代表跨种子统计显著性。

## 待检验假设
论文原始函数
```text
w(r) = tanh(r), r<1
w(r) = 1,       r>=1
```
在 `r=1` 发生跃迁：`lim(r→1-) w≈0.7616`，而 `w(1)=1`。
即使对可靠参考的损失变化很小也可能产生超过23%的梯度系数变化。这可能误伤困难但正确的跨服装样本，但**尚无可靠因果证据**。

## 本分支唯一新增算法选项
默认 `MODEL.NPC.WEIGHT_POLICY=original` 完全保持旧训练公式。

对照 `MODEL.NPC.WEIGHT_POLICY=continuous` 定义：
```text
r = 0.5 * (p_before / p_after + q_before / q_after)
risk = max(0, 1-r)
w = exp(-beta * risk)
```
默认 `beta=1.0`；该式在 `r=1` 连续，轻微恶化只轻微降权。

实验不得同时修改 GMM、记忆库、临时更新、Triplet、lambda_MB、归一化方式或评估协议。该分支直接基于 main；没有合入 PR #1 的 batch_mean 版本。因此以**原 N3（weight_sum）**作为主对照。

## 正式训练前必要检查
1. 执行 `python -m pytest -q tests/test_npc.py tests/test_npc_continuous_weight.py`；在 PPU 执行真实短迭代。新默认模式必须与旧 N3 数值一致。
2. 检查风险信号本身的可重复性：从现有训练检查点初始化临时模型，将 `MODEL.NPC.TEMP_LR_SCALE=0.0`，确认存在有效记忆参考且无参数更新，统计 `r` 偏离1的比例和权重分布。该诊断为**单独短程任务**，不能代替正式训练。
3. 由于预更新/后更新前向可能有训练态随机性和 AMP 精度差异，应记录两次前向的相同输入、随机状态、参数与运行模式；若零更新有明显假阳性，先修复测量流程后再评价权重策略。
4. 记录 `npc_weights.csv` 中低权重样本的身份/服装状态、参考有效率与 pre/post 损失变化，并分析是否误伤困难跨服装样本。
5. 若预检通过，再从同一 EVA02 预训练初始化、相同 Seed1234、相同训练协议，跑唯一新组 N3-continuous 60 epoch。模型选择规则与 N2/N3 一致。不得复用 N3 检查点继续训练。
6. 完成后分析 N3-continuous − 原N3、N3-continuous − N2 的 Rank-1 和 mAP；必要时开展 Seed4321 严格配对。

## 仍需独立审计
- GMM 的全局低 CE 筛选存在跨身份偏差，不属于本次权重消融修改范围。
- 现有日志中部分 `total_grad_norm` 为 NaN/Inf，且正式 `updates` 变量可能将 AMP 跳步计为已更新：需要实际参数步数审计，不能仅依据计数推断优化成功。
- 记忆梯度比率是观察量；未经确认实际损失系数是否纳入范数，不应宣称为有效梯度贡献比。
