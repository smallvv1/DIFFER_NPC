# PHASE NPC CE REDUCTION 消融验证报告

> 生成时间（北京时间）：2026-10-17T07:16:24+08:00

## 1. 实验范围与代码版本

本报告只验证 NPC 身份 CE reduction 的 batch_mean 消融。GMM、记忆库、临时更新、NPC 权重函数和记忆损失系数保持不变；没有引入新的 NPC 算法，也没有加载历史 N3 checkpoint 继续训练。
- 实验分支：`experiment/npc-ce-batch-mean-20261016`。
- 正式训练实际代码提交：`7f95e6933cf76f27f232e26fbba8c51b0327f280`。
- 报告生成时工作区提交：`67dba0ccc6e20814484b9de5ec084993093b229d`；该提交只包含汇总脚本，不改变已启动训练。
- 固定配置：LTCC、Seed1234、EVA02 初始化、60 epoch、原 20261014 N3 优化器/数据划分/评估协议、`TEST.WEIGHT=''`。
- 唯一实验差异：`MODEL.NPC.CE_REDUCTION=batch_mean`，即 `(w * CE).mean()`；默认 `weight_sum` 保留历史行为。
- 受保护 NPC 文件与基线提交相同：`是`。

代码文件 SHA256：
- `config/defaults.py`：`b3e2179b83090a0bb95ffc38d8d54ce5b6b1b8a0dc7ab5db2047c765105f2ec4`。
- `loss/make_loss.py`：`312929ee87f67c4ccd5fcc413102c3db21a6d351ae14cdddccff7d433e0f008f`。
- `loss/npc_weighting.py`：`978d3e8da538ade5eed148c0f53a8335680be0c8aa928066a626cc1fa666662e`。
- `tests/test_npc_ce_reduction.py`：`90f886bab7ac125dc2e6e0152ca76a3312a7ca52363782cf8302a3cbf212606d`。
- `scripts/run_npc_ce_batch_mean_formal.sh`：`bdf82dcde892518d58f3ba7b40b57b597a7230bea8378f4ff25c27212d9443cf`。

## 2. 测试与公式一致性

- 新增 CE reduction 测试：4 个基础测试加 3 个 dtype 参数化测试，共 7 项；与完整 NPC 测试合计 18 项，全部通过。
- `weight_sum` 与历史公式 `sum(w_i * CE_i) / max(sum(w_i), 1e-12)` 在 FP16、FP32、FP64 的 loss 和梯度一致；helper 不再隐式把 values 升为 FP32。
- `batch_mean` 数值严格等于 `(w * CE).mean()`，梯度为 `w / batch_size` 乘以逐样本 CE 的梯度。

## 3. 真实 LTCC PPU 短程预检

- 预检目录：`/home/DIFFER_NPC_worktrees/npc-ce-batch-mean-20261016/outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256`；真实更新数：60 个 batch，训练循环记录 1 个 epoch。
- 退出状态：`completed`；正式更新数：60。
- PPU：`PPU-ZW810E`，设备数量 1；未发生 CPU 模型回退。
- clothes-changing：Rank-1 0.1301，mAP 0.0524；general：Rank-1 0.3063，mAP 0.0840。
- 训练损失、CE、Triplet、memory loss、梯度范数和 AMP scale 均为有限值；`updates=60`，未见跳过优化器更新。
- 首个 epoch GMM 选中比例 0.0000，有效参考覆盖率 0.0000；因此该 60 batch 预检中的 NPC 权重全为 1 是无有效参考时的协议默认值，不应误解为动态权重实现被强制固定。

## 4. 与既有 B0/N2/N3 的 LTCC 对比

以下先列 E60，再列同一训练记录中按 clothes-changing Rank-1 选择的 best epoch；两者不拼接为同一模型。
- B0 E60：Rank-1 0.4796，Rank-5 0.6148，Rank-10 0.6709，mAP 0.2532，训练损失 1.4845。
- B0 best（epoch 23）：Rank-1 0.5026，Rank-5 0.6097，Rank-10 0.6582，mAP 0.2715，训练损失 3.0646。
- N2 E60：Rank-1 0.5306，Rank-5 0.6709，Rank-10 0.7219，mAP 0.3041，训练损失 1.3965。
- N2 best（epoch 18）：Rank-1 0.5638，Rank-5 0.7296，Rank-10 0.7806，mAP 0.3151，训练损失 3.4707。
- N3 E60：Rank-1 0.4694，Rank-5 0.5969，Rank-10 0.6505，mAP 0.2525，训练损失 1.5008。
- N3 best（epoch 17）：Rank-1 0.5077，Rank-5 0.6378，Rank-10 0.7041，mAP 0.2854，训练损失 3.8384。
- N3-batch_mean E60：Rank-1 0.4592，Rank-5 0.6071，Rank-10 0.6633，mAP 0.2546，训练损失 1.4953。
- N3-batch_mean best（epoch 17）：Rank-1 0.5077，Rank-5 0.6454，Rank-10 0.6811，mAP 0.2739，训练损失 3.8749。

- N3-batch_mean 相对原 N3（E60）：Rank-1 -0.0102，mAP 0.0021。
- N3-batch_mean 相对 N2（E60）：Rank-1 -0.0714，mAP -0.0495。
- 以上是同一评估协议下的观察性差异；单种子不能证明跨种子稳定性或因果收益。

## 5. NPC 机制诊断

- 每轮 GMM 选中比例、身份覆盖率、有效参考覆盖率、跨服装参考数以及权重统计均保存在 `metrics/metrics_per_epoch.csv`；逐样本 pre/post 诊断保存在 `npc_diagnostics/npc_weights.csv`。
- 正式逐样本权重记录：572160 行；其中有效参考 226898 行，非单位权重 68735 行。
- 正式训练最终轮：有效参考覆盖率 0.5382，NPC 权重均值 0.9374，中位数 1.0000，标准差 0.1050，权重等于 1 比例 0.7379，小于 0.9 比例 0.2621，小于 0.5 比例 0.0000。
- 正式训练最终轮有效参考子集：均值 0.8839，中位数 1.0000，标准差 0.1193，小于 0.9 比例 0.4863。
- CE reduction 的变化只改变有 NPC 权重时 CE 标量的归一化分母；它不改变 GMM 选择、记忆参考构建或临时模型更新。

## 6. 数值稳定性与异常记录

- 正式 metrics 中记录的非有限字段：22 个 epoch。若为 0，表示本次正式日志未出现 NaN/Inf；若大于 0，将在下方列出。
- epoch 1：total_grad_norm, memory_grad_ratio。
- epoch 6：total_grad_norm, memory_grad_ratio。
- epoch 9：total_grad_norm。
- epoch 11：total_grad_norm。
- epoch 13：total_grad_norm, memory_grad_ratio。
- epoch 15：total_grad_norm。
- epoch 16：total_grad_norm, memory_grad_ratio。
- epoch 20：total_grad_norm, memory_grad_ratio。
- epoch 22：total_grad_norm。
- epoch 26：total_grad_norm。
- epoch 32：total_grad_norm, memory_grad_ratio。
- epoch 34：total_grad_norm, memory_grad_ratio。
- epoch 35：total_grad_norm, memory_grad_ratio。
- epoch 37：total_grad_norm。
- epoch 41：total_grad_norm。
- epoch 43：total_grad_norm, memory_grad_ratio。
- epoch 49：total_grad_norm。
- epoch 51：total_grad_norm。
- epoch 52：total_grad_norm。
- epoch 54：total_grad_norm, memory_grad_ratio。
- epoch 57：total_grad_norm, memory_grad_ratio。
- epoch 59：total_grad_norm。

## 7. 结论

- 当前实验结论：**N3-batch_mean 相对原 N3 的 Rank-1 与 mAP 结论不一致**。
- 本报告不把观察到的指标差异解释为必然因果；特别需要区分 CE reduction 的梯度尺度变化与 NPC 机制本身的收益。
- 是否达到 N2：以 E60 的 Rank-1 与 mAP 分别判断，不能用不同 epoch 的最优值拼接。
- 后续建议：若 batch_mean 在 E60 相对 N3 有一致的 Rank-1/mAP 改善，下一步应使用 Seed4321 做严格复核；若只改善单一指标或仅 best epoch 改善，应先视为不确定，不修改 GMM/NPC 参数。

## 8. 产物路径

- 正式日志：`/home/DIFFER_NPC_worktrees/npc-ce-batch-mean-20261016/logs` 下对应 `NPC批均值正式训练_*.log`。
- 正式输出：`/home/DIFFER_NPC_worktrees/npc-ce-batch-mean-20261016/outputs/NPC_CE_REDUCTION_ABLATION_20261016_140800`。
- 预检输出：`/home/DIFFER_NPC_worktrees/npc-ce-batch-mean-20261016/outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256`。
- 正式 checkpoint 保留在服务器本地，按要求不纳入 GitHub 提交。
