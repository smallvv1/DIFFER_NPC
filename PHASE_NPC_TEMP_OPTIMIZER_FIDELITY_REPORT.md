# PHASE NPC 临时 SGD 优化器保真度报告

日期：2026-10-20

实验分支：`experiment/npc-temp-sgd-state-fidelity-20261019`

PPU 诊断代码提交：`e3378c2c9bfb4c34ac048373aeb67e9236ffabac`

## 结论

真实 PPU 短迭代显示，继承正式 SGD 的 `momentum_buffer` 明显提高了临时更新对实际正式更新的方向保真度，并改善了可靠参考样本损失变化的预测：临时/正式更新余弦从 `0.5363` 提高到 `0.9625`；参考一致性损失预测的平均绝对误差从 `0.0003081` 降至 `0.0001477`；逐样本改善方向一致率从 `83.33%` 提高到 `100%`。因此建议下一阶段进行独立、受控的正式消融，但本阶段没有启动正式训练，也没有把默认 N3 改为 True。

该结论基于一个固定正式采样批次的 6 个可靠参考条目，属于短诊断证据；在正式消融完成前，原 N3 默认 `TEMP_COPY_OPT_STATE=False` 保持不变。

本阶段没有修改 GMM、权重函数、记忆监督系数或 60 轮训练协议。

## GitHub 同步与执行范围

开始阶段通过 HTTPS 检查到：

| ref | 提交 |
|---|---|
| `refs/heads/main` | `9d4cb06fde48aeab53e3aab3bb833e50ad2ca875` |
| `refs/pull/4/head` | `d333fbba63b482b4648416177fca0e3a003ecad4` |
| `refs/pull/5/head` | `ad8f058d97d33032f63dda039343803c167030bc` |

PR #4 的四组 Best/Final PPU 证据已在独立检查点分支完成，并在本分支带入同一份复核报告。PR #5 代码提交链包括 `32224ff`（PPU 诊断脚本）、`5d67d78`（配对 RNG 恢复）和 `e3378c2`（改用正式 sampler 批次）。

实际检查点为 Best：

`/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth`

SHA256：`0b33bc4d02b70a1d8d83a7ecdc8dd801a479c307685fa537a39276e2188ee7db`。

## 测试结果

官方 PPU Python 环境中运行：

```text
PYTHONPATH="$PWD:/home/DIFFER_NPC:$DEPS_DIR" \
  /usr/local/bin/python3 -m pytest -q \
  tests/test_npc.py tests/test_npc_audit_checkpoint.py tests/test_npc_temp_sgd_state.py
..................                                                       [100%]
18 passed, 1 warning in 7.33s
```

警告为既有 `utils/metrics.py` 的 `addmm_` 弃用警告，不是本次改动失败。PR #5 新增状态继承测试为 `4 passed`；PR #4 checkpoint AST 测试为 `2 passed`。

## 真实 PPU 短迭代协议

- 设备：`PPU-ZW810E`，PyTorch `2.4.0`，单设备。
- 数据：LTCC train-only；记忆库使用 9576 条记录、77 个身份，N3 的原始 GMM/权重/记忆监督配置。
- 先运行 3 个正式 SGD 预热步，再冻结正式模型、正式优化器、批次、随机状态、学习率和 FP32 anchor 配置，分别运行 False/True。
- 正式 SGD 为 `SGD(momentum=0.9)`，学习率组为原配置的 `2e-6/4e-6`；临时学习率倍率 `1.0`，临时步数 1，`PREAWARE_FP32_ANCHOR=True`。
- 预热后实际形成 513 个 `momentum_buffer`，共 303,409,165 个元素，L2 `309.24658294`，最大绝对值 `9.49696255`。
- 固定比较批次来自正式 sampler-backed `train_loader`：batch size 8，有效参考 6/8（75%），3 个身份且每个身份 2 个样本；两种模式完全相同，临时子批满足 Triplet 条件。

第一次试运行曾错误使用顺序 memory loader，得到单身份批次并触发原 Triplet hard-mining 的协议错误；该次没有形成结果，随后已在提交 `e3378c2` 中改用正式 sampler 批次并重新运行。最终报告只使用 retry 目录的成功结果。

## False/True PPU 数值对照

| 指标 | `TEMP_COPY_OPT_STATE=False` | `TEMP_COPY_OPT_STATE=True` |
|---|---:|---:|
| 有效参考 / batch | 6/8 | 6/8 |
| 临时 loss 分支 | CE+Triplet | CE+Triplet |
| 正式 loss 分支 | full-batch CE+Triplet+memory | full-batch CE+Triplet+memory |
| 临时更新 L2 | 0.0001132095 | 0.0006325311 |
| 实际正式更新 L2 | 0.0006568594 | 0.0006569534 |
| 临时/正式更新余弦 | 0.5362868 | 0.9624854 |
| risk ratio 均值 | 1.00054272 | 1.00092989 |
| p ratio 均值 | 1.00029926 | 1.00034442 |
| q ratio 均值 | 1.00078621 | 1.00151538 |
| 有效参考动态权重均值 | 1.00000000 | 0.96025478 |
| 有效参考动态权重最小值 | 1.00000000 | 0.76152867 |
| 有效参考 `w<0.99` | 0/6 | 1/6（16.67%） |
| 预测 identity loss Δ 均值 | -0.00054280 | -0.00062883 |
| 实际正式 identity loss Δ 均值 | -0.00059072 | -0.00058585 |
| identity Δ 平均绝对误差 | 0.00014051 | 0.00013135 |
| identity 改善方向一致率 | 100% | 100% |
| 预测 reference consistency Δ 均值 | -0.00012232 | -0.00024720 |
| 实际正式 reference consistency Δ 均值 | -0.00039446 | -0.00039486 |
| reference Δ 平均绝对误差 | 0.00030809 | 0.00014766 |
| reference 改善方向一致率 | 83.33% | 100% |

两种模式的正式更新都使可靠参考 identity loss 平均下降；True 的临时更新更接近正式更新，尤其是 reference consistency 的逐样本风险方向。风险比值本身仍按现有权重函数定义报告，没有被重新解释或修改。

## 污染隔离

在调用 `preaware_weights` 后、正式更新之前进行精确状态核验：

| 模式 | 正式模型变化 | 正式 optimizer 状态变化 |
|---|---:|---:|
| False | 0 tensors，max abs 0 | 0 entries，max abs 0 |
| True | 0 tensors，max abs 0 | 0 entries，max abs 0 |

临时 optimizer 的 state 是深复制并按临时参数设备重建；正式 optimizer 的 513 个动量缓冲没有被临时步写回。之后的正式更新只在各模式自己的恢复状态上执行。

## 原因拆分

1. **动量状态差异**：这是 False/True 唯一有意改变的优化器因素。False 的临时 optimizer 没有正式动量；True 复制了预热得到的 513 个正式 `momentum_buffer`，所以更新范数和方向显著接近正式更新。
2. **有效子批差异**：两种模式均为相同的 6/8、3 个身份、每个身份 2 个样本，均满足临时 Triplet 条件。因此本次 False/True 差异不是有效子批差异。
3. **损失函数差异**：两种模式临时路径都使用 CE+Triplet；正式路径都使用 full-batch CE+Triplet+memory。该差异在两种模式中相同，不能把它归因给动量继承。
4. **精度路径**：两种模式都使用相同的 `PREAWARE_FP32_ANCHOR=True`；正式更新保留正式 AMP/GradScaler 路径，FP32 anchor 不被混入模式差异。

## 结果文件与建议

成功的机器可读结果：

`outputs/PHASE_NPC_TEMP_OPTIMIZER_FIDELITY_PPU_20261020_retry/temp_optimizer_fidelity_ppu.json`

配套的 PPU 日志、GMM/memory 明细、可靠性 CSV 也在该目录。第一次 sampler 协议错误的日志保存在 `outputs/PHASE_NPC_TEMP_OPTIMIZER_FIDELITY_PPU_20261020/`，用于保留诊断过程证据；它不作为数值结论。

本次 PPU 诊断支持进行下一阶段的正式 False/True 消融，因为 True 同时改善了更新方向、reference loss 预测误差和方向一致率。但正式消融应使用独立输出目录并显式保留原 N3 False 对照；本阶段不启动 60 轮训练，默认设置仍保持 `TEMP_COPY_OPT_STATE=False`。
