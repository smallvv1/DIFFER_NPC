# PHASE NPC 检查点复核报告

日期：2026-10-20

实验分支：`experiment/npc-checkpoint-risk-audit-fix-20261019`

实际代码提交：`d333fbba63b482b4648416177fca0e3a003ecad4`

## 结论

PR #4 的 Best/Final 配对零更新审计已经在官方 PPU 完成，不再使用旧 Final 审计结果。四组运行均使用实际解析的检查点路径，记录了对应 SHA256；正式模型状态没有发生任何变化，临时零学习率模型也没有变化。PR #4 的检查点加载复核通过。

本次只做诊断，没有启动 60 轮正式训练，也没有修改 GMM、权重函数、记忆监督系数或原始检查点。

## GitHub 同步

开始本阶段时通过 HTTPS 检查到：

| ref | 提交 |
|---|---|
| `refs/heads/main` | `9d4cb06fde48aeab53e3aab3bb833e50ad2ca875` |
| `refs/heads/experiment/npc-checkpoint-risk-audit-fix-20261019` | `d333fbba63b482b4648416177fca0e3a003ecad4` |
| `refs/heads/experiment/npc-temp-sgd-state-fidelity-20261019` | `ad8f058d97d33032f63dda039343803c167030bc` |
| `refs/pull/4/head` | `d333fbba63b482b4648416177fca0e3a003ecad4` |
| `refs/pull/5/head` | `ad8f058d97d33032f63dda039343803c167030bc` |

## 实际检查点

| 检查点 | 实际加载路径 | SHA256 | epoch | 模型状态 |
|---|---|---|---:|---|
| Best | `/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth` | `0b33bc4d02b70a1d8d83a7ecdc8dd801a479c307685fa537a39276e2188ee7db` | 17 | `OrderedDict`，515 张量，303,933,965 elements |
| Final | `/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_final.pth` | `e275cdcabc199988fbe5f77a17e3a70abcf9c28b4c03bb5d70d0befe5b08137b` | 60 | `OrderedDict`，515 张量，303,933,965 elements |

Best 与 Final 的键集合相同，但 513/515 个张量不同，最大绝对差 `0.9905719757`，全局 L2 差 `20.1788710065`。因此本次 Final 结果是独立重新加载并运行的证据。

## PPU 协议

- 设备：`PPU-ZW810E`，设备数 1，PyTorch `2.4.0`。
- 数据：LTCC train-only，9576 条记录、77 个身份；完整记忆库 batch size 16。
- 每组：16 个固定批次、4 次重复；临时步数 1；临时学习率 0（零更新门禁）。
- 比较：历史 AMP 锚点和 `PREAWARE_FP32_ANCHOR=True`；GMM 阈值 0.99，原始权重策略。
- 每组都记录了 forward 重复性、AMP/FP32 q 差异、实际 preaware 条目及正式模型状态差异。

## Best/Final 数值结果

下表的 risk/weight 是有效参考条目上的均值；`r<1` 是严格零更新假阳性，`r<0.999` 和 `w<0.99` 是实用阈值。`p` 为身份风险比，`q` 为参考一致性风险比。

| 检查点 / 锚点 | 有效参考 / GMM 选中 | 跨服装有效参考 | 有效条目 | r 均值 | p 均值 | q 均值 | w 均值 | r<1 | r<0.999 | w<0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best / legacy AMP | 1793 / 1269 | 100.00% | 152 | 0.99991263 | 1.00000000 | 0.99982525 | 0.90574979 | 39.47% | 10.53% | 39.47% |
| Best / FP32 anchor | 1793 / 1269 | 100.00% | 152 | 1.00000000 | 1.00000000 | 1.00000000 | 1.00000000 | 0.00% | 0.00% | 0.00% |
| Final / legacy AMP | 4976 / 3324 | 98.83% | 232 | 0.99994230 | 1.00000000 | 0.99988460 | 0.87243420 | 53.45% | 10.34% | 53.45% |
| Final / FP32 anchor | 4976 / 3324 | 98.83% | 232 | 1.00000000 | 1.00000000 | 1.00000065 | 1.00000000 | 0.00% | 0.00% | 0.00% |

Best 的有效参考覆盖率为 `18.723893%`，可靠身份覆盖率 `5.194805%`；Final 分别为 `51.962198%` 和 `22.077922%`。Best/Final 均无自引用和身份错配；Best 有效参考 1793/1793 跨服装，Final 为 4918/4976。

## 状态与加载核验

| 检查点 / 锚点 | 正式模型变化 | 临时零 LR 比较 | 实际 preaware 条目 |
|---|---|---|---:|
| Best / legacy | 0 tensors，max abs 0 | 126 次，0 次变化，max abs 0 | AMP 64，FP32 64 |
| Best / FP32 anchor | 0 tensors，max abs 0 | 126 次，0 次变化，max abs 0 | AMP 64，FP32 64 |
| Final / legacy | 0 tensors，max abs 0 | 128 次，0 次变化，max abs 0 | AMP 64，FP32 64 |
| Final / FP32 anchor | 0 tensors，max abs 0 | 128 次，0 次变化，max abs 0 | AMP 64，FP32 64 |

这确认了 PR #4 修复实际生效：Best/Final 不会静默复用同一文件；正式模型和临时零更新路径均保持不变。

## 测试与上传

PR #4 checkpoint AST 回归测试：`2 passed`。四组 PPU 原始 JSON 和日志保存在：

`outputs/PHASE_NPC_CHECKPOINT_REVALIDATION_20261020/{best_legacy,best_fp32_anchor,final_legacy,final_fp32_anchor}/`

另有压缩机器可读汇总：`diagnostics/phase_npc_checkpoint_revalidation_ppu_20261020.json`。这些新结果将与本报告一起提交并推送到本独立实验分支；主分支、历史实验结果和原始检查点未修改。
