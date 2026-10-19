# PHASE NPC 临时 SGD 优化器保真度报告

日期：2026-10-19  
实验分支：`experiment/npc-temp-sgd-state-fidelity-20261019`  
代码基线：PR #5 `d5e075e676d2438d6455bc83df0868ba427a1fc5`  
结论：**CPU 代码级状态继承测试通过，但真实 PPU 诊断未能启动；不建议进行正式消融，保留原 N3 设置 `TEMP_COPY_OPT_STATE=False`。**

## 1. GitHub 同步与执行边界

本次先通过 HTTPS 只读 refs 检查 GitHub：

| ref | 提交 |
|---|---|
| `refs/heads/main` | `9d4cb06fde48aeab53e3aab3bb833e50ad2ca875` |
| `refs/pull/4/head` | `83d453a0fa0a188f302c05f45d15c0235e90deb2` |
| `refs/pull/5/head` | `d5e075e676d2438d6455bc83df0868ba427a1fc5` |

PR #5 的提交链包含 PR #4 的路径修复、PR #5 的 `TEMP_COPY_OPT_STATE` 开关、SGD 状态复制实现及新增测试。本报告和诊断 JSON 均在独立 worktree 及独立实验分支中生成；主分支、旧实验结果和原始检查点没有修改。

先带入 PR #4 的复核证据提交 `d333fbb`，其中已重新读取 Best/Final 实际路径和 SHA256；该报告明确指出 PPU zero-update 尚未执行，未把旧 Final 审计当作证据。

## 2. 代码检查点

PR #5 的实现只改变临时优化器：

- 默认 `MODEL.NPC.TEMP_COPY_OPT_STATE=False`，保留历史 N3 行为；
- 打开时只接受正式 SGD；
- 按参数名映射正式/临时参数，错配立即报错；
- 深复制参数级 state，并将张量移动到临时参数设备；
- 临时 optimizer 只作用于临时模型，不回写正式模型或正式 optimizer。

本阶段没有修改 GMM、权重函数、记忆监督系数、正式损失函数或 60 轮训练协议。

## 3. 测试结果

在独立 CPU venv 中运行完整门禁：

```text
PYTHONPATH="$PWD:/home/DIFFER_NPC" \
  /home/DIFFER_NPC_worktrees/.venv-npc-cpu/bin/python -m pytest -q \
  tests/test_npc.py tests/test_npc_audit_checkpoint.py tests/test_npc_temp_sgd_state.py
..................                                                       [100%]
18 passed, 1 warning in 3.77s
```

其中新增 PR #5 状态继承测试单独结果为 `4 passed in 2.91s`；PR #4 checkpoint AST 回归测试为 `2 passed in 0.02s`。唯一警告是既有 `utils/metrics.py` 的 `addmm_` 弃用警告，不是本次改动引入的失败。

这些是 CPU 回归测试，不是 PPU 结果。

## 4. Best/Final 检查点证据

两个真实文件均通过 CPU `torch.load(..., map_location="cpu", weights_only=False)`：

| 检查点 | 实际路径 | SHA256 | 状态字典 |
|---|---|---|---|
| Best | `/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth` | `0b33bc4d02b70a1d8d83a7ecdc8dd801a479c307685fa537a39276e2188ee7db` | `OrderedDict`，515 张量 |
| Final | `/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_final.pth` | `e275cdcabc199988fbe5f77a17e3a70abcf9c28b4c03bb5d70d0befe5b08137b` | `OrderedDict`，515 张量 |

二者键集合相同，但 513/515 个张量不同，最大绝对差 `0.9905719757`、全局 L2 差 `20.1788710065`。检查点只保存模型 state，不含可恢复的正式 SGD optimizer/momentum state，因此不能从 Final 或 Best 伪造历史 momentum；正式 momentum 必须由一次新的受控短迭代真实积累。

## 5. CPU 代码级 False/True 数值对照

以下是同一 Tiny ReID 模型、同一批次、同一随机种子、同一学习率、同一 FP32 anchor 下的受控 sanity check。它验证状态复制语义和污染隔离，**不代表 PPU 风险预测效果**。

| 指标 | `False` | `True` |
|---|---:|---:|
| 有效参考数 / 比例 | 4 / 1.0000 | 4 / 1.0000 |
| pre identity loss 均值 | 0.785612 | 0.785612 |
| 临时预测参考 loss 变化 | -0.035068 | -0.031186 |
| 实际正式更新参考 loss 变化 | -0.031186 | -0.031186 |
| 预测/实际变化同号 | 是 | 是 |
| 预测与实际变化绝对差 | 0.003882 | 0.000000 |
| `risk ratio` 均值 | 1.037821 | 1.242485 |
| `p_ratio` 均值 | 1.045276 | 1.048445 |
| `q_ratio` 均值 | 1.030366 | 1.436526 |
| 动态权重均值 / 最小值 / 最大值 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| 临时参数更新 L2 | 0.042365 | 0.395651 |
| 对应正式更新 L2 | 0.395651 | 0.395651 |
| 临时/正式更新余弦 | 0.093466 | 1.000000 |
| 临时/正式更新 L2 差 | 0.393956 | 0.000000 |
| 临时步后正式模型是否变化 | 否 | 否 |
| 临时步后正式 optimizer 是否变化 | 否 | 否 |

解释：在该受控小例中，`True` 完整继承正式 SGD 动量后，临时更新与正式更新逐元素一致；`False` 重新初始化临时 SGD，方向和范数明显不同。`True` 的 `q_ratio` 也明显改变，说明状态继承会改变风险估计，而不是一个无效开关。不过该例全部 4 个样本都是有效参考、权重也都恰好为 1，不能外推到真实 LTCC 的稀疏参考和动态筛选。

机器可读结果见 `diagnostics/phase_npc_temp_optimizer_cpu_sanity.json`。

## 6. 真实 PPU 诊断状态

本执行环境的真实观测：

- PyTorch 为 `2.3.1+cpu`，`torch.cuda.is_available() == False`；
- 没有可见 CUDA/PPU/NPU 设备；
- 外部 `/home/processor` 运行包不存在；
- PR #4 审计脚本显式拒绝 CPU fallback。

因此没有运行下面任何一项 PPU 证据：

| 要求的 PPU 项目 | 本次状态 |
|---|---|
| 短迭代积累正式 SGD `momentum_buffer` | 未执行 |
| False/True 临时更新方向与更新范数 | 未执行（上节仅为 CPU 小例） |
| 可靠参考样本 pre/post loss、风险比值、动态权重 | 未执行 |
| 实际正式更新对可靠参考样本的 loss 变化 | 未执行 |
| 临时更新对正式模型/正式 optimizer 的污染审计 | 未执行（仅有 CPU 单元/小例隔离证据） |
| PPU 上区分 momentum、有效子批、loss 分支的贡献 | 未执行 |

这不是把“未运行”解释为“没有差异”，也不是把 CPU 小例解释为 PPU 改善。完整 PPU 阻塞记录见 `diagnostics/phase_npc_ppu_environment_gate.json`。

## 7. 原因拆分与消融结论

三类因素保持分开：

1. **动量状态差异**：CPU 测试证明 `True` 会复制正式 SGD 的 `momentum_buffer`，相同梯度下能复现正式步；`False` 则保持历史新建临时 SGD 语义。
2. **有效子批差异**：真实代码的临时步只使用有可靠参考的当前样本；CPU 小例中有效率为 100%，所以没有量化真实稀疏有效子批的影响。
3. **损失函数差异**：有效子批不满足 Triplet 条件时，临时路径回退到 identity CE；CPU 小例也没有复现真实 PPU 的 Triplet/CE 分支组合。因此不能把任何未测的风险差异归因于 momentum。

当前没有真实 PPU 证据证明状态继承改善实际风险预测；相反，唯一可验证的结论是它改变了风险估计并提高了相同梯度下的轨迹保真度。故本阶段：

- 不进行 60 轮正式训练或正式消融；
- 保留原 N3：`TEMP_COPY_OPT_STATE=False`；
- 不修改 GMM、权重函数或记忆监督系数；
- 待具备 PPU、`/home/processor` 和新的受控 optimizer state 后，再按 PR #4 通过的 Best/Final 配对门禁重做真实短迭代。

## 8. 已上传的新结果

本报告及以下 JSON 已提交到本独立实验分支：

- `PHASE_NPC_CHECKPOINT_REVALIDATION_REPORT.md`
- `PHASE_NPC_TEMP_OPTIMIZER_FIDELITY_REPORT.md`
- `diagnostics/phase_npc_checkpoint_revalidation_load.json`
- `diagnostics/phase_npc_temp_optimizer_cpu_sanity.json`
- `diagnostics/phase_npc_ppu_environment_gate.json`
