# NPC 临时 SGD 动量状态一致性：只读诊断与可选消融方案

## 当前可核对的远程证据（2026-10-19 20:19 北京时间）

- 可访问的 GitHub `main` 最近提交仍为 `9d4cb06fde48aeab53e3aab3bb833e50ad2ca875`。
- 已有 PR #4（`experiment/npc-checkpoint-risk-audit-fix-20261019`）修复了不同检查点的实际装载路径，但截至本次检查，远程仓库尚未出现 **PHASE_NPC_CHECKPOINT_REVALIDATION_REPORT.md** 或新 PPU 复核日志。
- 旧 best epoch17 的 FP32 锚点零步测试已有支持性结果；曾标注 final epoch60 的结果因为 `torch.load` 固定读取 best 而不能作为独立证据，需在 PR #4 修复版上重做。
- 已完成 LTCC Seed1234 60轮的 B0/N2/N3：最优 Rank-1（同轮 mAP）分别是 50.26%(27.15%) / 56.38%(31.51%) / 50.77%(28.54%)。CE batch_mean 单变量 N3 未改善（50.77%/27.39%）。
- 上述数据不能推断新的临时优化器状态改动已经改善效果。

## 已确认的代码差异

旧 `npc/engine.py::_make_temporary_optimizer` 为每个批次创建新 SGD 优化器，只复制正式优化器当前各组超参数（包括 LR、momentum 系数等），**不复制动量缓冲区**。

正式 LTCC SGD 的 momentum 配置为0.9。如果正式优化器已有历史动量 m，那么真实 SGD 更新与临时优化器的首步更新可能不同，即使样本梯度一致。这个问题与预感知测量的 AMP/FP32 不一致不同，必须作为单独假设检查。

此外即便复制历史动量，NPC 临时步仍使用**仅有有效参考的锚点子批**，而正式更新在**全部锚点批次**进行，并且临时 loss 在 Triplet 不兼容时可能退化为 CE；因此不能宣称复制动量后便模拟了完整真实训练更新。

## 本分支代码

父分支为 PR #4 的检查点修复分支。本分支默认保持历史行为：

```yaml
MODEL:
  NPC:
    TEMP_COPY_OPT_STATE: false
```

仅设置为 `true` 时：
1. 必须是 SGD 正式优化器；不支持的优化器明确报错。
2. 依据参数名将每个正式优化器参数组准确映射到隔离的临时模型参数；遇到错配立即失败。
3. 深复制正式 SGD 参数级优化器状态（包括 momentum_buffer）；无张量别名，移动到临时模型设备。
4. 只在临时模型、临时优化器上进行步进；绝不把状态写回正式优化器。

新增 `tests/test_npc_temp_sgd_state.py` 涵盖：默认不复制、状态数值/存储隔离、同模型同梯度下一步轨迹一致、非 SGD 拒绝。

## 先做的诊断（不得直接开展60轮训练）

1. 等待 PR #4 的 checkpoint best/final 真正重测，确认实际哈希及零步校准。
2. 在服务器独立 worktree 执行：
   ```bash
   PYTHONPATH=.:/home/DIFFER_NPC python3 -m pytest -q tests/test_npc.py tests/test_npc_audit_checkpoint.py tests/test_npc_temp_sgd_state.py
   ```
3. 在 PPU 使用已完成的 LTCC N3 **训练阶段中间检查点（如果有优化器状态则直接读取）**。现有 `model_best.pth`/ `model_final.pth` 只保存 model.state_dict，**不能恢复 SGD 历史动量**。如果没有 optimizer checkpoint，不能伪称有真实历史动量；必须进行受控真实短迭代训练以重新积累正式动量状态，且不得覆盖旧检查点。
4. 在同一次受控正式优化器状态下，分别生成 `TEMP_COPY_OPT_STATE=false/true` 临时步；保持相同样本、初始权重、fp32 校准、优化器组超参和梯度，比较：
   - 临时步参数位移方向/范数与正式步的差异；
   - 参考条目 pre/post p/q 损失及 r 和原 NPC 权重差异；
   - 有效参考比例及对应 PID/衣服状态；
   - 正式模型/优化器是否受到临时分支污染。
5. 区分“相同梯度下数值模拟是否正确”与“真实训练不同子批/不同损失下，风险预测是否更有效”两种结论。
6. 若短程证据不足，则保持新分支 Draft，不进行60轮消融。只有在风险信号与实际更新效果吻合度有可信改善且 FP32 校准已通过的前提下，才设计 N3-TEMP-MOMENTUM 单变量完整实验，绝不把 GMM、CE reduction、连续权重同时加入。

## 安全约束

新字段默认关闭，不修改 `main`、旧 B0/N2/N3 结果、PR #1–4 原分支，不终止现有进程。上述代码是**待测试的候选修复**，本对话环境未在 PPU 执行测试。
