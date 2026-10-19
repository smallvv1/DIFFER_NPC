# PHASE NPC-CHECKPOINT-AUDIT-FIX — FP32 零更新多检查点审计纠错

## 证据及确认的错误

2026-10-19 最新 `PHASE_NPC_FP32_CALIBRATION_REPORT.md` 声称检查了 epoch17 best 与 epoch60 final 两个不同模型，但它所使用的审计脚本虽然先解析了 `--checkpoint`：

```python
checkpoint_path = args.checkpoint.expanduser().resolve()
```

实际 `torch.load` 却使用 `CHECKPOINT_PATH` 常量（旧 best 模型），导致 **传 final 路径时，实际上仍装载旧 best 权重**。输出报告把传入的 final 文件名及 SHA256 记下来，并不代表模型真正来自 final。best/final GMM 数量完全一致可由此解释。

**原报告的校准实验仍能支持旧 best 权重上的配对结论；final 独立校准结论无效，必须撤回重跑。**

## 修改

- `scripts/diagnose_npc_zero_update_fp32.py` 改为 `torch.load(checkpoint_path, ...)`，在加载前打印实际文件路径及 SHA256。
- 新增 `tests/test_npc_audit_checkpoint.py`，解析脚本 AST 阻止后续再次使用硬编码常量装载模型，并检查报告记录的实际加载路径哈希一致。

本分支基于 `experiment/npc-fp32-anchor-preaware-20261019`（已做 FP32 anchor 校准），保持已有 PR #3 不变。

## 服务器上复核流程

1. 独立 worktree 检出此分支，确认未干扰已运行任务。
2. 执行 `python3 -m pytest -q tests/test_npc_audit_checkpoint.py tests/test_npc.py`。
3. 使用 LTCC train-only 数据，在旧 epoch17 best **及真实 epoch60 final** 上各运行两次审计（legacy/FP32），命令例子：

```bash
# 仓库独立 worktree 根目录内，确保主仓库只读代码和 checkpoint 可访问。
BEST=/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth
FINAL=/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_final.pth
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_zero_update_fp32.py --checkpoint "$BEST" --checkpoint-epoch 17 --num-batches 16
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_zero_update_fp32.py --checkpoint "$BEST" --checkpoint-epoch 17 --num-batches 16 --fp32-anchor
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_zero_update_fp32.py --checkpoint "$FINAL" --checkpoint-epoch 60 --num-batches 16
PYTHONPATH=.:/home/DIFFER_NPC python3 scripts/diagnose_npc_zero_update_fp32.py --checkpoint "$FINAL" --checkpoint-epoch 60 --num-batches 16 --fp32-anchor
```

4. 校验 best/final 两个 SHA256 **来自实际 torch.load 路径**；比对参数张量或全参数哈希，确认两模型确实不同；两组各自 no-flag/flag 样本对集合保持相同。
5. 各自产出：GMM 可靠样本及 PID 覆盖、有效参考覆盖、自引用、q/r 与 w<0.9 的严格零更新误判率、模型和优化器状态变化、全量独立 pair 数及训练耗时。
6. 如 final 模型权重与 best 不同且 FP32 校准也恢复 r=w=1，可认为**两个不同检查点**均通过测试。否则只报告已实测范围。
7. 复核成功后，再进行**真实非零临时更新风险辨识试验**，量化更新造成的参考 p/q 变化，评估筛选到的危险样本是否关联身份检索结构或有效交叉服装正样本。该试验与60轮性能实验独立，不得将零更新校准成功直接等价为性能收益。

## 约束

- 禁止覆盖旧结果、改动原 checkpoint、删除报告或自动合并 PR。
- 不修改 GMM、参考选择、原权重公式、动态损失系数及原 DIFFER 协议。
- 当前只有代码提交，没有完成 PPU 重跑；本报告是纠错与复核计划，不是假装已完成的测试结果。
