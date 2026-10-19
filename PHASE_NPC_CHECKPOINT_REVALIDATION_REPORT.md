# PHASE NPC 检查点复核报告

日期：2026-10-19  
范围：PR #4 `83d453a`（`experiment/npc-checkpoint-risk-audit-fix-20261019`）  
状态：**部分完成；PPU 零更新审计被当前执行环境阻塞，不能宣称 PR #4 的完整 Best/Final 通过**

## 1. GitHub 同步检查

本次通过 HTTPS 只读 refs 检查 GitHub，而不是使用主工作区的 SSH remote：

| ref | 实际远端提交 |
|---|---|
| `refs/heads/main` | `9d4cb06fde48aeab53e3aab3bb833e50ad2ca875` |
| `refs/pull/4/head` | `83d453a0fa0a188f302c05f45d15c0235e90deb2` |
| `refs/pull/5/head` | `d5e075e676d2438d6455bc83df0868ba427a1fc5` |

截至本复核，GitHub 的 PR #4 分支没有 `PHASE_NPC_CHECKPOINT_REVALIDATION_REPORT.md`，也没有新的 PPU 复核日志。PR #5 当前只包含候选代码、测试和诊断计划，不能作为已经完成 PPU 诊断的证据。

## 2. PR #4 代码与测试

PR #4 的关键修复提交为：

- `7af391f`：审计脚本按 `--checkpoint` 实际参数加载权重并记录真实来源哈希；
- `f92b042`：审计侧防止静默回退到旧 best 检查点；
- `83d453a`：记录错误原因及 paired repeat protocol。

在独立 worktree 中运行：

```text
PYTHONPATH=. /home/DIFFER_NPC_worktrees/.venv-npc-cpu/bin/python -m pytest -q tests/test_npc_audit_checkpoint.py
..                                                                       [100%]
2 passed in 0.02s
```

AST 回归测试确认唯一的 `torch.load` 使用解析后的 `checkpoint_path`，且报告的 SHA256 也来自同一路径。这个测试通过只证明代码路径正确，不等于 PPU 审计已经运行。

## 3. Best/Final 实际检查点配对

以下是本次实际读取的文件；没有使用旧 Final 审计结果替代本次证据。

| 检查点 | 实际 `torch.load` 路径 | 文件字节数 | SHA256 | CPU `torch.load` |
|---|---|---:|---|---|
| Best | `/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth` | 1,215,917,953 | `0b33bc4d02b70a1d8d83a7ecdc8dd801a479c307685fa537a39276e2188ee7db` | PASS |
| Final | `/home/DIFFER_NPC/outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_final.pth` | 1,215,918,472 | `e275cdcabc199988fbe5f77a17e3a70abcf9c28b4c03bb5d70d0befe5b08137b` | PASS |

两者均加载为 `OrderedDict`，各含 515 个张量、共 303,933,965 个元素。逐张量比较得到：键集合相同，但 513/515 个张量不同，最大绝对差 `0.9905719757`，全局 L2 差 `20.1788710065`。因此 Best 与 Final 确实是两个不同模型；不能再用“二者结果相同”支持任何独立 Final 结论。

## 4. 真实零更新审计的阻塞边界

PR #4 的真实脚本仍明确要求 PPU/CUDA，并在 `torch.cuda.is_available()` 为假时拒绝 CPU 回退。本执行容器的事实是：

- PyTorch 仅安装了 `2.3.1+cpu`，`torch.cuda.is_available() == False`；
- `/dev` 中没有可见 CUDA/PPU/NPU 设备；
- 项目真实训练所需的外部 `/home/processor` 运行包不存在；
- 因此无法在此环境构造与正式协议一致的 LTCC train-only loader、模型运行态、FP32 锚点、GMM 和 zero-update 风险统计。

所以本报告**没有**伪造以下结果，也没有读取旧 Final 审计作为替代：Best/Final 的 PPU zero-update、两种锚点模式下的 GMM/参考覆盖、`r=w=1`、模型/优化器状态变化及 paired sample 集合。

## 5. 结论

- **已确认**：PR #4 的实际加载路径修复存在；Best/Final 实际文件路径和 SHA256 已重新读取；二者模型权重确实不同；PR #4 AST 回归测试通过。
- **未确认**：真实 PPU 上的 Best/Final paired zero-update audit 尚未完成，因此 PR #4 的“完整复核通过”不能成立。
- **实验门控**：在 PPU 环境和 `/home/processor` 运行包恢复前，不启动 PR #5 的真实短迭代，也不把任何旧日志升级为本次证据。

原始检查点和主分支工作区均未修改；本报告只写入本独立实验分支。
