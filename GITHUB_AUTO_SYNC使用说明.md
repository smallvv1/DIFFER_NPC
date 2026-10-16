# 实验结束自动同步 GitHub

仓库：`git@github.com:smallvv1/DIFFER_NPC.git`

## 默认规则

每次实验结束后，使用白名单同步：

- 上传源代码目录中的代码修改和新增源文件；
- 上传指定运行目录中的 `.log`、`.md`、`.txt`、`.json`、`.csv`、`.yml`、`.yaml`；
- 不上传 checkpoint、模型权重、数据集、缓存、`.npy/.npz` 特征矩阵或超过 100 MiB 的单文件；
- 不使用 `git add .`，不强制 push，不覆盖远端历史。

## 推荐用法

```bash
/home/DIFFER_NPC/scripts/run_with_github_sync.sh \
  --run-dir /home/DIFFER_NPC/outputs/某次实验目录 \
  --label "实验名称" \
  -- python3 /home/DIFFER_NPC/scripts/实验入口.py
```

命令成功或失败后都会尝试同步。若只需要同步已经完成的实验：

```bash
/home/DIFFER_NPC/scripts/auto_sync_github.sh \
  /home/DIFFER_NPC/outputs/某次实验目录 \
  "实验名称" \
  /home/DIFFER_NPC/reports/本次报告.md
```

仅在明确需要离线运行时设置 `NDP_DISABLE_GITHUB_SYNC=1`；默认不跳过同步。
