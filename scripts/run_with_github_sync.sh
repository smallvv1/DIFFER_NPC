#!/usr/bin/env bash
set -u

# 通用实验包装器：命令结束（成功或失败）后都尝试上传代码和指定运行目录日志。
# 用法：
#   run_with_github_sync.sh --run-dir outputs/RUN --label "实验名称" -- command arg1 arg2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR=""
RUN_LABEL="实验运行"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --label) RUN_LABEL="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "未知参数：$1" >&2; exit 2 ;;
    esac
done

if [[ -z "${RUN_DIR}" || $# -eq 0 ]]; then
    echo "用法：$0 --run-dir <运行目录> --label <标签> -- <实验命令> [参数...]" >&2
    exit 2
fi

case "${RUN_DIR}" in
    /*) ABS_RUN_DIR="${RUN_DIR}" ;;
    *) ABS_RUN_DIR="${ROOT}/${RUN_DIR}" ;;
esac

"$@"
RUN_STATUS=$?

if [[ "${NDP_DISABLE_GITHUB_SYNC:-0}" == "1" ]]; then
    echo "NDP_DISABLE_GITHUB_SYNC=1：跳过自动 GitHub 同步。" >&2
    exit "${RUN_STATUS}"
fi

"${ROOT}/scripts/auto_sync_github.sh" "${ABS_RUN_DIR}" "${RUN_LABEL}"
SYNC_STATUS=$?

if [[ "${RUN_STATUS}" -ne 0 ]]; then
    exit "${RUN_STATUS}"
fi
exit "${SYNC_STATUS}"
