#!/usr/bin/env bash
set -euo pipefail

# 实验结束后的白名单 GitHub 同步器。
# 用法：auto_sync_github.sh <运行目录> <提交标签> [额外文件...]
# 只同步源代码和文本型实验产物；模型、数据、缓存和特征矩阵会被拒绝。

ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="${1:-}"
RUN_LABEL="${2:-手动同步}"
shift 2 || true

if [[ -z "${RUN_DIR}" ]]; then
    echo "缺少运行目录；同步已拒绝" >&2
    exit 2
fi

if [[ "${RUN_DIR}" != /* ]]; then
    RUN_DIR="${ROOT}/${RUN_DIR}"
fi

case "${RUN_DIR}" in
    "${ROOT}"/*) ;;
    *) echo "运行目录不在仓库内：${RUN_DIR}" >&2; exit 2 ;;
esac

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "运行目录不存在：${RUN_DIR}" >&2
    exit 2
fi

REMOTE_URL="$(git remote get-url origin)"
if [[ "${REMOTE_URL}" != "git@github.com:smallvv1/DIFFER_NPC.git" ]]; then
    echo "origin 不是受控目标仓库，拒绝同步：${REMOTE_URL}" >&2
    exit 3
fi

BRANCH="$(git branch --show-current)"
if [[ "${BRANCH}" != "main" ]]; then
    echo "当前分支不是 main，拒绝自动推送：${BRANCH}" >&2
    exit 3
fi

# 先纳入已跟踪文件的代码修改，再纳入代码目录中新增加的源文件。
git add -u -- .
for SOURCE_DIR in config configs loss model npc ndp scripts solver utils tools; do
    if [[ -d "${ROOT}/${SOURCE_DIR}" ]]; then
        git add -- "${SOURCE_DIR}"
    fi
done

# 运行目录只允许提交日志、报告、配置、manifest、CSV/JSON 诊断等文本产物。
# 特征矩阵 .npy/.npz 已在 .gitignore 中排除；下面的拒绝检查是第二道门。
while IFS= read -r -d '' FILE_PATH; do
    git add -- "${FILE_PATH}"
done < <(
    find -P "${RUN_DIR}" -type f \( \
        -name '*.log' -o -name '*.md' -o -name '*.txt' -o -name '*.json' \
        -o -name '*.csv' -o -name '*.yml' -o -name '*.yaml' \
    \) -print0
)

# 调用方可以显式指定本次运行之外的单个报告或日志。
for EXTRA_PATH in "$@"; do
    case "${EXTRA_PATH}" in
        "${ROOT}"/*) SAFE_EXTRA="${EXTRA_PATH}" ;;
        *) SAFE_EXTRA="${ROOT}/${EXTRA_PATH}" ;;
    esac
    case "${SAFE_EXTRA}" in
        "${ROOT}"/*) ;;
        *) echo "额外文件不在仓库内：${EXTRA_PATH}" >&2; exit 2 ;;
    esac
    if [[ ! -f "${SAFE_EXTRA}" ]]; then
        echo "额外文件不存在：${SAFE_EXTRA}" >&2
        exit 2
    fi
    git add -- "${SAFE_EXTRA}"
done

STAGED_FILES="$(git -c core.quotePath=false diff --cached --name-only)"
if [[ -z "${STAGED_FILES}" ]]; then
    echo "没有需要同步的代码或实验日志。"
    exit 0
fi

FORBIDDEN_FILES="$(printf '%s\n' "${STAGED_FILES}" | rg -i '(^|/)(data|cache|checkpoints?)(/|$)|\.(pth|pt|safetensors|bin|npy|npz|deb)$' || true)"
if [[ -n "${FORBIDDEN_FILES}" ]]; then
    echo "检测到禁止上传的文件，拒绝提交：" >&2
    echo "${FORBIDDEN_FILES}" >&2
    exit 4
fi

while IFS= read -r FILE_PATH; do
    [[ -n "${FILE_PATH}" ]] || continue
    FILE_SIZE="$(stat -c '%s' "${ROOT}/${FILE_PATH}")"
    if (( FILE_SIZE > 100 * 1024 * 1024 )); then
        echo "单文件超过 100 MiB，拒绝自动上传：${FILE_PATH}" >&2
        exit 4
    fi
done <<< "${STAGED_FILES}"

echo "即将同步以下文件："
printf '%s\n' "${STAGED_FILES}"
git commit -m "实验同步：${RUN_LABEL}"
git push origin "${BRANCH}"
echo "GitHub 同步完成：${REMOTE_URL}，分支=${BRANCH}，提交=$(git rev-parse --short HEAD)"
