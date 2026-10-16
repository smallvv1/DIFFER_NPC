#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIME="${1:?需要传入正式训练北京时间 RUN_TIME，例如 20261016_140800}"
RUN_ROOT="${ROOT}/outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}"
STATUS_FILE="${RUN_ROOT}/正式训练状态_${RUN_TIME}.txt"
PRE_ROOT="${ROOT}/outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256"
FINAL_LOG="${ROOT}/logs/NPC批均值自动收尾_${RUN_TIME}.log"

{
    echo "中文自动收尾日志：等待 NPC CE reduction 正式训练结束"
    echo "北京时间启动标识=${RUN_TIME}"
    echo "正式目录=${RUN_ROOT}"
} >> "${FINAL_LOG}"

while [[ ! -f "${STATUS_FILE}" || "$(<"${STATUS_FILE}")" == "运行中" ]]; do
    sleep 60
done

echo "训练状态=$(<"${STATUS_FILE}")" | tee -a "${FINAL_LOG}"
PYTHONPATH="${ROOT}:/home/DIFFER_NPC" python3 "${ROOT}/scripts/generate_npc_ce_reduction_report.py" \
    --root "${RUN_ROOT}" \
    --preflight "${PRE_ROOT}" \
    --baseline "${ROOT}/outputs/NPC_CONFIRMATORY_20261014_140116" \
    --output "${ROOT}/PHASE_NPC_CE_REDUCTION_ABLATION_REPORT.md" \
    2>&1 | tee -a "${FINAL_LOG}"

# 上传代码、报告、日志、配置、指标和轻量诊断；checkpoints 由 .gitignore 排除，
# 不进入 GitHub。逐样本权重与 reliability 全量 CSV 体积较大，完整文件保留在
# 服务器实验目录，报告使用其离线汇总结果。
git -C "${ROOT}" add \
    "PHASE_NPC_CE_REDUCTION_ABLATION_REPORT.md" \
    "logs/NPC批均值正式训练启动_${RUN_TIME}.log" \
    "logs/NPC批均值正式训练_${RUN_TIME}.log" \
    "logs/NPC批均值自动收尾_${RUN_TIME}.log" \
    "logs/NPC批均值真实LTCC短程预检_20261016_140256.log" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/正式训练状态_${RUN_TIME}.txt" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/ltcc/seed1234/N3/config.yml" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/ltcc/seed1234/N3/train_log.txt" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/ltcc/seed1234/N3/metrics/metrics_per_epoch.csv" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/ltcc/seed1234/N3/metrics/run_summary.json" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/ltcc/seed1234/N3/npc_diagnostics/gmm_epoch_*.json" \
    "outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}/ltcc/seed1234/N3/npc_diagnostics/memory_epoch_*.json" \
    "outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256/ltcc/seed1234/N3/config.yml" \
    "outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256/ltcc/seed1234/N3/train_log.txt" \
    "outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256/ltcc/seed1234/N3/metrics/metrics_per_epoch.csv" \
    "outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256/ltcc/seed1234/N3/metrics/run_summary.json" \
    "outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256/ltcc/seed1234/N3/npc_diagnostics/gmm_epoch_*.json" \
    "outputs/NPC_CE_REDUCTION_ABLATION_20261016_140256/ltcc/seed1234/N3/npc_diagnostics/memory_epoch_*.json"

if ! git -C "${ROOT}" diff --cached --quiet; then
    git -C "${ROOT}" commit -m "NPC: archive batch mean ablation logs and report"
    git -C "${ROOT}" push origin experiment/npc-ce-batch-mean-20261016
    echo "中文自动收尾：报告与非checkpoint产物已推送实验分支" | tee -a "${FINAL_LOG}"
else
    echo "中文自动收尾：没有需要提交的新产物" | tee -a "${FINAL_LOG}"
fi
