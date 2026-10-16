#!/usr/bin/env bash
set -euo pipefail

# NPC CE reduction 消融：只改变 CE reduction，保持 N3 的 GMM、记忆库、
# 临时更新、权重函数和记忆损失系数不变。脚本只创建独立实验目录。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIME="${RUN_TIME:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${ROOT}/outputs/NPC_CE_REDUCTION_ABLATION_${RUN_TIME}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/NPC批均值正式训练_${RUN_TIME}.log"
STATUS_FILE="${RUN_ROOT}/正式训练状态_${RUN_TIME}.txt"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
{
    echo "中文实验日志：DIFFER+NPC N3 batch_mean LTCC 正式训练"
    echo "北京时间=${RUN_TIME}"
    echo "代码提交=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "实验分支=$(git -C "${ROOT}" branch --show-current)"
    echo "配置文件=${ROOT}/configs/ltcc/eva02_l_bio.yml"
    echo "唯一消融差异=MODEL.NPC.CE_REDUCTION batch_mean"
    echo "固定协议=Seed1234、60 epoch、EVA02初始化、空TEST.WEIGHT、20261014 N3数据与评估协议"
    echo "禁止项=不修改GMM、记忆库、临时更新、权重函数、记忆损失系数；不加载历史N3 checkpoint"
    echo "输出目录=${RUN_ROOT}"
    echo "日志文件=${LOG_FILE}"
    echo "启动PID=$$"
} | tee -a "${LOG_FILE}"

printf '运行中\n' > "${STATUS_FILE}"
trap 'status=$?; if [[ ${status} -eq 0 ]]; then printf "已完成\n" > "${STATUS_FILE}"; else printf "失败，退出码=%s\n" "${status}" > "${STATUS_FILE}"; fi; echo "正式训练退出码=${status}" | tee -a "${LOG_FILE}"' EXIT

source /home/ppu_display_env.sh
echo "PPU_HOME=${PPU_HOME}" | tee -a "${LOG_FILE}"
echo "PPU_SDK=${PPU_SDK}" | tee -a "${LOG_FILE}"
echo "PPU设备数=$(PYTHONPATH="${ROOT}:/home/DIFFER_NPC" python3 -c 'import torch; print(torch.cuda.device_count())')" | tee -a "${LOG_FILE}"

PYTHONPATH="${ROOT}:/home/DIFFER_NPC" python3 "${ROOT}/npc/runner.py" \
    --config_file "${ROOT}/configs/ltcc/eva02_l_bio.yml" \
    --variant N3 \
    --output-root "${RUN_ROOT}" \
    --max-epochs 60 \
    DATA.ROOT /home/data \
    DATA.CAPTION_DIR /home/data/DIFFER_TextCaption/LTCC_ReID \
    DATA.TEST_BATCH 16 \
    DATA.NUM_WORKERS 1 \
    DATA.PIN_MEMORY False \
    SOLVER.SEED 1234 \
    SOLVER.MAX_EPOCHS 60 \
    TRAIN.START_EPOCH 1 \
    TEST.WEIGHT '' \
    MODEL.NPC.CE_REDUCTION batch_mean \
    2>&1 | tee -a "${LOG_FILE}"
