#!/bin/bash
# 启动 vllm DeepSeek-V4-Flash 服务脚本 (SM120/Blackwell) — 优化版 V2
# 基于 ds4-sm120-preview-dev 分支
# Changelog:
#   V2: +FlashInfer SM120 decode, +NCCL PCIe tuning, +SM120 fusion tables, +gpu_mem 0.95

# ==================== 配置区域 ====================
INDEX="1"                           # 脚本序号
MODEL_SHORT="dsv4"       # 日志文件中的模型名
PORT=8006                          # 实验模型端口（8006-8009）

MODEL_PATH="/models/models/deepseek-ai/DeepSeek-V4-Flash/"    # 模型绝对路径
MODEL_NAME="DeepSeek-V4-Flash"            # API上展示的模型名
GPUS="0,1,2,3"                         # 使用的GPU设备 (TP=4, 95GB * 4)
TP_SIZE=4                              # 张量并行大小
PP_SIZE=1                          # 流水线并行大小
VRAM_RATE=0.95                       # 显存使用率（ref: 0.91→6.0x, 0.95→~6.4x concurrency）
CONTEXT_LENGTH=1048576               # 单序列最大长度
MAX_NUM_SEQ=1024                      # 同时生成的序列数量
KV_CACHE_DTYPE="fp8"               # KV cache dtype
BLOCK_SIZE=256                     # Attention block size
CUDAGRAPH_MODE="FULL_AND_PIECEWISE" # CUDAGraph mode
LOG_POSTFIX="TP${TP_SIZE}PP${PP_SIZE}"             # 日志名尾缀
LOG_FILE_NAME="LOG_${INDEX}_${MODEL_SHORT}_${LOG_POSTFIX}.log"  # 日志文件
PID_FILE_NAME="PID_${INDEX}_${MODEL_SHORT}.pid"      # PID文件

# ===== NCCL / PCIe 通信优化 =====
# PCIe Gen5 x16, 4 GPUs, no NVLink — Ring algo + Simple proto for stable PCIe BW
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NTHREADS=256
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8
export NCCL_CHECKS_DISABLE=1
# =================================

# ===== FlashInfer / SM120 优化 =====
# 启用 FlashInfer 官方的 SM120 packed sparse-MLA decode kernel
export VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE=1
# 启用 FlashInfer allreduce backend（实验性）
export VLLM_ALLREDUCE_USE_FLASHINFER=1
# FlashInfer allreduce 选择 trtllm 后端
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
# FlashInfer workspace buffer（默认 394MB）
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1048576000
# =================================

# CUDA / 基础环境变量
export CUDA_HOME="/opt/cuda-13.0.3"
export PATH="/opt/cuda-13.0.3/bin:$PATH"
export TRITON_PTXAS_PATH="/opt/cuda-13.0.3/bin/ptxas"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "启动 vllm DeepSeek-V4-Flash 服务 (SM120) — 优化版 V2"
echo "模型路径: ${MODEL_PATH}"
echo "模型名称: ${MODEL_NAME}"
echo "GPU设备:  ${CUDA_VISIBLE_DEVICES}"
echo "张量并行: ${TP_SIZE}"
echo "流水线并行: ${PP_SIZE}"
echo "服务端口: ${PORT}"
echo "显存利用率: ${VRAM_RATE}"
echo "上下文长度: ${CONTEXT_LENGTH}"
echo "KV Cache: ${KV_CACHE_DTYPE}"
echo "Block Size: ${BLOCK_SIZE}"
echo "CUDAGraph: ${CUDAGRAPH_MODE}"
echo "NCCL Algo: ${NCCL_ALGO}"
echo "FlashInfer SM120 Decode: ${VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE}"
echo "FlashInfer Allreduce: ${VLLM_ALLREDUCE_USE_FLASHINFER}"
echo "============================================"

# 检查端口是否被占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo "ERROR: 端口 ${PORT} 已被占用!"
    exit 1
fi

# 启动服务
nohup setsid ../.venv/bin/vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --trust-remote-code \
    --kv-cache-dtype ${KV_CACHE_DTYPE} \
    --block-size ${BLOCK_SIZE} \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --gpu-memory-utilization "${VRAM_RATE}" \
    --max-model-len "${CONTEXT_LENGTH}" \
    --max-num-seqs "${MAX_NUM_SEQ}" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\", \"custom_ops\":[\"all\"]}" \
    --async-scheduling \
    --enable-prefix-caching \
    --load-format auto \
    --enable-expert-parallel \
    --host 0.0.0.0 \
    --port ${PORT} \
    > "${LOG_FILE_NAME}" 2>&1 &

echo $! > "$PID_FILE_NAME"
echo "DeepSeek-V4-Flash 已启动, PID: $(cat ${PID_FILE_NAME})"
echo "日志: ${LOG_FILE_NAME}"
