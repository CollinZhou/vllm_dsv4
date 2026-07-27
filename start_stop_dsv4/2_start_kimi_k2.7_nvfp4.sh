#!/bin/bash
# 启动 vllm Kimi-K2.7-Code NVFP4 服务脚本 (SM120/Blackwell, 6卡)
# 基于 ds4-sm120-preview-dev 分支
# NVFP4 量化 (modelopt_fp4), TP=2, EP=3 (自动分配)
# 6卡 -> TP=2 × EP=3 × PP=1
# EP 不分割 attention heads，只分割 expert，TP=2 平分 64 heads → 32 heads/卡

# ==================== 配置区域 ====================
INDEX="2"                           # 脚本序号
MODEL_SHORT="kimi"                  # 日志文件中的模型名
PORT=8025                           # 服务端口

MODEL_PATH="/models/models/nv-community/Kimi-K2.7-Code-NVFP4/"    # 模型绝对路径
MODEL_NAME="Kimi-K2.7-Code-NVFP4"   # API上展示的模型名
GPUS="0,1,2,3,4,5"                  # 使用的GPU设备 (6张全部)
TP_SIZE=2                           # 张量并行大小 (64 heads / 2 = 32 heads/卡)
PP_SIZE=1                           # 流水线并行大小 (EP模式不需要PP)
VRAM_RATE=0.95                      # 显存使用率 (不要降低，否则权重放不下)
CONTEXT_LENGTH=262144               # 单序列最大长度 (256K)
MAX_NUM_SEQ=512                     # 同时生成的序列数量
KV_CACHE_DTYPE="fp8"               # KV cache dtype
BLOCK_SIZE=256                      # Attention block size
CUDAGRAPH_MODE="FULL_AND_PIECEWISE" # CUDAGraph模式
LOG_POSTFIX="TP${TP_SIZE}EP3PP${PP_SIZE}"             # 日志名尾缀
LOG_FILE_NAME="LOG_${INDEX}_${MODEL_SHORT}_${LOG_POSTFIX}.log"  # 日志文件
PID_FILE_NAME="PID_${INDEX}_${MODEL_SHORT}.pid"      # PID文件

# ===== NCCL / 通信优化 =====
# EP 模式需要频繁 all-to-all，NCCL auto-tune 即可
# =================================

# ===== FlashInfer / SM120 优化 =====
export VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLREDUCE_USE_FLASHINFER=0
# =================================

# CUDA / 基础环境变量
export CUDA_HOME="/opt/cuda-13.0.3"
export PATH="/opt/cuda-13.0.3/bin:$PATH"
export TRITON_PTXAS_PATH="/opt/cuda-13.0.3/bin/ptxas"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "启动 Kimi-K2.7-Code NVFP4 服务 (SM120) — TP=${TP_SIZE} EP=auto(3) PP=${PP_SIZE}"
echo "模型路径: ${MODEL_PATH}"
echo "模型名称: ${MODEL_NAME}"
echo "GPU设备:  ${CUDA_VISIBLE_DEVICES}"
echo "张量并行: ${TP_SIZE} (不切分)"
echo "流水线并行: ${PP_SIZE}"
echo "专家并行:  auto (总数=6, TP=2, 自动分配 EP=3)"
echo "服务端口: ${PORT}"
echo "显存利用率: ${VRAM_RATE}"
echo "上下文长度: ${CONTEXT_LENGTH}"
echo "KV Cache: ${KV_CACHE_DTYPE}"
echo "Block Size: ${BLOCK_SIZE}"
echo "CUDAGraph: ${CUDAGRAPH_MODE}"
echo "FlashInfer SM120 Decode: ${VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE}"
echo "============================================"

# 检查端口是否被占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo "ERROR: 端口 ${PORT} 已被占用!"
    exit 1
fi

# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: 模型路径 ${MODEL_PATH} 不存在!"
    echo "请先下载模型: huggingface-cli download nvidia/Kimi-K2.7-Code-NVFP4 --local-dir ${MODEL_PATH}"
    exit 1
fi

# 启动服务
# - NVFP4 会被 auto-detect (ModelOptNvFp4Config.override_quantization_method)
# - EP=3 自动分配（总GPU/TP=6/2=3个EP组，每组2卡各128个专家）
# - --trust-remote-code 因为模型需要自定义 configuration_*.py
# - --dtype bfloat16 因为模型 config.json 中 dtype=bfloat16，NVFP4 只在 routed experts 上生效
nohup setsid ../.venv/bin/vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype ${KV_CACHE_DTYPE} \
    --block-size ${BLOCK_SIZE} \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --gpu-memory-utilization "${VRAM_RATE}" \
    --max-model-len "${CONTEXT_LENGTH}" \
    --max-num-seqs "${MAX_NUM_SEQ}" \
    --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\", \"custom_ops\":[\"all\"]}" \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --language-model-only \
    --host 0.0.0.0 \
    --port ${PORT} \
    > "${LOG_FILE_NAME}" 2>&1 &

echo $! > "$PID_FILE_NAME"
echo "Kimi-K2.7-Code-NVFP4 已启动, PID: $(cat ${PID_FILE_NAME})"
echo "日志: ${LOG_FILE_NAME}"
