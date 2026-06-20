#!/bin/bash
# 停止 vllm DeepSeek-V4-Flash 服务脚本（INDEX=1）

MODEL_SHORT="vllm_deepseekv4_flash"
PORT=8006

INDEX="1"
PID_FILE_NAME="PID_${INDEX}_${MODEL_SHORT}.pid"

# 读取 PID 文件，获取主进程号
if [ -f "$PID_FILE_NAME" ]; then
    PID=$(cat "$PID_FILE_NAME")
    PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')

    if [ -n "$PGID" ]; then
        # 正常终止：向整个进程组发送 SIGTERM（含 EngineCore/Worker）
        echo "正在停止进程组 PGID: $PGID, 主 PID: $PID"
        kill -- -"$PGID" 2>/dev/null

        # 等待 vLLM 正常退出
        sleep 3

        # 强制杀死残留进程
        kill -9 -- -"$PGID" 2>/dev/null
    else
        echo "PID $PID 不存在，跳过进程组停止"
    fi

    rm -f "$PID_FILE_NAME"
fi

# 端口级别兜底清理（fuser 针对本服务端口，不会误杀其他 vllm 服务）
fuser -k "$PORT"/tcp 2>/dev/null

echo "服务已停止，请检查 GPU 显存是否释放"
nvidia-smi --query-gpu=index,memory.used --format=csv
nvidia-smi --query-gpu=index,memory.used --format=csv 2>&1
