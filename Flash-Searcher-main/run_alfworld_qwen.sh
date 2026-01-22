#!/bin/bash
# ============================================================================
# Run Alfworld tasks with Qwen2.5-7B-Instruct model on GPU 0 and 1
# 
# This script:
# 1. Starts a vLLM server with Qwen2.5-7B-Instruct on GPU 0,1
# 2. Runs Alfworld evaluation tasks
# 3. Cleans up the server on exit
#
# Requirements:
# - vLLM installed: pip install vllm
# - CUDA-capable GPUs (0 and 1)
# - Alfworld data downloaded
# ============================================================================

set -e  # Exit on error

# ==================== Configuration ====================
# GPU configuration
CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# If you have the model locally, you can set:
# MODEL_NAME="/path/to/local/Qwen2.5-7B-Instruct"

# vLLM server configuration
VLLM_HOST="localhost"
VLLM_PORT=8000
VLLM_API_KEY="EMPTY"  # vLLM doesn't require API key by default
VLLM_MAX_MODEL_LEN=8192  # Maximum sequence length

# Alfworld task configuration
ALFWORLD_SPLIT="test"  # Options: train, dev, test
SAMPLE_NUM=10  # Number of tasks to run (set to null or empty for all)
MAX_STEPS=50
CONCURRENCY=1  # Number of concurrent tasks (each uses separate env)
MEMORY_PROVIDER=""  # e.g., "lightweight_memory", "agent_kb", or leave empty

# Output configuration
OUTPUT_DIR="./alfworld_output"
OUTPUT_FILE="${OUTPUT_DIR}/qwen_results.jsonl"

# Log directory
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# ==================== Functions ====================

# Function to check if port is in use
check_port() {
    local port=$1
    if command -v lsof > /dev/null 2>&1; then
        if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            return 0  # Port is in use
        fi
    elif command -v netstat > /dev/null 2>&1; then
        if netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            return 0  # Port is in use
        fi
    fi
    return 1  # Port is free
}

# Function to wait for server to be ready
wait_for_server() {
    local max_attempts=120  # 4 minutes max wait time
    local attempt=0
    echo "Waiting for vLLM server to be ready (this may take a few minutes for model loading)..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo ""
            echo "✓ vLLM server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $((attempt % 10)) -eq 0 ]; then
            echo "  Still waiting... (${attempt}/${max_attempts})"
        else
            echo -n "."
        fi
        sleep 2
    done
    
    echo ""
    echo "ERROR: vLLM server failed to start within $((max_attempts * 2)) seconds"
    echo "Check logs: ${LOG_DIR}/vllm_server.log"
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    if [ ! -z "${VLLM_PID}" ]; then
        echo "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill ${VLLM_PID} 2>/dev/null || true
        # Wait a bit for graceful shutdown
        sleep 3
        # Force kill if still running
        kill -9 ${VLLM_PID} 2>/dev/null || true
        wait ${VLLM_PID} 2>/dev/null || true
        echo "vLLM server stopped."
    fi
    echo "Cleanup complete."
    echo "=========================================="
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# ==================== Main Script ====================

echo "=========================================="
echo "Alfworld Evaluation with Qwen2.5-7B-Instruct"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "Split: ${ALFWORLD_SPLIT}"
if [ ! -z "${SAMPLE_NUM}" ] && [ "${SAMPLE_NUM}" != "null" ]; then
    echo "Sample Num: ${SAMPLE_NUM}"
else
    echo "Sample Num: All tasks"
fi
echo "Max Steps: ${MAX_STEPS}"
echo "Concurrency: ${CONCURRENCY}"
if [ ! -z "${MEMORY_PROVIDER}" ]; then
    echo "Memory Provider: ${MEMORY_PROVIDER}"
else
    echo "Memory Provider: None"
fi
echo "=========================================="
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "ERROR: vLLM is not installed."
    echo ""
    echo "Please install it with:"
    echo "  pip install vllm"
    echo ""
    echo "Or with specific CUDA version:"
    echo "  pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# Check if curl is available (for health check)
if ! command -v curl &> /dev/null; then
    echo "WARNING: curl is not installed. Health check will be skipped."
    echo "Please install curl or modify the script to use another method."
fi

# Check if port is available
if check_port ${VLLM_PORT}; then
    echo "WARNING: Port ${VLLM_PORT} is already in use."
    echo "Please stop the service using this port or change VLLM_PORT in the script."
    echo ""
    echo "To find what's using the port:"
    echo "  lsof -i :${VLLM_PORT}"
    echo "  or"
    echo "  netstat -tuln | grep ${VLLM_PORT}"
    exit 1
fi

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.device_count()} GPUs')" 2>/dev/null; then
    echo "WARNING: CUDA may not be available. The script will continue but may fail."
    echo ""
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# Start vLLM server
echo "=========================================="
echo "Starting vLLM server..."
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Host: ${VLLM_HOST}"
echo "Port: ${VLLM_PORT}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Tensor Parallel Size: ${NUM_GPUS}"
echo "Max Model Length: ${VLLM_MAX_MODEL_LEN}"
echo "Logs: ${LOG_DIR}/vllm_server.log"
echo ""

# Start vLLM server in background
# Note: --trust-remote-code is needed for Qwen models
vllm serve ${MODEL_NAME} \
    --host ${VLLM_HOST} \
    --port ${VLLM_PORT} \
    --tensor-parallel-size ${NUM_GPUS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.9 \
    > ${LOG_DIR}/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: ${VLLM_PID}"
echo ""

# Wait for server to be ready
if ! wait_for_server; then
    echo ""
    echo "Failed to start vLLM server."
    echo "Last 50 lines of server log:"
    echo "----------------------------------------"
    tail -50 ${LOG_DIR}/vllm_server.log
    echo "----------------------------------------"
    exit 1
fi

# Set environment variables for the evaluation script
export DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
export OPENAI_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${VLLM_API_KEY}"

echo ""
echo "=========================================="
echo "Environment Configuration"
echo "=========================================="
echo "DEFAULT_MODEL=${DEFAULT_MODEL}"
echo "OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "OPENAI_API_KEY=${OPENAI_API_KEY}"
echo "=========================================="
echo ""

# Prepare output directory
mkdir -p ${OUTPUT_DIR}

# Build command arguments
CMD_ARGS=(
    "--split" "${ALFWORLD_SPLIT}"
    "--outfile" "${OUTPUT_FILE}"
    "--max_steps" "${MAX_STEPS}"
    "--concurrency" "${CONCURRENCY}"
    "--direct_output_dir" "${OUTPUT_DIR}/qwen_runs"
)

if [ ! -z "${SAMPLE_NUM}" ] && [ "${SAMPLE_NUM}" != "null" ]; then
    CMD_ARGS+=("--sample_num" "${SAMPLE_NUM}")
fi

if [ ! -z "${MEMORY_PROVIDER}" ]; then
    CMD_ARGS+=("--memory_provider" "${MEMORY_PROVIDER}")
fi

# Run Alfworld evaluation
echo "=========================================="
echo "Starting Alfworld Evaluation"
echo "=========================================="
echo "Command:"
echo "  python run_flash_searcher_alfworld.py ${CMD_ARGS[*]}"
echo ""
echo "Evaluation logs: ${LOG_DIR}/alfworld_eval.log"
echo "=========================================="
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Run evaluation
python run_flash_searcher_alfworld.py "${CMD_ARGS[@]}" 2>&1 | tee ${LOG_DIR}/alfworld_eval.log

EVAL_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo ""
    echo "Results:"
    echo "  - Results file: ${OUTPUT_FILE}"
    echo "  - Report: ${OUTPUT_DIR}/qwen_runs/report.txt"
    echo "  - Individual task results: ${OUTPUT_DIR}/qwen_runs/"
    echo "  - Evaluation log: ${LOG_DIR}/alfworld_eval.log"
else
    echo "✗ Evaluation failed with exit code: ${EVAL_EXIT_CODE}"
    echo ""
    echo "Check logs:"
    echo "  - Evaluation log: ${LOG_DIR}/alfworld_eval.log"
    echo "  - Server log: ${LOG_DIR}/vllm_server.log"
fi
echo "=========================================="

# Exit with evaluation exit code
exit ${EVAL_EXIT_CODE}
