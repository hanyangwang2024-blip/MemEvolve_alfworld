#!/bin/bash
# ============================================================================
# Run Alfworld tasks with Qwen2.5-7B-Instruct model on GPU 0 and 1
# 
# This script uses FastChat (official method) to serve the model:
# 1. Starts FastChat controller
# 2. Starts FastChat model worker with Qwen2.5-7B-Instruct on GPU 0,1
# 3. Starts FastChat OpenAI API server
# 4. Runs Alfworld evaluation tasks
# 5. Cleans up all services on exit
#
# Requirements:
# - FastChat installed (from ETO/fastchat or install: pip install fschat)
# - CUDA-capable GPUs (0 and 1)
# - Alfworld data downloaded
# ============================================================================

set -e  # Exit on error

# ==================== Configuration ====================
# GPU configuration
CUDA_VISIBLE_DEVICES="1,3"
NUM_GPUS=2

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# If you have the model locally, you can set:
# MODEL_NAME="/path/to/local/Qwen2.5-7B-Instruct"

# FastChat server configuration
CONTROLLER_HOST="localhost"
CONTROLLER_PORT=21001
WORKER_HOST="localhost"
WORKER_PORT=21002
API_SERVER_HOST="localhost"
API_SERVER_PORT=8000
API_KEY="EMPTY"  # FastChat doesn't require API key by default

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

# FastChat path (adjust if needed)
# If FastChat is in ETO directory
if [ -d "../ETO/fastchat" ]; then
    FASTCHAT_PATH="../ETO/fastchat"
    PYTHON_CMD="python -u -m fastchat.serve"
elif [ -d "../../ETO/fastchat" ]; then
    FASTCHAT_PATH="../../ETO/fastchat"
    PYTHON_CMD="python -u -m fastchat.serve"
else
    # Assume FastChat is installed as package
    FASTCHAT_PATH=""
    PYTHON_CMD="python -u -m fastchat.serve"
fi

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

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=120  # 4 minutes max wait time
    local attempt=0
    
    echo "Waiting for ${service_name} to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s ${url} > /dev/null 2>&1; then
            echo ""
            echo "✓ ${service_name} is ready!"
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
    echo "ERROR: ${service_name} failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up FastChat services..."
    
    # Kill API server
    if [ ! -z "${API_SERVER_PID}" ]; then
        echo "Stopping OpenAI API server (PID: ${API_SERVER_PID})..."
        kill ${API_SERVER_PID} 2>/dev/null || true
        sleep 2
        kill -9 ${API_SERVER_PID} 2>/dev/null || true
        wait ${API_SERVER_PID} 2>/dev/null || true
    fi
    
    # Kill model worker
    if [ ! -z "${WORKER_PID}" ]; then
        echo "Stopping model worker (PID: ${WORKER_PID})..."
        kill ${WORKER_PID} 2>/dev/null || true
        sleep 2
        kill -9 ${WORKER_PID} 2>/dev/null || true
        wait ${WORKER_PID} 2>/dev/null || true
    fi
    
    # Kill controller
    if [ ! -z "${CONTROLLER_PID}" ]; then
        echo "Stopping controller (PID: ${CONTROLLER_PID})..."
        kill ${CONTROLLER_PID} 2>/dev/null || true
        sleep 2
        kill -9 ${CONTROLLER_PID} 2>/dev/null || true
        wait ${CONTROLLER_PID} 2>/dev/null || true
    fi
    
    echo "Cleanup complete."
    echo "=========================================="
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# ==================== Main Script ====================

echo "=========================================="
echo "Alfworld Evaluation with Qwen2.5-7B-Instruct"
echo "Using FastChat (Official Method)"
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

# Check if FastChat is available
if ! python -c "import fastchat" 2>/dev/null; then
    echo "ERROR: FastChat is not installed or not in Python path."
    echo ""
    echo "Please ensure FastChat is available:"
    echo "  1. If using ETO/fastchat, make sure it's in PYTHONPATH"
    echo "  2. Or install: pip install fschat"
    echo ""
    echo "To use ETO's FastChat, you can:"
    echo "  export PYTHONPATH=\$PYTHONPATH:/path/to/ETO"
    exit 1
fi

# Check if curl is available (for health check)
if ! command -v curl &> /dev/null; then
    echo "WARNING: curl is not installed. Health check will be skipped."
fi

# Check if ports are available
PORTS_TO_CHECK=("${CONTROLLER_PORT}" "${WORKER_PORT}" "${API_SERVER_PORT}")
for port in "${PORTS_TO_CHECK[@]}"; do
    if check_port ${port}; then
        echo "WARNING: Port ${port} is already in use."
        echo "Please stop the service using this port or change the port in the script."
        exit 1
    fi
done

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.device_count()} GPUs')" 2>/dev/null; then
    echo "WARNING: CUDA may not be available. The script will continue but may fail."
    echo ""
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# Start FastChat services
echo "=========================================="
echo "Starting FastChat Services"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Controller: ${CONTROLLER_HOST}:${CONTROLLER_PORT}"
echo "Worker: ${WORKER_HOST}:${WORKER_PORT}"
echo "API Server: ${API_SERVER_HOST}:${API_SERVER_PORT}"
echo "=========================================="
echo ""

# 1. Start Controller
echo "Starting FastChat controller..."
${PYTHON_CMD}.controller \
    --host ${CONTROLLER_HOST} \
    --port ${CONTROLLER_PORT} \
    > ${LOG_DIR}/fastchat_controller.log 2>&1 &

CONTROLLER_PID=$!
echo "Controller started with PID: ${CONTROLLER_PID}"
sleep 5

# 2. Start Model Worker
echo "Starting FastChat model worker..."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON_CMD}.model_worker \
    --model-path ${MODEL_NAME} \
    --controller-address http://${CONTROLLER_HOST}:${CONTROLLER_PORT} \
    --worker-address http://${WORKER_HOST}:${WORKER_PORT} \
    --host ${WORKER_HOST} \
    --port ${WORKER_PORT} \
    --num-gpus ${NUM_GPUS} \
    --trust-remote-code \
    > ${LOG_DIR}/fastchat_worker.log 2>&1 &

WORKER_PID=$!
echo "Model worker started with PID: ${WORKER_PID}"
echo "Waiting for model to load (this may take several minutes)..."
sleep 30

# Wait for worker to be ready (check if it registered with controller)
echo "Checking worker status..."
for i in {1..60}; do
    if curl -s http://${CONTROLLER_HOST}:${CONTROLLER_PORT}/list_models > /dev/null 2>&1; then
        MODELS=$(curl -s http://${CONTROLLER_HOST}:${CONTROLLER_PORT}/list_models)
        if echo "${MODELS}" | grep -q "${MODEL_NAME}" || echo "${MODELS}" | grep -q "model"; then
            echo "✓ Model worker is ready!"
            break
        fi
    fi
    if [ $i -eq 60 ]; then
        echo "WARNING: Worker may not be fully ready, but continuing..."
    fi
    sleep 2
done

# 3. Start OpenAI API Server
echo "Starting FastChat OpenAI API server..."
${PYTHON_CMD}.openai_api_server \
    --controller-address http://${CONTROLLER_HOST}:${CONTROLLER_PORT} \
    --host ${API_SERVER_HOST} \
    --port ${API_SERVER_PORT} \
    > ${LOG_DIR}/fastchat_api_server.log 2>&1 &

API_SERVER_PID=$!
echo "OpenAI API server started with PID: ${API_SERVER_PID}"

# Wait for API server to be ready
if ! wait_for_service "http://${API_SERVER_HOST}:${API_SERVER_PORT}/health" "OpenAI API server"; then
    echo ""
    echo "Failed to start OpenAI API server."
    echo "Last 50 lines of API server log:"
    echo "----------------------------------------"
    tail -50 ${LOG_DIR}/fastchat_api_server.log
    echo "----------------------------------------"
    exit 1
fi

# Set environment variables for the evaluation script
export DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
export OPENAI_API_BASE="http://${API_SERVER_HOST}:${API_SERVER_PORT}/v1"
export OPENAI_API_KEY="${API_KEY}"

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
    echo "  - Controller log: ${LOG_DIR}/fastchat_controller.log"
    echo "  - Worker log: ${LOG_DIR}/fastchat_worker.log"
    echo "  - API server log: ${LOG_DIR}/fastchat_api_server.log"
fi
echo "=========================================="

# Exit with evaluation exit code
exit ${EVAL_EXIT_CODE}
