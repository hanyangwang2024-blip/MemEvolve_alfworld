# 使用 Qwen2.5-7B-Instruct 运行 Alfworld 任务指南

## 前置要求

### 1. 安装 vLLM

```bash
# 基础安装
pip install vllm

# 或指定 CUDA 版本（推荐）
# 对于 CUDA 11.8
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. 检查 GPU

确保 GPU 0 和 1 可用：

```bash
nvidia-smi
```

### 3. 下载 Alfworld 数据

```bash
export ALFWORLD_DATA=./eval_agent/data/alfworld
alfworld-download
```

## 使用方法

### 基本使用

```bash
# 使用默认配置（测试集，10个任务）
bash run_alfworld_qwen.sh
```

### 自定义配置

编辑 `run_alfworld_qwen.sh` 文件，修改以下变量：

```bash
# GPU 配置
CUDA_VISIBLE_DEVICES="0,1"  # 使用的 GPU
NUM_GPUS=2                   # GPU 数量

# 模型配置
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"  # 模型名称或路径

# 任务配置
ALFWORLD_SPLIT="test"        # train, dev, 或 test
SAMPLE_NUM=10                # 任务数量（设为 null 运行全部）
MAX_STEPS=50                 # 最大步数
CONCURRENCY=1                # 并发数
MEMORY_PROVIDER=""           # 内存系统（如 "lightweight_memory"）

# 输出配置
OUTPUT_DIR="./alfworld_output"
```

### 命令行参数示例

如果需要更灵活的控制，可以直接运行 Python 脚本：

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES="0,1"
export DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

# 在一个终端启动 vLLM 服务器
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 8192

# 在另一个终端运行评估
python run_flash_searcher_alfworld.py \
    --split test \
    --sample_num 10 \
    --max_steps 50 \
    --outfile ./alfworld_output/qwen_results.jsonl
```

## 配置说明

### GPU 配置

- `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU，如 "0,1" 表示使用 GPU 0 和 1
- `NUM_GPUS`: 必须与 `CUDA_VISIBLE_DEVICES` 中的 GPU 数量一致

### vLLM 服务器配置

- `VLLM_HOST`: 服务器地址（默认 localhost）
- `VLLM_PORT`: 服务器端口（默认 8000）
- `VLLM_MAX_MODEL_LEN`: 最大序列长度（默认 8192）

### 模型配置

可以使用 HuggingFace 模型 ID 或本地路径：

```bash
# 使用 HuggingFace 模型
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# 使用本地模型
MODEL_NAME="/path/to/local/Qwen2.5-7B-Instruct"
```

### 内存系统

支持所有 MemEvolve 内存系统：

```bash
MEMORY_PROVIDER="lightweight_memory"      # 轻量级内存
MEMORY_PROVIDER="agent_kb"                 # Agent-KB
MEMORY_PROVIDER="cerebra_fusion_memory"   # Cerebra Fusion
# ... 等等
```

## 输出文件

运行完成后，会在以下位置生成文件：

```
alfworld_output/
├── qwen_results.jsonl          # 所有任务的结果
└── qwen_runs/
    ├── report.txt               # 评估报告
    ├── 1.json                   # 任务 1 的详细结果
    ├── 2.json                   # 任务 2 的详细结果
    └── ...
```

日志文件：

```
logs/
├── vllm_server.log              # vLLM 服务器日志
└── alfworld_eval.log            # 评估过程日志
```

## 故障排除

### 1. vLLM 服务器启动失败

**问题**: 服务器无法启动或模型加载失败

**解决方案**:
- 检查 GPU 内存是否足够（7B 模型需要约 14GB+ 显存）
- 检查模型路径是否正确
- 查看日志: `logs/vllm_server.log`
- 尝试减少 `gpu-memory-utilization`（默认 0.9）

### 2. 端口被占用

**问题**: 端口 8000 已被使用

**解决方案**:
```bash
# 修改脚本中的 VLLM_PORT
VLLM_PORT=8001  # 或其他可用端口

# 或停止占用端口的进程
lsof -ti:8000 | xargs kill -9
```

### 3. CUDA 错误

**问题**: CUDA out of memory 或其他 CUDA 错误

**解决方案**:
- 减少 `gpu-memory-utilization`（如改为 0.8）
- 减少 `MAX_STEPS` 或 `CONCURRENCY`
- 检查其他进程是否占用 GPU

### 4. 模型下载慢

**问题**: 首次运行时模型下载很慢

**解决方案**:
```bash
# 预先下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 或设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 5. 评估脚本连接失败

**问题**: 评估脚本无法连接到 vLLM 服务器

**解决方案**:
- 检查服务器是否正常启动: `curl http://localhost:8000/health`
- 检查环境变量是否正确设置
- 等待服务器完全启动（首次加载模型需要时间）

## 性能优化

### 1. 增加并发

如果 GPU 内存充足，可以增加并发数：

```bash
CONCURRENCY=2  # 或更多
```

### 2. 调整 vLLM 参数

在脚本中修改 vLLM 启动参数：

```bash
vllm serve ${MODEL_NAME} \
    --host ${VLLM_HOST} \
    --port ${VLLM_PORT} \
    --tensor-parallel-size ${NUM_GPUS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096 \  # 增加批处理大小
    --max-num-seqs 256               # 增加并发序列数
```

### 3. 使用量化模型

如果显存不足，可以使用量化版本：

```bash
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-AWQ"  # 如果可用
```

## 示例输出

成功运行后，你会看到类似输出：

```
==========================================
Alfworld Evaluation Summary
==========================================
Success Rate: 8/10 = 80.00%
Total Time: 15.3m | Total Tokens: 150,000
----------------------------------------
  clean: 2/2 (100.0%)
  put: 3/4 (75.0%)
  heat: 2/3 (66.7%)
  cool: 1/1 (100.0%)
==========================================
```

## 注意事项

1. **首次运行**: 首次运行需要下载模型，可能需要较长时间
2. **GPU 内存**: 确保有足够的 GPU 内存（建议每张卡至少 8GB）
3. **网络**: 如果使用 HuggingFace 模型，需要网络连接
4. **清理**: 脚本会在退出时自动清理 vLLM 服务器

## 更多信息

- vLLM 文档: https://docs.vllm.ai/
- Qwen 模型: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Alfworld 文档: 查看 `docs/alfworld_guide.md`
