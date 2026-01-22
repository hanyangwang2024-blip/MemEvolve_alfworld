# Alfworld 评估指南

本指南介绍如何在 MemEvolve 框架中运行 Alfworld 数据集评估。

## 环境准备

### 1. 安装依赖

```bash
pip install alfworld
```

### 2. 下载 Alfworld 数据

```bash
# 下载 Alfworld 数据集
export ALFWORLD_DATA=./eval_agent/data/alfworld
alfworld-download
```

或者手动下载数据：
- 从 https://github.com/alfworld/alfworld 下载
- 将 `json_2.1.1` 目录放到 `eval_agent/data/alfworld/` 下

### 3. 数据目录结构

确保数据目录结构如下：

```
eval_agent/data/alfworld/
├── base_config.yaml
├── logic/
│   ├── alfred.pddl
│   └── alfred.twl2
└── json_2.1.1/           # 需要下载
    ├── train/
    ├── valid_seen/
    └── valid_unseen/
```

## 运行评估

### 基本用法

```bash
# 运行测试集的前10个任务
python run_flash_searcher_alfworld.py \
    --split test \
    --sample_num 10 \
    --max_steps 50
```

### 使用内存系统

```bash
# 使用 lightweight_memory 内存系统
python run_flash_searcher_alfworld.py \
    --split test \
    --sample_num 20 \
    --memory_provider lightweight_memory \
    --max_steps 50
```

### 按任务类型过滤

Alfworld 包含以下任务类型：
- `put`: 拿起并放置物品
- `clean`: 清洁并放置物品
- `heat`: 加热并放置物品
- `cool`: 冷却并放置物品
- `examine`: 在灯光下检查物品
- `puttwo`: 拿起两个物品并放置

```bash
# 只运行 "clean" 类型的任务
python run_flash_searcher_alfworld.py \
    --split test \
    --task_type clean \
    --max_steps 50
```

### 完整参数列表

```bash
python run_flash_searcher_alfworld.py --help
```

主要参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--split` | 数据集分割 (train/dev/test) | test |
| `--sample_num` | 处理的任务数量 | None (全部) |
| `--task_indices` | 指定任务索引 (如 "1-10,15,20") | None |
| `--task_type` | 按任务类型过滤 | None |
| `--max_steps` | Agent 最大步数 | 50 |
| `--memory_provider` | 内存系统类型 | None |
| `--concurrency` | 并发任务数 | 1 |
| `--outfile` | 输出文件路径 | ./alfworld_output/alfworld_results.jsonl |

## 导出任务到 JSONL

如果不想在运行时依赖 Alfworld 环境，可以预先导出任务：

```bash
# 导出测试集任务
python scripts/export_alfworld_tasks.py \
    --split test \
    --output ./data/alfworld/test_tasks.jsonl

# 使用导出的文件运行评估
python run_flash_searcher_alfworld.py \
    --infile ./data/alfworld/test_tasks.jsonl \
    --sample_num 10
```

## 输出格式

每个任务的输出包含：

```json
{
    "task_id": 0,
    "item_index": 1,
    "task_type": "clean",
    "question": "Your task is to: put a clean apple in fridge.",
    "full_observation": "...",
    "game_file": "path/to/game/file",
    "agent_result": "...",
    "is_correct": true,
    "task_completed": true,
    "status": "success",
    "steps_taken": 15,
    "agent_trajectory": [...],
    "metrics": {
        "elapsed_time": 45.2,
        "total_tokens": 5000,
        "prompt_tokens": 4000,
        "completion_tokens": 1000
    }
}
```

## 评估报告

运行完成后会生成评估报告，包含：
- 总体成功率
- 各任务类型的成功率
- 资源使用统计 (时间、Token 数量、步数)

报告示例：

```
================================================================================
Alfworld Evaluation Summary
================================================================================
Success Rate: 85/134 = 63.43%
Total Time: 120.5m | Total Tokens: 1,500,000
----------------------------------------
  clean: 15/20 (75.0%)
  cool: 12/18 (66.7%)
  examine: 10/15 (66.7%)
  heat: 18/25 (72.0%)
  put: 20/36 (55.6%)
  puttwo: 10/20 (50.0%)
================================================================================
```

## 内存系统支持

Alfworld 支持所有 MemEvolve 内存系统：

- `agent_kb`: Agent-KB 知识库
- `lightweight_memory`: 轻量级内存系统
- `cerebra_fusion_memory`: Cerebra Fusion 内存
- `skillweaver`: SkillWeaver
- `expel`: ExpeL
- `voyager`: Voyager
- 等等

内存系统会记录任务执行轨迹，用于后续任务的经验指导。

## 常见问题

### Q: 出现 "alfworld is not installed" 错误

安装 alfworld：
```bash
pip install alfworld
```

### Q: 出现数据路径错误

确保设置了正确的数据路径：
```bash
export ALFWORLD_DATA=./eval_agent/data/alfworld
```

### Q: 并发运行时出现错误

Alfworld 环境是有状态的，每个并发任务需要独立的环境实例。
框架已自动处理此问题，但如果内存不足，请降低 `--concurrency` 参数。

### Q: Agent 总是失败

- 增加 `--max_steps` 参数
- 检查模型配置是否正确
- 查看日志了解具体失败原因
