# Alfworld 数据集迁移到 MemEvolve 框架方案

## 一、架构对比分析

### ETO/eval_agent 架构
```
Task (AlfWorldTask) 
  ↓ 加载任务
Env (AlfWorldEnv) 
  ↓ 管理环境交互
State (历史对话 + 状态)
  ↓ 交互循环
Agent (LMAgent) 
  ↓ 生成动作
Env.step() → 执行动作
```

### MemEvolve 架构
```
数据文件 (JSONL/CSV)
  ↓ 读取
process_item()
  ↓ 创建
Agent (SearchAgent/MMSearchAgent)
  ↓ 执行任务
Agent.run() → 返回结果
  ↓ 转换为
TrajectoryData
  ↓ 存入
MemoryProvider.take_in_memory()
```

## 二、关键差异点

### 1. 环境交互方式

**ETO/eval_agent:**
- 使用 `AlfWorldEnv` 直接管理环境
- 通过 `env.step(action)` 执行动作
- 返回 `(observation, reward, done, info)`
- 状态通过 `State` 对象管理

**MemEvolve:**
- 使用 `Tool` 系统封装环境操作
- Agent 通过工具调用与环境交互
- 需要创建 `AlfworldTool` 来封装环境操作

### 2. 任务加载方式

**ETO/eval_agent:**
```python
# 从环境直接加载任务
tasks, n_tasks = AlfWorldTask.load_tasks(split, part_num, part_idx)
for task in tasks:
    obs = task.observation  # 初始观察
    env = task.env  # 环境对象
```

**MemEvolve:**
```python
# 从文件加载任务列表
data = read_jsonl(args.infile)
for item in data:
    question = item["question"]
    # 需要保存环境状态以便后续使用
```

### 3. 执行流程

**ETO/eval_agent:**
```python
# 交互式循环
while not state.finished:
    llm_output = agent(state.history)
    observation, state = env.step(llm_output)
```

**MemEvolve:**
```python
# 一次性执行
result = search_agent(question)
# Agent 内部处理所有步骤
```

## 三、迁移实现方案

### 方案一：创建 AlfworldTool（推荐）

**优点：**
- 符合 MemEvolve 的 Tool 架构
- 可以复用现有的 Agent 框架
- 内存系统可以无缝集成

**实现步骤：**

#### 1. 创建 AlfworldTool

```python
# FlashOAgents/tools.py 或新建 alfworld_tool.py
from FlashOAgents.tools import Tool
import alfworld
import alfworld.agents.environment as envs
import yaml
import os
import re

class AlfworldTool(Tool):
    """Alfworld 环境交互工具"""
    
    def __init__(self, env_config_path: str, split: str = "test"):
        super().__init__(
            name="alfworld_action",
            description="Execute actions in Alfworld household environment. "
                       "Available actions: go to {recep}, take {obj} from {recep}, "
                       "put {obj} in/on {recep}, open {recep}, close {recep}, "
                       "toggle {obj} {recep}, clean {obj} with {recep}, "
                       "heat {obj} with {recep}, cool {obj} with {recep}",
        )
        self.env_config_path = env_config_path
        self.split = split
        self.env = None
        self.current_task_id = None
        self.current_observation = None
        self._initialize_env()
    
    def _initialize_env(self):
        """初始化环境"""
        os.environ["ALFWORLD_DATA"] = "eval_agent/data/alfworld"
        with open(self.env_config_path) as f:
            config = yaml.safe_load(f)
        
        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=self.split
        )
        self.env = env.init_env(batch_size=1)
    
    def reset_task(self, task_id: int):
        """重置到指定任务"""
        if self.current_task_id != task_id:
            # 跳转到指定任务
            self.env.skip(task_id)
            self.current_task_id = task_id
        
        obs, info = self.env.reset()
        obs = "\n".join(obs[0].split("\n\n")[1:])
        self.current_observation = obs
        return obs, info
    
    def __call__(self, action: str) -> str:
        """
        执行动作
        
        Args:
            action: 动作字符串，格式如 "go to fridge"
        
        Returns:
            观察结果字符串
        """
        # 解析动作（如果包含 "Action: " 前缀）
        action = action.strip()
        if action.startswith("Action:"):
            action = re.sub(r"Action:\s*", "", action, flags=re.IGNORECASE).strip()
        
        try:
            observation, reward, done, info = self.env.step([action])
            observation = observation[0]
            reward = info['won'][0]
            done = done[0]
            
            # 处理观察
            if observation.startswith('You arrive at loc '):
                observation = observation[observation.find('. ')+2:]
            
            result = f"Observation: {observation}"
            
            if done:
                if reward > 0:
                    result += "\n[Task completed successfully!]"
                else:
                    result += "\n[Task failed]"
            
            return result
            
        except Exception as e:
            return f"Error: {str(e)}. Please check your action format."
    
    def get_current_observation(self) -> str:
        """获取当前观察"""
        return self.current_observation or ""
```

#### 2. 创建 AlfworldAgent

```python
# base_agent.py 中添加
class AlfworldAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, 
                 memory_provider=None, env_config_path=None, split="test", **kwargs):
        super().__init__(model)
        
        # 创建 Alfworld 工具
        alfworld_tool = AlfworldTool(env_config_path, split)
        tools = [alfworld_tool]
        
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider
        )
        self.alfworld_tool = alfworld_tool
    
    def forward(self, task_id: int, initial_observation: str, **kwargs):
        """
        执行 Alfworld 任务
        
        Args:
            task_id: 任务ID
            initial_observation: 初始观察（包含任务描述）
        """
        # 重置到指定任务
        obs, info = self.alfworld_tool.reset_task(task_id)
        
        # 构建初始提示
        task_prompt = f"""{initial_observation}

You are an intelligent agent in a household environment. Your goal is to complete the task described above.

For each turn, you will receive an observation. You should think about the current condition and plan your actions, then output your action.

Your response must follow this format:
Thought: <your thoughts>
Action: <your next action>

Available actions:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}

Begin by reading the observation and planning your first action."""
        
        # 执行任务
        result = self.agent_fn.run(task_prompt)
        
        # 检查任务是否完成
        # 需要从工具中获取最终状态
        # 这里可能需要修改 ToolCallingAgent 来暴露工具状态
        
        return {
            "agent_result": result,
            **self.capture_trajectory()
        }
```

#### 3. 创建运行脚本

```python
# run_flash_searcher_alfworld.py
#!/usr/bin/env python
import os
import argparse
import json
import logging
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from FlashOAgents import OpenAIServerModel
from base_agent import AlfworldAgent
from utils import read_jsonl, write_jsonl
from EvolveLab.memory_types import MemoryType, TrajectoryData, PROVIDER_MAPPING
from EvolveLab.config import get_memory_config
from eval_utils import (
    TaskTimer, TokenCounter, save_task_result,
    generate_unified_report, enrich_result_with_metrics, create_run_directory
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(override=True)

def load_memory_provider(memory_type_str, model=None):
    """加载内存提供者（与GAIA等脚本相同）"""
    # ... 复用现有代码 ...

def process_item(item, model_config, summary_interval, prompts_type, max_steps,
                 memory_type_str=None, item_index=None, enable_memory_evolution=True,
                 env_config_path=None, split="test"):
    """处理单个 Alfworld 任务"""
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    
    memory_provider = None
    if memory_type_str:
        memory_provider = load_memory_provider(memory_type_str, task_model)
    
    timer = TaskTimer()
    timer.start()
    
    # 创建 AlfworldAgent
    search_agent = AlfworldAgent(
        task_model,
        summary_interval=summary_interval,
        prompts_type=prompts_type,
        max_steps=max_steps,
        memory_provider=memory_provider,
        env_config_path=env_config_path,
        split=split
    )
    
    task_id = item.get("task_id", item_index)
    initial_observation = item.get("observation", item.get("question", ""))
    game_file = item.get("game_file", "")
    
    try:
        # 执行任务
        result = search_agent.forward(
            task_id=task_id,
            initial_observation=initial_observation
        )
        
        # 获取轨迹
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(
                include_system_prompt=False
            )
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        # 判断任务是否成功
        # 需要从工具或轨迹中提取成功状态
        is_correct = False
        # 检查轨迹中是否有成功标记
        for step in trajectory:
            if isinstance(step, dict):
                obs = step.get("obs", "")
                if "[Task completed successfully!]" in str(obs):
                    is_correct = True
                    break
        
        # 存入内存
        if memory_provider and enable_memory_evolution:
            try:
                trajectory_data = TrajectoryData(
                    query=initial_observation,
                    trajectory=agent_messages,
                    result=result.get("agent_result"),
                    metadata={
                        "task_id": task_id,
                        "status": "success",
                        "is_correct": is_correct,
                        "game_file": game_file,
                    }
                )
                success, msg = memory_provider.take_in_memory(trajectory_data)
                if success:
                    logger.debug(f"Memory ingested: {msg}")
                else:
                    logger.warning(f"Memory ingestion failed: {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed: {e}")
        
        token_counter = TokenCounter.from_model(task_model)
        
        task_result = {
            "agent_result": result.get("agent_result"),
            "is_correct": is_correct,
            "task_id": task_id,
            "item_index": item_index,
            "question": initial_observation,
            "game_file": game_file,
            "status": "success",
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
        }
        
        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception: {error_msg}")
        
        # 错误处理（类似其他脚本）
        # ...
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)

def load_alfworld_tasks(split: str, part_num: int = 1, part_idx: int = -1):
    """加载 Alfworld 任务列表"""
    import alfworld
    import alfworld.agents.environment as envs
    import yaml
    
    os.environ["ALFWORLD_DATA"] = "eval_agent/data/alfworld"
    alfworld_data_path = os.environ.get("ALFWORLD_DATA")
    
    with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
        config = yaml.safe_load(f)
    
    if split == 'train':
        split_env = "train"
        N_TASKS = 3321
    elif split == 'dev':
        split_env = "eval_in_distribution"
        N_TASKS = 140
    elif split == 'test':
        split_env = "eval_out_of_distribution"
        N_TASKS = 134
    
    env = getattr(alfworld.agents.environment, config["env"]["type"])(
        config, train_eval=split_env
    )
    env = env.init_env(batch_size=1)
    
    if part_num > 1:
        assert part_idx != -1
        part_inst_num = [N_TASKS // part_num] * part_num
        part_inst_num[-1] += N_TASKS % part_num
        env.skip(sum(part_inst_num[:part_idx]))
        N_TASKS = part_inst_num[part_idx]
    
    tasks = []
    for idx in range(N_TASKS):
        obs, info = env.reset()
        obs = "\n".join(obs[0].split("\n\n")[1:])
        game_file = info["extra.gamefile"][0]
        
        tasks.append({
            "task_id": idx,
            "observation": obs,
            "game_file": game_file,
            "_global_index": idx + 1
        })
    
    return tasks

def main(args):
    # 模型配置（与其他脚本相同）
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }
    
    # 加载任务
    if args.infile:
        # 从文件加载（如果已预处理）
        data = read_jsonl(args.infile)
    else:
        # 直接从环境加载
        data = load_alfworld_tasks(args.split, args.part_num, args.part_idx)
    
    # 过滤任务
    if args.task_indices:
        # 解析任务索引
        # ...
        pass
    elif args.sample_num is not None:
        data = data[:args.sample_num]
    
    # 创建输出目录
    # ...
    
    # 处理任务（与其他脚本类似）
    # ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alfworld evaluation with Flash Searcher')
    parser.add_argument('--infile', type=str, default=None,
                       help='Input path (optional, if not provided, load from env)')
    parser.add_argument('--outfile', type=str, default="./alfworld_output/results.jsonl",
                       help='Output path')
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'], default='test',
                       help='Dataset split')
    parser.add_argument('--part_num', type=int, default=1, help='Number of parts')
    parser.add_argument('--part_idx', type=int, default=-1, help='Part index')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Prompts type')
    parser.add_argument('--concurrency', type=int, default=1, help='Concurrency')
    parser.add_argument('--max_steps', type=int, default=40, help='Max steps')
    parser.add_argument('--memory_provider', type=str, default=None, help='Memory provider')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True)
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false')
    parser.add_argument('--env_config_path', type=str,
                       default="eval_agent/data/alfworld/base_config.yaml",
                       help='Alfworld environment config path')
    
    args = parser.parse_args()
    main(args)
```

### 方案二：适配现有 Agent（备选）

如果不想创建新工具，可以修改 `ToolCallingAgent` 来支持环境交互，但这需要更大的改动。

## 四、关键实现细节

### 1. 任务状态管理

**问题：** Alfworld 需要维护环境状态，而 MemEvolve 的 Agent 是无状态的。

**解决方案：**
- 在 `AlfworldTool` 中维护当前任务的环境状态
- 每次调用 `reset_task()` 时重置环境
- 工具内部跟踪当前任务ID

### 2. 动作解析

**问题：** Agent 可能输出 "Thought: ... Action: ..." 格式，需要解析。

**解决方案：**
- 在 `AlfworldTool.__call__()` 中解析动作
- 支持多种格式：纯动作、带 "Action:" 前缀、带 "Thought:" 前缀

### 3. 成功判断

**问题：** 需要从轨迹中判断任务是否成功。

**解决方案：**
- 在工具返回的观察中包含成功标记
- 从轨迹的观察中提取成功状态
- 或者修改工具返回结构，包含 `done` 和 `reward` 信息

### 4. 并发处理

**问题：** 每个线程需要独立的环境实例。

**解决方案：**
- 在 `process_item()` 中为每个任务创建新的 `AlfworldTool` 实例
- 或者使用环境池（如果支持）

## 五、数据格式转换

### 输入数据格式

可以创建预处理脚本将 Alfworld 任务导出为 JSONL：

```python
# preprocess_alfworld.py
def export_alfworld_to_jsonl(split: str, output_path: str):
    tasks = load_alfworld_tasks(split)
    data = []
    for task in tasks:
        data.append({
            "task_id": task["task_id"],
            "observation": task["observation"],
            "game_file": task["game_file"]
        })
    write_jsonl(output_path, data)
```

### 输出数据格式

与其他数据集保持一致：

```json
{
    "task_id": 0,
    "item_index": 1,
    "question": "初始观察...",
    "game_file": "xxx.json",
    "agent_result": "...",
    "is_correct": true,
    "status": "success",
    "agent_trajectory": [...],
    "agent_messages": [...],
    "metrics": {...}
}
```

## 六、测试计划

1. **单元测试**
   - `AlfworldTool` 的动作执行
   - 动作解析逻辑
   - 环境重置

2. **集成测试**
   - 完整任务执行流程
   - 内存系统集成
   - 多任务处理

3. **性能测试**
   - 并发处理
   - 内存使用
   - 执行时间

## 七、迁移步骤

1. ✅ 创建 `AlfworldTool` 类
2. ✅ 创建 `AlfworldAgent` 类
3. ✅ 创建 `run_flash_searcher_alfworld.py` 脚本
4. ✅ 创建数据预处理脚本（可选）
5. ⏳ 测试基本功能
6. ⏳ 测试内存系统集成
7. ⏳ 性能优化
8. ⏳ 文档更新

## 八、潜在问题和解决方案

### 问题1：环境状态持久化
- **问题：** 环境状态无法跨进程/线程共享
- **解决：** 每个任务创建独立环境实例

### 问题2：动作格式不一致
- **问题：** Agent 可能输出不同格式的动作
- **解决：** 在工具中实现健壮的解析逻辑

### 问题3：任务索引管理
- **问题：** 需要正确跳转到指定任务
- **解决：** 使用环境的 `skip()` 方法

## 九、总结

迁移 Alfworld 到 MemEvolve 的核心是：
1. **创建 AlfworldTool** 封装环境交互
2. **创建 AlfworldAgent** 适配 Agent 框架
3. **创建运行脚本** 遵循 MemEvolve 的模式
4. **数据格式转换** 统一输入输出格式

这样可以在保持 MemEvolve 架构统一性的同时，支持 Alfworld 的特殊需求。
