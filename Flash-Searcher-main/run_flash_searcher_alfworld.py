#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Alfworld evaluation script for MemEvolve framework.

This script runs Alfworld household simulation tasks using the Flash Searcher
agent framework with support for various memory providers.

Usage:
    python run_flash_searcher_alfworld.py \
        --split test \
        --sample_num 10 \
        --memory_provider lightweight_memory \
        --max_steps 50
"""

import os
import argparse
import json
import logging
import yaml
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from FlashOAgents import OpenAIServerModel
from FlashOAgents.alfworld_tool import AlfworldEnvWrapper
from base_agent import AlfworldAgent, SimpleAlfworldAgent
from utils import read_jsonl, write_jsonl
from EvolveLab.memory_types import MemoryType, TrajectoryData, PROVIDER_MAPPING
from EvolveLab.config import get_memory_config
from eval_utils import (
    TaskTimer, TokenCounter, save_task_result,
    generate_unified_report, enrich_result_with_metrics, create_run_directory
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


# Alfworld task type prefixes
TASK_TYPES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def get_task_type(game_file: str) -> str:
    """Extract task type from game file path"""
    for task_key, task_name in TASK_TYPES.items():
        if task_key in game_file:
            return task_name
    return "unknown"


def parse_task_indices(indices_str):
    """Parse index string like "5", "1-10" or "1,3,5-8,10" into a 1-based index set."""
    if not indices_str:
        return None
    
    indices = set()
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                indices.update(range(start, end + 1))
            except ValueError as e:
                logger.error(f"Invalid range format: {part}. Error: {e}")
                raise
        else:
            try:
                indices.add(int(part))
            except ValueError:
                logger.error(f"Invalid number format: {part}")
                raise
    
    return indices


def load_memory_provider(memory_type_str, model=None):
    """Load and initialize memory provider from type string"""
    if not memory_type_str:
        return None
    
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        logger.error(f"Invalid memory type: {memory_type_str}")
        return None
    
    if memory_type not in PROVIDER_MAPPING:
        logger.error(f"Memory type {memory_type_str} not found in PROVIDER_MAPPING")
        return None
    
    try:
        class_name, module_name = PROVIDER_MAPPING[memory_type]
        module = __import__(f"EvolveLab.providers.{module_name}", fromlist=[class_name])
        provider_class = getattr(module, class_name)
        config = get_memory_config(memory_type)
        if model is not None:
            try:
                config["model"] = model
            except Exception:
                pass
        provider = provider_class(config=config)
        
        if not provider.initialize():
            logger.error(f"Failed to initialize memory provider: {memory_type_str}")
            return None
        
        logger.info(f"Memory provider loaded: {memory_type_str}")
        return provider
    except Exception as e:
        logger.error(f"Failed to load memory provider {memory_type_str}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_alfworld_tasks(config_path: str, split: str = "test", 
                        part_num: int = 1, part_idx: int = -1):
    """
    Load Alfworld tasks from the environment.
    
    Args:
        config_path: Path to Alfworld config file
        split: Dataset split ('train', 'dev', 'test')
        part_num: Number of parts to split the dataset
        part_idx: Index of the part to load
        
    Returns:
        List of task dictionaries and total count
    """
    try:
        import alfworld
        import alfworld.agents.environment as envs
        from alfworld.agents.environment import get_environment
    except ImportError:
        raise ImportError(
            "alfworld is not installed. Please install it with: "
            "pip install alfworld"
        )
    
    # Set environment data path (only if not already set)
    if "ALFWORLD_DATA" not in os.environ:
        alfworld_data_path = os.path.dirname(config_path)
        os.environ["ALFWORLD_DATA"] = alfworld_data_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Map split names
    if split == 'train':
        split_env = "train"
        n_tasks = 3321
    elif split == 'dev':
        split_env = "eval_in_distribution"
        n_tasks = 140
    elif split == 'test':
        split_env = "eval_out_of_distribution"
        n_tasks = 134
    else:
        raise ValueError(f"Unknown split: {split}")
    
    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    AlfredEnvClass = get_environment(env_type)
    env = AlfredEnvClass(config, train_eval=split_env)
    env = env.init_env(batch_size=1)
    
    # Handle partitioning
    if part_num > 1:
        assert part_idx != -1, "part_idx must be specified when part_num > 1"
        part_inst_num = [n_tasks // part_num] * part_num
        part_inst_num[-1] += n_tasks % part_num
        env.skip(sum(part_inst_num[:part_idx]))
        n_tasks = part_inst_num[part_idx]
    
    # Load all tasks
    tasks = []
    for idx in range(n_tasks):
        obs, info = env.reset()
        # Handle different return formats from alfworld
        if isinstance(obs, list):
            obs_text = obs[0]
        elif isinstance(obs, tuple):
            obs_text = obs[0]
        else:
            obs_text = obs
        # Handle nested structures
        while isinstance(obs_text, (list, tuple)):
            obs_text = obs_text[0]
        obs_lines = obs_text.split("\n\n")
        if len(obs_lines) > 1:
            obs_text = "\n".join(obs_lines[1:])
        
        game_file = info.get("extra.gamefile", [""])[0]
        task_type = get_task_type(game_file)
        
        tasks.append({
            "task_id": idx,
            "observation": obs_text,
            "game_file": game_file,
            "task_type": task_type,
            "_global_index": idx + 1
        })
    
    return tasks, n_tasks


def process_item(item, model_config, summary_interval, prompts_type, max_steps,
                 memory_type_str=None, env_config_path=None, split_env=None,
                 enable_memory_evolution=True, use_simple_agent=False):
    """
    Process a single Alfworld task.

    Args:
        item: Task dictionary with task_id, observation, game_file
        model_config: Model configuration dict
        summary_interval: Interval for agent summaries
        prompts_type: Type of prompts to use
        max_steps: Maximum steps for the agent
        memory_type_str: Memory provider type string
        env_config_path: Path to Alfworld environment config
        split_env: Environment split name
        enable_memory_evolution: Whether to enable memory evolution
        use_simple_agent: Use simple text-based agent without tool calling

    Returns:
        Task result dictionary
    """
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    
    memory_provider = None
    if memory_type_str:
        memory_provider = load_memory_provider(memory_type_str, task_model)
    
    timer = TaskTimer()
    timer.start()
    
    # Create environment wrapper for this task
    env_wrapper = AlfworldEnvWrapper(env_config_path, split=split_env)
    env_wrapper.max_steps = max_steps

    # Create Alfworld agent
    if use_simple_agent:
        # Simple text-based agent without tool calling (better for smaller models)
        agent = SimpleAlfworldAgent(
            task_model,
            env_wrapper=env_wrapper,
            max_steps=max_steps,
            memory_provider=memory_provider
        )
    else:
        # Tool-calling based agent
        agent = AlfworldAgent(
            task_model,
            env_wrapper=env_wrapper,
            summary_interval=summary_interval,
            prompts_type=prompts_type,
            max_steps=max_steps,
            memory_provider=memory_provider
        )
    
    task_id = item.get("task_id")
    item_index = item.get("_global_index", task_id)
    observation = item.get("observation", "")
    game_file = item.get("game_file", "")
    task_type = item.get("task_type", "unknown")
    
    try:
        # Always need to reset environment for actual execution
        # The observation from file is just for display; we need the real environment
        obs, info = agent.reset_task(task_id)

        # Use file observation if available, otherwise use environment observation
        if observation:
            actual_observation = observation
        else:
            actual_observation = obs if obs else ""

        # Update game_file from environment if available
        if info.get("game_file"):
            game_file = info.get("game_file")
        
        # Extract task description from observation
        # The first line usually contains the task goal
        task_lines = actual_observation.strip().split('\n')
        task_description = task_lines[0] if task_lines else actual_observation
        
        # Execute the task
        result = agent.forward(
            task_description=task_description,
            observation=actual_observation
        )
        
        # Get trajectory
        try:
            agent_messages = agent.agent_fn.write_memory_to_messages(
                include_system_prompt=False
            )
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        # Determine success
        is_correct = result.get("task_success", False)
        task_completed = result.get("task_completed", False)

        # Set status based on actual task success
        status = "success" if is_correct else "failed"

        # Store in memory
        if memory_provider and enable_memory_evolution:
            try:
                trajectory_data = TrajectoryData(
                    query=task_description,
                    trajectory=agent_messages,
                    result=result.get("agent_result"),
                    metadata={
                        "task_id": task_id,
                        "status": status,
                        "is_correct": is_correct,
                        "task_type": task_type,
                        "game_file": game_file,
                        "full_query": actual_observation,
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
            "task_completed": task_completed,
            "task_id": task_id,
            "item_index": item_index,
            "task_type": task_type,
            "question": task_description,
            "full_observation": actual_observation,
            "game_file": game_file,
            "status": status,
            "steps_taken": result.get("steps_taken", 0),
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
        }
        
        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception occurred while processing task {task_id}: {error_msg}")
        
        try:
            agent_messages = agent.agent_fn.write_memory_to_messages(
                include_system_prompt=False
            )
        except Exception:
            agent_messages = []
        
        # Store failed trajectory in memory
        if memory_provider and enable_memory_evolution:
            try:
                trajectory_data = TrajectoryData(
                    query=observation,
                    trajectory=agent_messages,
                    result=None,
                    metadata={
                        "task_id": task_id,
                        "status": "error",
                        "is_correct": False,
                        "task_type": task_type,
                        "game_file": game_file,
                    }
                )
                success, msg = memory_provider.take_in_memory(trajectory_data)
            except Exception as e:
                logger.warning(f"take_in_memory failed (error case): {e}")
        
        task_result = {
            "agent_result": None,
            "is_correct": False,
            "task_completed": False,
            "task_id": task_id,
            "item_index": item_index,
            "task_type": task_type,
            "question": observation,
            "game_file": game_file,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)


def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 4096,  # Reduced for Qwen2.5-7B (32K context)
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }
    
    model = OpenAIServerModel(**model_config)
    
    # Determine environment split
    if args.split == 'train':
        split_env = "train"
    elif args.split == 'dev':
        split_env = "eval_in_distribution"
    elif args.split == 'test':
        split_env = "eval_out_of_distribution"
    else:
        raise ValueError(f"Unknown split: {args.split}")
    
    # Load tasks
    if args.infile:
        # Load from pre-exported file
        if args.infile.endswith('.json'):
            with open(args.infile, 'r') as f:
                data = json.load(f)
        else:
            data = read_jsonl(args.infile)
        
        # Add global index if not present
        for idx, item in enumerate(data):
            if "_global_index" not in item:
                item["_global_index"] = idx + 1
        
        logger.info(f"Loaded {len(data)} tasks from {args.infile}")
    else:
        # Load directly from environment
        logger.info(f"Loading Alfworld tasks from environment (split={args.split})")
        data, total = load_alfworld_tasks(
            args.env_config_path, 
            args.split,
            args.part_num,
            args.part_idx
        )
        logger.info(f"Loaded {len(data)} tasks from Alfworld environment")
    
    # Filter by task type
    if args.task_type:
        before = len(data)
        data = [d for d in data if d.get("task_type") == args.task_type]
        logger.info(f"Filtered by task_type={args.task_type}: {len(data)}/{before}")
    
    # Filter by indices
    if args.task_indices:
        try:
            selected_indices = parse_task_indices(args.task_indices)
            data = [data[i-1] for i in sorted(selected_indices) if 0 < i <= len(data)]
            logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            return
    elif args.sample_num is not None:
        data = data[:args.sample_num]
        logger.info(f"Limited to first {args.sample_num} tasks")
    
    data_to_run = data
    logger.info(f"Total tasks to process: {len(data_to_run)}")
    
    # Setup output directory
    memory_name = ""
    if args.memory_provider:
        try:
            temp_provider = load_memory_provider(args.memory_provider, model)
            if temp_provider:
                memory_name = temp_provider.get_memory_type().value + "_"
        except Exception as e:
            logger.warning(f"Failed to create temp provider for memory name: {e}")
            try:
                memory_type = MemoryType(args.memory_provider)
                memory_name = memory_type.value + "_"
            except Exception:
                pass
    
    if args.direct_output_dir:
        run_dir = args.direct_output_dir
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Using direct output directory: {run_dir}")
    else:
        out_dir = os.path.dirname(args.outfile) or "."
        base_name = os.path.splitext(os.path.basename(args.outfile))[0]
        run_dir = create_run_directory(out_dir, base_name, memory_name)
        logger.info(f"Run directory created: {run_dir}")
    
    results = []
    file_lock = threading.Lock()
    
    def safe_write(result):
        """Thread-safe result saving"""
        with file_lock:
            idx = result.get("item_index")
            filename = f"{idx}.json" if idx is not None else None
            save_task_result(result, run_dir, filename)
    
    # Process tasks
    # Note: Alfworld typically needs sequential processing due to environment state
    # But we support concurrency for different environment instances
    effective_concurrency = args.concurrency
    if args.memory_provider:
        logger.info(f"Memory provider enabled: {args.memory_provider}")
    
    # For Alfworld, concurrency > 1 requires separate environment instances
    # which we handle by creating new env_wrapper in each process_item call
    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        futures = [
            executor.submit(
                process_item,
                item,
                model_config,
                args.summary_interval,
                args.prompts_type,
                args.max_steps,
                args.memory_provider,
                args.env_config_path,
                split_env,
                args.enable_memory_evolution,
                args.simple_agent
            ) for item in data_to_run
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Alfworld"):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    safe_write(result)
                    
                    metrics = result.get("metrics", {})
                    success_str = "✓" if result.get("is_correct") else "✗"
                    if result.get("status") == "success":
                        logger.info(
                            f"Task done [{len(results)}/{len(futures)}] {success_str}: "
                            f"{result.get('task_type', 'unknown')} | "
                            f"Steps: {result.get('steps_taken', 0)} | "
                            f"Time: {metrics.get('elapsed_time', 0):.1f}s"
                        )
                    else:
                        logger.warning(
                            f"Task error [{len(results)}/{len(futures)}]: "
                            f"{result.get('error', 'Unknown')[:100]}"
                        )
            except Exception as e:
                import traceback
                logger.error(f"Failed to get result from future: {traceback.format_exc()}")
    
    logger.info(f"Processing completed. Total results: {len(results)}")
    
    # Save combined results
    write_jsonl(args.outfile, results)
    logger.info(f"Results saved to {args.outfile}")
    
    # Generate report
    report_path = os.path.join(run_dir, "report.txt")
    
    # Calculate Alfworld-specific statistics
    success_count = sum(1 for r in results if r.get("is_correct"))
    total_count = len(results)
    
    # Statistics by task type
    by_task_type = {}
    for r in results:
        task_type = r.get("task_type", "unknown")
        if task_type not in by_task_type:
            by_task_type[task_type] = {"total": 0, "success": 0}
        by_task_type[task_type]["total"] += 1
        if r.get("is_correct"):
            by_task_type[task_type]["success"] += 1
    
    # Write custom report for Alfworld
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Alfworld Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Tasks: {total_count}\n")
        f.write(f"Successful: {success_count} ({success_count/total_count*100:.1f}%)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("By Task Type\n")
        f.write("-" * 80 + "\n")
        for task_type in sorted(by_task_type.keys()):
            stats = by_task_type[task_type]
            acc = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"  {task_type}: {stats['success']}/{stats['total']} ({acc:.1f}%)\n")
        f.write("\n")
        
        # Resource usage
        total_time = sum(r.get("metrics", {}).get("elapsed_time", 0) for r in results)
        total_tokens = sum(r.get("metrics", {}).get("total_tokens", 0) for r in results)
        total_steps = sum(r.get("steps_taken", 0) for r in results)
        
        f.write("-" * 80 + "\n")
        f.write("Resource Usage\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)\n")
        f.write(f"Total Tokens: {total_tokens:,}\n")
        f.write(f"Total Steps: {total_steps}\n")
        f.write(f"Average Steps per Task: {total_steps/total_count:.1f}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Alfworld Evaluation Summary")
    print("=" * 80)
    print(f"Success Rate: {success_count}/{total_count} = {success_count/total_count*100:.2f}%")
    print(f"Total Time: {total_time/60:.1f}m | Total Tokens: {total_tokens:,}")
    print("-" * 40)
    for task_type in sorted(by_task_type.keys()):
        stats = by_task_type[task_type]
        acc = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {task_type}: {stats['success']}/{stats['total']} ({acc:.1f}%)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alfworld evaluation with MemEvolve framework')
    
    # Input/Output
    parser.add_argument('--infile', type=str, default=None,
                        help='Input path (JSONL/JSON). If not provided, loads from environment')
    parser.add_argument('--outfile', type=str, default="./alfworld_output/alfworld_results.jsonl",
                        help='Output path for results')
    parser.add_argument('--direct_output_dir', type=str, default=None,
                        help='Direct output directory (skips timestamped nesting)')
    
    # Data selection
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'], default='test',
                        help='Dataset split')
    parser.add_argument('--part_num', type=int, default=1,
                        help='Number of parts to split the dataset')
    parser.add_argument('--part_idx', type=int, default=-1,
                        help='Index of the part to process')
    parser.add_argument('--sample_num', type=int, default=None,
                        help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None,
                        help='Task indices to run (e.g., "1-10", "1,3,5-8")')
    parser.add_argument('--task_type', type=str, 
                        choices=['put', 'clean', 'heat', 'cool', 'examine', 'puttwo'],
                        default=None, help='Filter by task type')
    
    # Agent configuration
    parser.add_argument('--summary_interval', type=int, default=5,
                        help='Summary interval for agent')
    parser.add_argument('--prompts_type', type=str, default="default",
                        help='Type of prompts to use')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps for agent')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of concurrent tasks (each uses separate env)')
    
    # Memory configuration
    parser.add_argument('--memory_provider', type=str, default=None,
                        help='Memory provider type (e.g., "agent_kb", "lightweight_memory")')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory)')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution',
                        action='store_false',
                        help='Disable memory system evolution')
    
    # Environment configuration
    parser.add_argument('--env_config_path', type=str,
                        default="eval_agent/data/alfworld/base_config.yaml",
                        help='Path to Alfworld environment config')

    # Agent type
    parser.add_argument('--simple_agent', action='store_true', default=False,
                        help='Use simple text-based agent without tool calling (recommended for smaller models)')

    args = parser.parse_args()
    
    main(args)
