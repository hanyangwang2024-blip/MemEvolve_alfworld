#!/usr/bin/env python
# coding=utf-8
"""
Export Alfworld tasks to JSONL format for use without the full environment.

This script loads tasks from the Alfworld environment and exports them to a JSONL file
that can be used for evaluation without requiring the Alfworld environment at runtime.

Usage:
    python scripts/export_alfworld_tasks.py \
        --split test \
        --output ./data/alfworld/test_tasks.jsonl
"""

import os
import sys
import json
import yaml
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Task type mapping
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


def export_tasks(config_path: str, split: str, output_path: str):
    """
    Export Alfworld tasks to JSONL format.
    
    Args:
        config_path: Path to Alfworld config file
        split: Dataset split ('train', 'dev', 'test')
        output_path: Output JSONL file path
    """
    try:
        import alfworld
        import alfworld.agents.environment as envs
    except ImportError:
        logger.error(
            "alfworld is not installed. Please install it with:\n"
            "pip install alfworld\n"
            "And download the data from: https://github.com/alfworld/alfworld"
        )
        sys.exit(1)
    
    # Set environment data path
    alfworld_data_path = os.path.dirname(config_path)
    os.environ["ALFWORLD_DATA"] = alfworld_data_path
    
    logger.info(f"ALFWORLD_DATA set to: {alfworld_data_path}")
    
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
    
    logger.info(f"Loading {n_tasks} tasks from split: {split} ({split_env})")
    
    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    env = getattr(alfworld.agents.environment, env_type)(
        config, train_eval=split_env
    )
    env = env.init_env(batch_size=1)
    
    # Export tasks
    tasks = []
    for idx in range(n_tasks):
        obs, info = env.reset()
        
        # Process observation
        obs_text = obs[0] if isinstance(obs, list) else obs
        obs_lines = obs_text.split("\n\n")
        
        # Full observation
        full_observation = obs_text
        
        # Task description (usually after the first paragraph)
        if len(obs_lines) > 1:
            task_observation = "\n".join(obs_lines[1:])
        else:
            task_observation = obs_text
        
        game_file = info.get("extra.gamefile", [""])[0]
        task_type = get_task_type(game_file)
        
        task_data = {
            "task_id": idx,
            "observation": task_observation,
            "full_observation": full_observation,
            "game_file": game_file,
            "task_type": task_type,
            "split": split,
        }
        tasks.append(task_data)
        
        if (idx + 1) % 20 == 0:
            logger.info(f"Processed {idx + 1}/{n_tasks} tasks")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    logger.info(f"Exported {len(tasks)} tasks to {output_path}")
    
    # Print statistics
    type_counts = {}
    for task in tasks:
        t = task["task_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\nTask type distribution:")
    for task_type, count in sorted(type_counts.items()):
        print(f"  {task_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Export Alfworld tasks to JSONL')
    
    parser.add_argument('--config', type=str,
                        default="eval_agent/data/alfworld/base_config.yaml",
                        help='Path to Alfworld config file')
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'],
                        default='test', help='Dataset split')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output JSONL file path')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"./data/alfworld/{args.split}_tasks.jsonl"
    
    export_tasks(args.config, args.split, args.output)


if __name__ == '__main__':
    main()
