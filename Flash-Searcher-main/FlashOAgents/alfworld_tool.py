#!/usr/bin/env python
# coding=utf-8
"""
Alfworld environment tool for MemEvolve framework.
Wraps the Alfworld environment to work with ToolCallingAgent.
"""

import os
import re
import yaml
import logging
from typing import Optional, Tuple, Dict, Any

from .tools import Tool

logger = logging.getLogger(__name__)


class AlfworldEnvWrapper:
    """
    Wrapper for Alfworld environment that manages state across tool calls.
    This is a singleton-like wrapper that maintains environment state.
    """
    
    def __init__(self, config_path: str, split: str = "eval_out_of_distribution"):
        self.config_path = config_path
        self.split = split
        self.env = None
        self.current_task_idx = -1
        self.current_observation = ""
        self.current_game_file = ""
        self.done = False
        self.reward = 0
        self.steps = 0
        self.max_steps = 50
        self._initialized = False
        
    def initialize(self):
        """Initialize the Alfworld environment"""
        if self._initialized:
            return
            
        try:
            import alfworld
            import alfworld.agents.environment as envs
        except ImportError:
            raise ImportError(
                "alfworld is not installed. Please install it with: "
                "pip install alfworld"
            )
        
        # Set environment data path
        alfworld_data_path = os.path.dirname(self.config_path)
        os.environ["ALFWORLD_DATA"] = alfworld_data_path
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        self.env = getattr(alfworld.agents.environment, env_type)(
            config, train_eval=self.split
        )
        self.env = self.env.init_env(batch_size=1)
        self._initialized = True
        logger.info(f"Alfworld environment initialized with split: {self.split}")
    
    def reset(self, task_idx: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment to a new task.
        
        Args:
            task_idx: Optional task index to skip to. If None, just reset to next task.
                     Note: This assumes sequential task access. For non-sequential access,
                     create a new wrapper instance for each task.
            
        Returns:
            Tuple of (observation, info dict)
        """
        if not self._initialized:
            self.initialize()
        
        # Skip to specific task if needed
        # Only skip if task_idx is provided and different from expected next task
        if task_idx is not None:
            expected_next = self.current_task_idx + 1
            if task_idx != expected_next:
                skip_count = task_idx - expected_next
                if skip_count > 0:
                    self.env.skip(skip_count)
                elif skip_count < 0:
                    # Cannot go backwards, need to recreate environment
                    logger.warning(
                        f"Cannot skip backwards from {self.current_task_idx} to {task_idx}. "
                        f"Creating new environment instance is recommended for non-sequential access."
                    )
        
        obs, info = self.env.reset()
        
        # Process observation - remove initial description part
        obs_text = obs[0] if isinstance(obs, list) else obs
        obs_lines = obs_text.split("\n\n")
        if len(obs_lines) > 1:
            obs_text = "\n".join(obs_lines[1:])
        
        self.current_observation = obs_text
        self.current_game_file = info.get("extra.gamefile", [""])[0]
        self.current_task_idx = task_idx if task_idx is not None else self.current_task_idx + 1
        self.done = False
        self.reward = 0
        self.steps = 0
        
        return self.current_observation, {
            "game_file": self.current_game_file,
            "task_idx": self.current_task_idx
        }
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action string to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.done:
            return "Task already completed.", self.reward, True, {"won": self.reward > 0}
        
        observation, reward, done, info = self.env.step([action])
        
        # Process outputs
        obs_text = observation[0] if isinstance(observation, list) else observation
        won = info.get('won', [False])[0]
        done = done[0] if isinstance(done, list) else done
        
        # Process observation text
        if obs_text.startswith('You arrive at loc '):
            obs_text = obs_text[obs_text.find('. ')+2:]
        
        self.current_observation = obs_text
        self.done = done
        self.reward = 1 if won else 0
        self.steps += 1
        
        # Check max steps
        if self.steps >= self.max_steps and not done:
            self.done = True
            
        return obs_text, self.reward, self.done, {"won": won, "steps": self.steps}


class AlfworldActionTool(Tool):
    """
    Tool for executing actions in Alfworld environment.
    
    This tool wraps the Alfworld environment and allows agents to interact
    with household simulation tasks.
    """
    
    name = "alfworld_action"
    description = """Execute an action in the Alfworld household environment.

Available actions:
1. go to {recep} - Navigate to a receptacle (e.g., "go to fridge 1")
2. take {obj} from {recep} - Pick up an object (e.g., "take apple 1 from fridge 1")
3. put {obj} in/on {recep} - Place an object (e.g., "put apple 1 in/on countertop 1")
4. open {recep} - Open a receptacle (e.g., "open fridge 1")
5. close {recep} - Close a receptacle (e.g., "close fridge 1")
6. toggle {obj} {recep} - Toggle an object (e.g., "toggle desklamp 1")
7. clean {obj} with {recep} - Clean an object (e.g., "clean apple 1 with sinkbasin 1")
8. heat {obj} with {recep} - Heat an object (e.g., "heat apple 1 with microwave 1")
9. cool {obj} with {recep} - Cool an object (e.g., "cool apple 1 with fridge 1")
10. examine {obj} - Examine an object (e.g., "examine apple 1")
11. inventory - Check your inventory
12. look - Look around the current location

Returns the observation after executing the action."""

    inputs = {
        "action": {
            "type": "string",
            "description": "The action to execute in the environment. Must be one of the available action formats."
        }
    }
    output_type = "string"
    
    def __init__(self, env_wrapper: AlfworldEnvWrapper):
        super().__init__()
        self.env_wrapper = env_wrapper
        self.task_completed = False
        self.task_success = False
    
    def forward(self, action: str) -> str:
        """
        Execute an action in the Alfworld environment.
        
        Args:
            action: The action string to execute
            
        Returns:
            Observation string after executing the action
        """
        # Clean up action string
        action = action.strip()
        
        # Parse action if it contains "Action:" prefix
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', action, re.IGNORECASE | re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        
        # Remove any trailing thought or extra content
        action = action.split('\n')[0].strip()
        
        try:
            observation, reward, done, info = self.env_wrapper.step(action)
            
            self.task_completed = done
            self.task_success = info.get("won", False)
            
            result = f"{observation}"
            
            if done:
                if self.task_success:
                    result += "\n\n[SUCCESS] Task completed successfully!"
                else:
                    result += "\n\n[FAILED] Task failed or max steps reached."
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing action '{action}': {e}")
            return f"Error: Invalid action '{action}'. Please use one of the available action formats."
    
    def is_task_done(self) -> bool:
        """Check if the current task is done"""
        return self.task_completed
    
    def is_task_successful(self) -> bool:
        """Check if the current task was successful"""
        return self.task_success
    
    def get_current_observation(self) -> str:
        """Get the current observation"""
        return self.env_wrapper.current_observation


class AlfworldLookTool(Tool):
    """
    Tool for observing the current state in Alfworld.
    Useful for getting the initial observation or refreshing current state.
    """
    
    name = "alfworld_look"
    description = "Look around the current location in Alfworld to see what objects and receptacles are present."
    inputs = {}
    output_type = "string"
    
    # Skip forward signature validation since we have no inputs
    skip_forward_signature_validation = True
    
    def __init__(self, env_wrapper: AlfworldEnvWrapper):
        super().__init__()
        self.env_wrapper = env_wrapper
    
    def forward(self) -> str:
        """Look around the current location"""
        try:
            observation, _, _, _ = self.env_wrapper.step("look")
            return observation
        except Exception as e:
            return f"Error: {e}"


__all__ = [
    "AlfworldEnvWrapper",
    "AlfworldActionTool",
    "AlfworldLookTool"
]
