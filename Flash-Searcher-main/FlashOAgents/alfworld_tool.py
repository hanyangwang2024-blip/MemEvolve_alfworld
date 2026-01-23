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
        self.current_admissible_actions = []
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
            from alfworld.agents.environment import get_environment
        except ImportError:
            raise ImportError(
                "alfworld is not installed. Please install it with: "
                "pip install alfworld"
            )

        # Set environment data path (only if not already set)
        if "ALFWORLD_DATA" not in os.environ:
            alfworld_data_path = os.path.dirname(self.config_path)
            os.environ["ALFWORLD_DATA"] = alfworld_data_path

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        AlfredEnvClass = get_environment(env_type)
        self.env = AlfredEnvClass(config, train_eval=self.split)
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

        # Process observation - handle different return formats
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
        
        self.current_observation = obs_text
        self.current_game_file = info.get("extra.gamefile", [""])[0]
        self.current_task_idx = task_idx if task_idx is not None else self.current_task_idx + 1
        self.done = False
        self.reward = 0
        self.steps = 0

        # Extract admissible actions for initial state
        admissible_actions = info.get('admissible_commands', [])
        if isinstance(admissible_actions, list):
            admissible_actions = admissible_actions[0] if len(admissible_actions) > 0 else []

        # Store current admissible actions
        self.current_admissible_actions = admissible_actions

        return self.current_observation, {
            "game_file": self.current_game_file,
            "task_idx": self.current_task_idx,
            "admissible_actions": admissible_actions
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

        # Process outputs - handle different return formats
        # Official Alfworld returns batch format: [obs], reward, [done], info
        if isinstance(observation, list):
            obs_text = observation[0]
        elif isinstance(observation, tuple):
            obs_text = observation[0]
        else:
            obs_text = observation
        # Handle nested structures
        while isinstance(obs_text, (list, tuple)):
            obs_text = obs_text[0]

        # Extract reward signal - may be list or scalar
        if isinstance(reward, (list, tuple)):
            reward = reward[0] if len(reward) > 0 else 0
        reward = float(reward) if reward is not None else 0.0

        # Extract won signal - handle both list and scalar formats
        # Official format: info['won'] = [True/False] (list for batch)
        won_raw = info.get('won', [False])
        if isinstance(won_raw, (list, tuple)):
            won = won_raw[0] if len(won_raw) > 0 else False
        else:
            won = won_raw

        # Extract done signal - handle both list and scalar formats
        # Official format: done = [1.0/0.0] (list of floats for batch)
        done = done[0] if isinstance(done, (list, tuple)) else done

        # Convert to proper types
        won = bool(won)
        done = bool(done) or (float(done) > 0.5)  # Handle 0.0/1.0 format

        # Detailed logging - CRITICAL: log all done/won signals
        if done or won:
            logger.info(f"⚠️  ALFWORLD SIGNAL: Step {self.steps}: action='{action}', done={done}, won={won}, reward={reward}")
        else:
            logger.debug(f"Step {self.steps}: action='{action}', done={done}, won={won}, reward={reward}, obs='{obs_text[:100]}...'")

        # Process observation text
        if isinstance(obs_text, str) and obs_text.startswith('You arrive at loc '):
            obs_text = obs_text[obs_text.find('. ')+2:]

        self.current_observation = obs_text
        self.done = done
        self.reward = reward  # Use actual reward from environment, not derived
        self.steps += 1

        # Extract admissible actions if available
        admissible_actions = info.get('admissible_commands', [])
        if isinstance(admissible_actions, list):
            admissible_actions = admissible_actions[0] if len(admissible_actions) > 0 else []

        # Store current admissible actions
        self.current_admissible_actions = admissible_actions

        # Check max steps - only mark done if we haven't already succeeded
        max_steps_reached = self.steps >= self.max_steps and not done
        if max_steps_reached:
            self.done = True
            # Don't mark as won when max steps reached - keep won as Alfworld determined
            done = True

        return obs_text, self.reward, self.done, {
            "won": won,
            "steps": self.steps,
            "max_steps_reached": max_steps_reached,
            "admissible_actions": admissible_actions
        }


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

            # Check if task is done FIRST before showing actions
            if done:
                # Make success/failure signal VERY prominent
                result += "\n\n" + "="*60
                if self.task_success:
                    result += "\n[SUCCESS] Task completed successfully!"
                    result += "\n[SUCCESS] You MUST now call the final_answer tool!"
                    result += "\n[SUCCESS] Use: {\"name\": \"final_answer\", \"arguments\": {\"answer\": \"success\"}}"
                else:
                    result += "\n[FAILED] Task failed or max steps reached."
                    result += "\n[FAILED] You MUST now call the final_answer tool!"
                    result += "\n[FAILED] Use: {\"name\": \"final_answer\", \"arguments\": {\"answer\": \"failed\"}}"
                result += "\n" + "="*60
                # Do NOT show available actions when task is done
            else:
                # Only show available actions if task is NOT done
                admissible_actions = info.get("admissible_actions", [])
                if admissible_actions and len(admissible_actions) > 0:
                    result += "\n\n[AVAILABLE ACTIONS]"
                    # Show up to 15 most relevant actions to keep context manageable
                    for i, act in enumerate(admissible_actions[:15], 1):
                        result += f"\n{i}. {act}"
                    if len(admissible_actions) > 15:
                        result += f"\n... and {len(admissible_actions) - 15} more actions available"

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
