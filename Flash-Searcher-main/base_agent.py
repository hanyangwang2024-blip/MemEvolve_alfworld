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

from dotenv import load_dotenv
from utils import safe_json_loads
import time

from FlashOAgents import ToolCallingAgent
from FlashOAgents import ActionStep, PlanningStep, TaskStep, SummaryStep
from FlashOAgents import WebSearchTool, CrawlPageTool, VisualInspectorTool, AudioInspectorTool, TextInspectorTool
from FlashOAgents import AgentMemory, ChatMessage, MessageRole, ToolCall

load_dotenv(override=True)

class BaseAgent:
    def __init__(self, model):
        self.model = model
        self.agent_fn = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def capture_trajectory(self, ):
        if not hasattr(self, 'agent_fn'):
            raise ValueError("[capture_trajectory] agent_fn is not defined.")
        if not isinstance(self.agent_fn, ToolCallingAgent):
            raise ValueError("[capture_trajectory] agent_fn must be an instance of ToolCallingAgent.")
        trajectory = []
        for step_num, step in enumerate(self.agent_fn.memory.steps):
            if isinstance(step, TaskStep):
                continue
            elif isinstance(step, PlanningStep):
                traj = {
                    "name": "plan", 
                    "value": step.plan, 
                    "think": step.plan_think, 
                    "cot_think": step.plan_reasoning,
                    "memory_guidance": step.memory_guidance
                }
                trajectory.append(traj)
            elif isinstance(step, SummaryStep):
                traj = {"name": "summary", "value": step.summary, "cot_think": step.summary_reasoning}
                trajectory.append(traj)
            elif isinstance(step, ActionStep):
                safe_tool_calls = step.tool_calls if step.tool_calls is not None else []
                traj = {
                    "name": "action", 
                    "tool_calls": [st.dict() for st in safe_tool_calls], 
                    "obs": step.observations,
                    "think": step.action_think, 
                    "cot_think": step.action_reasoning,
                    "memory_guidance": step.memory_guidance
                }
                trajectory.append(traj)
            else:
                raise ValueError("[capture_trajectory] Unknown Step:", step)

        return {
            "agent_trajectory": trajectory,
        }

    def forward(self, task, answer=None, return_json=False, max_retries=3):
        last_error = None
        for _ in range(max_retries):
            try:
                if answer is not None:
                    result = self.agent_fn.run(task, answer=answer)
                else:
                    result = self.agent_fn.run(task)
                if return_json and isinstance(result, str):
                    result = safe_json_loads(result)
                elif not return_json and isinstance(result, dict):
                    result = str(result)
                return {
                    "agent_result": result, **self.capture_trajectory()
                }
            except Exception as e:
                if "TERMINATE_QUOTA" in str(e):
                    return {"error": "quota_exceeded"}
                last_error = e
                print(f"[BaseAgent] error: {e}")
                continue
        return {"error": str(last_error)}


class SearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, memory_provider=None, **kwargs):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        tools = [web_tool, crawl_tool]
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider
        )

class MMSearchAgent(BaseAgent):
    def __init__(self, model, summary_interval, prompts_type, max_steps, memory_provider=None, **kwargs):
        super().__init__(model)

        web_tool = WebSearchTool()
        crawl_tool = CrawlPageTool(model=model)
        visual_tool = VisualInspectorTool(model, 100000)
        text_tool = TextInspectorTool(model, 100000) 
        audio_tool = AudioInspectorTool(model, 100000) 
        tools = [web_tool, crawl_tool, visual_tool,text_tool,audio_tool] 

        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider
        )


class AnalysisAgent(BaseAgent):
    """Agent for analyzing task execution trajectories"""
    
    def __init__(self, model, task_logs_dir, prompts_type="default", max_steps=15, summary_model_id="gpt-4o-mini", **kwargs):
        super().__init__(model)
        
        from MemEvolve.utils.trajectory_tools import (
            TrajectoryViewerTool,
            StepViewerTool,
            MemoryDatabaseViewerTool
        )
        
        tools = [
            TrajectoryViewerTool(task_logs_dir, max_tasks=3, model_id=summary_model_id),
            StepViewerTool(task_logs_dir),
            MemoryDatabaseViewerTool()
        ]
        
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=5,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=None
        )


class ReviewAgent(BaseAgent):
    """Agent for code review and validation"""
    
    def __init__(self, model, prompts_type="default", max_steps=10, **kwargs):
        super().__init__(model)
        
        # No tools needed - pure reasoning agent
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=[],
            summary_interval=5,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=None
        )


class AlfworldAgent(BaseAgent):
    """
    Agent for Alfworld household simulation tasks.
    
    This agent uses AlfworldActionTool to interact with the Alfworld environment
    and complete household tasks like cleaning, heating, cooling objects, etc.
    """
    
    # Alfworld task instruction template
    ALFWORLD_INSTRUCTION = """You are an intelligent agent in a household environment. Your goal is to complete the task described below.

TASK: {task_description}

CURRENT OBSERVATION:
{observation}

# YOU HAVE EXACTLY TWO TOOLS AVAILABLE:

1. **alfworld_action** - Execute an action in the household environment
   - Input: {{"action": "action string"}}
   - The tool name is ALWAYS "alfworld_action"
   - Pass the complete action as the "action" argument

2. **final_answer** - Submit your final result
   - Use when you see "[SUCCESS]" or "[FAILED]" in the observation

# HOW TO USE AVAILABLE ACTIONS:

The observation includes an [AVAILABLE ACTIONS] section that lists all valid actions you can take in the current state.
- **ALWAYS choose your action from this list** - these are the ONLY actions that will work
- The list is updated after every action you take
- Choosing an action not in the list will result in an error
- Common action formats include:
  * go to {{receptacle}} (e.g., "go to fridge 1")
  * open {{receptacle}} (e.g., "open cabinet 1")
  * close {{receptacle}} (e.g., "close drawer 1")
  * take {{object}} from {{receptacle}} (e.g., "take apple 1 from fridge 1")
  * put {{object}} on {{receptacle}} (e.g., "put plate 1 on countertop 1") - Use "on" for surfaces
  * put {{object}} in {{receptacle}} (e.g., "put apple 1 in fridge 1") - Use "in" for containers
  * clean {{object}} with {{receptacle}} (e.g., "clean apple 1 with sinkbasin 1")
  * heat {{object}} with {{receptacle}} (e.g., "heat bread 1 with microwave 1")
  * cool {{object}} with {{receptacle}} (e.g., "cool tomato 1 with fridge 1")
  * examine {{object}}, inventory, look

# CRITICAL TASK REQUIREMENTS:

**For tasks involving object state changes, you MUST perform the state-changing action:**

1. **"put a CLEAN [object] in [place]"** tasks:
   - Step 1: Find and take the object
   - Step 2: **CLEAN the object with sinkbasin** (e.g., "clean ladle 1 with sinkbasin 1")
   - Step 3: Put the cleaned object in the target place
   - ⚠️ Simply placing the object WITHOUT cleaning will FAIL the task!

2. **"put a HEATED [object] in [place]"** tasks:
   - Step 1: Find and take the object
   - Step 2: **HEAT the object with microwave** (e.g., "heat bread 1 with microwave 1")
   - Step 3: Put the heated object in the target place
   - ⚠️ Simply placing the object WITHOUT heating will FAIL the task!

3. **"put a COOLED [object] in [place]"** tasks:
   - Step 1: Find and take the object
   - Step 2: **COOL the object with fridge** (e.g., "cool apple 1 with fridge 1")
   - Step 3: Put the cooled object in the target place
   - ⚠️ Simply placing the object WITHOUT cooling will FAIL the task!

**Read the task description carefully! If it says "clean", "heat", or "cool", you MUST perform that action.**

# CORRECT USAGE EXAMPLES:

Example 1 - Choosing from available actions:
Observation: You are in the kitchen.
[AVAILABLE ACTIONS]
1. go to cabinet 1
2. go to fridge 1
3. go to countertop 1
4. inventory

Action:
{{
  "think": "I see 4 available actions. I need to find a plate, so let me check cabinet 1 first.",
  "tools": [
    {{
      "name": "alfworld_action",
      "arguments": {{"action": "go to cabinet 1"}}
    }}
  ]
}}

Example 2 - Opening a container:
Observation: You arrive at cabinet 1. The cabinet 1 is closed.
[AVAILABLE ACTIONS]
1. open cabinet 1
2. go to fridge 1
3. examine cabinet 1

Action:
{{
  "think": "The cabinet is closed. The available actions show I can open it. Let me do that.",
  "tools": [
    {{
      "name": "alfworld_action",
      "arguments": {{"action": "open cabinet 1"}}
    }}
  ]
}}

Example 3 - Taking an object:
Observation: You open cabinet 1. In it, you see a plate 1.
[AVAILABLE ACTIONS]
1. take plate 1 from cabinet 1
2. close cabinet 1
3. go to fridge 1

Action:
{{
  "think": "Perfect! I found a plate. The available actions show I can take it. Let me pick it up.",
  "tools": [
    {{
      "name": "alfworld_action",
      "arguments": {{"action": "take plate 1 from cabinet 1"}}
    }}
  ]
}}

Example 4 - Task completion (SUCCESS):
Observation: You put the plate on the countertop.
============================================================
[SUCCESS] Task completed successfully!
[SUCCESS] You MUST now call the final_answer tool!
[SUCCESS] Use: {{"name": "final_answer", "arguments": {{"answer": "success"}}}}
============================================================

Action:
{{
  "think": "I see [SUCCESS] markers. The task is complete and successful. I must call final_answer with 'success'.",
  "tools": [
    {{
      "name": "final_answer",
      "arguments": {{"answer": "success"}}
    }}
  ]
}}

Example 5 - Task completion (FAILED):
Observation: You are still in the kitchen.
============================================================
[FAILED] Task failed or max steps reached.
[FAILED] You MUST now call the final_answer tool!
[FAILED] Use: {{"name": "final_answer", "arguments": {{"answer": "failed"}}}}
============================================================

Action:
{{
  "think": "I see [FAILED] markers. The task failed or I ran out of steps. I must call final_answer with 'failed', NOT 'success'.",
  "tools": [
    {{
      "name": "final_answer",
      "arguments": {{"answer": "failed"}}
    }}
  ]
}}

# CRITICAL RULES - READ CAREFULLY:
1. Tool name is ALWAYS "alfworld_action" (NOT "go to", "open", "take", etc.)
2. The action string is passed as the "action" argument value
3. **ALWAYS select your action from the [AVAILABLE ACTIONS] list in the observation**
4. Execute ONLY ONE alfworld_action tool per turn
5. Object/receptacle numbers matter (e.g., "plate 1" not "plate")
6. **UNDERSTAND THE TASK REQUIREMENTS:**
   - Read the task description carefully for keywords: "clean", "heat", "cool"
   - "put a CLEAN ladle" means: find ladle → CLEAN it with sinkbasin → put it
   - "put a HEATED bread" means: find bread → HEAT it with microwave → put it
   - "put a COOLED apple" means: find apple → COOL it with fridge → put it
   - DO NOT skip the state-changing action (clean/heat/cool) - the task will FAIL!
7. **RECOGNIZING TASK COMPLETION:**
   - [SUCCESS] markers = Task completed SUCCESSFULLY → call final_answer with "success"
   - [FAILED] markers = Task FAILED or ran out of steps → call final_answer with "failed"
   - These markers are surrounded by ============================================================
   - They appear when the episode ends (success OR failure)
8. **WHEN YOU SEE [SUCCESS] OR [FAILED] MARKERS:**
   - The task execution is COMPLETE (no more actions possible)
   - You MUST IMMEDIATELY call the final_answer tool
   - [SUCCESS] → use {{"answer": "success"}}
   - [FAILED] → use {{"answer": "failed"}}  ← DO NOT use "success" when you see [FAILED]!
   - Do NOT call alfworld_action after seeing these markers
   - The observation will show you the exact format to use
9. Do NOT invent new tool names - only use "alfworld_action" and "final_answer"
10. If no [AVAILABLE ACTIONS] section is shown, use standard action formats

Begin by thinking about the task and calling the appropriate tool."""

    def __init__(
        self, 
        model, 
        env_wrapper,
        summary_interval: int = 5,
        prompts_type: str = "default", 
        max_steps: int = 50,
        memory_provider=None,
        **kwargs
    ):
        super().__init__(model)
        
        from FlashOAgents.alfworld_tool import AlfworldActionTool
        
        self.env_wrapper = env_wrapper
        
        # Create Alfworld action tool
        self.alfworld_tool = AlfworldActionTool(env_wrapper)
        tools = [self.alfworld_tool]
        
        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            summary_interval=summary_interval,
            max_steps=max_steps,
            prompts_type=prompts_type,
            memory_provider=memory_provider
        )
    
    def reset_task(self, task_idx: int = None):
        """
        Reset the environment to a new task.
        
        Args:
            task_idx: Optional task index to reset to
            
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env_wrapper.reset(task_idx)
        # Reset tool state
        self.alfworld_tool.task_completed = False
        self.alfworld_tool.task_success = False
        return obs, info
    
    def forward(self, task_description: str, observation: str = None, **kwargs):
        """
        Execute an Alfworld task.

        Args:
            task_description: Description of the task to complete
            observation: Initial observation (if None, uses current env observation)

        Returns:
            Dict with agent_result, agent_trajectory, and task status
        """
        if observation is None:
            observation = self.env_wrapper.current_observation

        # Enhance observation with available actions from the environment
        admissible_actions = self.env_wrapper.current_admissible_actions
        if admissible_actions and len(admissible_actions) > 0:
            observation += "\n\n[AVAILABLE ACTIONS]"
            # Show up to 15 actions to keep context manageable
            for i, act in enumerate(admissible_actions[:15], 1):
                observation += f"\n{i}. {act}"
            if len(admissible_actions) > 15:
                observation += f"\n... and {len(admissible_actions) - 15} more actions available"

        # Build the task prompt
        task_prompt = self.ALFWORLD_INSTRUCTION.format(
            task_description=task_description,
            observation=observation
        )

        # Run the agent
        result = super().forward(task_prompt, **kwargs)

        # Add task completion status
        result["task_completed"] = self.alfworld_tool.is_task_done()
        result["task_success"] = self.alfworld_tool.is_task_successful()
        result["steps_taken"] = self.env_wrapper.steps

        # CRITICAL FIX: Correct agent_result if it contradicts the environment's judgment
        # This handles cases where the agent misunderstands task completion
        agent_result = result.get("agent_result", "")
        task_success = result["task_success"]

        if task_success and agent_result not in ["success", "Success", "SUCCESS"]:
            # Environment says success but agent didn't report it correctly
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Agent reported '{agent_result}' but environment indicates SUCCESS. "
                f"Correcting agent_result to 'success'."
            )
            result["agent_result"] = "success"
        elif not task_success and agent_result in ["success", "Success", "SUCCESS"]:
            # Environment says failure but agent reported success - WRONG!
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Agent reported 'success' but environment indicates FAILURE (won=False). "
                f"Correcting agent_result to 'failed'."
            )
            result["agent_result"] = "failed"

        return result
    
    def is_task_done(self) -> bool:
        """Check if the current task is done"""
        return self.alfworld_tool.is_task_done()
    
    def is_task_successful(self) -> bool:
        """Check if the current task was successful"""
        return self.alfworld_tool.is_task_successful()


class SimpleAlfworldAgent(BaseAgent):
    """
    Simple text-based agent for Alfworld without tool calling.
    Uses ReAct format: Think -> Action -> Observation loop.
    Integrated with MemEvolve memory system.
    """

    SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Think: your thoughts.
Action: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} on {recep} (for surfaces like countertop, shelf, table)
4. put {obj} in {recep} (for containers like fridge, cabinet, drawer)
5. open {recep}
6. close {recep}
7. toggle {obj} {recep}
8. clean {obj} with {recep}
9. heat {obj} with {recep}
10. cool {obj} with {recep}
11. use {recep}
12. look
13. inventory
14. examine {obj}

where {obj} and {recep} correspond to objects and receptacles.

IMPORTANT: Use "put X on Y" for surfaces (countertop, shelf, table, etc.) and "put X in Y" for containers (fridge, cabinet, drawer, etc.). Do NOT use "in/on" format.

Here is an example:
Task: put some spraybottle on toilet.
Think: To solve the task, I need to find and take a spraybottle, then put it on toilet.
Think: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
Action: go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Action: go to cabinet 2
Observation: The cabinet 2 is closed.
Action: open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Think: Now I find a spraybottle (2). Next, I need to take it.
Action: take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Think: Now I take a spraybottle (2). Next, I need to put it on toilet 1.
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Action: put spraybottle 2 on toilet 1
Observation: You put the spraybottle 2 on the toilet 1.

Here is your task:"""

    def __init__(self, model, env_wrapper, max_steps: int = 50, memory_provider=None):
        super().__init__(model)
        self.env_wrapper = env_wrapper
        self.max_steps = max_steps
        self.memory_provider = memory_provider
        self.trajectory = []
        self.task_completed = False
        self.task_success = False

        # Initialize memory system (for MemEvolve compatibility)
        self.memory = AgentMemory(system_prompt=self.SYSTEM_PROMPT)
        self.agent_fn = self  # Self-reference for compatibility with run script

    def reset_task(self, task_idx: int = None):
        """Reset the environment to a new task."""
        obs, info = self.env_wrapper.reset(task_idx)
        self.trajectory = []
        self.task_completed = False
        self.task_success = False
        self.memory.reset()  # Reset memory steps for new task
        return obs, info

    def _parse_action(self, response: str) -> str:
        """Extract action from ReAct format model response."""
        import re

        # Try to find "Action:" in the response
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # Fallback: take the last non-empty line or first line
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            action = lines[-1] if lines else response.strip()

        # Remove any remaining prefixes
        prefixes = ["Action:", "action:", "I will", "Let me", "Next,"]
        for prefix in prefixes:
            if action.lower().startswith(prefix.lower()):
                action = action[len(prefix):].strip()

        # Remove quotes if present
        if action.startswith('"') and action.endswith('"'):
            action = action[1:-1]
        if action.startswith("'") and action.endswith("'"):
            action = action[1:-1]

        return action

    def _build_prompt_with_history(self, task_description: str, current_obs: str, history: list, admissible_actions: list = None, history_len: int = 2):
        """Build prompt with recent action history in ReAct format.

        Args:
            task_description: The task to complete
            current_obs: Current observation before generating next action
            history: List of completed steps (each has 'action', 'observation', 'think')
            admissible_actions: List of valid actions available in current state
            history_len: Number of recent steps to include in context (default: 2)
        """
        prompt = f"{task_description}\n\n"

        # Add recent history (last N steps) in ReAct format
        recent_history = history[-history_len:] if len(history) > history_len else history

        if recent_history:
            prompt += "Recent experience:\n"
            for i, h in enumerate(recent_history, 1):
                if h.get('think'):
                    prompt += f"Step {len(history) - len(recent_history) + i}:\n"
                    prompt += f"Think: {h['think']}\n"
                    prompt += f"Action: {h['action']}\n"
                    prompt += f"Observation: {h['observation'][:200]}\n\n"
                else:
                    prompt += f"Step {len(history) - len(recent_history) + i}:\n"
                    prompt += f"Action: {h['action']}\n"
                    prompt += f"Observation: {h['observation'][:200]}\n\n"

        # Add current observation and prompt for action
        prompt += f"Current observation: {current_obs}\n"

        # Add admissible actions if available
        if admissible_actions and len(admissible_actions) > 0:
            prompt += "\nYour admissible actions in this situation are:\n"
            for i, action in enumerate(admissible_actions[:15], 1):  # Show up to 15 actions
                prompt += f"  - {action}\n"
            prompt += "\n"

        prompt += "What is your next action? Respond in format:\nThink: <reasoning>\nAction: <action>"
        return prompt

    def forward(self, task_description: str, observation: str = None, history_len: int = 2, **kwargs):
        """Execute an Alfworld task using simple text generation."""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 STARTING NEW TASK")
        logger.info(f"Task: {task_description[:100]}...")
        logger.info(f"Initial obs: {observation[:100] if observation else 'None'}...")
        logger.info(f"Max steps: {self.max_steps}, History len: {history_len}")
        logger.info(f"{'='*80}\n")

        if observation is None:
            observation = self.env_wrapper.current_observation

        self.trajectory = []
        step = 0
        current_obs = observation

        # Get initial admissible actions from environment wrapper's current state
        admissible_actions = getattr(self.env_wrapper, 'current_admissible_actions', [])

        while step < self.max_steps and not self.task_completed:
            # Retrieve memory guidance if memory provider is available
            memory_guidance = ""
            if self.memory_provider and hasattr(self.memory_provider, 'provide_memory'):
                try:
                    guidance = self.memory_provider.provide_memory(
                        query=f"{task_description}\n\nCurrent observation: {current_obs}",
                        phase="in"
                    )
                    if guidance:
                        memory_guidance = f"\n\n[Memory Guidance]\n{guidance}\n"
                        logger.info(f"[Step {step + 1}] Retrieved memory guidance: {guidance[:100]}...")
                except Exception as e:
                    logger.warning(f"Memory retrieval failed: {e}")

            # Build prompt with recent history and current admissible actions
            user_prompt = self._build_prompt_with_history(task_description, current_obs, self.trajectory, admissible_actions, history_len)

            # Append memory guidance to prompt if available
            if memory_guidance:
                user_prompt += memory_guidance

            # Log prompt for debugging (only show for first few steps and every 10th step)
            if step < 3 or step % 10 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"STEP {step + 1} PROMPT:")
                logger.info(f"{'='*80}")
                logger.info(f"Task: {task_description}")
                if self.trajectory:
                    logger.info(f"Recent history ({len(self.trajectory[-history_len:])} steps):")
                    for i, h in enumerate(self.trajectory[-history_len:], 1):
                        logger.info(f"  [{step - (history_len - i)}] Action: {h['action']}")
                        logger.info(f"       Observation: {h['observation'][:100]}...")
                logger.info(f"Current obs: {current_obs[:100]}...")
                if admissible_actions:
                    logger.info(f"Admissible actions ({len(admissible_actions)} total):")
                    for i, action in enumerate(admissible_actions[:5], 1):
                        logger.info(f"  {i}. {action}")
                    if len(admissible_actions) > 5:
                        logger.info(f"  ... and {len(admissible_actions) - 5} more")
                logger.info(f"{'='*80}\n")

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            # Generate action from model
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    action_text = response.content
                else:
                    action_text = str(response)
                logger.info(f"[Step {step + 1}] Model response: {action_text[:150]}...")
            except Exception as e:
                logger.error(f"Model generation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                break

            # Parse think and action from response
            import re
            think_match = re.search(r'Think:\s*(.+?)(?=Action:|$)', action_text, re.IGNORECASE | re.DOTALL)
            think = think_match.group(1).strip() if think_match else ""

            action = self._parse_action(action_text)
            logger.info(f"[Step {step + 1}] Parsed - Think: '{think[:80]}...' | Action: '{action}'")

            # Execute action in environment
            obs, reward, done, info = self.env_wrapper.step(action)

            # Extract admissible actions for next step
            admissible_actions = info.get("admissible_actions", [])
            if isinstance(admissible_actions, list) and len(admissible_actions) > 0:
                # admissible_actions might be nested
                if isinstance(admissible_actions[0], list):
                    admissible_actions = admissible_actions[0]

            # Handle done being a tuple/list (alfworld returns tuple)
            if isinstance(done, (list, tuple)):
                done = done[0]

            # Check if done was forced due to max_steps
            max_steps_reached = info.get("max_steps_reached", False)
            won = info.get("won", False)

            logger.info(f"[Step {step + 1}] Env - done={done}, reward={reward}, won={won}, max_steps_reached={max_steps_reached}")
            if step < 3 or step % 10 == 0:
                logger.info(f"       Observation: {obs[:120]}...")
                if admissible_actions and len(admissible_actions) > 0:
                    logger.info(f"       Admissible actions: {admissible_actions[:3]}{'...' if len(admissible_actions) > 3 else ''}")

            # Record trajectory
            self.trajectory.append({
                "step": step + 1,
                "think": think,
                "action": action,
                "observation": obs,
                "done": done,
                "reward": reward,
                "admissible_actions": admissible_actions
            })

            # Record to memory for MemEvolve integration
            action_step = ActionStep(step_number=step)
            action_step.action_think = think
            action_step.observations = obs
            action_step.action_reasoning = think
            action_step.start_time = time.time()
            action_step.end_time = time.time()
            action_step.duration = 0

            self.memory.steps.append(action_step)

            # Check if task is done
            if done:
                self.task_completed = True
                # Task is successful only if Alfworld explicitly marked it won
                # (not just because we hit max_steps)
                self.task_success = info.get("won", False)
                max_steps_reached = info.get("max_steps_reached", False)

                logger.info(f"\n{'='*80}")
                logger.info(f"✓ TASK TERMINATION SIGNAL DETECTED")
                logger.info(f"   done=True, won={self.task_success}, max_steps_reached={max_steps_reached}")

                if max_steps_reached:
                    logger.info(f"⏱️  REASON: MAX STEPS REACHED ({self.max_steps})")
                    logger.info(f"   Result: {self.task_success}")
                elif self.task_success:
                    logger.info(f"🎯 REASON: TASK COMPLETED SUCCESSFULLY")
                    logger.info(f"   ✓✓✓ SUCCESS DETECTED ✓✓✓")
                else:
                    logger.info(f"❌ REASON: TASK FAILED (done=True but won=False)")

                logger.info(f"   Steps taken: {step + 1}/{self.max_steps}")
                logger.info(f"   Total trajectory length: {len(self.trajectory)}")
                logger.info(f"{'='*80}\n")
                break

            # Update current observation for next step
            current_obs = obs
            step += 1

            # Log progress every 10 steps
            if step % 10 == 0:
                logger.info(f"Progress: {step}/{self.max_steps} steps completed")

        # SAFEGUARD: Check if there's any success signal in trajectory that we might have missed
        success_in_trajectory = any(t.get("reward") == 1 for t in self.trajectory)

        if success_in_trajectory and not self.task_success:
            logger.warning(f"\n{'!'*80}")
            logger.warning(f"⚠️  POTENTIAL ISSUE DETECTED:")
            logger.warning(f"   Trajectory contains reward=1 at some step")
            logger.warning(f"   But task_success is marked as False")
            logger.warning(f"   This may indicate a detection issue")
            for i, t in enumerate(self.trajectory):
                if t.get("reward") == 1:
                    logger.warning(f"   Step {i+1}: reward=1 detected")
            logger.warning(f"{'!'*80}\n")

        result = {
            "agent_result": "success" if self.task_success else "failed",
            "agent_trajectory": self.trajectory,
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "steps_taken": step + 1
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 TASK SUMMARY")
        logger.info(f"   Result: {result['agent_result'].upper()}")
        logger.info(f"   Completed: {result['task_completed']}")
        logger.info(f"   Success marked: {self.task_success}")
        logger.info(f"   Success in trajectory: {success_in_trajectory}")
        logger.info(f"   Steps: {result['steps_taken']}/{self.max_steps}")
        logger.info(f"   Trajectory length: {len(self.trajectory)}")
        logger.info(f"   Memory steps recorded: {len(self.memory.steps)}")
        logger.info(f"{'='*80}\n")

        return result

    def is_task_done(self) -> bool:
        return self.task_completed

    def is_task_successful(self) -> bool:
        return self.task_success

    def write_memory_to_messages(self, include_system_prompt: bool = True):
        """Extract memory steps as messages for MemEvolve integration.

        Returns:
            List[Dict] - messages in format compatible with TrajectoryData
        """
        messages = []

        if include_system_prompt:
            messages.append({
                "role": "system",
                "content": self.SYSTEM_PROMPT
            })

        # Convert trajectory to messages (each action-observation pair)
        for traj_item in self.trajectory:
            if traj_item.get('think'):
                messages.append({
                    "role": "assistant",
                    "content": f"Think: {traj_item['think']}"
                })
            messages.append({
                "role": "assistant",
                "content": f"Action: {traj_item['action']}"
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {traj_item['observation']}"
            })

        return messages

    def capture_trajectory(self):
        """Capture trajectory in MemEvolve format."""
        trajectory = []
        for step_data in self.trajectory:
            traj = {
                "name": "action",
                "action": step_data.get("action", ""),
                "think": step_data.get("think", ""),
                "observation": step_data.get("observation", ""),
                "done": step_data.get("done", False),
                "reward": step_data.get("reward", 0)
            }
            trajectory.append(traj)

        return {
            "agent_trajectory": trajectory,
        }

class SimpleWebShopAgent(BaseAgent):
    """
    Simple text-based agent for WebShop without tool calling.
    Uses ReAct format: Think -> Action -> Observation loop.
    Integrated with MemEvolve memory system.
    """

    SYSTEM_PROMPT = """You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought: I think ...
Action: click[something]"""

    def __init__(self, model, env_wrapper, max_steps: int = 10, memory_provider=None,
                 use_icl: bool = False, icl_path: str = None):
        super().__init__(model)
        self.env_wrapper = env_wrapper
        self.max_steps = max_steps
        self.memory_provider = memory_provider
        self.trajectory = []
        self.task_completed = False
        self.task_success = False
        self.use_icl = use_icl
        self.icl_examples = []

        # Load ICL examples if enabled
        if self.use_icl:
            self._load_icl_examples(icl_path)

        # Initialize memory system (for MemEvolve compatibility)
        self.memory = AgentMemory(system_prompt=self.SYSTEM_PROMPT)
        self.agent_fn = self  # Self-reference for compatibility with run script

    def _load_icl_examples(self, icl_path: str = None):
        """Load in-context learning examples from JSON file."""
        import json
        import os

        if icl_path is None:
            # Default path
            icl_path = os.path.join(
                os.path.dirname(__file__),
                "FlashOAgents/webshop_icl_examples.json"
            )

        if os.path.exists(icl_path):
            with open(icl_path, 'r') as f:
                self.icl_examples = json.load(f)
            logger.info(f"Loaded {len(self.icl_examples)} ICL examples from {icl_path}")
        else:
            logger.warning(f"ICL examples file not found at {icl_path}")
            self.icl_examples = []

    def reset_task(self, session_id: int = None):
        """Reset the environment to a new task."""
        obs, info = self.env_wrapper.reset(session_id)
        self.trajectory = []
        self.task_completed = False
        self.task_success = False
        self.memory.reset()  # Reset memory steps for new task
        return obs, info

    def _parse_action(self, response: str) -> str:
        """Extract action from ReAct format model response."""
        import re

        # Try to find "Action:" in the response
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # Fallback: take the last non-empty line or first line
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            action = lines[-1] if lines else response.strip()

        # Remove any remaining prefixes
        prefixes = ["Action:", "action:", "I will", "Let me", "Next,"]
        for prefix in prefixes:
            if action.lower().startswith(prefix.lower()):
                action = action[len(prefix):].strip()

        # Remove quotes if present
        if action.startswith('"') and action.endswith('"'):
            action = action[1:-1]
        if action.startswith("'") and action.endswith("'"):
            action = action[1:-1]

        return action

    def _build_prompt_with_history(self, task_description: str, current_obs: str,
                                   history: list, clickables: list = None,
                                   has_search_bar: bool = True, history_len: int = 2,
                                   is_first_step: bool = False):
        """Build prompt with recent action history in ReAct format."""
        # For the first step, format like ETO's initial observation
        if is_first_step:
            prompt = f"Observation:\n{current_obs}"
            return prompt

        # For subsequent steps, show recent history
        prompt = ""

        # Add recent history (last N steps)
        recent_history = history[-history_len:] if len(history) > history_len else history

        # For history, use the ReAct format from previous steps (like ETO)
        if recent_history:
            for h in recent_history:
                # Add previous Thought and Action
                if h.get('think'):
                    prompt += f"Thought: {h['think']}\n"
                prompt += f"Action: {h['action']}\n"
                # Add observation from that action
                prompt += f"Observation:\n{h['observation']}\n"

        # Add current observation (using ETO's format: "Observation:\n{obs}")
        prompt += f"Observation:\n{current_obs}"
        return prompt

    def forward(self, task_description: str, observation: str = None, history_len: int = 2, **kwargs):
        """Execute a WebShop task using simple text generation."""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 STARTING WEBSHOP TASK")
        logger.info(f"Task: {task_description[:100]}...")
        logger.info(f"Max steps: {self.max_steps}")
        logger.info(f"{'='*80}\n")

        if observation is None:
            observation = self.env_wrapper.current_observation

        self.trajectory = []
        step = 0
        current_obs = observation
        clickables = self.env_wrapper.current_clickables
        has_search_bar = self.env_wrapper.has_search_bar

        while step < self.max_steps and not self.task_completed:
            # Build prompt
            is_first_step = (step == 0)
            user_prompt = self._build_prompt_with_history(
                task_description, current_obs, self.trajectory,
                clickables, has_search_bar, history_len, is_first_step
            )

            # Build messages with optional ICL examples
            if self.use_icl and is_first_step and self.icl_examples:
                # Use ICL format: system + ICL examples + current task
                messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
                # Add ICL examples (first example from the list)
                if len(self.icl_examples) > 0:
                    messages.extend(self.icl_examples[0])
                # Add current task
                messages.append({"role": "user", "content": user_prompt})
            else:
                # Standard format: system + user prompt
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

            # Generate action from model
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    action_text = response.content
                else:
                    action_text = str(response)
                logger.info(f"[Step {step + 1}] Model response: {action_text[:150]}...")
            except Exception as e:
                logger.error(f"Model generation error: {e}")
                break

            # Parse think and action
            import re
            think_match = re.search(r'Think:\s*(.+?)(?=Action:|$)', action_text, re.IGNORECASE | re.DOTALL)
            think = think_match.group(1).strip() if think_match else ""

            action = self._parse_action(action_text)
            logger.info(f"[Step {step + 1}] Action: '{action}'")

            # Execute action
            obs, reward, done, info = self.env_wrapper.step(action)

            # Extract new clickables
            clickables = info.get("clickables", [])
            has_search_bar = info.get("has_search_bar", False)

            logger.info(f"[Step {step + 1}] done={done}, reward={reward}")

            # Record trajectory
            self.trajectory.append({
                "step": step + 1,
                "think": think,
                "action": action,
                "observation": obs,
                "done": done,
                "reward": reward
            })

            # Record to memory
            action_step = ActionStep(step_number=step)
            action_step.action_think = think
            action_step.observations = obs
            self.memory.steps.append(action_step)

            # Check if done
            if done:
                self.task_completed = True
                self.task_success = (reward > 0)
                logger.info(f"✓ TASK DONE - Success: {self.task_success}")
                break

            current_obs = obs
            step += 1

        result = {
            "agent_result": "success" if self.task_success else "failed",
            "agent_trajectory": self.trajectory,
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "steps_taken": step + 1
        }

        return result

    def is_task_done(self) -> bool:
        return self.task_completed

    def is_task_successful(self) -> bool:
        return self.task_success

    def write_memory_to_messages(self, include_system_prompt: bool = True):
        """Extract memory steps as messages for MemEvolve integration."""
        messages = []
        if include_system_prompt:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

        for traj_item in self.trajectory:
            if traj_item.get('think'):
                messages.append({"role": "assistant", "content": f"Think: {traj_item['think']}"})
            messages.append({"role": "assistant", "content": f"Action: {traj_item['action']}"})
            messages.append({"role": "user", "content": f"Observation: {traj_item['observation']}"})

        return messages

    def capture_trajectory(self):
        """Capture trajectory in MemEvolve format."""
        trajectory = []
        for step_data in self.trajectory:
            traj = {
                "name": "action",
                "action": step_data.get("action", ""),
                "think": step_data.get("think", ""),
                "observation": step_data.get("observation", ""),
                "done": step_data.get("done", False),
                "reward": step_data.get("reward", 0)
            }
            trajectory.append(traj)

        return {"agent_trajectory": trajectory}
