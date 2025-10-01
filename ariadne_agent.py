"""
Ariadne Cognitive Architecture - The main agent implementing planning and decision-making
with AriGraph memory system
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from draft import Memory, MemoryType, RetrievalQuery, AdvancedMemoryManager
from arigraph_implementation import AriGraphStorage
from arigraph_retrieval import AriGraphRetrieval


class ActionType(Enum):
    MOVE = "move"
    EXAMINE = "examine"
    TAKE = "take"
    USE = "use"
    OPEN = "open"
    CLOSE = "close"
    EXPLORE = "explore"


@dataclass
class Plan:
    """Represents the agent's current plan with hierarchical sub-goals"""
    main_goal: str
    sub_goals: List[Dict[str, str]] = field(default_factory=list)
    current_step: int = 0
    completed_goals: List[str] = field(default_factory=list)


@dataclass
class WorkingMemory:
    """Working memory for current planning and decision-making"""
    current_observation: str = ""
    current_location: str = ""
    inventory: List[str] = field(default_factory=list)
    plan: Optional[Plan] = None
    recent_actions: List[str] = field(default_factory=list)
    retrieved_memories: List[Memory] = field(default_factory=list)
    available_actions: List[str] = field(default_factory=list)


class AriadneLLMPlanner:
    """Simplified LLM-based planning module (using rule-based logic for demo)"""

    def __init__(self):
        pass

    def create_plan(self, goal: str, working_memory: WorkingMemory) -> Plan:
        """Create a plan to achieve the given goal"""
        plan = Plan(main_goal=goal)

        # Simple rule-based planning for treasure hunt
        if "treasure" in goal.lower():
            if "key" not in working_memory.inventory:
                plan.sub_goals = [
                    {"sub_goal": "explore_rooms", "reason": "Need to find the first key"},
                    {"sub_goal": "find_key", "reason": "Key is needed to unlock lockers"},
                    {"sub_goal": "follow_clues", "reason": "Keys provide clues to next location"},
                    {"sub_goal": "find_treasure", "reason": "Ultimate goal is the treasure"}
                ]
            else:
                plan.sub_goals = [
                    {"sub_goal": "follow_clues", "reason": "Use current key to find next clue"},
                    {"sub_goal": "find_treasure", "reason": "Ultimate goal is the treasure"}
                ]

        elif "clean" in goal.lower():
            plan.sub_goals = [
                {"sub_goal": "explore_house", "reason": "Need to map out all rooms"},
                {"sub_goal": "identify_misplaced_items", "reason": "Find what needs cleaning"},
                {"sub_goal": "return_items", "reason": "Put items in correct locations"}
            ]

        return plan

    def update_plan(self, current_plan: Plan, working_memory: WorkingMemory) -> Plan:
        """Update existing plan based on new information"""
        # Check if current sub-goal is completed
        if current_plan.sub_goals and current_plan.current_step < len(current_plan.sub_goals):
            current_goal = current_plan.sub_goals[current_plan.current_step]

            # Simple completion detection
            goal_text = current_goal["sub_goal"]
            if goal_text == "find_key" and any("key" in item for item in working_memory.inventory):
                current_plan.completed_goals.append(goal_text)
                current_plan.current_step += 1

            elif goal_text == "explore_rooms" and len(working_memory.retrieved_memories) > 3:
                current_plan.completed_goals.append(goal_text)
                current_plan.current_step += 1

        return current_plan


class AriadneDecisionMaker:
    """Decision-making module using ReAct-style reasoning"""

    def __init__(self, retrieval: AriGraphRetrieval):
        self.retrieval = retrieval

    def select_action(self, working_memory: WorkingMemory) -> Tuple[str, str]:
        """
        Select the best action based on current state and plan
        Returns (action, reasoning)
        """
        if not working_memory.available_actions:
            return "wait", "No actions available"

        current_goal = self._get_current_goal(working_memory)
        reasoning = f"Working towards: {current_goal}"

        # Priority 1: Follow current plan
        action = self._plan_based_action(working_memory, current_goal)
        if action:
            return action, f"{reasoning}. Plan suggests: {action}"

        # Priority 2: Exploration if no clear plan action
        action = self._exploration_action(working_memory)
        if action:
            return action, f"{reasoning}. Exploring: {action}"

        # Priority 3: Default to first available action
        return working_memory.available_actions[0], f"{reasoning}. Default action"

    def _get_current_goal(self, working_memory: WorkingMemory) -> str:
        """Get the current sub-goal from the plan"""
        if not working_memory.plan or not working_memory.plan.sub_goals:
            return "explore"

        current_step = working_memory.plan.current_step
        if current_step < len(working_memory.plan.sub_goals):
            return working_memory.plan.sub_goals[current_step]["sub_goal"]

        return "completed"

    def _plan_based_action(self, working_memory: WorkingMemory, goal: str) -> Optional[str]:
        """Select action based on current goal"""
        actions = working_memory.available_actions

        if goal == "explore_rooms" or goal == "explore_house":
            # Look for movement actions to unexplored areas
            movement_actions = [a for a in actions if any(direction in a.lower()
                              for direction in ["north", "south", "east", "west", "go"])]
            if movement_actions:
                return movement_actions[0]

        elif goal == "find_key":
            # Look for examine or take actions
            if any("examine" in a.lower() for a in actions):
                examine_actions = [a for a in actions if "examine" in a.lower()]
                return examine_actions[0]
            elif any("take" in a.lower() for a in actions):
                take_actions = [a for a in actions if "take" in a.lower()]
                return take_actions[0]

        elif goal == "follow_clues":
            # If we have a key, try to use it
            if any("key" in item for item in working_memory.inventory):
                use_actions = [a for a in actions if "open" in a.lower() or "unlock" in a.lower()]
                if use_actions:
                    return use_actions[0]

        return None

    def _exploration_action(self, working_memory: WorkingMemory) -> Optional[str]:
        """Select exploration action when no clear plan direction"""
        actions = working_memory.available_actions

        # Prefer movement to unknown areas
        unexplored_exits = self.retrieval.get_exploration_info(working_memory.current_location)

        for exit_direction in unexplored_exits:
            matching_actions = [a for a in actions if exit_direction.lower() in a.lower()]
            if matching_actions:
                return matching_actions[0]

        # Fall back to any movement action
        movement_actions = [a for a in actions if any(direction in a.lower()
                          for direction in ["north", "south", "east", "west", "go"])]
        if movement_actions:
            return movement_actions[0]

        return None


class AriadneAgent:
    """
    Main Ariadne agent combining AriGraph memory with planning and decision-making
    """

    def __init__(self):
        # Initialize memory components
        self.storage = AriGraphStorage()
        self.retrieval = AriGraphRetrieval(self.storage)
        self.memory_manager = AdvancedMemoryManager(
            storage=self.storage,
            retrieval=self.retrieval
        )

        # Initialize cognitive components
        self.planner = AriadneLLMPlanner()
        self.decision_maker = AriadneDecisionMaker(self.retrieval)

        # Working memory
        self.working_memory = WorkingMemory()

        self.step_count = 0

    async def set_goal(self, goal: str):
        """Set the main goal for the agent"""
        self.working_memory.plan = self.planner.create_plan(goal, self.working_memory)

    async def step(self, observation: str, available_actions: List[str]) -> Tuple[str, str]:
        """
        Process one step: observation -> memory update -> planning -> decision
        Returns (selected_action, reasoning)
        """
        self.step_count += 1

        # Update working memory with new observation
        self.working_memory.current_observation = observation
        self.working_memory.available_actions = available_actions

        # Extract location and inventory from observation
        self._update_location_and_inventory(observation)

        # Store observation in AriGraph memory
        await self._update_memory(observation)

        # Retrieve relevant memories for current situation
        await self._retrieve_relevant_memories()

        # Update plan based on new information
        if self.working_memory.plan:
            self.working_memory.plan = self.planner.update_plan(
                self.working_memory.plan,
                self.working_memory
            )

        # Select action based on current state and plan
        action, reasoning = self.decision_maker.select_action(self.working_memory)

        # Update recent actions
        self.working_memory.recent_actions.append(action)
        if len(self.working_memory.recent_actions) > 5:
            self.working_memory.recent_actions.pop(0)

        return action, reasoning

    def _update_location_and_inventory(self, observation: str):
        """Extract location and inventory info from observation"""
        # Simple pattern matching for location
        if "you are in" in observation.lower():
            parts = observation.lower().split("you are in")
            if len(parts) > 1:
                location_part = parts[1].split(".")[0].split(",")[0]
                self.working_memory.current_location = location_part.strip()

        # Simple pattern matching for inventory
        if "inventory:" in observation.lower():
            parts = observation.lower().split("inventory:")
            if len(parts) > 1:
                inventory_text = parts[1]
                # Extract items (simplified)
                items = [item.strip() for item in inventory_text.replace("\n", ",").split(",")
                        if item.strip() and item.strip() not in ["", "empty", "nothing"]]
                self.working_memory.inventory = items

    async def _update_memory(self, observation: str):
        """Store observation in AriGraph memory"""
        context = {
            "step": self.step_count,
            "current_location": self.working_memory.current_location,
            "inventory": self.working_memory.inventory
        }

        await self.memory_manager.add_memory(
            content=observation,
            memory_type=MemoryType.EPISODIC,
            metadata=context
        )

    async def _retrieve_relevant_memories(self):
        """Retrieve memories relevant to current situation and goal"""
        # Create query based on current goal and location
        query_parts = []

        if self.working_memory.plan and self.working_memory.plan.sub_goals:
            current_goal = self.working_memory.plan.sub_goals[
                self.working_memory.plan.current_step
            ] if self.working_memory.plan.current_step < len(self.working_memory.plan.sub_goals) else None

            if current_goal:
                query_parts.append(current_goal["sub_goal"])

        if self.working_memory.current_location:
            query_parts.append(self.working_memory.current_location)

        query_text = " ".join(query_parts) if query_parts else "recent observations"

        # Retrieve relevant memories
        query = RetrievalQuery(
            query=query_text,
            limit=5,
            similarity_threshold=0.3
        )

        self.working_memory.retrieved_memories = await self.retrieval.search(query)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory state"""
        stats = self.storage.get_semantic_graph_stats()
        stats["step_count"] = self.step_count
        stats["current_location"] = self.working_memory.current_location
        stats["inventory_size"] = len(self.working_memory.inventory)

        if self.working_memory.plan:
            stats["plan_progress"] = f"{self.working_memory.plan.current_step}/{len(self.working_memory.plan.sub_goals)}"

        return stats

    def print_current_state(self):
        """Print current agent state for debugging"""
        print("\n" + "="*50)
        print(f"Step {self.step_count}")
        print(f"Location: {self.working_memory.current_location}")
        print(f"Inventory: {self.working_memory.inventory}")

        if self.working_memory.plan:
            current_step = self.working_memory.plan.current_step
            if current_step < len(self.working_memory.plan.sub_goals):
                current_goal = self.working_memory.plan.sub_goals[current_step]
                print(f"Current Goal: {current_goal['sub_goal']} - {current_goal['reason']}")

        print(f"Memory Stats: {self.get_memory_stats()}")
        print("="*50)