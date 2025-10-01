"""
Simple Treasure Hunt Example using AriGraph and Ariadne Agent

This example demonstrates the AriGraph memory system in action with a simple
text-based treasure hunt game environment.
"""

import asyncio
import random
from typing import Dict, List, Tuple

from ariadne_agent import AriadneAgent


class TreasureHuntEnvironment:
    """
    Simple treasure hunt environment for testing AriGraph
    """

    def __init__(self):
        # Define rooms and their connections
        self.rooms = {
            "entrance": {
                "description": "You are in the entrance hall. There is a dusty table here.",
                "exits": {"north": "library", "east": "kitchen"},
                "items": ["note"],
                "examined": []
            },
            "library": {
                "description": "You are in a library filled with old books. There is a red locker here.",
                "exits": {"south": "entrance", "east": "study"},
                "items": ["red_key"],
                "examined": []
            },
            "kitchen": {
                "description": "You are in a kitchen. There is a blue locker here and a stove.",
                "exits": {"west": "entrance", "north": "study"},
                "items": ["blue_key"],
                "examined": []
            },
            "study": {
                "description": "You are in a study room. There is a golden locker here that gleams mysteriously.",
                "exits": {"west": "library", "south": "kitchen"},
                "items": ["treasure"],
                "examined": []
            }
        }

        # Define keys and what they unlock
        self.key_locks = {
            "note": "red_key",
            "red_key": "blue_key",
            "blue_key": "treasure"
        }

        # Define clues
        self.clues = {
            "note": "The first key is hidden in the room filled with knowledge.",
            "red_key": "The blue key waits where meals are prepared.",
            "blue_key": "The treasure lies where scholars contemplate."
        }

        self.current_location = "entrance"
        self.inventory = []
        self.game_completed = False
        self.steps = 0

    def get_observation(self) -> str:
        """Get current observation string"""
        room = self.rooms[self.current_location]
        obs = room["description"]

        # Add available items
        visible_items = [item for item in room["items"] if item not in self.inventory]
        if visible_items:
            obs += f" You see: {', '.join(visible_items)}."

        # Add exits
        exits = list(room["exits"].keys())
        obs += f" Exits are to the: {', '.join(exits)}."

        # Add inventory
        if self.inventory:
            obs += f" Inventory: {', '.join(self.inventory)}."
        else:
            obs += " Inventory: empty."

        return obs

    def get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        actions = []
        room = self.rooms[self.current_location]

        # Movement actions
        for direction in room["exits"]:
            actions.append(f"go {direction}")

        # Item actions
        visible_items = [item for item in room["items"] if item not in self.inventory]
        for item in visible_items:
            actions.append(f"take {item}")
            actions.append(f"examine {item}")

        # Inventory actions
        for item in self.inventory:
            actions.append(f"use {item}")
            actions.append(f"examine {item}")

        # General actions
        actions.extend(["look around", "check inventory"])

        return actions

    def execute_action(self, action: str) -> Tuple[str, bool]:
        """
        Execute action and return (result_description, game_over)
        """
        self.steps += 1
        action = action.lower().strip()

        # Movement
        if action.startswith("go "):
            direction = action.split(" ", 1)[1]
            return self._move(direction)

        # Taking items
        elif action.startswith("take "):
            item = action.split(" ", 1)[1]
            return self._take_item(item)

        # Examining items
        elif action.startswith("examine "):
            item = action.split(" ", 1)[1]
            return self._examine_item(item)

        # Using items
        elif action.startswith("use "):
            item = action.split(" ", 1)[1]
            return self._use_item(item)

        # General actions
        elif action == "look around":
            return self.get_observation(), False

        elif action == "check inventory":
            if self.inventory:
                return f"You have: {', '.join(self.inventory)}", False
            else:
                return "Your inventory is empty.", False

        else:
            return "I don't understand that command.", False

    def _move(self, direction: str) -> Tuple[str, bool]:
        """Move in specified direction"""
        room = self.rooms[self.current_location]

        if direction in room["exits"]:
            self.current_location = room["exits"][direction]
            return f"You go {direction}. {self.get_observation()}", False
        else:
            return f"You can't go {direction} from here.", False

    def _take_item(self, item: str) -> Tuple[str, bool]:
        """Take an item"""
        room = self.rooms[self.current_location]

        if item in room["items"] and item not in self.inventory:
            self.inventory.append(item)
            room["items"].remove(item)

            # Check for treasure
            if item == "treasure":
                self.game_completed = True
                return "Congratulations! You found the treasure and won the game!", True

            return f"You take the {item}.", False
        else:
            return f"There is no {item} here to take.", False

    def _examine_item(self, item: str) -> Tuple[str, bool]:
        """Examine an item"""
        room = self.rooms[self.current_location]

        # Check if item is in current room or inventory
        if item in room["items"] or item in self.inventory:
            if item in self.clues:
                clue = self.clues[item]
                room["examined"].append(item)
                return f"You examine the {item}. It says: '{clue}'", False
            else:
                return f"You examine the {item}. Nothing special about it.", False
        else:
            return f"There is no {item} here to examine.", False

    def _use_item(self, item: str) -> Tuple[str, bool]:
        """Use an item"""
        if item not in self.inventory:
            return f"You don't have a {item}.", False

        # Check if this item unlocks something in current room
        room = self.rooms[self.current_location]

        if item in self.key_locks:
            target_item = self.key_locks[item]
            if target_item in room["items"]:
                # Success! Remove the key and reveal the target
                self.inventory.remove(item)
                return f"You use the {item} and unlock access to the {target_item}!", False
            else:
                return f"The {item} doesn't seem to work here.", False
        else:
            return f"You can't use the {item} here.", False

    def is_completed(self) -> bool:
        """Check if game is completed"""
        return self.game_completed

    def get_stats(self) -> Dict:
        """Get game statistics"""
        return {
            "steps": self.steps,
            "location": self.current_location,
            "inventory_size": len(self.inventory),
            "completed": self.game_completed
        }


async def run_treasure_hunt_demo(max_steps: int = 50):
    """
    Run the treasure hunt demo with Ariadne agent
    """
    print("🏴‍☠️ Starting Treasure Hunt with AriGraph Memory System 🏴‍☠️")
    print("="*60)

    # Initialize environment and agent
    env = TreasureHuntEnvironment()
    agent = AriadneAgent()

    # Set agent goal
    await agent.set_goal("Find the hidden treasure by following the clues")

    print(f"Goal: Find the hidden treasure")
    print(f"Starting location: {env.current_location}")
    print("-"*60)

    # Main game loop
    step = 0
    while step < max_steps and not env.is_completed():
        step += 1

        # Get current state
        observation = env.get_observation()
        available_actions = env.get_available_actions()

        print(f"\n📍 Step {step}")
        print(f"Observation: {observation}")

        # Agent processes observation and selects action
        action, reasoning = await agent.step(observation, available_actions)

        print(f"🧠 Agent reasoning: {reasoning}")
        print(f"🎬 Selected action: {action}")

        # Execute action in environment
        result, game_over = env.execute_action(action)
        print(f"📄 Result: {result}")

        # Print agent's internal state
        if step % 10 == 0 or game_over:  # Every 10 steps or at end
            agent.print_current_state()

        if game_over:
            break

        # Small delay for readability
        await asyncio.sleep(0.1)

    # Final results
    print("\n" + "="*60)
    print("🏁 GAME FINISHED 🏁")
    print(f"Completed: {'Yes' if env.is_completed() else 'No'}")
    print(f"Steps taken: {step}")

    final_stats = agent.get_memory_stats()
    print(f"📊 Final Memory Stats:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    # Display semantic knowledge learned
    print(f"\n🧠 Semantic Knowledge Acquired:")
    for edge_id, edge in agent.storage.semantic_edges.items():
        subject = agent.storage.semantic_vertices[edge.subject_id].name
        obj = agent.storage.semantic_vertices[edge.object_id].name
        print(f"  • {subject} → {edge.relation} → {obj}")

    return env.is_completed(), step


async def main():
    """Main function"""
    print("AriGraph Implementation Demo")
    print("Based on: 'AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents'")
    print()

    # Run the treasure hunt demo
    success, steps = await run_treasure_hunt_demo()

    print(f"\n🎯 Demo Results:")
    print(f"  Success: {'✅' if success else '❌'}")
    print(f"  Steps: {steps}")

    if success:
        print("🎉 The agent successfully found the treasure using AriGraph memory!")
    else:
        print("🤔 The agent didn't complete the treasure hunt. Try tweaking the implementation!")


if __name__ == "__main__":
    asyncio.run(main())