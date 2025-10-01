"""
Simple Cleaning Example using AriGraph - Shows the memory system working properly

This demonstrates the semantic and episodic memory integration with a simpler task.
"""

import asyncio
from typing import Dict, List, Tuple

from ariadne_agent import AriadneAgent


class SimpleCleaningEnvironment:
    """
    Simple cleaning environment to demonstrate AriGraph memory
    """

    def __init__(self):
        self.rooms = {
            "living_room": {
                "description": "You are in the living room. There's a comfortable sofa here.",
                "items": ["book", "tv_remote"],  # tv_remote belongs in living room, book doesn't
                "correct_items": ["tv_remote"],
                "exits": {"north": "kitchen", "east": "bedroom"}
            },
            "kitchen": {
                "description": "You are in the kitchen. There's a refrigerator and stove here.",
                "items": ["apple", "toothbrush"],  # apple belongs in kitchen, toothbrush doesn't
                "correct_items": ["apple"],
                "exits": {"south": "living_room", "east": "bathroom"}
            },
            "bedroom": {
                "description": "You are in the bedroom. There's a comfortable bed here.",
                "items": ["pillow", "coffee_mug"],  # pillow belongs in bedroom, coffee_mug doesn't
                "correct_items": ["pillow"],
                "exits": {"west": "living_room", "north": "bathroom"}
            },
            "bathroom": {
                "description": "You are in the bathroom. There's a sink and mirror here.",
                "items": [],
                "correct_items": ["toothbrush"],
                "exits": {"west": "kitchen", "south": "bedroom"}
            }
        }

        self.current_location = "living_room"
        self.inventory = []
        self.cleaned_items = []
        self.steps = 0

    def get_observation(self) -> str:
        """Get current observation string"""
        room = self.rooms[self.current_location]
        obs = room["description"]

        # Add items in room
        visible_items = [item for item in room["items"] if item not in self.inventory]
        if visible_items:
            misplaced = [item for item in visible_items if item not in room["correct_items"]]
            if misplaced:
                obs += f" You notice these items seem out of place: {', '.join(misplaced)}."
            correct = [item for item in visible_items if item in room["correct_items"]]
            if correct:
                obs += f" These items belong here: {', '.join(correct)}."

        # Add exits
        exits = list(room["exits"].keys())
        obs += f" Exits: {', '.join(exits)}."

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

        # Drop actions
        for item in self.inventory:
            actions.append(f"drop {item}")

        actions.append("look around")

        return actions

    def execute_action(self, action: str) -> Tuple[str, bool]:
        """Execute action and return (result_description, game_over)"""
        self.steps += 1
        action = action.lower().strip()

        if action.startswith("go "):
            direction = action.split(" ", 1)[1]
            return self._move(direction)

        elif action.startswith("take "):
            item = action.split(" ", 1)[1]
            return self._take_item(item)

        elif action.startswith("drop "):
            item = action.split(" ", 1)[1]
            return self._drop_item(item)

        elif action == "look around":
            return self.get_observation(), False

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
            return f"You take the {item}.", False
        else:
            return f"There is no {item} here to take.", False

    def _drop_item(self, item: str) -> Tuple[str, bool]:
        """Drop an item"""
        if item not in self.inventory:
            return f"You don't have a {item}.", False

        room = self.rooms[self.current_location]
        self.inventory.remove(item)
        room["items"].append(item)

        # Check if item belongs in this room
        if item in room["correct_items"]:
            if item not in self.cleaned_items:
                self.cleaned_items.append(item)
                result = f"You drop the {item}. Great! The {item} belongs here."

                # Check if game is complete
                total_misplaced = 3  # book, toothbrush, coffee_mug
                if len(self.cleaned_items) >= total_misplaced:
                    return result + " Congratulations! You've cleaned up all misplaced items!", True

                return result, False
            else:
                return f"You drop the {item}. It already belongs here.", False
        else:
            return f"You drop the {item}. This item doesn't belong in this room.", False

    def is_completed(self) -> bool:
        """Check if cleaning is completed"""
        return len(self.cleaned_items) >= 3

    def get_stats(self) -> Dict:
        """Get game statistics"""
        return {
            "steps": self.steps,
            "location": self.current_location,
            "inventory_size": len(self.inventory),
            "cleaned_items": len(self.cleaned_items),
            "completed": self.is_completed()
        }


async def run_cleaning_demo(max_steps: int = 30):
    """Run the cleaning demo with Ariadne agent"""
    print("🧹 Starting Simple Cleaning with AriGraph Memory System 🧹")
    print("="*60)

    # Initialize environment and agent
    env = SimpleCleaningEnvironment()
    agent = AriadneAgent()

    # Set agent goal
    await agent.set_goal("Clean the house by putting misplaced items in their correct rooms")

    print(f"Goal: Clean the house by returning misplaced items to correct locations")
    print(f"Misplaced items: book (should be in bedroom), toothbrush (should be in bathroom), coffee_mug (should be in kitchen)")
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

        if game_over:
            break

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"   Progress: {len(env.cleaned_items)}/3 items cleaned")

        # Small delay for readability
        await asyncio.sleep(0.1)

    # Final results
    print("\n" + "="*60)
    print("🏁 CLEANING FINISHED 🏁")
    print(f"Completed: {'Yes' if env.is_completed() else 'No'}")
    print(f"Items cleaned: {len(env.cleaned_items)}/3")
    print(f"Steps taken: {step}")

    final_stats = agent.get_memory_stats()
    print(f"\n📊 Final Memory Stats:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    # Display some semantic knowledge learned
    print(f"\n🧠 Sample Semantic Knowledge Acquired:")
    knowledge_count = 0
    for edge_id, edge in agent.storage.semantic_edges.items():
        if knowledge_count >= 10:  # Limit output
            break
        subject = agent.storage.semantic_vertices[edge.subject_id].name
        obj = agent.storage.semantic_vertices[edge.object_id].name
        print(f"  • {subject} → {edge.relation} → {obj}")
        knowledge_count += 1

    if len(agent.storage.semantic_edges) > 10:
        print(f"  ... and {len(agent.storage.semantic_edges) - 10} more relationships")

    return env.is_completed(), step


async def main():
    """Main function"""
    print("AriGraph Implementation Demo - Simple Cleaning")
    print("Based on: 'AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents'")
    print()

    # Run the cleaning demo
    success, steps = await run_cleaning_demo()

    print(f"\n🎯 Demo Results:")
    print(f"  Success: {'✅' if success else '❌'}")
    print(f"  Steps: {steps}")

    if success:
        print("🎉 The agent successfully cleaned the house using AriGraph memory!")
    else:
        print("🧹 The agent made progress but didn't complete all cleaning. The AriGraph system is learning and building knowledge!")


if __name__ == "__main__":
    asyncio.run(main())