# AriGraph Implementation with Advanced Memory Management API

This repository contains a complete implementation of the AriGraph system from the paper "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents" using the Advanced Memory Management API.

## 🏛️ Architecture Overview

The implementation follows the original AriGraph paper architecture with these key components:

### 1. **Memory Graph Structure** (`arigraph_implementation.py`)
- **Semantic Memory**: Knowledge graph with vertices (concepts/objects) and edges (relationships)
- **Episodic Memory**: Specific observations/experiences linked to extracted semantic triplets
- **Dynamic Updates**: Outdated knowledge detection and replacement
- **Triplet Extraction**: Pattern-based extraction of (subject, relation, object) triplets

### 2. **Retrieval System** (`arigraph_retrieval.py`)
- **Semantic Search**: Graph traversal with similarity scoring and depth/width controls
- **Episodic Search**: Relevance scoring based on associated semantic triplets
- **Integrated Retrieval**: Combines both semantic and episodic results

### 3. **Cognitive Architecture** (`ariadne_agent.py`)
- **Planning Module**: Goal-based hierarchical planning with sub-goals
- **Decision Making**: ReAct-style reasoning with working memory
- **Memory Integration**: Automatic storage and retrieval during interaction

### 4. **Examples**
- **Treasure Hunt** (`treasure_hunt_example.py`): Complex multi-step puzzle solving
- **Simple Cleaning** (`simple_cleaning_example.py`): Item placement and spatial reasoning

## 🎯 Key Features Implemented

### ✅ Core AriGraph Features
- [x] **Semantic Knowledge Graph**: Vertices and edges representing world knowledge
- [x] **Episodic Vertices**: Storing specific observations and experiences
- [x] **Episodic Edges**: Linking observations to extracted semantic triplets
- [x] **Triplet Extraction**: Converting text observations into structured knowledge
- [x] **Outdated Knowledge Detection**: Identifying and replacing conflicting information
- [x] **Graph-based Retrieval**: Semantic search with graph traversal
- [x] **Episodic Retrieval**: Finding relevant past experiences
- [x] **Memory Growth Tracking**: Monitoring semantic graph expansion

### ✅ Ariadne Agent Features
- [x] **Hierarchical Planning**: Multi-level goal decomposition
- [x] **Working Memory**: Current context and retrieved memories
- [x] **ReAct Decision Making**: Reasoning before action selection
- [x] **Exploration**: Discovering new areas and updating spatial knowledge
- [x] **Memory-Guided Actions**: Using retrieved memories for decision making

### ✅ Integration with Memory API
- [x] **Storage Backend**: AriGraph as MemoryStorage implementation
- [x] **Retrieval Backend**: AriGraph search as MemoryRetrieval implementation
- [x] **Memory Types**: Proper semantic/episodic memory categorization
- [x] **Async Operations**: Non-blocking memory operations
- [x] **Statistics**: Memory system monitoring and debugging

## 🧠 How AriGraph Works

### Memory Construction
1. **Observation Processing**: Text observations are processed through triplet extraction
2. **Semantic Update**: New triplets update the semantic knowledge graph
3. **Conflict Resolution**: Outdated triplets are detected and removed
4. **Episodic Linking**: Observations are stored with links to their extracted triplets

### Retrieval Process
1. **Semantic Search**: Query matches are found using graph traversal and similarity
2. **Episodic Search**: Past experiences are ranked by relevance to semantic results
3. **Integrated Results**: Both types of memories are returned for decision making

### Agent Cognition
1. **Memory Update**: Each observation updates both semantic and episodic memory
2. **Planning**: Goals are decomposed into sub-goals using current knowledge
3. **Retrieval**: Relevant memories are retrieved based on current situation
4. **Decision**: Actions are selected using ReAct-style reasoning

## 📊 Demonstration Results

The implementation successfully demonstrates:
- **Knowledge Graph Growth**: Dynamic expansion of semantic relationships
- **Memory Integration**: Seamless combination of semantic and episodic information
- **Spatial Reasoning**: Learning room connections and object locations
- **Pattern Recognition**: Extracting structured knowledge from text observations

### Example Knowledge Learned
```
• living room → has_exit → north
• living room → has_exit → east
• kitchen → has_exit → south
• player → location → living room
• book → is_out_of_place → true
• toothbrush → belongs_in → bathroom
```

## 🚀 Usage

### Basic AriGraph Usage
```python
from arigraph_implementation import AriGraphStorage
from arigraph_retrieval import AriGraphRetrieval
from draft import AdvancedMemoryManager, Memory, MemoryType

# Initialize AriGraph system
storage = AriGraphStorage()
retrieval = AriGraphRetrieval(storage)
memory_manager = AdvancedMemoryManager(storage=storage, retrieval=retrieval)

# Store observations
await memory_manager.add_memory(
    content="You are in the kitchen. There is an apple on the table.",
    memory_type=MemoryType.EPISODIC
)

# Retrieve relevant memories
memories = await retrieval.search(RetrievalQuery(query="kitchen items"))
```

### Ariadne Agent Usage
```python
from ariadne_agent import AriadneAgent

# Initialize agent
agent = AriadneAgent()
await agent.set_goal("Find the hidden treasure")

# Process environment step
action, reasoning = await agent.step(observation, available_actions)
```

### Run Examples
```bash
# Treasure hunt example
python3 treasure_hunt_example.py

# Simple cleaning example
python3 simple_cleaning_example.py
```

## 🔬 Technical Implementation Details

### Semantic Graph Structure
- **Vertices**: Represent entities/concepts with unique IDs and names
- **Edges**: Represent relationships with subject/relation/object triplets
- **Adjacency Lists**: Efficient graph traversal for retrieval
- **Name Resolution**: Mapping between entity names and vertex IDs

### Episodic Memory Structure
- **Episodic Vertices**: Store complete observations with timestamps
- **Episodic Edges**: Link observations to sets of semantic triplets
- **Temporal Indexing**: Step-based ordering of experiences
- **Metadata Support**: Additional context information

### Retrieval Algorithms
- **Semantic Search**: BFS graph traversal with similarity scoring
- **Episodic Ranking**: Information-theoretic relevance scoring
- **Depth/Width Controls**: Configurable search parameters
- **Score Combination**: Weighted integration of different similarity measures

## 🎓 Research Paper Implementation

This implementation faithfully follows the AriGraph paper:

**Paper**: "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents"
**Authors**: Petr Anokhin, Nikita Semenov, et al.
**Key Concepts Implemented**:
- Semantic + Episodic memory integration
- Knowledge graph world model construction
- Graph-based retrieval with episodic ranking
- Cognitive architecture for planning and decision-making

### Differences from Paper
- **Triplet Extraction**: Uses rule-based patterns instead of LLM (can be easily swapped)
- **Planning**: Simplified rule-based planning (production system would use LLM)
- **Environment**: Custom text-based games (paper used TextWorld)
- **Evaluation**: Qualitative demonstration vs. quantitative benchmarks

## 🔧 Customization

The system is designed to be modular and extensible:

### Custom Triplet Extraction
Replace `TripletExtractor` in `arigraph_implementation.py` with your own LLM-based extraction.

### Custom Planning
Replace `AriadneLLMPlanner` in `ariadne_agent.py` with your own planning algorithm.

### Custom Environments
Create new environments following the pattern in the examples.

### Custom Retrieval
Extend `AriGraphRetrieval` to implement domain-specific search strategies.

## 📈 Future Enhancements

- **LLM Integration**: Replace rule-based components with actual LLM calls
- **Multi-modal Support**: Extend to handle images and other modalities
- **Procedural Memory**: Add learned skills and procedures
- **Advanced Graph Operations**: More sophisticated graph algorithms
- **Performance Optimization**: Scalability improvements for larger knowledge graphs
- **Evaluation Framework**: Quantitative benchmarking against paper results

## 🤝 Contributing

This implementation demonstrates the core concepts from the AriGraph paper integrated with the Advanced Memory Management API. It provides a solid foundation for further research and development in memory-augmented LLM agents.