from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core Memory Types
class MemoryType(Enum):
    EPISODIC = "episodic"      # Specific events/interactions
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"        # Short-term context

@dataclass
class Memory:
    """Core memory unit"""
    id: str
    content: Union[str, Dict[str, Any]]
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    decay_rate: float = 0.1  # For forgetting mechanisms

@dataclass
class RetrievalQuery:
    """Query object for memory retrieval"""
    query: Union[str, Dict[str, Any]]
    memory_types: Optional[List[MemoryType]] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    time_range: Optional[tuple] = None
    filters: Dict[str, Any] = field(default_factory=dict)

# 1. MEMORY ACQUISITION
class MemoryAcquisition(ABC):
    """Handles how memories are captured and preprocessed"""
    
    @abstractmethod
    def extract_memory(self, input_data: Any, context: Dict[str, Any] = None) -> List[Memory]:
        """Extract memories from input data"""
        pass
    
    @abstractmethod
    def should_remember(self, memory: Memory, context: Dict[str, Any] = None) -> bool:
        """Determine if something should be stored"""
        pass

class DefaultAcquisition(MemoryAcquisition):
    """Default implementation with configurable extractors"""
    
    def __init__(self, extractors: List[Callable] = None):
        self.extractors = extractors or []
    
    def extract_memory(self, input_data: Any, context: Dict[str, Any] = None) -> List[Memory]:
        memories = []
        for extractor in self.extractors:
            memories.extend(extractor(input_data, context or {}))
        return memories
    
    def should_remember(self, memory: Memory, context: Dict[str, Any] = None) -> bool:
        # Simple importance threshold
        return memory.importance > 0.5

# 2. MEMORY STORAGE
class MemoryStorage(ABC):
    """Abstract storage backend"""
    
    @abstractmethod
    async def store(self, memory: Memory) -> bool:
        pass
    
    @abstractmethod
    async def store_batch(self, memories: List[Memory]) -> bool:
        pass
    
    @abstractmethod
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        pass

# 3. MEMORY MAINTENANCE
class MemoryMaintenance(ABC):
    """Handles memory consolidation, forgetting, and cleanup"""
    
    @abstractmethod
    async def consolidate(self, memories: List[Memory]) -> List[Memory]:
        """Merge similar memories"""
        pass
    
    @abstractmethod
    async def forget(self, criteria: Dict[str, Any]) -> List[str]:
        """Remove memories based on criteria"""
        pass
    
    @abstractmethod
    async def update_importance(self, memory_id: str, importance: float) -> bool:
        """Update memory importance"""
        pass

class DefaultMaintenance(MemoryMaintenance):
    """Default maintenance with decay-based forgetting"""
    
    async def consolidate(self, memories: List[Memory]) -> List[Memory]:
        # Simple duplicate removal based on content similarity
        # In practice, this would use embeddings
        return list({m.content: m for m in memories}.values())
    
    async def forget(self, criteria: Dict[str, Any]) -> List[str]:
        # Implement decay-based or criteria-based forgetting
        return []
    
    async def update_importance(self, memory_id: str, importance: float) -> bool:
        # Update importance score
        return True

# 4. MEMORY RETRIEVAL
class MemoryRetrieval(ABC):
    """Handles memory search and retrieval"""
    
    @abstractmethod
    async def search(self, query: RetrievalQuery) -> List[Memory]:
        pass
    
    @abstractmethod
    async def get_recent(self, limit: int = 10, memory_type: MemoryType = None) -> List[Memory]:
        pass
    
    @abstractmethod
    async def get_important(self, limit: int = 10, threshold: float = 0.8) -> List[Memory]:
        pass

# MAIN MEMORY MANAGER
class AdvancedMemoryManager:
    """Main memory management interface"""
    
    def __init__(self, 
                 storage: MemoryStorage,
                 acquisition: MemoryAcquisition = None,
                 maintenance: MemoryMaintenance = None,
                 retrieval: MemoryRetrieval = None):
        self.storage = storage
        self.acquisition = acquisition or DefaultAcquisition()
        self.maintenance = maintenance or DefaultMaintenance()
        self.retrieval = retrieval
        
    # Core operations
    async def add_memory(self, content: Any, memory_type: MemoryType = MemoryType.EPISODIC, **kwargs) -> str:
        """Add a single memory"""
        memory = Memory(
            id=self._generate_id(),
            content=content,
            memory_type=memory_type,
            **kwargs
        )
        await self.storage.store(memory)
        return memory.id
    
    async def process_input(self, input_data: Any, context: Dict[str, Any] = None) -> List[str]:
        """Process input through acquisition pipeline"""
        memories = self.acquisition.extract_memory(input_data, context)
        stored_ids = []
        for memory in memories:
            if self.acquisition.should_remember(memory, context):
                await self.storage.store(memory)
                stored_ids.append(memory.id)
        return stored_ids
    
    async def retrieve(self, query: Union[str, RetrievalQuery]) -> List[Memory]:
        """Retrieve memories based on query"""
        if isinstance(query, str):
            query = RetrievalQuery(query=query)
        return await self.retrieval.search(query)
    
    async def get_context(self, limit: int = 10) -> List[Memory]:
        """Get recent context for agents"""
        return await self.retrieval.get_recent(limit=limit)
    
    # Maintenance operations
    async def maintain(self):
        """Run maintenance operations"""
        # This would be called periodically
        # Implementation depends on specific maintenance strategy
        pass
    
    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

# COMPATIBILITY LAYER FOR EXISTING FRAMEWORKS

class LangChainMemoryAdapter:
    """Adapter to make our system compatible with LangChain's memory interface"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager):
        self.memory_manager = memory_manager
        self.chat_memory = []  # LangChain expects this
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """LangChain BaseMemory interface"""
        import asyncio
        context = {"inputs": inputs, "outputs": outputs}
        asyncio.create_task(self.memory_manager.process_input(context))
    
    def clear(self) -> None:
        """Clear memory - implement based on your needs"""
        self.chat_memory.clear()
    
    @property
    def memory_variables(self) -> List[str]:
        """Variables this memory class will add to chain inputs"""
        return ["history", "context"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for the chain"""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in async context, you might need to handle this differently
            context = []
        else:
            context = loop.run_until_complete(self.memory_manager.get_context())
        
        return {
            "history": "\n".join([str(m.content) for m in context]),
            "context": context
        }

class CamelMemoryAdapter:
    """Adapter for CAMEL-AI framework"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager):
        self.memory_manager = memory_manager
    
    async def store_message(self, message: Any) -> None:
        """Store a message in CAMEL format"""
        await self.memory_manager.add_memory(
            content=message,
            memory_type=MemoryType.EPISODIC,
            tags=["message"]
        )
    
    async def get_chat_history(self, limit: int = 10) -> List[Any]:
        """Get chat history in CAMEL format"""
        memories = await self.memory_manager.get_context(limit)
        return [m.content for m in memories if "message" in m.tags]

# EXAMPLE USAGE
async def example_usage():
    # Setup with a storage backend (you'd implement specific ones)
    from memory_backends import VectorStorage, PostgresStorage
    
    storage = VectorStorage()  # Your vector DB implementation
    memory_manager = AdvancedMemoryManager(storage)
    
    # Direct usage
    memory_id = await memory_manager.add_memory("User prefers Python over JavaScript")
    context = await memory_manager.get_context()
    
    # LangChain compatibility
    langchain_memory = LangChainMemoryAdapter(memory_manager)
    # Use with LangChain chains as normal
    
    # CAMEL compatibility  
    camel_memory = CamelMemoryAdapter(memory_manager)
    await camel_memory.store_message("Hello from CAMEL")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
