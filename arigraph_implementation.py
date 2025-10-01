"""
AriGraph Implementation using the Advanced Memory Management API

This implementation creates the AriGraph system described in the paper,
integrating semantic and episodic memories in a knowledge graph framework.
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import numpy as np

# Import our base memory management system
from draft import (
    Memory, MemoryType, RetrievalQuery, AdvancedMemoryManager,
    MemoryStorage, MemoryAcquisition, MemoryMaintenance, MemoryRetrieval
)

# === SEMANTIC GRAPH STRUCTURES ===

@dataclass
class SemanticVertex:
    """Represents a semantic concept/object in the knowledge graph"""
    id: str
    name: str
    entity_type: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SemanticEdge:
    """Represents a relationship between two semantic vertices"""
    id: str
    subject_id: str
    relation: str
    object_id: str
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_triplet(self) -> str:
        """Convert to triplet string format: 'subject, relation, object'"""
        return f"{self.subject_id}, {self.relation}, {self.object_id}"

@dataclass
class EpisodicVertex:
    """Represents an episodic memory - a specific observation/experience"""
    id: str
    observation: str
    step: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EpisodicEdge:
    """Links episodic vertices to semantic triplets that were extracted from them"""
    id: str
    episodic_vertex_id: str
    semantic_edge_ids: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

# === TRIPLET EXTRACTION ===

class TripletExtractor:
    """Extracts semantic triplets from textual observations"""

    def __init__(self, llm_backend=None):
        self.llm_backend = llm_backend

    def extract_triplets(self, observation: str, context: Dict[str, Any] = None) -> List[Tuple[str, str, str]]:
        """
        Extract triplets from observation text.
        Returns list of (subject, relation, object) tuples.

        For now, implements basic pattern matching. In production, this would use LLM.
        """
        triplets = []

        # Simple pattern-based extraction for demo
        # Pattern: "X contains Y", "X is in Y", "X has Y", etc.
        patterns = [
            r"(\w+(?:\s+\w+)*)\s+contains\s+(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+is\s+in\s+(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+has\s+(?:an?\s+)?(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+is\s+(?:to\s+the\s+)?(\w+)\s+of\s+(\w+(?:\s+\w+)*)",
            r"go\s+(\w+).*?to\s+(\w+(?:\s+\w+)*)",
            r"you\s+are\s+in\s+(?:the\s+)?(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+(?:leads?|goes?)\s+(\w+)",
        ]

        # Extract location/containment relationships
        if "you are in" in observation.lower():
            match = re.search(r"you are in (?:the )?(\w+(?:\s+\w+)*)", observation.lower())
            if match:
                location = match.group(1).strip()
                triplets.append(("player", "location", location))

        if "contains" in observation.lower():
            matches = re.findall(r"(\w+(?:\s+\w+)*)\s+contains?\s+(?:an?\s+)?(\w+(?:\s+\w+)*)", observation.lower())
            for subject, obj in matches:
                triplets.append((subject.strip(), "contains", obj.strip()))

        if "is on" in observation.lower():
            matches = re.findall(r"(\w+(?:\s+\w+)*)\s+is\s+on\s+(?:the\s+)?(\w+(?:\s+\w+)*)", observation.lower())
            for subject, obj in matches:
                triplets.append((subject.strip(), "is_on", obj.strip()))

        # Extract exits/directions
        direction_words = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
        for direction in direction_words:
            if direction in observation.lower():
                if context and "current_location" in context:
                    location = context["current_location"]
                    triplets.append((location, "has_exit", direction))

        return triplets

    def detect_outdated_triplets(self, existing_triplets: List[Tuple[str, str, str]],
                                new_triplets: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Detect which existing triplets are outdated by new observations.
        Returns list of triplets to remove.
        """
        outdated = []

        # Simple heuristic: if new triplet contradicts existing one about same subject+relation
        existing_dict = defaultdict(list)
        for subj, rel, obj in existing_triplets:
            existing_dict[(subj, rel)].append(obj)

        for subj, rel, obj in new_triplets:
            # Check for location updates (player can only be in one place)
            if rel == "location" and subj == "player":
                for existing_obj in existing_dict.get(("player", "location"), []):
                    if existing_obj != obj:
                        outdated.append(("player", "location", existing_obj))

            # Check for containment conflicts (object can only be in one container)
            if rel in ["is_in", "location"]:
                for existing_rel in ["is_in", "location", "is_on"]:
                    for existing_obj in existing_dict.get((subj, existing_rel), []):
                        if existing_obj != obj and existing_rel != rel:
                            outdated.append((subj, existing_rel, existing_obj))

        return outdated

# === ARIGRAPH STORAGE BACKEND ===

class AriGraphStorage(MemoryStorage):
    """
    Storage backend that implements the AriGraph memory architecture
    combining semantic knowledge graph with episodic memories
    """

    def __init__(self):
        # Semantic memory structures
        self.semantic_vertices: Dict[str, SemanticVertex] = {}
        self.semantic_edges: Dict[str, SemanticEdge] = {}
        self.vertex_name_to_id: Dict[str, str] = {}  # For quick lookup

        # Episodic memory structures
        self.episodic_vertices: Dict[str, EpisodicVertex] = {}
        self.episodic_edges: Dict[str, EpisodicEdge] = {}

        # Adjacency lists for graph traversal
        self.semantic_adjacency: Dict[str, List[str]] = defaultdict(list)  # vertex_id -> [edge_ids]
        self.triplet_to_episodic: Dict[str, List[str]] = defaultdict(list)  # edge_id -> [episodic_vertex_ids]

        self.step_counter = 0

    def _get_or_create_vertex(self, name: str, entity_type: str = None) -> str:
        """Get existing vertex ID or create new vertex for entity name"""
        clean_name = name.strip().lower()

        if clean_name in self.vertex_name_to_id:
            return self.vertex_name_to_id[clean_name]

        vertex_id = str(uuid.uuid4())
        vertex = SemanticVertex(
            id=vertex_id,
            name=clean_name,
            entity_type=entity_type
        )

        self.semantic_vertices[vertex_id] = vertex
        self.vertex_name_to_id[clean_name] = vertex_id
        return vertex_id

    def _create_semantic_edge(self, subject: str, relation: str, obj: str) -> str:
        """Create semantic edge between two entities"""
        subject_id = self._get_or_create_vertex(subject)
        object_id = self._get_or_create_vertex(obj)

        edge_id = str(uuid.uuid4())
        edge = SemanticEdge(
            id=edge_id,
            subject_id=subject_id,
            relation=relation.strip().lower(),
            object_id=object_id
        )

        self.semantic_edges[edge_id] = edge
        self.semantic_adjacency[subject_id].append(edge_id)
        self.semantic_adjacency[object_id].append(edge_id)

        return edge_id

    def _remove_semantic_edge(self, subject: str, relation: str, obj: str) -> bool:
        """Remove semantic edge if it exists"""
        subject_id = self.vertex_name_to_id.get(subject.strip().lower())
        object_id = self.vertex_name_to_id.get(obj.strip().lower())

        if not subject_id or not object_id:
            return False

        # Find matching edge
        for edge_id in list(self.semantic_edges.keys()):
            edge = self.semantic_edges[edge_id]
            if (edge.subject_id == subject_id and
                edge.object_id == object_id and
                edge.relation == relation.strip().lower()):

                # Remove from adjacency lists
                self.semantic_adjacency[subject_id].remove(edge_id)
                self.semantic_adjacency[object_id].remove(edge_id)

                # Remove edge
                del self.semantic_edges[edge_id]
                return True

        return False

    async def store(self, memory: Memory) -> bool:
        """Store memory by updating both semantic and episodic structures"""
        try:
            # Create episodic vertex for this observation
            episodic_id = str(uuid.uuid4())
            episodic_vertex = EpisodicVertex(
                id=episodic_id,
                observation=str(memory.content),
                step=self.step_counter,
                metadata=memory.metadata
            )
            self.episodic_vertices[episodic_id] = episodic_vertex

            # Extract semantic information from memory content
            extractor = TripletExtractor()
            new_triplets = extractor.extract_triplets(str(memory.content), memory.metadata)

            if new_triplets:
                # Get existing triplets for outdated detection
                existing_triplets = [(e.subject_id, e.relation, e.object_id)
                                   for e in self.semantic_edges.values()]

                # Convert vertex IDs back to names for comparison
                existing_named_triplets = []
                for subj_id, rel, obj_id in existing_triplets:
                    subj_name = next((v.name for v in self.semantic_vertices.values() if v.id == subj_id), subj_id)
                    obj_name = next((v.name for v in self.semantic_vertices.values() if v.id == obj_id), obj_id)
                    existing_named_triplets.append((subj_name, rel, obj_name))

                # Detect and remove outdated triplets
                outdated_triplets = extractor.detect_outdated_triplets(existing_named_triplets, new_triplets)
                for subj, rel, obj in outdated_triplets:
                    self._remove_semantic_edge(subj, rel, obj)

                # Add new semantic edges
                new_edge_ids = []
                for subject, relation, obj in new_triplets:
                    edge_id = self._create_semantic_edge(subject, relation, obj)
                    new_edge_ids.append(edge_id)

                    # Track which episodic vertex this triplet came from
                    self.triplet_to_episodic[edge_id].append(episodic_id)

                # Create episodic edge linking observation to extracted triplets
                if new_edge_ids:
                    episodic_edge_id = str(uuid.uuid4())
                    episodic_edge = EpisodicEdge(
                        id=episodic_edge_id,
                        episodic_vertex_id=episodic_id,
                        semantic_edge_ids=new_edge_ids
                    )
                    self.episodic_edges[episodic_edge_id] = episodic_edge

            self.step_counter += 1
            return True

        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    async def store_batch(self, memories: List[Memory]) -> bool:
        """Store multiple memories"""
        for memory in memories:
            success = await self.store(memory)
            if not success:
                return False
        return True

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID - could be semantic or episodic"""
        # Check episodic vertices first
        if memory_id in self.episodic_vertices:
            vertex = self.episodic_vertices[memory_id]
            return Memory(
                id=vertex.id,
                content=vertex.observation,
                memory_type=MemoryType.EPISODIC,
                timestamp=vertex.timestamp,
                metadata=vertex.metadata
            )

        # Check semantic edges
        if memory_id in self.semantic_edges:
            edge = self.semantic_edges[memory_id]
            subject_name = self.semantic_vertices[edge.subject_id].name
            object_name = self.semantic_vertices[edge.object_id].name
            content = f"{subject_name}, {edge.relation}, {object_name}"

            return Memory(
                id=edge.id,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                timestamp=edge.created_at
            )

        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete memory by ID"""
        if memory_id in self.episodic_vertices:
            del self.episodic_vertices[memory_id]
            return True

        if memory_id in self.semantic_edges:
            edge = self.semantic_edges[memory_id]
            # Remove from adjacency lists
            self.semantic_adjacency[edge.subject_id].remove(memory_id)
            self.semantic_adjacency[edge.object_id].remove(memory_id)
            del self.semantic_edges[memory_id]
            return True

        return False

    def get_semantic_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the current semantic graph"""
        return {
            "vertices": len(self.semantic_vertices),
            "edges": len(self.semantic_edges),
            "episodic_vertices": len(self.episodic_vertices),
            "episodic_edges": len(self.episodic_edges)
        }

# Continue in next file...