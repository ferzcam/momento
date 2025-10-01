"""
AriGraph Retrieval System - Implements semantic and episodic search
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
import math
import re

from draft import Memory, MemoryType, RetrievalQuery, MemoryRetrieval
from arigraph_implementation import AriGraphStorage, SemanticEdge, EpisodicVertex


class AriGraphRetrieval(MemoryRetrieval):
    """
    Retrieval system implementing the AriGraph search algorithm
    Combines semantic graph traversal with episodic memory ranking
    """

    def __init__(self, storage: AriGraphStorage):
        self.storage = storage

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words (placeholder for embeddings)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _semantic_search(self, query: str, depth: int = 2, width: int = 10) -> List[str]:
        """
        Semantic search using graph traversal
        Returns list of relevant semantic edge IDs
        """
        relevant_edges = []
        visited_vertices = set()
        search_queue = deque()

        # Find initial vertices that match the query
        query_words = set(query.lower().split())
        initial_vertices = []

        for vertex_id, vertex in self.storage.semantic_vertices.items():
            vertex_words = set(vertex.name.split())
            if vertex_words.intersection(query_words):
                initial_vertices.append((vertex_id, 0))  # (vertex_id, depth)

        # Start BFS from initial vertices
        search_queue.extend(initial_vertices)

        edge_scores = {}

        while search_queue and len(relevant_edges) < width:
            vertex_id, current_depth = search_queue.popleft()

            if vertex_id in visited_vertices or current_depth > depth:
                continue

            visited_vertices.add(vertex_id)

            # Get all edges connected to this vertex
            for edge_id in self.storage.semantic_adjacency.get(vertex_id, []):
                if edge_id in edge_scores:
                    continue

                edge = self.storage.semantic_edges[edge_id]

                # Calculate relevance score
                edge_text = f"{self.storage.semantic_vertices[edge.subject_id].name} {edge.relation} {self.storage.semantic_vertices[edge.object_id].name}"
                similarity = self._calculate_text_similarity(query, edge_text)

                # Weight by graph distance (closer is better)
                distance_weight = 1.0 / (current_depth + 1)
                final_score = similarity * distance_weight

                edge_scores[edge_id] = final_score

                # Add connected vertices to search queue for next depth
                if current_depth < depth:
                    other_vertex = edge.object_id if edge.subject_id == vertex_id else edge.subject_id
                    if other_vertex not in visited_vertices:
                        search_queue.append((other_vertex, current_depth + 1))

        # Sort edges by score and return top results
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_edges = [edge_id for edge_id, score in sorted_edges[:width] if score > 0.1]

        return relevant_edges

    def _episodic_search(self, semantic_edge_ids: List[str], limit: int = 10) -> List[str]:
        """
        Episodic search based on semantic triplets
        Returns list of relevant episodic vertex IDs
        """
        if not semantic_edge_ids:
            return []

        episodic_scores = defaultdict(float)

        # For each semantic edge, find associated episodic vertices
        for edge_id in semantic_edge_ids:
            episodic_vertex_ids = self.storage.triplet_to_episodic.get(edge_id, [])

            for episodic_id in episodic_vertex_ids:
                if episodic_id in self.storage.episodic_vertices:
                    episodic_vertex = self.storage.episodic_vertices[episodic_id]

                    # Count how many relevant triplets this episodic vertex has
                    relevant_triplet_count = 0
                    total_triplets = 0

                    # Find all episodic edges for this vertex
                    for ep_edge_id, ep_edge in self.storage.episodic_edges.items():
                        if ep_edge.episodic_vertex_id == episodic_id:
                            total_triplets = len(ep_edge.semantic_edge_ids)
                            relevant_triplet_count = sum(1 for edge_id in ep_edge.semantic_edge_ids
                                                       if edge_id in semantic_edge_ids)
                            break

                    if total_triplets > 0:
                        # Calculate relevance score using the formula from the paper
                        relevance = relevant_triplet_count / max(total_triplets, 1)
                        # Add log weighting for information content
                        log_weight = math.log2(max(total_triplets, 1)) if total_triplets > 1 else 0
                        final_score = relevance * (1 + log_weight)

                        episodic_scores[episodic_id] = final_score

        # Sort by score and return top results
        sorted_episodes = sorted(episodic_scores.items(), key=lambda x: x[1], reverse=True)
        return [ep_id for ep_id, score in sorted_episodes[:limit] if score > 0]

    async def search(self, query: RetrievalQuery) -> List[Memory]:
        """Main search method combining semantic and episodic search"""
        memories = []

        query_text = query.query if isinstance(query.query, str) else str(query.query)

        # Step 1: Semantic search
        relevant_semantic_edges = self._semantic_search(
            query_text,
            depth=2,
            width=query.limit * 2  # Get more semantic results for better episodic search
        )

        # Convert semantic edges to Memory objects
        for edge_id in relevant_semantic_edges[:query.limit // 2]:  # Reserve half for episodic
            edge = self.storage.semantic_edges[edge_id]
            subject_name = self.storage.semantic_vertices[edge.subject_id].name
            object_name = self.storage.semantic_vertices[edge.object_id].name
            content = f"{subject_name}, {edge.relation}, {object_name}"

            memory = Memory(
                id=edge.id,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                timestamp=edge.created_at,
                importance=1.0
            )
            memories.append(memory)

        # Step 2: Episodic search based on semantic results
        relevant_episodic_vertices = self._episodic_search(
            relevant_semantic_edges,
            limit=query.limit - len(memories)
        )

        # Convert episodic vertices to Memory objects
        for vertex_id in relevant_episodic_vertices:
            vertex = self.storage.episodic_vertices[vertex_id]
            memory = Memory(
                id=vertex.id,
                content=vertex.observation,
                memory_type=MemoryType.EPISODIC,
                timestamp=vertex.timestamp,
                importance=1.0,
                metadata=vertex.metadata
            )
            memories.append(memory)

        return memories[:query.limit]

    async def get_recent(self, limit: int = 10, memory_type: MemoryType = None) -> List[Memory]:
        """Get recent memories, optionally filtered by type"""
        memories = []

        if memory_type is None or memory_type == MemoryType.EPISODIC:
            # Get recent episodic memories
            sorted_episodic = sorted(
                self.storage.episodic_vertices.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )

            for vertex in sorted_episodic[:limit]:
                memory = Memory(
                    id=vertex.id,
                    content=vertex.observation,
                    memory_type=MemoryType.EPISODIC,
                    timestamp=vertex.timestamp,
                    metadata=vertex.metadata
                )
                memories.append(memory)

        if memory_type is None or memory_type == MemoryType.SEMANTIC:
            # Get recent semantic memories
            remaining_limit = limit - len(memories)
            if remaining_limit > 0:
                sorted_semantic = sorted(
                    self.storage.semantic_edges.values(),
                    key=lambda x: x.updated_at,
                    reverse=True
                )

                for edge in sorted_semantic[:remaining_limit]:
                    subject_name = self.storage.semantic_vertices[edge.subject_id].name
                    object_name = self.storage.semantic_vertices[edge.object_id].name
                    content = f"{subject_name}, {edge.relation}, {object_name}"

                    memory = Memory(
                        id=edge.id,
                        content=content,
                        memory_type=MemoryType.SEMANTIC,
                        timestamp=edge.updated_at
                    )
                    memories.append(memory)

        return memories[:limit]

    async def get_important(self, limit: int = 10, threshold: float = 0.8) -> List[Memory]:
        """Get important memories based on connectivity and recency"""
        memories = []

        # For semantic memories: importance based on node connectivity
        edge_importance = {}
        for edge_id, edge in self.storage.semantic_edges.items():
            # Calculate importance based on how connected the vertices are
            subject_connections = len(self.storage.semantic_adjacency.get(edge.subject_id, []))
            object_connections = len(self.storage.semantic_adjacency.get(edge.object_id, []))
            connectivity_score = (subject_connections + object_connections) / 2

            # Normalize and threshold
            normalized_score = min(connectivity_score / 10.0, 1.0)  # Assuming max 10 connections
            if normalized_score >= threshold:
                edge_importance[edge_id] = normalized_score

        # Sort by importance and convert to Memory objects
        sorted_important = sorted(edge_importance.items(), key=lambda x: x[1], reverse=True)

        for edge_id, importance in sorted_important[:limit]:
            edge = self.storage.semantic_edges[edge_id]
            subject_name = self.storage.semantic_vertices[edge.subject_id].name
            object_name = self.storage.semantic_vertices[edge.object_id].name
            content = f"{subject_name}, {edge.relation}, {object_name}"

            memory = Memory(
                id=edge.id,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                timestamp=edge.updated_at,
                importance=importance
            )
            memories.append(memory)

        return memories

    def get_exploration_info(self, current_location: str) -> List[str]:
        """Get unexplored exits from current location"""
        unexplored_exits = []

        # Find current location vertex
        location_id = self.storage.vertex_name_to_id.get(current_location.lower())
        if not location_id:
            return unexplored_exits

        # Find all exits from current location
        for edge_id in self.storage.semantic_adjacency.get(location_id, []):
            edge = self.storage.semantic_edges[edge_id]

            # Check if this is an exit edge
            if edge.relation in ["has_exit", "leads_to", "connects_to"]:
                if edge.subject_id == location_id:
                    # This is an outgoing exit
                    exit_name = self.storage.semantic_vertices[edge.object_id].name
                    unexplored_exits.append(exit_name)

        return unexplored_exits