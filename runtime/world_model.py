"""Causal world model -- maintains a structured representation of how
the world works, seeded from geolocation.

The world model is a directed graph where:
  - Nodes are entities (agent, users, tools, concepts, objects)
  - Edges are causal/relational links (causes, enables, constrains, etc.)
  - Every node and edge has a confidence score and a timestamp

The geolocation module provides the root nodes. From there, the model
grows as the agent observes the world and infers causal structure.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from runtime.geolocation import GeoLocation

log = structlog.get_logger()


class Node:
    """An entity in the world model."""

    __slots__ = ("id", "type", "name", "properties", "confidence", "created", "updated")

    def __init__(
        self,
        node_type: str,
        name: str,
        properties: dict | None = None,
        confidence: float = 1.0,
        node_id: str | None = None,
    ):
        self.id = node_id or str(uuid.uuid4())[:8]
        self.type = node_type
        self.name = name
        self.properties = properties or {}
        self.confidence = confidence
        self.created = time.time()
        self.updated = self.created

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "properties": self.properties,
            "confidence": self.confidence,
            "created": self.created,
            "updated": self.updated,
        }


class Edge:
    """A causal or relational link between two nodes."""

    __slots__ = ("source", "target", "relation", "strength", "evidence", "created")

    def __init__(
        self,
        source: str,
        target: str,
        relation: str,
        strength: float = 1.0,
        evidence: str = "",
    ):
        self.source = source
        self.target = target
        self.relation = relation
        self.strength = strength
        self.evidence = evidence
        self.created = time.time()

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "strength": self.strength,
            "evidence": self.evidence,
            "created": self.created,
        }


# Relation types
CAUSES = "causes"
ENABLES = "enables"
CONSTRAINS = "constrains"
BELONGS_TO = "belongs_to"
LOCATED_IN = "located_in"
RUNS_ON = "runs_on"
KNOWS = "knows"
TALKS_TO = "talks_to"
GOVERNS = "governs"
PRODUCES = "produces"
DEPENDS_ON = "depends_on"


class WorldModel:
    """Causal graph of the agent's understanding of the world.

    Seeded from GeoLocation, then grows through observation and inference.
    Supports forward simulation (predict) and abductive reasoning (explain).
    """

    def __init__(self, geolocation: GeoLocation, persist_path: str | None = None):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.geolocation = geolocation
        self.persist_path = Path(persist_path) if persist_path else None
        self.inference_count = 0

        # Seed from geolocation
        self._seed_from_geolocation()

        # Load persisted state if available
        if self.persist_path and self.persist_path.exists():
            self._load()

        log.info(
            "world_model_initialized",
            nodes=len(self.nodes),
            edges=len(self.edges),
        )

    def _seed_from_geolocation(self) -> None:
        """Build the initial world graph from the five geolocation dimensions."""
        geo = self.geolocation

        # Root: the agent itself
        agent = self._add_node("agent", "self", {
            "model": geo.informational["model_name"],
            "training_cutoff": geo.informational["model_training_cutoff"],
        }, node_id="self")

        # Computational location
        dc = self._add_node("datacenter", geo.computational["datacenter"], {
            "region": geo.computational["region"],
            "provider": "modal",
        })
        gpu = self._add_node("hardware", geo.computational["gpu"], {
            "vram_gb": geo.computational["gpu_vram_gb"],
            "type": "gpu",
        })
        self._add_edge(agent.id, dc.id, LOCATED_IN, evidence="runtime environment")
        self._add_edge(agent.id, gpu.id, RUNS_ON, evidence="modal gpu allocation")

        # Jurisdictional location
        jurisdiction = self._add_node("jurisdiction", geo.jurisdictional["primary_jurisdiction"])
        self._add_edge(jurisdiction.id, agent.id, GOVERNS, evidence="datacenter location")
        for reg in geo.jurisdictional["applicable_regulations"]:
            reg_node = self._add_node("regulation", reg)
            self._add_edge(jurisdiction.id, reg_node.id, PRODUCES, evidence="legal framework")
            self._add_edge(reg_node.id, agent.id, CONSTRAINS, evidence="compliance requirement")

        # Relational location
        owner = self._add_node("organization", geo.relational["owner_org"], {
            "workspace": geo.relational["owner_workspace"],
        })
        self._add_edge(agent.id, owner.id, BELONGS_TO, evidence="deployment config")

        if geo.relational["current_interlocutor"]:
            user = self._add_node("user", geo.relational["current_interlocutor"])
            self._add_edge(agent.id, user.id, TALKS_TO, evidence="active session")

        # Informational location -- tools
        for tool_name in geo.informational["available_tools"]:
            tool = self._add_node("tool", tool_name)
            self._add_edge(agent.id, tool.id, KNOWS, evidence="tool registry")

        # Temporal -- not a node, it's a property of everything
        # (timestamps on all nodes/edges serve this purpose)

    def _add_node(
        self,
        node_type: str,
        name: str,
        properties: dict | None = None,
        confidence: float = 1.0,
        node_id: str | None = None,
    ) -> Node:
        """Add a node to the graph, or update if exists."""
        # Check for existing node with same type+name
        for n in self.nodes.values():
            if n.type == node_type and n.name == name:
                if properties:
                    n.properties.update(properties)
                n.updated = time.time()
                return n

        node = Node(node_type, name, properties, confidence, node_id)
        self.nodes[node.id] = node
        return node

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        strength: float = 1.0,
        evidence: str = "",
    ) -> Edge:
        """Add a causal/relational edge."""
        edge = Edge(source, target, relation, strength, evidence)
        self.edges.append(edge)
        return edge

    def observe(self, observation: str, entities: list[dict] | None = None) -> None:
        """Update the world model from an observation.

        entities: list of {"type": str, "name": str, "properties": dict}
        """
        if entities:
            for ent in entities:
                self._add_node(
                    ent.get("type", "concept"),
                    ent["name"],
                    ent.get("properties"),
                    confidence=ent.get("confidence", 0.7),
                )

        # Store the observation itself as a node
        obs_node = self._add_node("observation", observation[:100], {
            "full_text": observation[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, confidence=0.9)

        # Link observation to the agent
        if "self" in self.nodes:
            self._add_edge("self", obs_node.id, "observed", evidence="direct observation")

        self._save()

    def predict(self, action_description: str) -> list[dict]:
        """Forward simulation: predict consequences of an action.

        Uses the causal graph to trace likely effects.
        Returns a list of predicted outcomes with confidence scores.
        """
        self.inference_count += 1
        predictions = []

        # Find nodes related to concepts in the action
        action_lower = action_description.lower()
        relevant_nodes = [
            n for n in self.nodes.values()
            if n.name.lower() in action_lower or action_lower in n.name.lower()
        ]

        # Trace causal chains from relevant nodes
        for node in relevant_nodes:
            effects = self._trace_effects(node.id, depth=3)
            for effect_id, path, confidence in effects:
                effect_node = self.nodes.get(effect_id)
                if effect_node:
                    predictions.append({
                        "entity": effect_node.name,
                        "type": effect_node.type,
                        "causal_path": path,
                        "confidence": confidence,
                    })

        # Always predict that regulations constrain actions
        regulation_nodes = [n for n in self.nodes.values() if n.type == "regulation"]
        for reg in regulation_nodes:
            predictions.append({
                "entity": reg.name,
                "type": "constraint",
                "causal_path": ["action", "jurisdiction", reg.name],
                "confidence": 0.9,
            })

        log.info("prediction", action=action_description[:50], predictions=len(predictions))
        return predictions

    def explain(self, observation: str) -> list[dict]:
        """Abductive reasoning: what could have caused this observation?

        Traces backwards through causal edges to find possible causes.
        """
        self.inference_count += 1
        explanations = []

        obs_lower = observation.lower()
        relevant_nodes = [
            n for n in self.nodes.values()
            if n.name.lower() in obs_lower or obs_lower in n.name.lower()
        ]

        for node in relevant_nodes:
            causes = self._trace_causes(node.id, depth=3)
            for cause_id, path, confidence in causes:
                cause_node = self.nodes.get(cause_id)
                if cause_node:
                    explanations.append({
                        "cause": cause_node.name,
                        "type": cause_node.type,
                        "causal_path": path,
                        "confidence": confidence,
                    })

        log.info("explanation", observation=observation[:50], explanations=len(explanations))
        return explanations

    def _trace_effects(
        self, node_id: str, depth: int = 3, visited: set | None = None
    ) -> list[tuple[str, list[str], float]]:
        """Trace forward causal effects from a node."""
        if visited is None:
            visited = set()
        if depth <= 0 or node_id in visited:
            return []

        visited.add(node_id)
        results = []

        for edge in self.edges:
            if edge.source == node_id and edge.relation in (CAUSES, ENABLES, PRODUCES):
                node = self.nodes.get(node_id)
                source_name = node.name if node else node_id
                path = [source_name, f"--{edge.relation}-->"]
                target_node = self.nodes.get(edge.target)
                if target_node:
                    path.append(target_node.name)

                results.append((edge.target, path, edge.strength))

                # Recurse
                deeper = self._trace_effects(edge.target, depth - 1, visited)
                for d_id, d_path, d_conf in deeper:
                    results.append((d_id, path + d_path, edge.strength * d_conf))

        return results

    def _trace_causes(
        self, node_id: str, depth: int = 3, visited: set | None = None
    ) -> list[tuple[str, list[str], float]]:
        """Trace backward to find causes of a node."""
        if visited is None:
            visited = set()
        if depth <= 0 or node_id in visited:
            return []

        visited.add(node_id)
        results = []

        for edge in self.edges:
            if edge.target == node_id and edge.relation in (CAUSES, ENABLES, PRODUCES, CONSTRAINS):
                source_node = self.nodes.get(edge.source)
                if source_node:
                    path = [source_node.name, f"--{edge.relation}-->"]
                    results.append((edge.source, path, edge.strength))
                    deeper = self._trace_causes(edge.source, depth - 1, visited)
                    for d_id, d_path, d_conf in deeper:
                        results.append((d_id, d_path + path, edge.strength * d_conf))

        return results

    def get_context_for_reasoning(self, query: str, max_nodes: int = 20) -> str:
        """Extract relevant world context for a reasoning query.

        Returns a text summary of relevant nodes and relationships.
        """
        query_lower = query.lower()

        # Score nodes by relevance to query
        scored = []
        for node in self.nodes.values():
            score = 0.0
            if node.name.lower() in query_lower:
                score += 1.0
            if node.type in ("agent", "organization", "jurisdiction"):
                score += 0.3  # always somewhat relevant
            for word in query_lower.split():
                if word in node.name.lower() or word in str(node.properties).lower():
                    score += 0.2
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [n for _, n in scored[:max_nodes]]

        if not top_nodes:
            return ""

        lines = ["## World Model Context"]
        node_ids = {n.id for n in top_nodes}

        for node in top_nodes:
            props = ", ".join(f"{k}={v}" for k, v in node.properties.items()) if node.properties else ""
            lines.append(f"- [{node.type}] {node.name}{f' ({props})' if props else ''} (conf={node.confidence:.1f})")

        # Include edges between relevant nodes
        relevant_edges = [
            e for e in self.edges
            if e.source in node_ids or e.target in node_ids
        ]
        if relevant_edges:
            lines.append("\nRelationships:")
            for edge in relevant_edges[:15]:
                src = self.nodes.get(edge.source)
                tgt = self.nodes.get(edge.target)
                if src and tgt:
                    lines.append(f"  {src.name} --{edge.relation}--> {tgt.name}")

        return "\n".join(lines)

    def summary(self) -> dict:
        """Summary statistics of the world model."""
        type_counts: dict[str, int] = {}
        for n in self.nodes.values():
            type_counts[n.type] = type_counts.get(n.type, 0) + 1

        relation_counts: dict[str, int] = {}
        for e in self.edges:
            relation_counts[e.relation] = relation_counts.get(e.relation, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": type_counts,
            "relation_types": relation_counts,
            "inference_count": self.inference_count,
        }

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            for nd in data.get("nodes", []):
                node = Node(nd["type"], nd["name"], nd.get("properties", {}),
                            nd.get("confidence", 1.0), nd.get("id"))
                node.created = nd.get("created", time.time())
                node.updated = nd.get("updated", time.time())
                self.nodes[node.id] = node
            for ed in data.get("edges", []):
                edge = Edge(ed["source"], ed["target"], ed["relation"],
                            ed.get("strength", 1.0), ed.get("evidence", ""))
                edge.created = ed.get("created", time.time())
                self.edges.append(edge)
            log.info("world_model_loaded", nodes=len(self.nodes), edges=len(self.edges))
        except Exception as e:
            log.warning("world_model_load_failed", error=str(e))
