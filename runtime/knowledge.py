"""Semantic knowledge graph -- structured fact store with relational queries.

Stores facts as (subject, predicate, object) triples with provenance and
confidence. Supports:
  - Fact assertion and retraction
  - Relational queries (who, what, where, how)
  - Conflict detection (contradictory facts)
  - Fact decay (confidence drops without reinforcement)
  - LLM-assisted extraction from text
"""

import json
import re
import time
from pathlib import Path

import structlog

log = structlog.get_logger()


class Fact:
    """A single (subject, predicate, object) triple."""

    __slots__ = ("subject", "predicate", "object", "confidence", "source",
                 "timestamp", "access_count")

    def __init__(self, subject: str, predicate: str, obj: str,
                 confidence: float = 0.8, source: str = "observation"):
        self.subject = subject.lower().strip()
        self.predicate = predicate.lower().strip()
        self.object = obj.lower().strip()
        self.confidence = confidence
        self.source = source
        self.timestamp = time.time()
        self.access_count = 0

    def matches(self, subject: str | None = None, predicate: str | None = None,
                obj: str | None = None) -> bool:
        """Check if fact matches a query pattern (None = wildcard)."""
        if subject and self.subject != subject.lower().strip():
            return False
        if predicate and self.predicate != predicate.lower().strip():
            return False
        if obj and self.object != obj.lower().strip():
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "predicate": self.predicate,
            "object": self.object, "confidence": self.confidence,
            "source": self.source, "timestamp": self.timestamp,
            "access_count": self.access_count,
        }

    def __repr__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object}) [{self.confidence:.2f}]"


class KnowledgeGraph:
    """Semantic knowledge store using SPO triples.

    Facts are indexed by subject for fast lookup. Confidence decays
    over time unless reinforced by repeated observation.
    """

    DECAY_RATE = 0.001       # confidence lost per hour
    MIN_CONFIDENCE = 0.1     # below this, fact is pruned
    MAX_FACTS = 10000

    def __init__(self, persist_path: str | None = None):
        self.facts: list[Fact] = []
        self._subject_index: dict[str, list[int]] = {}  # subject -> fact indices
        self.persist_path = Path(persist_path) if persist_path else None

        if self.persist_path and self.persist_path.exists():
            self._load()

    def assert_fact(self, subject: str, predicate: str, obj: str,
                    confidence: float = 0.8, source: str = "observation") -> Fact:
        """Add a fact. If it already exists, reinforce its confidence."""
        # Check for existing
        for fact in self.facts:
            if fact.matches(subject, predicate, obj):
                # Reinforce
                fact.confidence = min(1.0, fact.confidence + 0.1)
                fact.timestamp = time.time()
                fact.access_count += 1
                self._save()
                return fact

        # Check for contradiction
        contradictions = self.query(subject=subject, predicate=predicate)
        for c in contradictions:
            if c.object != obj.lower().strip():
                # Different object for same subject+predicate -- reduce old confidence
                c.confidence *= 0.7
                log.info("fact_conflict",
                         existing=repr(c),
                         new=f"({subject}, {predicate}, {obj})")

        fact = Fact(subject, predicate, obj, confidence, source)
        idx = len(self.facts)
        self.facts.append(fact)

        # Index
        key = fact.subject
        if key not in self._subject_index:
            self._subject_index[key] = []
        self._subject_index[key].append(idx)

        # Prune if over limit
        if len(self.facts) > self.MAX_FACTS:
            self._prune()

        self._save()
        return fact

    def retract(self, subject: str, predicate: str, obj: str) -> bool:
        """Remove a specific fact."""
        for i, fact in enumerate(self.facts):
            if fact.matches(subject, predicate, obj):
                self.facts.pop(i)
                self._rebuild_index()
                self._save()
                return True
        return False

    def query(self, subject: str | None = None, predicate: str | None = None,
              obj: str | None = None, min_confidence: float = 0.0) -> list[Fact]:
        """Query facts matching a pattern. None = wildcard."""
        # Use index if querying by subject
        if subject and not predicate and not obj:
            indices = self._subject_index.get(subject.lower().strip(), [])
            results = [self.facts[i] for i in indices
                       if i < len(self.facts) and self.facts[i].confidence >= min_confidence]
        else:
            results = [f for f in self.facts
                       if f.matches(subject, predicate, obj) and f.confidence >= min_confidence]

        # Bump access count
        for f in results:
            f.access_count += 1

        return sorted(results, key=lambda f: f.confidence, reverse=True)

    def query_text(self, question: str) -> list[Fact]:
        """Natural language query -- find facts relevant to a question."""
        words = set(question.lower().split())
        # Remove stop words
        stops = {"is", "a", "the", "what", "who", "where", "how", "does", "do",
                 "are", "was", "were", "has", "have", "of", "in", "on", "for",
                 "to", "and", "or", "not", "it", "this", "that"}
        words -= stops

        scored: list[tuple[float, Fact]] = []
        for fact in self.facts:
            score = 0.0
            fact_words = set(f"{fact.subject} {fact.predicate} {fact.object}".split())
            overlap = words & fact_words
            score = len(overlap) * fact.confidence
            if score > 0:
                scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [f for _, f in scored[:10]]

        for f in results:
            f.access_count += 1

        return results

    def get_context(self, query: str, max_facts: int = 10) -> str:
        """Get relevant facts as context for the LLM."""
        facts = self.query_text(query)[:max_facts]
        if not facts:
            return ""

        lines = ["## Known Facts"]
        for f in facts:
            lines.append(f"- {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f})")
        return "\n".join(lines)

    async def extract_facts_from_text(self, text: str, llm_fn) -> list[Fact]:
        """Use the LLM to extract structured facts from text."""
        try:
            response = await llm_fn([
                {"role": "system", "content": (
                    "Extract factual statements from the text as (subject, predicate, object) triples. "
                    'Return JSON array: [{"subject": "...", "predicate": "...", "object": "..."}]. '
                    "Only clear factual claims, not opinions or questions. JSON only, no markdown."
                )},
                {"role": "user", "content": text[:500]},
            ], max_tokens=256, temperature=0.1)

            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []

            triples = json.loads(json_match.group())
            facts = []
            for t in triples:
                if isinstance(t, dict) and "subject" in t and "predicate" in t and "object" in t:
                    fact = self.assert_fact(
                        t["subject"], t["predicate"], t["object"],
                        confidence=0.7, source="llm_extraction",
                    )
                    facts.append(fact)
            return facts

        except Exception as e:
            log.debug("fact_extraction_failed", error=str(e))
            return []

    def decay(self) -> int:
        """Apply confidence decay to all facts. Returns number pruned."""
        now = time.time()
        pruned = 0
        remaining = []
        for fact in self.facts:
            hours_elapsed = (now - fact.timestamp) / 3600
            # Accessed facts decay slower
            decay_factor = self.DECAY_RATE / (1 + fact.access_count * 0.1)
            fact.confidence -= decay_factor * hours_elapsed
            if fact.confidence >= self.MIN_CONFIDENCE:
                remaining.append(fact)
            else:
                pruned += 1

        if pruned > 0:
            self.facts = remaining
            self._rebuild_index()
            self._save()
            log.info("knowledge_decay", pruned=pruned, remaining=len(self.facts))
        return pruned

    def _prune(self) -> None:
        """Remove lowest-confidence facts to stay under MAX_FACTS."""
        self.facts.sort(key=lambda f: f.confidence)
        excess = len(self.facts) - self.MAX_FACTS
        if excess > 0:
            self.facts = self.facts[excess:]
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._subject_index.clear()
        for i, fact in enumerate(self.facts):
            if fact.subject not in self._subject_index:
                self._subject_index[fact.subject] = []
            self._subject_index[fact.subject].append(i)

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [f.to_dict() for f in self.facts]
        self.persist_path.write_text(json.dumps(data))

    def _load(self) -> None:
        try:
            data = json.loads(self.persist_path.read_text())
            for fd in data:
                fact = Fact(
                    fd["subject"], fd["predicate"], fd["object"],
                    fd.get("confidence", 0.8), fd.get("source", "loaded"),
                )
                fact.timestamp = fd.get("timestamp", time.time())
                fact.access_count = fd.get("access_count", 0)
                self.facts.append(fact)
            self._rebuild_index()
            log.info("knowledge_loaded", facts=len(self.facts))
        except Exception as e:
            log.warning("knowledge_load_failed", error=str(e))

    def summary(self) -> dict:
        subjects = set(f.subject for f in self.facts)
        predicates = set(f.predicate for f in self.facts)
        return {
            "total_facts": len(self.facts),
            "unique_subjects": len(subjects),
            "unique_predicates": len(predicates),
            "avg_confidence": round(
                sum(f.confidence for f in self.facts) / max(1, len(self.facts)), 3
            ),
            "most_accessed": sorted(
                [{"fact": repr(f), "accesses": f.access_count}
                 for f in self.facts],
                key=lambda x: x["accesses"], reverse=True,
            )[:5],
        }

    def __len__(self) -> int:
        return len(self.facts)
