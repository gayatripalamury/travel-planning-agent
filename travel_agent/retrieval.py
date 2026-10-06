"""Simple TF-IDF retrieval over the bundled destination knowledge base."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class DestinationRetriever:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.term_counts = [Counter(_tokens(self._text(doc))) for doc in documents]
        self.document_frequency = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())

    @staticmethod
    def _text(document: dict[str, Any]) -> str:
        return " ".join(str(document.get(key, "")) for key in ("name", "country", "summary", "activities", "tips"))

    def _vector(self, counts: Counter) -> dict[str, float]:
        size = max(sum(counts.values()), 1)
        return {
            token: (count / size) * math.log((1 + len(self.documents)) / (1 + self.document_frequency[token])) + 1
            for token, count in counts.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_vector = self._vector(Counter(_tokens(query)))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        scored = []
        for document, counts in zip(self.documents, self.term_counts):
            vector = self._vector(counts)
            score = sum(query_vector.get(token, 0) * vector.get(token, 0) for token in query_vector)
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            scored.append((score / (query_norm * norm), document))
        return [
            {**document, "retrieval_score": round(score, 4)}
            for score, document in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        ]


def load_destinations(path: str | Path | None = None) -> list[dict[str, Any]]:
    import json
    source = Path(path or Path(__file__).parent / "data" / "destinations.json")
    return json.loads(source.read_text(encoding="utf-8"))
