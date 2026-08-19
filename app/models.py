from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    source: str
    text: str


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    strategy: str
    chunk_index: int
    metadata: dict[str, Any]


@dataclass
class Retrieval:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[dict]
    should_answer: bool
    confidence: float
    latency_ms: dict[str, float]
    transcript: str = ""
    refusal_reason: str | None = None

    def to_dict(self):
        return self.__dict__