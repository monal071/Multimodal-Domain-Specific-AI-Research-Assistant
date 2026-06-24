from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RetrievedChunk:
    chunk_id:     str
    doc_id:       str
    source_file:  str
    page_start:   int
    page_end:     int
    section_path: list[str]
    heading:      Optional[str]
    text:         str
    score:        float
    rerank_score: Optional[float] = None
    has_table:    bool = False
    has_formula:  bool = False

@dataclass
class RAGResult:
    query:   str
    answer:  str
    chunks:  list[RetrievedChunk] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
