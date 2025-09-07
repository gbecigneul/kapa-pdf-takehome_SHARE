"""
In-memory LanceDB hybrid store
──────────────────────────────
• Dense vectors  : all-mpnet-base-v2  (HuggingFace, 768-d)
• Sparse index   : BM25 on ``text`` column
• Hybrid query   : LanceDB blends BM25 + cosine in one call
• Reset support  : drops the table so UI can re-index
"""

from __future__ import annotations

import os
import uuid
from typing import List, Tuple, Optional
import time

import lancedb
import pandas as pd

from .schema import Document

TOP_K = int(os.getenv("TOP_K", "5"))


class InMemoryVectorStore:
    DEFAULT_TABLE_NAME = "chunks"

    def __init__(self, table_name: Optional[str] = None, db_uri: Optional[str] = None) -> None:
        self._db = lancedb.connect(db_uri or ":memory:")
        self._table_name = table_name or f"{InMemoryVectorStore.DEFAULT_TABLE_NAME}_{uuid.uuid4().hex[:8]}"

        self._table = self._db.create_table(
            self._table_name,
            schema=Document,
            mode="overwrite",
        )
        self._table.create_fts_index("text", replace=True)
        self.texts: List[str] = []

    def add_texts(self, texts: List[str]) -> None:
        batch = []
        for t in texts:
            if t:
                batch.append({"text": t})

        if batch:
            # Retry on transient commit conflicts from concurrent overwrites/reruns
            max_attempts = 5
            delay = 0.2
            for attempt in range(max_attempts):
                try:
                    self._table.add(batch)
                    self.texts.extend(texts)
                    break
                except RuntimeError as e:
                    msg = str(e)
                    if "Commit conflict" in msg or "incompatible with concurrent transaction" in msg:
                        if attempt < max_attempts - 1:
                            time.sleep(delay)
                            delay *= 2
                            continue
                    raise

    def search(self, query: str, k: int = TOP_K) -> List[Tuple[str, float]]:
        if self._table is None:
            return []

        df = (
            self._table.search(
                query,
                query_type="hybrid",
            )
            .limit(k)
            .to_list()
        )

        return [(doc["text"], doc["_relevance_score"]) for doc in df]

    def reset(self) -> None:
        if self._table is not None:
            self._db.drop_table(self._table.name)
        self._table = None
        self.texts.clear()
