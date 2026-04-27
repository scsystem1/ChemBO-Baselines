from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings

from .kimi_utils import build_kimi_client


DEFAULT_EMBED_MODEL = os.getenv("BOICL_EMBED_MODEL", "text-embedding-v4")
DEFAULT_EMBED_DIM = int(os.getenv("BOICL_EMBED_DIM", "1536"))


class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str | None = None, dimensions: int | None = None):
        self.model = model or DEFAULT_EMBED_MODEL
        self.dimensions = int(dimensions or DEFAULT_EMBED_DIM)
        self.client = build_kimi_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            model=self.model,
            input=clean_texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_default_embeddings() -> Embeddings:
    return DashScopeEmbeddings()
