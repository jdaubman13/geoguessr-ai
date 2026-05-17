import faiss
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.knowledge.embedder import ClueEntry, load_clues
from app.config import settings


@dataclass
class RetrievedClue:
    clue: ClueEntry
    similarity: float
    effective_weight: float


class ClueRetriever:
    def __init__(self, model: SentenceTransformer):
        self.embedder = model
        self.clues = load_clues(settings.kb_path)
        self.index = faiss.read_index(str(settings.faiss_index_path))
        print(f"Retriever loaded {len(self.clues)} clues from FAISS index.")

    def retrieve(
        self,
        query_features: list[str],
        top_k: int = None,
        threshold: float = None,
    ) -> list[RetrievedClue]:
        top_k = top_k or settings.retrieval_top_k
        threshold = threshold or settings.retrieval_threshold

        query_text = ". ".join(query_features)
        q_emb = self.embedder.encode(
            [query_text], normalize_embeddings=True
        )
        scores, indices = self.index.search(
            q_emb.astype(np.float32), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue
            clue = self.clues[idx]
            results.append(RetrievedClue(
                clue=clue,
                similarity=float(score),
                effective_weight=clue.weight * float(score),
            ))

        return sorted(
            results,
            key=lambda r: r.effective_weight,
            reverse=True,
        )