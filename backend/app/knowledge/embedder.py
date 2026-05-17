import json
import faiss
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from app.config import settings


@dataclass
class ClueEntry:
    id: str
    feature: str
    description: str
    countries: list[str]
    regions: list[str] = field(default_factory=list)
    weight: float = 0.5
    tags: list[str] = field(default_factory=list)


def load_clues(path: Path) -> list[ClueEntry]:
    with open(path, "r") as f:
        data = json.load(f)
    return [
        ClueEntry(
            id=entry["id"],
            feature=entry["feature"],
            description=entry.get("description", ""),
            countries=entry["countries"],
            regions=entry.get("regions", []),
            weight=entry.get("weight", 0.5),
            tags=entry.get("tags", []),
        )
        for entry in data
    ]


def build_faiss_index(
    clues: list[ClueEntry],
    model: SentenceTransformer,
    save_path: Path,
):
    texts = [f"{c.feature}. {c.description}" for c in clues]
    print(f"Embedding {len(texts)} clues...")
    embeddings = model.encode(texts, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_path))
    print(f"FAISS index saved to {save_path}")
    return index