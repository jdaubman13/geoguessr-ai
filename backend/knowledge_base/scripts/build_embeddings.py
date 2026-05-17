import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from sentence_transformers import SentenceTransformer
from app.config import settings
from app.knowledge.embedder import load_clues, build_faiss_index


def main():
    print("Loading sentence transformer model...")
    model = SentenceTransformer(settings.embed_model)

    print(f"Loading clues from {settings.kb_path}...")
    clues = load_clues(settings.kb_path)
    print(f"Loaded {len(clues)} clues.")

    build_faiss_index(clues, model, settings.faiss_index_path)
    print("Done!")


if __name__ == "__main__":
    main()