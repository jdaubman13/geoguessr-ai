import json
import hashlib
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from app.features.extractor import FeatureExtractor
from app.knowledge.retriever import ClueRetriever
from app.reasoning.scorer import ClueScorer
from app.config import settings
from sentence_transformers import SentenceTransformer

router = APIRouter()

_extractor = None
_retriever = None
_scorer = None
_priors = None


def get_components():
    global _extractor, _retriever, _scorer, _priors
    if _extractor is None:
        print("Initializing components...")
        _extractor = FeatureExtractor()
        model = SentenceTransformer(settings.embed_model)
        _retriever = ClueRetriever(model)
        _scorer = ClueScorer()
        priors_path = Path("knowledge_base/country_priors.json")
        with open(priors_path) as f:
            _priors = json.load(f)
        print("Components ready.")
    return _extractor, _retriever, _scorer, _priors


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WebP images are supported."
        )

    contents = await file.read()
    image_hash = hashlib.sha256(contents).hexdigest()[:16]

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    max_dim = settings.max_image_dim
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))

    extractor, retriever, scorer, priors = get_components()

    features = extractor.extract(image)

    query_features = [
        f for f in [
            features.pole_type,
            features.vegetation,
            features.terrain,
            features.road_line_color,
            features.bollard_style,
            f"driving on the {features.driving_side} side" if features.driving_side else None,
            f"{features.script_type} script on signs" if features.script_type and features.script_type != "unknown" else None,
            f"{features.sky_condition}" if features.sky_condition else None,
        ] if f is not None
    ]

    clues = retriever.retrieve(query_features, threshold=0.1)
    result = scorer.score(features, clues, priors)

    return {
        "session_id": image_hash,
        "prediction": {
            "country": result.country,
            "region": result.region,
            "direction": result.direction,
            "confidence": result.confidence,
            "country_probabilities": dict(
                list(result.country_probabilities.items())[:5]
            ),
        },
        "explanation": {
            "top_clues": result.top_clues,
            "driving_side": features.driving_side,
            "script_type": features.script_type,
            "vegetation": features.vegetation,
            "terrain": features.terrain,
        },
    }