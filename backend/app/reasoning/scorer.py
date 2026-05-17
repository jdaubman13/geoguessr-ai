from dataclasses import dataclass, field
from collections import defaultdict
from app.features.extractor import FeatureDict
from app.knowledge.retriever import RetrievedClue


@dataclass
class PredictionResult:
    country: str
    region: str | None
    direction: str | None
    confidence: float
    country_probabilities: dict[str, float]
    region_probabilities: dict[str, float]
    top_clues: list[dict]


LHT_COUNTRIES = {
    "Japan", "Australia", "UK", "India", "New Zealand",
    "Thailand", "Indonesia", "Malaysia", "South Africa",
    "Kenya", "Ireland", "Bangladesh", "Pakistan",
}

CYRILLIC_COUNTRIES = {
    "Russia", "Ukraine", "Bulgaria", "Belarus",
    "Mongolia", "Serbia", "Kazakhstan",
}

JAPANESE_COUNTRIES = {"Japan"}
KOREAN_COUNTRIES = {"South Korea", "North Korea"}
ARABIC_COUNTRIES = {
    "Saudi Arabia", "Egypt", "Jordan", "UAE",
    "Iraq", "Syria", "Tunisia", "Morocco",
}


class ClueScorer:
    def score(
        self,
        features: FeatureDict,
        retrieved_clues: list[RetrievedClue],
        country_priors: dict[str, float],
    ) -> PredictionResult:

        country_scores: dict[str, float] = defaultdict(float)
        region_scores: dict[str, float] = defaultdict(float)

        # Accumulate weighted clue votes
        # Single country clues get a specificity boost
        for rc in retrieved_clues:
            specificity_boost = 1.5 if len(rc.clue.countries) == 1 else 1.0
            for country in rc.clue.countries:
                country_scores[country] += rc.effective_weight * specificity_boost
            for region in (rc.clue.regions or []):
                region_scores[region] += rc.effective_weight

        # Apply driving side elimination
        if features.driving_side == "left":
            for country in list(country_scores.keys()):
                if country not in LHT_COUNTRIES:
                    country_scores[country] *= 0.05

        # Apply script constraints
        if features.script_type == "cyrillic":
            for country in list(country_scores.keys()):
                if country not in CYRILLIC_COUNTRIES:
                    country_scores[country] *= 0.05

        if features.script_type == "japanese":
            for country in list(country_scores.keys()):
                if country not in JAPANESE_COUNTRIES:
                    country_scores[country] *= 0.01

        if features.script_type == "korean":
            for country in list(country_scores.keys()):
                if country not in KOREAN_COUNTRIES:
                    country_scores[country] *= 0.01

        if features.script_type == "arabic":
            for country in list(country_scores.keys()):
                if country not in ARABIC_COUNTRIES:
                    country_scores[country] *= 0.05

        # Blend with priors
        for country, prior in country_priors.items():
            country_scores[country] = (
                country_scores.get(country, 0.0) + prior * 0.15
            )

        # Normalize to probabilities
        total = sum(country_scores.values()) or 1.0
        normalized = {
            k: round(v / total, 4)
            for k, v in country_scores.items()
        }

        # Sort by probability
        sorted_countries = dict(
            sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        )

        top_country = next(iter(sorted_countries))

        # Confidence penalty if top two are close
        values = list(sorted_countries.values())
        confidence = values[0]
        if len(values) >= 2 and values[1] > 0.3 * values[0]:
            confidence *= 0.80

        # Top region
        top_region = (
            max(region_scores, key=region_scores.get)
            if region_scores else None
        )

        # Direction estimate
        direction = self._estimate_direction(features)

        # Top clues for explanation
        top_clues = [
            {
                "feature": rc.clue.feature,
                "countries": rc.clue.countries,
                "weight": round(rc.effective_weight, 3),
            }
            for rc in sorted(
                retrieved_clues,
                key=lambda r: r.effective_weight,
                reverse=True,
            )[:5]
        ]

        return PredictionResult(
            country=top_country,
            region=top_region,
            direction=direction,
            confidence=round(confidence, 4),
            country_probabilities=sorted_countries,
            region_probabilities=dict(region_scores),
            top_clues=top_clues,
        )

    def _estimate_direction(self, features: FeatureDict) -> str | None:
        veg = features.vegetation or ""
        if "tundra" in veg or "coniferous" in veg:
            return "north"
        if "tropical" in veg or "palm" in veg:
            return "south"
        return None