from PIL import Image
from app.vision.clip_encoder import get_clip_encoder

TERRAIN_PROMPTS = [
    "flat terrain",
    "hilly terrain",
    "mountainous terrain",
    "coastal terrain",
    "desert terrain",
    "arctic tundra terrain",
]

VEGETATION_PROMPTS = [
    "tropical vegetation and palm trees",
    "dense pine and coniferous forest",
    "deciduous mixed forest",
    "dry savanna and grassland",
    "desert with sparse vegetation",
    "tundra with no trees",
    "rice paddy and farmland",
    "mediterranean scrubland",
]

SKY_PROMPTS = [
    "clear blue sky",
    "overcast grey sky",
    "partly cloudy sky",
    "heavy rain and storm",
    "foggy and misty sky",
]


class SceneClassifier:
    def __init__(self):
        self.encoder = get_clip_encoder()

    def classify_terrain(self, image: Image.Image) -> str:
        scores = self.encoder.zero_shot_classify(image, TERRAIN_PROMPTS)
        return max(scores, key=scores.get)

    def classify_vegetation(self, image: Image.Image) -> str:
        scores = self.encoder.zero_shot_classify(image, VEGETATION_PROMPTS)
        return max(scores, key=scores.get)

    def classify_sky(self, image: Image.Image) -> str:
        scores = self.encoder.zero_shot_classify(image, SKY_PROMPTS)
        return max(scores, key=scores.get)