from PIL import Image
from app.vision.clip_encoder import get_clip_encoder

DRIVING_SIDE_PROMPTS = [
    "driving on the left side of the road",
    "driving on the right side of the road",
]


class DrivingSideDetector:
    def __init__(self):
        self.encoder = get_clip_encoder()

    def estimate(self, image: Image.Image) -> str:
        scores = self.encoder.zero_shot_classify(image, DRIVING_SIDE_PROMPTS)
        best = max(scores, key=scores.get)
        if "left" in best:
            return "left"
        return "right"