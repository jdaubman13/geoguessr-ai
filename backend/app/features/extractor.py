from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

from app.vision.clip_encoder import get_clip_encoder, CLIPEncoder
from app.vision.scene_classifier import SceneClassifier
from app.vision.driving_side import DrivingSideDetector
from app.vision.ocr_engine import OCREngine

POLE_PROMPTS = [
    "concrete square utility pole with holes japan",
    "wooden utility pole with crossbar rural",
    "metal steel utility pole urban",
    "concrete round utility pole",
    "no utility poles visible in scene",
]

ROAD_PROMPTS = [
    "yellow painted center road line",
    "white painted center road line",
    "double yellow center road line",
    "no center road line visible",
]

BOLLARD_PROMPTS = [
    "white bollard with black cap on roadside",
    "yellow bollard on roadside",
    "red and white bollard on roadside",
    "no bollards visible",
]

ARCHITECTURE_PROMPTS = [
    "japanese style building architecture",
    "european style building architecture",
    "american style building architecture",
    "soviet brutalist concrete apartment block",
    "tropical developing world building",
    "middle eastern arabic architecture",
    "southeast asian building architecture",
    "african building architecture",
    "latin american colorful building",
    "no buildings visible",
]

ROAD_TYPE_PROMPTS = [
    "paved asphalt urban road",
    "paved asphalt rural road",
    "unpaved dirt gravel road",
    "red dirt laterite road africa australia",
    "cobblestone road european",
    "highway motorway road",
]


@dataclass
class FeatureDict:
    # Road
    road_line_color: Optional[str] = None
    road_line_conf: float = 0.0
    road_type: Optional[str] = None

    # Infrastructure
    pole_type: Optional[str] = None
    pole_type_conf: float = 0.0
    bollard_style: Optional[str] = None

    # Architecture
    architecture_style: Optional[str] = None

    # Environment
    vegetation: Optional[str] = None
    terrain: Optional[str] = None
    sky_condition: Optional[str] = None

    # Driving
    driving_side: Optional[str] = None

    # OCR
    detected_text: list[str] = field(default_factory=list)
    script_type: Optional[str] = None

    # Raw CLIP embedding for similarity search
    clip_embedding: list[float] = field(default_factory=list)


class FeatureExtractor:
    def __init__(self):
        self.encoder: CLIPEncoder = get_clip_encoder()
        self.scene = SceneClassifier()
        self.driving_side = DrivingSideDetector()
        self.ocr = OCREngine()

    def extract(self, image: Image.Image) -> FeatureDict:
        fd = FeatureDict()

        # Pole type
        pole_scores = self.encoder.zero_shot_classify(image, POLE_PROMPTS)
        best_pole = max(pole_scores, key=pole_scores.get)
        fd.pole_type = best_pole
        fd.pole_type_conf = pole_scores[best_pole]

        # Road line color
        road_scores = self.encoder.zero_shot_classify(image, ROAD_PROMPTS)
        best_road = max(road_scores, key=road_scores.get)
        fd.road_line_color = best_road
        fd.road_line_conf = road_scores[best_road]

        # Bollard style
        bollard_scores = self.encoder.zero_shot_classify(image, BOLLARD_PROMPTS)
        best_bollard = max(bollard_scores, key=bollard_scores.get)
        fd.bollard_style = best_bollard

        # Architecture style
        arch_scores = self.encoder.zero_shot_classify(image, ARCHITECTURE_PROMPTS)
        best_arch = max(arch_scores, key=arch_scores.get)
        fd.architecture_style = best_arch

        # Road type
        road_type_scores = self.encoder.zero_shot_classify(image, ROAD_TYPE_PROMPTS)
        best_road_type = max(road_type_scores, key=road_type_scores.get)
        fd.road_type = best_road_type

        # Scene
        fd.vegetation = self.scene.classify_vegetation(image)
        fd.terrain = self.scene.classify_terrain(image)
        fd.sky_condition = self.scene.classify_sky(image)

        # Driving side
        fd.driving_side = self.driving_side.estimate(image)

        # OCR
        fd.detected_text, fd.script_type = self.ocr.extract(image)

        # Raw CLIP embedding
        fd.clip_embedding = self.encoder.encode_image(image).tolist()

        return fd