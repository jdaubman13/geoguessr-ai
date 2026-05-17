import torch
import clip
import numpy as np
from PIL import Image
from functools import lru_cache
from app.config import settings


class CLIPEncoder:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading CLIP {settings.clip_model} on {self.device}...")
        self.model, self.preprocess = clip.load(
            settings.clip_model, device=self.device
        )
        self.model.eval()
        print("CLIP ready.")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def zero_shot_classify(
        self, image: Image.Image, candidates: list[str]
    ) -> dict[str, float]:
        img_feat = self.encode_image(image)
        txt_feats = self.encode_texts(candidates)
        sims = img_feat @ txt_feats.T
        probs = torch.softmax(torch.tensor(sims * 100), dim=0).numpy()
        return dict(zip(candidates, probs.tolist()))


@lru_cache(maxsize=1)
def get_clip_encoder() -> CLIPEncoder:
    return CLIPEncoder()