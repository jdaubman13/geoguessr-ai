import numpy as np
from PIL import Image
import easyocr


SCRIPT_MAP = {
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ru": "cyrillic",
    "uk": "cyrillic",
    "bg": "cyrillic",
    "ar": "arabic",
    "hi": "devanagari",
    "th": "thai",
    "en": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "pt": "latin",
    "pl": "latin",
}


class OCREngine:
    def __init__(self):
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(
            ["en", "ja", "ru", "ko", "ar", "hi", "th", "fr", "de", "es"],
            gpu=False,
        )
        print("EasyOCR ready.")

    def extract(self, image: Image.Image) -> tuple[list[str], str]:
        img_array = np.array(image)
        results = self.reader.readtext(img_array)

        texts = [text for _, text, conf in results if conf > 0.3]

        script = self._detect_script(texts)

        return texts, script

    def _detect_script(self, texts: list[str]) -> str:
        if not texts:
            return "unknown"

        combined = " ".join(texts)

        if any("\u3040" <= c <= "\u30ff" for c in combined):
            return "japanese"
        if any("\uac00" <= c <= "\ud7a3" for c in combined):
            return "korean"
        if any("\u4e00" <= c <= "\u9fff" for c in combined):
            return "chinese"
        if any("\u0400" <= c <= "\u04ff" for c in combined):
            return "cyrillic"
        if any("\u0600" <= c <= "\u06ff" for c in combined):
            return "arabic"
        if any("\u0900" <= c <= "\u097f" for c in combined):
            return "devanagari"
        if any("\u0e00" <= c <= "\u0e7f" for c in combined):
            return "thai"

        return "latin"