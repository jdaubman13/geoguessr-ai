import numpy as np
from PIL import Image
import easyocr


class OCREngine:
    def __init__(self):
        print("Loading EasyOCR...")
        self.reader_latin = easyocr.Reader(
            ["en", "fr", "de", "es", "pt", "pl"],
            gpu=False,
        )
        self.reader_cyrillic = easyocr.Reader(
            ["ru", "bg", "uk", "en"],
            gpu=False,
        )
        self.reader_ja = easyocr.Reader(["ja", "en"], gpu=False)
        self.reader_ko = easyocr.Reader(["ko", "en"], gpu=False)
        self.reader_th = easyocr.Reader(["th", "en"], gpu=False)
        print("EasyOCR ready.")

    def extract(self, image: Image.Image) -> tuple[list[str], str]:
        img_array = np.array(image)

        # Always run latin first as the default
        results = self.reader_latin.readtext(img_array)
        texts = [text for _, text, conf in results if conf > 0.3]
        script = self._detect_script(texts)

        # Re-run with the appropriate reader if non-latin detected
        if script == "cyrillic":
            results = self.reader_cyrillic.readtext(img_array)
            texts = [text for _, text, conf in results if conf > 0.3]
        elif script == "japanese":
            results = self.reader_ja.readtext(img_array)
            texts = [text for _, text, conf in results if conf > 0.3]
        elif script == "korean":
            results = self.reader_ko.readtext(img_array)
            texts = [text for _, text, conf in results if conf > 0.3]
        elif script == "thai":
            results = self.reader_th.readtext(img_array)
            texts = [text for _, text, conf in results if conf > 0.3]

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