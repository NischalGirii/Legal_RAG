import io
import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

def preprocess_devanagari_image(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    # Deskew
    coords = np.column_stack(np.where(gray > 128))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if angle != 0:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return Image.fromarray(thresh)

def resolve_ocr_lang_flag() -> str:
    try:
        installed = pytesseract.get_languages(config="")
        priority = ["nep", "script/Devanagari", "eng"]
        selected = [lang for lang in priority if lang in installed]
        return "+".join(selected) if selected else "eng"
    except Exception:
        return "eng"

def ocr_scanned_page(page: fitz.Page, lang_flag: str) -> str:
    pix = page.get_pixmap(dpi=300)
    raw_pil = Image.open(io.BytesIO(pix.tobytes("png")))
    processed_pil = preprocess_devanagari_image(raw_pil)
    config = r'--oem 1 --psm 6'
    return pytesseract.image_to_string(processed_pil, lang=lang_flag, config=config)