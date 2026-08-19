import io
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

def preprocess_devanagari_image(pil_img: Image.Image) -> Image.Image:
    """Enhanced for Devanagari: Rescales and uses adaptive threshold to preserve matras."""
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Scale up the image to make small matras clearer to Tesseract[cite: 1]
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Lighter denoising to preserve delicate character curves[cite: 1]
    denoised = cv2.fastNlMeansDenoising(gray, h=5)
    
    # Adaptive Thresholding works better for Devanagari than global Otsu[cite: 1]
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    
    return Image.fromarray(thresh)

def resolve_ocr_lang_flag() -> str:
    """Determines the best available language packs for pytesseract[cite: 1]."""
    try:
        installed = pytesseract.get_languages(config="")
        priority = ["nep", "script/Devanagari", "eng"]
        selected = [lang for lang in priority if lang in installed]
        return "+".join(selected) if selected else "eng"
    except Exception:
        return "eng"

def ocr_scanned_page(page: fitz.Page, lang_flag: str) -> str:
    """Renders page at 300 DPI and performs OpenCV binarized Tesseract OCR[cite: 1]."""
    pix = page.get_pixmap(dpi=300)
    raw_pil = Image.open(io.BytesIO(pix.tobytes("png")))
    processed_pil = preprocess_devanagari_image(raw_pil)
    
    # PSM 3 (Fully automatic page segmentation) and OEM 1 (LSTM neural net)[cite: 1]
    config = r'--oem 1 --psm 3'
    return pytesseract.image_to_string(processed_pil, lang=lang_flag, config=config)