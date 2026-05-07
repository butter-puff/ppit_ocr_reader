"""
agent_perceive.py
Perception layer of the HandwriteAI agent.
Analyses image quality and detects script type without any external API.
"""

import io
import math
from PIL import Image, ImageFilter, ImageStat


# ── Quality thresholds ────────────────────────────────────────────────────────
BLUR_THRESHOLD      = 80    # Laplacian variance; below = blurry
MIN_WIDTH           = 400   # pixels
MIN_HEIGHT          = 400
BRIGHTNESS_LOW      = 40    # 0-255
BRIGHTNESS_HIGH     = 220
CONTRAST_MIN        = 20    # std-dev of grayscale


def _laplacian_variance(gray_img: Image.Image) -> float:
    """Higher = sharper image. Uses FIND_EDGES as a Laplacian proxy."""
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return stat.var[0]


def _brightness_and_contrast(gray_img: Image.Image):
    stat = ImageStat.Stat(gray_img)
    return stat.mean[0], stat.stddev[0]


def perceive_image(image_bytes: bytes) -> dict:
    """
    Returns a perception dict consumed by the Decide layer:
    {
        "quality_score": 0.0-1.0,
        "issues": [...],          # human-readable warnings
        "width": int,
        "height": int,
        "brightness": float,
        "contrast": float,
        "sharpness": float,
        "needs_preprocessing": bool,
        "script_hint": "latin" | "arabic_urdu" | "mixed" | "unknown"
    }
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    gray = img.convert("L")

    width, height = img.size
    sharpness   = _laplacian_variance(gray)
    brightness, contrast = _brightness_and_contrast(gray)

    issues = []
    penalty = 0.0

    # ── Size check ───────────────────────────────────────────────────────────
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        issues.append(f"Low resolution ({width}×{height}px) — results may be inaccurate.")
        penalty += 0.25

    # ── Blur check ───────────────────────────────────────────────────────────
    if sharpness < BLUR_THRESHOLD:
        issues.append("Image appears blurry — consider re-scanning.")
        penalty += 0.30

    # ── Brightness check ─────────────────────────────────────────────────────
    if brightness < BRIGHTNESS_LOW:
        issues.append("Image is too dark — OCR accuracy may suffer.")
        penalty += 0.20
    elif brightness > BRIGHTNESS_HIGH:
        issues.append("Image is overexposed — text may be washed out.")
        penalty += 0.15

    # ── Contrast check ───────────────────────────────────────────────────────
    if contrast < CONTRAST_MIN:
        issues.append("Low contrast detected — text and background are hard to separate.")
        penalty += 0.20

    quality_score = max(0.0, round(1.0 - penalty, 2))

    # ── Script heuristic (aspect-ratio + pixel density proxy) ────────────────
    # Real script detection needs an ML model; this is a lightweight heuristic.
    # Arabic/Urdu text tends to be right-to-left with connected strokes;
    # we check pixel density in horizontal vs vertical bands as a weak proxy.
    script_hint = _estimate_script(gray)

    return {
        "quality_score":       quality_score,
        "issues":              issues,
        "width":               width,
        "height":              height,
        "brightness":          round(brightness, 1),
        "contrast":            round(contrast, 1),
        "sharpness":           round(sharpness, 1),
        "needs_preprocessing": quality_score < 0.65,
        "script_hint":         script_hint,
    }


def _estimate_script(gray_img: Image.Image) -> str:
    """
    Very lightweight heuristic:
    Arabic/Urdu strokes cluster differently in horizontal slices.
    Returns 'arabic_urdu', 'latin', or 'unknown'.
    """
    try:
        w, h = gray_img.size
        # Sample a central horizontal strip
        strip_h = max(10, h // 10)
        top_strip    = gray_img.crop((0, h//4,        w, h//4 + strip_h))
        middle_strip = gray_img.crop((0, h//2,        w, h//2 + strip_h))
        bottom_strip = gray_img.crop((0, 3*h//4,      w, 3*h//4 + strip_h))

        def dark_ratio(strip):
            stat = ImageStat.Stat(strip)
            return stat.mean[0]

        ratios = [dark_ratio(s) for s in [top_strip, middle_strip, bottom_strip]]
        variance = max(ratios) - min(ratios)

        # Arabic/Urdu tends to have more uniform horizontal density (connected script)
        if variance < 8:
            return "arabic_urdu"
        elif variance < 20:
            return "mixed"
        else:
            return "latin"
    except Exception:
        return "unknown"