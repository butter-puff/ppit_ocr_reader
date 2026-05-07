"""
agent_act.py
Action layer of the HandwriteAI agent.
Applies image preprocessing and calls Azure OCR.
"""

import io
import time
import os

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# ── Azure client ──────────────────────────────────────────────────────────────
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY      = os.getenv("AZURE_VISION_KEY")

_client = ComputerVisionClient(
    AZURE_ENDPOINT,
    CognitiveServicesCredentials(AZURE_KEY)
)


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes, ops: list[str]) -> bytes:
    """
    Applies ordered preprocessing operations and returns new image bytes.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    for op in ops:
        if op == "brighten":
            img = ImageEnhance.Brightness(img).enhance(1.4)

        elif op == "darken":
            img = ImageEnhance.Brightness(img).enhance(0.75)

        elif op == "enhance_contrast":
            img = ImageEnhance.Contrast(img).enhance(1.8)

        elif op == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)

        elif op == "upscale":
            w, h = img.size
            img = img.resize((w * 2, h * 2), Image.LANCZOS)

        elif op == "binarize":
            gray = img.convert("L")
            # Otsu-like threshold via histogram
            hist = gray.histogram()
            total = sum(hist)
            sumB = 0; wB = 0; maximum = 0.0; threshold = 128
            total_sum = sum(i * hist[i] for i in range(256))
            for i in range(256):
                wB += hist[i]
                if wB == 0:
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += i * hist[i]
                mB = sumB / wB
                mF = (total_sum - sumB) / wF
                between = wB * wF * (mB - mF) ** 2
                if between >= maximum:
                    threshold = i
                    maximum = between
            img = gray.point(lambda p: 255 if p > threshold else 0).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Azure OCR ─────────────────────────────────────────────────────────────────

def run_ocr(image_bytes: bytes, language: str | None = None) -> dict:
    """
    Calls Azure Read API.
    Returns:
    {
        "text":       str,
        "lines":      list[dict],   # [{text, confidence}]
        "avg_confidence": float,
        "status":     "succeeded" | "failed",
        "error":      str | None
    }
    """
    try:
        stream = io.BytesIO(image_bytes)

        kwargs = {}
        if language:
            kwargs["language"] = language

        response = _client.read_in_stream(stream, raw=True, **kwargs)
        op_location = response.headers["Operation-Location"]
        op_id = op_location.split("/")[-1]

        # Poll until done
        for _ in range(30):
            result = _client.get_read_result(op_id)
            if result.status not in ["notStarted", "running"]:
                break
            time.sleep(0.5)

        if result.status != "succeeded":
            return {"text": "", "lines": [], "avg_confidence": 0.0,
                    "status": "failed", "error": f"Azure status: {result.status}"}

        lines = []
        for page in result.analyze_result.read_results:
            for line in page.lines:
                # Azure provides per-word confidence in newer API; fall back gracefully
                word_confs = []
                for word in (line.words or []):
                    if hasattr(word, "confidence") and word.confidence is not None:
                        word_confs.append(word.confidence)
                line_conf = sum(word_confs) / len(word_confs) if word_confs else None
                lines.append({"text": line.text, "confidence": line_conf})

        full_text = "\n".join(l["text"] for l in lines)

        # Average confidence (only over lines that have it)
        conf_values = [l["confidence"] for l in lines if l["confidence"] is not None]
        avg_conf = round(sum(conf_values) / len(conf_values), 3) if conf_values else None

        return {
            "text": full_text.strip(),
            "lines": lines,
            "avg_confidence": avg_conf,
            "status": "succeeded",
            "error": None,
        }

    except Exception as e:
        return {"text": "", "lines": [], "avg_confidence": 0.0,
                "status": "failed", "error": str(e)}


# ── Post-processing (structure raw OCR text) ──────────────────────────────────

def structure_text(raw_text: str) -> str:
    """
    Rule-based structuring — no LLM needed.
    Detects headings (short ALL-CAPS or title-case lines), bullet markers,
    and numbered lists; returns cleaned markdown-ish text.
    """
    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    structured = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            structured.append("")
            continue

        words = stripped.split()

        # Heading heuristic: short line (≤6 words) that is mostly uppercase
        upper_ratio = sum(1 for c in stripped if c.isupper()) / max(len(stripped), 1)
        if len(words) <= 6 and upper_ratio > 0.6 and len(stripped) > 3:
            structured.append(f"## {stripped}")
            continue

        # Bullet normalisation: -, *, •, ·, –, —
        if stripped[0] in "-*•·–—" and len(stripped) > 2:
            structured.append(f"- {stripped[1:].strip()}")
            continue

        # Numbered list: starts with digit + . or )
        if len(words) >= 1 and words[0][0].isdigit() and words[0][-1] in ".):":
            structured.append(stripped)
            continue

        structured.append(stripped)

    return "\n".join(structured).strip()
