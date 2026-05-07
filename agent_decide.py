"""
agent_decide.py
Decision layer of the HandwriteAI agent.
Chooses OCR strategy and preprocessing steps based on perception output.
"""


def decide_strategy(perception: dict) -> dict:
    """
    Consumes a perception dict and returns a strategy dict:
    {
        "preprocess":      bool,
        "preprocess_ops":  list[str],   # ordered ops to apply
        "ocr_language":    str | None,  # hint for Azure (None = auto)
        "retry_on_low":    bool,        # retry with enhanced preprocessing if confidence < threshold
        "confidence_warn_threshold": float,
        "explanation":     str,         # plain-language explanation shown to user
        "badge":           str,         # short status label
    }
    """
    ops = []
    explanation_parts = []
    ocr_language = None
    retry_on_low = False
    badge = "✅ Ready"

    q = perception["quality_score"]
    script = perception["script_hint"]
    issues = perception["issues"]

    # ── Preprocessing decisions ───────────────────────────────────────────────

    if perception.get("needs_preprocessing"):
        badge = "⚙️ Auto-enhancing"

    if perception["brightness"] < 60:
        ops.append("brighten")
        explanation_parts.append("brightening the image")

    if perception["brightness"] > 210:
        ops.append("darken")
        explanation_parts.append("reducing overexposure")

    if perception["contrast"] < 25:
        ops.append("enhance_contrast")
        explanation_parts.append("boosting contrast")

    if perception["sharpness"] < 80:
        ops.append("sharpen")
        explanation_parts.append("sharpening edges")

    if perception["width"] < 600 or perception["height"] < 600:
        ops.append("upscale")
        explanation_parts.append("upscaling to improve resolution")

    # Binarisation helps on very noisy scans
    if q < 0.5:
        ops.append("binarize")
        explanation_parts.append("binarizing to improve text-background separation")
        retry_on_low = True
        badge = "⚠️ Low quality — enhancing"

    # ── Script-based routing ─────────────────────────────────────────────────
    if script == "arabic_urdu":
        ocr_language = "ur"   # Azure Read API language hint
        explanation_parts.append("routing to Urdu/Arabic-optimised OCR model")

    # ── Confidence warn threshold ─────────────────────────────────────────────
    # Lower quality → we warn at a higher threshold
    confidence_warn_threshold = 0.75 if q < 0.65 else 0.60

    # ── Build human-readable explanation ────────────────────────────────────
    if explanation_parts:
        explanation = (
            f"Agent detected quality score **{q:.0%}** — "
            f"automatically {', '.join(explanation_parts)}."
        )
    else:
        explanation = f"Image quality looks good (score: **{q:.0%}**). Sending to OCR directly."

    if issues:
        explanation += "\n\n⚠️ " + "  \n⚠️ ".join(issues)

    return {
        "preprocess":                ops != [],
        "preprocess_ops":            ops,
        "ocr_language":              ocr_language,
        "retry_on_low":              retry_on_low,
        "confidence_warn_threshold": confidence_warn_threshold,
        "explanation":               explanation,
        "badge":                     badge,
        "quality_score":             q,
    }
