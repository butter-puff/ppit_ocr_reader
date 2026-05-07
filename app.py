"""
app.py  —  HandwriteAI Phase 2: Agentic Transformation
Orchestrates the Perceive → Decide → Act → Feedback agent loop.
"""

import streamlit as st
from PIL import Image

from agent_perceive import perceive_image
from agent_decide  import decide_strategy
from agent_act     import preprocess_image, run_ocr, structure_text
from agent_memory  import init_memory, record_result, record_correction, get_session_summary

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HandwriteAI Agent",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #0d1117;
}

/* Agent badge pill */
.agent-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}
.badge-good    { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
.badge-warn    { background: #3a2a10; color: #fbbf24; border: 1px solid #92400e; }
.badge-enhance { background: #1a2a3a; color: #60a5fa; border: 1px solid #1e3a5f; }

/* Agent decision card */
.decision-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #388bfd;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Metric row */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 8px 0;
}
.metric-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    color: #8b949e;
}
.metric-chip span { color: #e6edf3; font-weight: 600; }

/* Quality bar */
.quality-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.quality-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6e7681;
    margin: 24px 0 8px;
}

/* Override Streamlit expander for dark mode */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Initialise agent memory
# ─────────────────────────────────────────────────────────────────────────────
init_memory()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 HandwriteAI Agent")
st.markdown(
    "An agentic OCR system that **perceives** image quality, **decides** a strategy, "
    "**acts** with preprocessing + OCR, and **learns** from your feedback."
)

col_upload, col_info = st.columns([3, 1])

with col_info:
    summary = get_session_summary()
    if summary:
        st.markdown('<div class="section-header">Session Memory</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-chip">Processed <span>{summary["total_processed"]}</span></div>'
            f'<div class="metric-chip">Warnings <span>{summary["total_warnings"]}</span></div>'
            f'<div class="metric-chip">Corrections <span>{summary["corrections"]}</span></div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────
with col_upload:
    st.info(
        "🔒 **Privacy notice:** Your images are sent to Microsoft Azure for OCR. "
        "Images are processed in-memory only and are never stored by us or Azure beyond the API call.",
        icon=None
    )

uploaded_files = st.file_uploader(
    "Upload handwritten note images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploader"
)

if not uploaded_files:
    st.markdown(
        '<div class="section-header">How it works</div>',
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, desc in [
        (c1, "👁️", "Perceive", "Checks quality, resolution, brightness, contrast, script type"),
        (c2, "🧠", "Decide",   "Chooses preprocessing steps and OCR routing automatically"),
        (c3, "⚙️", "Act",      "Enhances image if needed, runs Azure OCR, structures output"),
        (c4, "🔁", "Feedback", "You flag errors — agent learns confidence patterns per session"),
    ]:
        with col:
            st.markdown(f"**{icon} {label}**")
            st.caption(desc)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────────────────────────────────────
st.success(f"{len(uploaded_files)} image(s) uploaded")
preview_cols = st.columns(min(len(uploaded_files), 4))
for i, f in enumerate(uploaded_files):
    f.seek(0)
    with preview_cols[i % 4]:
        st.image(Image.open(f), caption=f.name, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Run Agent
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🚀 Run Agent", type="primary"):

    all_results = []   # (filename, perception, strategy, ocr_result, structured_text)
    progress = st.progress(0.0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files):
        file.seek(0)
        raw_bytes = file.read()

        # ── PERCEIVE ──────────────────────────────────────────────────────────
        status_text.markdown(f"👁️ **Perceiving** `{file.name}`…")
        perception = perceive_image(raw_bytes)

        # ── DECIDE ───────────────────────────────────────────────────────────
        status_text.markdown(f"🧠 **Deciding strategy** for `{file.name}`…")
        strategy = decide_strategy(perception)

        # ── ACT: preprocess ──────────────────────────────────────────────────
        if strategy["preprocess"]:
            status_text.markdown(f"⚙️ **Preprocessing** `{file.name}`…")
            try:
                processed_bytes = preprocess_image(raw_bytes, strategy["preprocess_ops"])
            except Exception:
                processed_bytes = raw_bytes   # fall back to original
        else:
            processed_bytes = raw_bytes

        # ── ACT: OCR ─────────────────────────────────────────────────────────
        status_text.markdown(f"📖 **Running OCR** on `{file.name}`…")
        ocr_result = run_ocr(processed_bytes, language=strategy["ocr_language"])

        # Retry with original if preprocessed result is worse (empty output)
        if strategy["retry_on_low"] and not ocr_result["text"] and processed_bytes != raw_bytes:
            status_text.markdown(f"🔄 **Retrying** with original `{file.name}`…")
            ocr_result = run_ocr(raw_bytes, language=strategy["ocr_language"])

        # ── ACT: structure ───────────────────────────────────────────────────
        structured = structure_text(ocr_result["text"])

        # ── FEEDBACK: record to memory ────────────────────────────────────────
        record_result(file.name, perception, strategy, ocr_result)

        all_results.append((file.name, perception, strategy, ocr_result, structured))
        progress.progress((idx + 1) / len(uploaded_files))

    status_text.empty()
    st.success("✅ Agent completed all files!")

    # ── Store results in session state for feedback UI ────────────────────────
    st.session_state["last_results"] = all_results


# ─────────────────────────────────────────────────────────────────────────────
# Display results (persisted in session state so feedback buttons work)
# ─────────────────────────────────────────────────────────────────────────────
if "last_results" in st.session_state:
    all_results = st.session_state["last_results"]

    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    combined_text = ""

    for filename, perception, strategy, ocr_result, structured in all_results:
        q = perception["quality_score"]
        bar_color = "#4ade80" if q >= 0.75 else "#fbbf24" if q >= 0.5 else "#f87171"
        bar_w = int(q * 100)

        with st.expander(f"📄 {filename}  —  {strategy['badge']}", expanded=True):

            # ── Agent decision summary ────────────────────────────────────────
            st.markdown(
                f'<div class="decision-card">🧠 <b>Agent Decision</b><br>{strategy["explanation"]}</div>',
                unsafe_allow_html=True
            )

            # ── Metrics row ──────────────────────────────────────────────────
            conf_display = (
                f"{ocr_result['avg_confidence']:.0%}"
                if ocr_result.get("avg_confidence") is not None
                else "N/A"
            )
            ops_display = ", ".join(strategy["preprocess_ops"]) or "none"

            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-chip">Quality <span>{q:.0%}</span></div>'
                f'<div class="metric-chip">Confidence <span>{conf_display}</span></div>'
                f'<div class="metric-chip">Script <span>{perception["script_hint"]}</span></div>'
                f'<div class="metric-chip">Preprocessing <span>{ops_display}</span></div>'
                f'</div>'
                f'<div class="quality-bar-bg">'
                f'<div class="quality-bar-fill" style="width:{bar_w}%;background:{bar_color};"></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── Confidence warning ───────────────────────────────────────────
            avg_conf = ocr_result.get("avg_confidence")
            if avg_conf is not None and avg_conf < strategy["confidence_warn_threshold"]:
                st.warning(
                    f"⚠️ OCR confidence is low ({avg_conf:.0%}). "
                    "Please review the output carefully — some words may be incorrect."
                )

            if ocr_result["status"] == "failed":
                st.error(f"OCR failed: {ocr_result['error']}")
            else:
                # ── Tabs: structured vs raw ──────────────────────────────────
                tab_struct, tab_raw = st.tabs(["📋 Structured Output", "📝 Raw OCR"])

                with tab_struct:
                    st.markdown(structured or "_No text detected._")

                with tab_raw:
                    st.text(ocr_result["text"] or "(empty)")

                # ── Feedback button ──────────────────────────────────────────
                col_dl, col_fb = st.columns([3, 1])
                with col_dl:
                    st.download_button(
                        "📥 Download TXT",
                        structured or ocr_result["text"],
                        file_name=filename.rsplit(".", 1)[0] + "_notes.txt",
                        mime="text/plain",
                        key=f"dl_{filename}"
                    )
                with col_fb:
                    if st.button("🚩 Flag as incorrect", key=f"flag_{filename}"):
                        record_correction(filename)
                        st.warning("Flagged — thank you! Agent memory updated.")

                combined_text += f"\n{'='*60}\n{filename}\n{'='*60}\n{structured}\n\n"

    # ── Batch download ────────────────────────────────────────────────────────
    if combined_text.strip():
        st.download_button(
            "📦 Download All as TXT",
            combined_text,
            "all_notes.txt",
            "text/plain"
        )

    # ── Session memory summary ────────────────────────────────────────────────
    summary = get_session_summary()
    if summary and summary["total_processed"] > 0:
        with st.expander("🧠 Agent Session Memory", expanded=False):
            st.markdown(f"""
**Files processed this session:** {summary['total_processed']}  
**Total quality warnings:** {summary['total_warnings']}  
**User corrections flagged:** {summary['corrections']}  
**Average image quality:** {summary['avg_quality']:.0%}  
**Average OCR confidence:** {f"{summary['avg_confidence']:.0%}" if summary['avg_confidence'] else 'N/A'}

*Session memory is cleared on page refresh. No data is stored permanently.*
            """)

            st.markdown("**Per-file log:**")
            for entry in summary["batch_history"]:
                conf_str = f"{entry['avg_confidence']:.0%}" if entry['avg_confidence'] else "N/A"
                st.caption(
                    f"• `{entry['filename']}` — quality {entry['quality_score']:.0%}, "
                    f"confidence {conf_str}, script: {entry['script_hint']}, "
                    f"preprocessing: {entry['preprocessing'] or 'none'}"
                )
