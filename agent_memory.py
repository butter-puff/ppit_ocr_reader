"""
agent_memory.py
Short-term memory layer of the HandwriteAI agent.
Uses Streamlit session_state — cleared on refresh (no persistent storage by default).
"""

import streamlit as st
from datetime import datetime


def init_memory():
    """Call once at app startup to initialise session state."""
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = {
            "batch_history":    [],   # list of per-file result dicts
            "session_start":    datetime.now().isoformat(),
            "total_processed":  0,
            "total_warnings":   0,
            "user_corrections": [],   # files where user flagged output as wrong
        }


def record_result(filename: str, perception: dict, strategy: dict, ocr_result: dict):
    """Store a processed file's full result into memory."""
    mem = st.session_state.agent_memory
    entry = {
        "filename":       filename,
        "timestamp":      datetime.now().isoformat(),
        "quality_score":  perception["quality_score"],
        "script_hint":    perception["script_hint"],
        "issues":         perception["issues"],
        "preprocessing":  strategy["preprocess_ops"],
        "avg_confidence": ocr_result.get("avg_confidence"),
        "status":         ocr_result["status"],
        "text_length":    len(ocr_result.get("text", "")),
    }
    mem["batch_history"].append(entry)
    mem["total_processed"] += 1
    if perception["issues"]:
        mem["total_warnings"] += len(perception["issues"])


def record_correction(filename: str):
    """User flagged this file's output as incorrect."""
    st.session_state.agent_memory["user_corrections"].append({
        "filename":  filename,
        "timestamp": datetime.now().isoformat(),
    })


def get_session_summary() -> dict:
    """Return a summary dict for display."""
    mem = st.session_state.agent_memory
    history = mem["batch_history"]
    if not history:
        return {}

    conf_values = [e["avg_confidence"] for e in history
                   if e["avg_confidence"] is not None]
    avg_conf = round(sum(conf_values) / len(conf_values), 2) if conf_values else None

    quality_values = [e["quality_score"] for e in history]
    avg_quality = round(sum(quality_values) / len(quality_values), 2)

    return {
        "total_processed":  mem["total_processed"],
        "total_warnings":   mem["total_warnings"],
        "corrections":      len(mem["user_corrections"]),
        "avg_confidence":   avg_conf,
        "avg_quality":      avg_quality,
        "batch_history":    history,
    }
