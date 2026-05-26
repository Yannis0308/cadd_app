"""Shared CSS loader for all CADD Streamlit pages.

Usage:
    from utils.style_loader import load_css
    load_css()

Must be called after st.set_page_config().
"""

from pathlib import Path

import streamlit as st


def load_css() -> None:
    """Inject shared style.css into the current Streamlit page."""
    css_file = Path(__file__).resolve().parent.parent / "static" / "style.css"
    if css_file.exists():
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
