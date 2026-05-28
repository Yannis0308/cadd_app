"""
Shared UI component wrappers for CADD Streamlit pages.

All text-rendering functions inject CSS classes (defined in static/style.css)
so that font sizes, colors, and spacing are controlled centrally.  Pages should
import from here instead of calling st.markdown / st.title / st.caption directly
for user-facing text.
"""

import re

import streamlit as st


def _md_inline(text: str) -> str:
    """Convert **bold** and *italic* to HTML <strong> / <em> tags."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


# ═══════════════════════════════════════════════════════════════
# Page-level components
# ═══════════════════════════════════════════════════════════════

def page_title(text: str) -> None:
    """Large page title (<h1>).  Use once at the top of the page."""
    st.markdown(
        f'<h1 class="ui-page-title">{text}</h1>',
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Section heading (<h3>).  Use for subsection titles."""
    st.markdown(
        f'<h3 class="ui-section-header">{text}</h3>',
        unsafe_allow_html=True,
    )


def description(text: str) -> None:
    """Body / description paragraph with comfortable line-height."""
    text = _md_inline(text)
    st.markdown(
        f'<p class="ui-description">{text}</p>',
        unsafe_allow_html=True,
    )


def label(text: str) -> None:
    """Bold inline label, e.g. 'Pose ID:' or 'Optimized SMILES:'."""
    st.markdown(
        f'<span class="ui-label">{text}</span>',
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    """Small / footnote text."""
    text = _md_inline(text)
    st.markdown(
        f'<p class="ui-caption">{text}</p>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    """Consistent gradient divider."""
    st.markdown(
        '<hr class="ui-divider">',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# Sidebar components
# ═══════════════════════════════════════════════════════════════

def sidebar_title(text: str) -> None:
    """Sidebar heading."""
    st.sidebar.markdown(
        f'<h2 class="ui-sidebar-title">{text}</h2>',
        unsafe_allow_html=True,
    )


def sidebar_text(text: str) -> None:
    """Sidebar body text."""
    st.sidebar.markdown(
        f'<p class="ui-sidebar-text">{text}</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# Helper / composite components
# ═══════════════════════════════════════════════════════════════

def metric_row(items: list[tuple[str, str]]) -> None:
    """Render a row of st.metric widgets in equal-width columns.

    Args:
        items: List of (label, value) pairs, e.g. [("QED", "0.85"), ("LogP", "2.3")].
    """
    cols = st.columns(len(items))
    for col, (label_text, value) in zip(cols, items):
        col.metric(label_text, value)


def info_callout(text: str) -> None:
    """Info callout box."""
    st.info(text)


def warning_callout(text: str) -> None:
    """Warning callout box."""
    st.warning(text)


def error_callout(text: str) -> None:
    """Error callout box."""
    st.error(text)
