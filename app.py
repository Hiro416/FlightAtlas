
import math
from collections import Counter
from datetime import datetime
from io import BytesIO

import airportsdata
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod


# =========================================================
# Streamlit config
# =========================================================

st.set_page_config(
    page_title="My Flight Atlas",
    layout="wide",
    page_icon="✈️",
)

st.markdown(
    """
    <style>
    :root {
        --atlas-ink: #172033;
        --atlas-muted: #64748b;
        --atlas-line: #d9e2ec;
        --atlas-panel: #ffffff;
        --atlas-surface: #f6f8fb;
        --atlas-accent: #c7372f;
        --atlas-blue: #1f5f8b;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 95, 139, 0.10), transparent 30rem),
            linear-gradient(180deg, #f8fbff 0%, #f4f7fb 46%, #ffffff 100%);
        color: var(--atlas-ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--atlas-line);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--atlas-line);
        border-radius: 8px;
        padding: 1rem 1rem 0.85rem;
        box-shadow: 0 10px 28px rgba(23, 32, 51, 0.06);
    }

    div[data-testid="stMetric"] label {
        color: var(--atlas-muted);
        font-weight: 700;
    }

    .atlas-hero {
        border-bottom: 1px solid var(--atlas-line);
        margin-bottom: 1.25rem;
        padding: 0.75rem 0 1.4rem;
    }

    .atlas-kicker {
        color: var(--atlas-blue);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .atlas-title {
        color: var(--atlas-ink);
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        font-weight: 850;
        letter-spacing: 0;
        line-height: 0.98;
        margin: 0.2rem 0 0.75rem;
    }

    .atlas-lead {
        color: #3d4b63;
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;

14m 1s作業しました





