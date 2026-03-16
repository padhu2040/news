import re
import json
import time
import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from supabase import create_client, Client
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="News Aggregator", page_icon="▪", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
    html, body, [class*="css"] { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SUPABASE INIT
# ---------------------------------------------------------------------------
try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    else:
        supabase = None
except Exception:
    st.warning("Supabase not connected. Weekly Radar will be disabled.")
    supabase = None

# ---------------------------------------------------------------------------
# GEMINI INIT (Dynamic Fallback)
# ---------------------------------------------------------------------------
GEMINI_FALLBACKS: list[str] = []

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    
    PREFERRED_KEYWORDS = [
        "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash", 
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"
    ]

    seen: set[str] = set()
    for keyword in PREFERRED_KEYWORDS:
        for m in available_models:
            if keyword in m and m not in seen:
                GEMINI_FALLBACKS.append(m)
                seen.add(m)

    for m in available_models:
        if m not in seen:
            GEMINI_FALLBACKS.append(m)
            seen.add(m)

    if not GEMINI_FALLBACKS:
        raise ValueError("No generateContent-capable Gemini models found.")

except Exception as e:
    st.error(f"Failed to initialise Gemini: {e}")

def _get_gemini_model(model_name: str):
    return genai.GenerativeModel(
        model_name,
        generation_config={"response_mime_type": "application/json"},
    )

# ---------------------------------------------------------------------------
# NLP MODEL
# ---------------------------------------------------------------------------
@st.cache_resource
def load_nlp_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp_model = load_nlp_model()

# ---------------------------------------------------------------------------
# RSS FEED SOURCES
# ---------------------------------------------------------------------------
RSS_FEEDS = {
    "Global": {
        "CNN": {"url": "http://rss.cnn.com/rss/cnn_topstories.rss", "bias": "Left"},
        "BBC": {"url": "http://feeds.bbci.co.uk/news/world/rss.xml", "bias": "Center"},
        "Fox": {"url": "http://feeds.foxnews.com/foxnews/world", "bias": "Right"},
    },
    "India": {
        "The Hindu": {"url": "https://news.google.com/rss/search?q=site:thehindu.com+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "bias": "Left"},
        "NDTV": {"url": "https://feeds.feedburner.com/ndtvnews-india-news", "bias": "Center"},
        "Times of India": {"url": "https://news.google.com/rss/search?q=site:timesofindia.indiatimes.com+india+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "bias": "Right"},
    },
    "Tamil Nadu": {
        "The Hindu TN": {"url": "https://news.google.com/rss/search?q=site:thehindu.com+tamil+nadu+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "bias": "Left"},
        "Indian Express TN": {"url": "https://news.google.com/rss/search?q=site:indianexpress.com+chennai+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "bias": "Center"},
        "TOI Chennai": {"url": "https://news.google.com/rss/search?q=site:timesofindia.indiatimes.com+chennai+when:1d&hl=en-IN&gl=IN&ceid=IN:en", "bias": "Right"},
    },
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _safe_parse_json(raw: str) -> dict | list:
    raw = raw.strip()
    raw = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0
