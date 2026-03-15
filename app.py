import streamlit as st
import feedparser
import json
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from supabase import create_client, Client
from datetime import datetime, timedelta

# --- PAGE CONFIG & CSS (MINIMALIST) ---
st.set_page_config(page_title="News Aggregator", page_icon="▪", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    html, body, [class*="css"]  { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- INIT SUPABASE ---
try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    else:
        supabase = None
except Exception as e:
    st.warning("Supabase not connected. Weekly radar will be disabled.")
    supabase = None

# --- INIT GEMINI (DYNAMIC FALLBACK) ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_model = 'gemini-1.0-pro' 
    if available_models:
        target_model = available_models[0]
        for m in available_models:
            if 'gemini-1.5-flash' in m: target_model = m; break
            elif '1.5-pro' in m: target_model = m
            elif '1.0-pro' in m: target_model = m

    gemini_model = genai.GenerativeModel(
        target_model,
        generation_config={"response_mime_type": "application/json"}
    )
except Exception as e:
    st.error(f"Failed to initialize AI model. Details: {e}")
    gemini_model = None

# --- LOAD NLP MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

nlp_model = load_model()

# --- REGIONAL RSS SOURCES ---
RSS_FEEDS = {
    "Global": {
        "CNN": {"url": "http://rss.cnn.com/rss/cnn_topstories.rss", "bias": "Left"},
        "BBC": {"url": "http://feeds.bbci.co.uk/news/world/rss.xml", "bias": "Center"},
        "Fox": {"url": "http://feeds.foxnews.com/foxnews/world", "bias": "Right"}
    },
    "India": {
        "The Hindu": {"url": "https://www.thehindu.com/news/national/feeder/default.rss", "bias": "Left"},
        "NDTV": {"url": "https://feeds.feedburner.com/ndtvnews-india-news", "bias": "Center"},
        "Times of India": {"url": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms", "bias": "Right"}
    },
    "Tamil Nadu": {
        "The Hindu TN": {"url": "https://www.thehindu.com/news/states/tamil-nadu/feeder/default.rss", "bias": "Left"},
        "Indian Express TN": {"url": "https://indianexpress.com/section/cities/chennai/feed/", "bias": "Center"},
        "TOI Chennai": {"url": "https://timesofindia.indiatimes.com/rssfeeds/2950623.cms", "bias": "Right"}
    }
}

# --- AI JSON FETCHER ---
def generate_ai_metadata(titles):
    if not gemini_model: return {"error": "AI Offline"}
        
    headlines = "\n".join(titles)
    
    # Updated prompt to handle single sources gracefully
    json_schema = """
    {
        "summary": "Factual 2-sentence summary of the event.",
        "insights": { "key_fact": "One verified fact.", "discrepancy": "Media differences (or 'Single source narrative' if only one article)." },
        "topics": ["Tag1", "Tag2"] 
    }
    """
    prompt = f"Analyze these headlines:\n{headlines}\nReturn JSON strictly following this structure. For topics, provide 1 to 3 broad categories (e.g. Politics, Tech, Business, Crime, Election):\n{json_schema}"
    
    try:
        resp = gemini_model.generate_content(prompt)
        raw = resp.text.strip()
        
        if raw.startswith('
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1

Once this is running, try switching over to your **Weekly Radar** tab. You should now be able to see exactly what was saved in the database earlier, and the live feeds will now beautifully populate even if only a single source is running the story!
