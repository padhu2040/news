import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --- PAGE CONFIG (Mobile First) ---
st.set_page_config(page_title="News Aggregator", page_icon="📰", layout="centered")

# --- MINIMALIST CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    html, body, [class*="css"]  {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Style tweaks for the new cards */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE GEMINI ---
# Make sure your API key is correctly set in Streamlit Cloud Secrets!
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error("API Key not found. If running locally or in Codespaces, ensure your secrets are configured.")

# --- LOAD NLP MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- SOURCES DICTIONARY ---
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

# --- FETCH DATA ---
@st.cache_data(ttl=1800)
def fetch_news(region):
    articles = []
    feeds = RSS_FEEDS[region]
    for source_name, data in feeds.items():
        try:
            feed = feedparser.parse(data["url"])
            for entry in feed.entries[:12]:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "source": source_name,
                    "bias": data["bias"]
                })
        except Exception:
            continue
    return articles

# --- CLUSTERING LOGIC ---
@st.cache_data(ttl=1800)
def cluster_articles(articles):
    if not articles: return []
    titles = [art['title'] for art in articles]
    embeddings = model.encode(titles)
    
    clusters = []
    used_indices = set()
    
    for i in range(len(articles)):
        if i in used_indices: continue
        current_cluster = [articles[i]]
        used_indices.add(i)
        
        for j in range(i + 1, len(articles)):
            if j not in used_indices:
                cosine_score = util.cos_sim(embeddings[i], embeddings[j]).item()
                if cosine_score > 0.55:
                    current_cluster.append(articles[j])
                    used_indices.add(j)
                    
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
    return clusters

# --- AI SUMMARIZATION (UPDATED FOR DEBUGGING) ---
@st.cache_data(ttl=3600)
def generate_summary(titles):
    headlines = "\n".join(titles)
    prompt = f"Read these headlines about a single event:\n{headlines}\nWrite a strictly flat, neutral, 2-sentence summary of the event. Do not use sensationalism."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Now it will print the actual reason it failed to the screen!
        return f"⚠️ **AI Error:** {str(e)}"

# --- UI APP LAYOUT ---
st.title("News Aggregator")
st.markdown("▪️ Bias tracking and AI event clustering.")

tab1, tab2, tab3 = st.tabs(["Global", "India", "Tamil Nadu"])

def render_feed(region):
    with st.spinner("Analyzing feeds..."):
        articles = fetch_news(region)
        clusters = cluster_articles(articles)
        
    if not clusters:
        st.write("Not enough overlapping coverage at the moment.")
        return

    for cluster in clusters:
        # --- THE NEW CARD DESIGN ---
        with st.container(border=True):
            st.markdown(f"#### {cluster[0]['title']}")
            
            # AI Summary
            titles_only = [art['title'] for art in cluster]
            summary = generate_summary(titles_only)
            st.caption(f"{summary}")
            
            # Calculate Bias
            left_count = sum(1 for a in cluster if a['bias'] == 'Left')
            center_count = sum(1 for a in cluster if a['bias'] == 'Center')
            right_count = sum(1 for a in cluster if a['bias'] == 'Right')
            total = len(cluster)
            
            # Bias Bar Snapshot
            st.markdown(f"<div style='display:flex; height: 6px; width: 100%; border-radius: 3px; overflow: hidden; margin-bottom: 4px; margin-top: 12px;'>"
                        f"<div style='width: {(left_count/total)*100}%; background-color: #3b82f6;'></div>"
                        f"<div style='width: {(center_count/total)*100}%; background-color: #eab308;'></div>"
                        f"<div style='width: {(right_count/total)*100}%; background-color: #ef4444;'></div>"
                        f"</div>", unsafe_allow_html=True)
            
            # Hidden Sources Expander to keep the card clean
            with st.expander(f"View {total} Sources ({left_count} Left, {center_count} Center, {right_count} Right)"):
                for art in cluster:
                    color = "#3b82f6" if art['bias'] == "Left" else "#eab308" if art['bias'] == "Center" else "#ef4444"
                    st.markdown(f"<small><b><span style='color:{color}'>▪ {art['bias']}</span> | {art['source']}</b>: <a href='{art['link']}' style='color:#333; text-decoration:none;'>{art['title']}</a></small>", unsafe_allow_html=True)

with tab1: render_feed("Global")
with tab2: render_feed("India")
with tab3: render_feed("Tamil Nadu")
