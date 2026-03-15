import streamlit as st
import feedparser
import json
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from supabase import create_client, Client

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="News Aggregator", page_icon="📰", layout="centered")

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
            if 'gemini-1.5-flash' in m: 
                target_model = m
                break
            elif '1.5-pro' in m: 
                target_model = m
            elif '1.0-pro' in m: 
                target_model = m

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
        "CNN": {"url": "[http://rss.cnn.com/rss/cnn_topstories.rss](http://rss.cnn.com/rss/cnn_topstories.rss)", "bias": "Left"},
        "BBC": {"url": "[http://feeds.bbci.co.uk/news/world/rss.xml](http://feeds.bbci.co.uk/news/world/rss.xml)", "bias": "Center"},
        "Fox": {"url": "[http://feeds.foxnews.com/foxnews/world](http://feeds.foxnews.com/foxnews/world)", "bias": "Right"}
    },
    "India": {
        "The Hindu": {"url": "[https://www.thehindu.com/news/national/feeder/default.rss](https://www.thehindu.com/news/national/feeder/default.rss)", "bias": "Left"},
        "NDTV": {"url": "[https://feeds.feedburner.com/ndtvnews-india-news](https://feeds.feedburner.com/ndtvnews-india-news)", "bias": "Center"},
        "Times of India": {"url": "[https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms](https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms)", "bias": "Right"}
    },
    "Tamil Nadu": {
        "The Hindu TN": {"url": "[https://www.thehindu.com/news/states/tamil-nadu/feeder/default.rss](https://www.thehindu.com/news/states/tamil-nadu/feeder/default.rss)", "bias": "Left"},
        "Indian Express TN": {"url": "[https://indianexpress.com/section/cities/chennai/feed/](https://indianexpress.com/section/cities/chennai/feed/)", "bias": "Center"},
        "TOI Chennai": {"url": "[https://timesofindia.indiatimes.com/rssfeeds/2950623.cms](https://timesofindia.indiatimes.com/rssfeeds/2950623.cms)", "bias": "Right"}
    }
}

# --- AI JSON FETCHER (TOPIC CLOUD READY) ---
def generate_ai_metadata(titles):
    if not gemini_model: 
        return {"error": "AI Offline"}
        
    headlines = "\n".join(titles)
    
    json_schema = """
    {
        "summary": "Factual 2-sentence summary of the event.",
        "insights": { "key_fact": "One verified fact.", "discrepancy": "Media differences or 'Coverage is consistent'." },
        "topics": ["Tag1", "Tag2"] 
    }
    """
    prompt = f"Analyze these headlines:\n{headlines}\nReturn JSON strictly following this structure. For topics, provide 1 to 3 broad categories (e.g. Politics, Tech, Business, Crime, Election):\n{json_schema}"
    
    try:
        resp = gemini_model.generate_content(prompt)
        raw = resp.text.strip()
        
        # FIX: Using single quotes to prevent markdown parser crashes during copy-paste
        if raw.startswith('```json'): 
            raw = raw[7:-3]
        elif raw.startswith('```'): 
            raw = raw[3:-3]
            
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# --- FETCH, CLUSTER & SAVE (TODAY'S NEWS) ---
@st.cache_data(ttl=3600)
def process_live_news(region):
    articles = []
    for source_name, data in RSS_FEEDS[region].items():
        try:
            feed = feedparser.parse(data["url"])
            for entry in feed.entries[:12]:
                articles.append({"title": entry.title, "link": entry.link, "source": source_name, "bias": data["bias"]})
        except: 
            continue
            
    if not articles: 
        return []
    
    titles = [art['title'] for art in articles]
    embeddings = nlp_model.encode(titles)
    
    processed_events = []
    used_indices = set()
    
    for i in range(len(articles)):
        if i in used_indices: continue
        cluster = [articles[i]]
        used_indices.add(i)
        for j in range(i + 1, len(articles)):
            if j not in used_indices:
                if util.cos_sim(embeddings[i], embeddings[j]).item() > 0.55:
                    cluster.append(articles[j])
                    used_indices.add(j)
                    
        if len(cluster) > 1:
            # Analyze with Gemini
            ai_data = generate_ai_metadata([art['title'] for art in cluster])
            if "error" in ai_data: continue
                
            event_title = cluster[0]['title']
            left_c = sum(1 for a in cluster if a['bias'] == 'Left')
            center_c = sum(1 for a in cluster if a['bias'] == 'Center')
            right_c = sum(1 for a in cluster if a['bias'] == 'Right')
            
            event_record = {
                "region": region,
                "title": event_title,
                "summary": ai_data.get('summary', 'Summary not available'),
                "key_fact": ai_data.get('insights', {}).get('key_fact', 'N/A'),
                "discrepancy": ai_data.get('insights', {}).get('discrepancy', 'N/A'),
                "topics": ai_data.get('topics', []),
                "left_count": left_c, "center_count": center_c, "right_count": right_c,
                "total_articles": len(cluster),
                "sources_json": cluster
            }
            
            processed_events.append(event_record)
            
            # Save to Supabase (Deduplication Check)
            if supabase:
                try:
                    existing = supabase.table("news_events").select("id").eq("title", event_title).execute()
                    if not existing.data:
                        supabase.table("news_events").insert(event_record).execute()
                except Exception as e:
                    print(f"Supabase error: {e}")
                    
    return processed_events

# --- FETCH WEEKLY RADAR (FROM DB) ---
@st.cache_data(ttl=600)
def fetch_weekly_radar(region):
    if not supabase: 
        return []
    try:
        res = supabase.table("news_events").select("*").eq("region", region).order("total_articles", desc=True).limit(20).execute()
        return res.data
    except:
        return []

# --- HTML CARD RENDERER ---
def render_event_card(event):
    topics_html = "".join([f"<span style='display:inline-block; background:#f3f4f6; color:#374151; padding:4px 10px; border-radius:16px; font-size:12px; margin-right:6px; font-weight:500;'>{t}</span>" for t in event.get('topics', [])])
    
    sources_html = ""
    for art in event['sources_json']:
        color = "#3b82f6" if art['bias'] == "Left" else "#eab308" if art['bias'] == "Center" else "#ef4444"
        sources_html += f"<div style='margin-bottom:8px;'><small><b><span style='color:{color}'>▪ {art['bias']}</span> | {art['source']}</b>: <a href='{art['link']}' style='color:#333; text-decoration:none;'>{art['title']}</a></small></div>"

    html_card = f"""
<style>
.custom-card {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; }}
.insight-tag {{ color: #4b5563; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 12px; }}
.bias-bar-container {{ display: flex; height: 6px; width: 100%; border-radius: 3px; overflow: hidden; margin-top: 16px; margin-bottom: 16px; }}
.sources-box {{ background: #f9fafb; padding: 12px; border-radius: 6px; margin-top: 16px; border: 1px solid #f3f4f6; }}
</style>

<div class="custom-card">
    {topics_html}
    <h3 style="margin-top: 12px; color: #111827;">{event['title']}</h3>
    <p style="color: #374151; font-size: 15px; line-height: 1.5;">{event['summary']}</p>
    
    <div class="insight-tag">Key Fact</div>
    <p style="margin: 4px 0 0 0; font-size: 14px; color: #4b5563;">{event['key_fact']}</p>
    
    <div class="insight-tag">Media Discrepancy</div>
    <p style="margin: 4px 0 0 0; font-size: 14px; color: #4b5563;">{event['discrepancy']}</p>
    
    <div class="bias-bar-container">
        <div style="width: {(event['left_count']/max(1, event['total_articles']))*100}%; background-color: #3b82f6;"></div>
        <div style="width: {(event['center_count']/max(1, event['total_articles']))*100}%; background-color: #eab308;"></div>
        <div style="width: {(event['right_count']/max(1, event['total_articles']))*100}%; background-color: #ef4444;"></div>
    </div>
    
    <details>
        <summary style="cursor:pointer; color:#6b7280; font-size:14px; font-weight:500;">View {event['total_articles']} Sources</summary>
        <div class="sources-box">{sources_html}</div>
    </details>
</div>
"""
    st.markdown(html_card, unsafe_allow_html=True)

# --- UI APP LAYOUT ---
st.title("News Aggregator")
st.markdown("▪️ Bias tracking and AI event clustering.")

tab_global, tab_india, tab_tn = st.tabs(["🌍 Global", "🇮🇳 India", "🐅 Tamil Nadu"])

def build_view(region):
    view_mode = st.radio("Timeframe", ["Top News Today (Live)", "Weekly Radar (Database)"], horizontal=True, label_visibility="collapsed", key=f"radio_{region}")
    
    with st.spinner(f"Loading {region} news..."):
        if "Live" in view_mode:
            events = process_live_news(region)
        else:
            events = fetch_weekly_radar(region)

    if not events:
        if "Live" in view_mode:
            st.info("Not enough overlapping coverage on live feeds right now.")
        else:
            st.info("No saved stories found in the database. Ensure Supabase is configured and data has been scraped.")
        return

    all_topics = []
    for ev in events:
        all_topics.extend(ev.get('topics', []))
    unique_topics = sorted(list(set(all_topics)))
    
    selected_topics = st.multiselect("☁️ Filter by Topic", unique_topics, placeholder="Select a topic to filter...", key=f"filter_{region}")
    st.divider()

    for event in events:
        if selected_topics:
            event_topics = event.get('topics', [])
            if not any(topic in selected_topics for topic in event_topics):
                continue
                
        render_event_card(event)

with tab_global: build_view("Global")
with tab_india: build_view("India")
with tab_tn: build_view("Tamil Nadu")
