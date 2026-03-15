import streamlit as st
import feedparser
import json
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="News Aggregator", page_icon="📰", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    html, body, [class*="css"]  {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE GEMINI (DYNAMIC FALLBACK) ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 1. Ask Google what models this specific API key can use
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    # 2. Set a safe default
    target_model = 'gemini-1.0-pro' 
    
    # 3. Scan the available models and grab the smartest one available
    if available_models:
        target_model = available_models[0] # Absolute fallback
        for m in available_models:
            if 'gemini-1.5-flash' in m: 
                target_model = m
                break # Stop searching, we found the best one!
            elif '1.5-pro' in m: 
                target_model = m
            elif '1.0-pro' in m:
                target_model = m

    # 4. Initialize the model using the guaranteed valid name AND the JSON config
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

model = load_model()

# --- SOURCES DICTIONARY ---
RSS_FEEDS = {
    "Global": {
        "CNN": {"url": "http://rss.cnn.com/rss/cnn_topstories.rss", "bias": "Left"},
        "BBC": {"url": "http://feeds.bbci.co.uk/news/world/rss.xml", "bias": "Center"},
        "Fox": {"url": "http://feeds.foxnews.com/foxnews/world", "bias": "Right"}
    }
}

# --- FETCH & CLUSTER DATA ---
@st.cache_data(ttl=1800)
def fetch_and_cluster():
    articles = []
    for source_name, data in RSS_FEEDS["Global"].items():
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

# --- AI JSON FETCHER ---
@st.cache_data(ttl=3600)
def generate_structured_summary(titles):
    if not gemini_model:
        return {"error": "AI Model failed to initialize."}
        
    headlines = "\n".join(titles)
    
    json_schema = """
    {
        "summary": "A strictly factual, neutral 2-sentence summary of the event.",
        "insights": {
            "key_fact": "One verified fact present across the headlines.",
            "discrepancy": "Any difference in how the headlines frame the event (or write 'Coverage is consistent' if none)."
        }
    }
    """
    
    prompt = f"Analyze these news headlines about a single event:\n{headlines}\n\nReturn a JSON object strictly following this structure:\n"
    prompt += json_schema
    
    try:
        resp = gemini_model.generate_content(prompt)
        raw_text = resp.text.strip()
        if raw_text.startswith("```json"): 
            raw_text = raw_text[7:-3]
        elif raw_text.startswith("```"): 
            raw_text = raw_text[3:-3]
            
        return json.loads(raw_text)
    except Exception as e:
        return {"error": str(e)}

# --- UI APP LAYOUT ---
st.title("News Aggregator")
st.markdown("▪️ Bias tracking and AI event clustering.")

with st.spinner("Analyzing feeds..."):
    clusters = fetch_and_cluster()

if not clusters:
    st.write("Not enough overlapping coverage at the moment.")
else:
    for cluster in clusters:
        titles_only = [art['title'] for art in cluster]
        ai_data = generate_structured_summary(titles_only)
        
        # Calculate Bias
        left_count = sum(1 for a in cluster if a['bias'] == 'Left')
        center_count = sum(1 for a in cluster if a['bias'] == 'Center')
        right_count = sum(1 for a in cluster if a['bias'] == 'Right')
        total = len(cluster)
        
        sources_html = ""
        for art in cluster:
            color = "#3b82f6" if art['bias'] == "Left" else "#eab308" if art['bias'] == "Center" else "#ef4444"
            sources_html += f"<div style='margin-bottom: 8px;'><small><b><span style='color:{color}'>▪ {art['bias']}</span> | {art['source']}</b>: <a href='{art['link']}' style='color:#333; text-decoration:none;'>{art['title']}</a></small></div>"

        # --- HTML RENDERER ---
        if "error" not in ai_data:
            html_card = f"""
<style>
.custom-card {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; }}
.insight-tag {{ color: #4b5563; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 12px; }}
.bias-bar-container {{ display: flex; height: 6px; width: 100%; border-radius: 3px; overflow: hidden; margin-top: 16px; margin-bottom: 16px; }}
.sources-box {{ background: #f9fafb; padding: 12px; border-radius: 6px; margin-top: 16px; border: 1px solid #f3f4f6; }}
</style>

<div class="custom-card">
<h3 style="margin-top: 0; color: #111827;">{cluster[0]['title']}</h3>
<p style="color: #374151; font-size: 15px; line-height: 1.5;">{ai_data['summary']}</p>

<div class="insight-tag">Key Fact</div>
<p style="margin: 4px 0 0 0; font-size: 14px; color: #4b5563;">{ai_data['insights']['key_fact']}</p>

<div class="insight-tag">Media Discrepancy</div>
<p style="margin: 4px 0 0 0; font-size: 14px; color: #4b5563;">{ai_data['insights']['discrepancy']}</p>

<div class="bias-bar-container">
    <div style="width: {(left_count/total)*100}%; background-color: #3b82f6;"></div>
    <div style="width: {(center_count/total)*100}%; background-color: #eab308;"></div>
    <div style="width: {(right_count/total)*100}%; background-color: #ef4444;"></div>
</div>

<details>
    <summary style="cursor: pointer; color: #6b7280; font-size: 14px; font-weight: 500;">View {total} Sources ({left_count} Left, {center_count} Center, {right_count} Right)</summary>
    <div class="sources-box">
        {sources_html}
    </div>
</details>
</div>
"""
            st.markdown(html_card, unsafe_allow_html=True)
        else:
            st.error(f"AI Error for this cluster: {ai_data['error']}")
