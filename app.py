import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util

# Set up the page
st.set_page_config(page_title="News Aggregator", page_icon="📰", layout="wide")

st.title("📰 Open News Aggregator")
st.markdown("Tracking bias and clustering global events using AI.")

# --- LOAD AI MODEL ---
# @st.cache_resource ensures the ML model only downloads once when the server starts
@st.cache_resource
def load_model():
    # This downloads a small, fast NLP model from Hugging Face
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- REAL DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_news():
    rss_feeds = {
        "Left": "http://rss.cnn.com/rss/cnn_topstories.rss",
        "Center": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Right": "http://feeds.foxnews.com/foxnews/world"
    }
    
    articles = []
    for bias, url in rss_feeds.items():
        feed = feedparser.parse(url)
        # We grab 15 articles from each source to increase the chance of overlap
        for entry in feed.entries[:15]: 
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "bias": bias
            })
    return articles

# --- AI CLUSTERING LOGIC ---
@st.cache_data(ttl=3600)
def cluster_articles(articles):
    if not articles:
        return []
        
    titles = [art['title'] for art in articles]
    # Convert text titles into mathematical embeddings
    embeddings = model.encode(titles) 
    
    clusters = []
    used_indices = set()
    
    for i in range(len(articles)):
        if i in used_indices:
            continue
            
        # Start a new cluster with the current article
        current_cluster = [articles[i]]
        used_indices.add(i)
        
        # Compare this article with all remaining articles
        for j in range(i + 1, len(articles)):
            if j not in used_indices:
                # Calculate how similar the two headlines are (0.0 to 1.0)
                cosine_score = util.cos_sim(embeddings[i], embeddings[j]).item()
                
                # If they are more than 55% similar, group them together
                if cosine_score > 0.55: 
                    current_cluster.append(articles[j])
                    used_indices.add(j)
                    
        # Only save the cluster if at least 2 different articles are covering it
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            
    return clusters

# --- EXECUTION ---
with st.spinner("Fetching news and running AI clustering... this takes a few seconds."):
    live_articles = fetch_news()
    clustered_events = cluster_articles(live_articles)

# --- UI DISPLAY ---
st.header("Clustered News Events")
st.write("Our NLP model has grouped these articles together because they are reporting on the exact same story.")

if not clustered_events:
    st.info("Not enough overlapping stories found between the networks right now. Try again later!")

# Loop through our AI-generated groups
for index, cluster in enumerate(clustered_events):
    st.divider()
    
    # We use the title of the first article as the overarching "Event Title"
    st.subheader(f"Event: {cluster[0]['title']}")
    
    # Create the bias columns
    left_col, center_col, right_col = st.columns(3)
    
    # Sort the articles in this specific cluster into their bias buckets
    for art in cluster:
        if art["bias"] == "Left":
            with left_col:
                st.info(f"**Left Leaning**\n\n[{art['title']}]({art['link']})")
        elif art["bias"] == "Center":
            with center_col:
                st.warning(f"**Center**\n\n[{art['title']}]({art['link']})")
        elif art["bias"] == "Right":
            with right_col:
                st.error(f"**Right Leaning**\n\n[{art['title']}]({art['link']})")
