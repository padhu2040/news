import streamlit as st
import feedparser

# Set up the page
st.set_page_config(page_title="News Aggregator", page_icon="📰", layout="wide")

st.title("📰 Open News Aggregator")
st.markdown("Tracking bias and clustering global events.")
st.divider()

# --- REAL DATA FETCHING ---
# We use @st.cache_data so Streamlit doesn't re-download the news every time you click a button
@st.cache_data(ttl=3600) # Caches the data for 1 hour
def fetch_news():
    # Dictionary of RSS feeds and their assigned bias
    rss_feeds = {
        "Left": "http://rss.cnn.com/rss/cnn_topstories.rss",
        "Center": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Right": "http://feeds.foxnews.com/foxnews/world"
    }
    
    articles = []
    
    for bias, url in rss_feeds.items():
        feed = feedparser.parse(url)
        # Grab the top 3 articles from each feed
        for entry in feed.entries[:3]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get('summary', 'No summary available.')[:100] + "...",
                "bias": bias
            })
            
    return articles

# Fetch the live articles
live_articles = fetch_news()

# --- UI DISPLAY ---
st.header("Live News Feed")
st.write("Here are the latest global headlines pulled directly from live RSS feeds.")

st.divider()

st.subheader("Latest Coverage by Bias")

# Create the 3 columns
left_col, center_col, right_col = st.columns(3)

# Sort the real articles into their respective columns
with left_col:
    st.info("### Left Leaning (CNN)")
    for art in live_articles:
        if art["bias"] == "Left":
            st.write(f"**{art['title']}**")
            st.caption(art["summary"])
            st.markdown(f"[Read Article]({art['link']})")
            st.divider()

with center_col:
    st.warning("### Center (BBC)")
    for art in live_articles:
        if art["bias"] == "Center":
            st.write(f"**{art['title']}**")
            st.caption(art["summary"])
            st.markdown(f"[Read Article]({art['link']})")
            st.divider()

with right_col:
    st.error("### Right Leaning (Fox)")
    for art in live_articles:
        if art["bias"] == "Right":
            st.write(f"**{art['title']}**")
            st.caption(art["summary"])
            st.markdown(f"[Read Article]({art['link']})")
            st.divider()
