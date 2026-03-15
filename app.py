return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

# --- FETCH WEEKLY RADAR (FROM DB) ---
@st.cache_data(ttl=600)
def fetch_weekly_radar(region):
    if not supabase: return []
    # Pulls the top 20 most covered stories from the last 7 days for this region
    res = supabase.table("news_events").select("*").eq("region", region).order("total_articles", desc=True).limit(20).execute()
    return res.data

# --- HTML CARD RENDERER ---
def render_event_card(event):
    topics_html = "".join([f"<span class='topic-pill'>{t}</span>" for t in event.get('topics', [])])
    
    sources_html = ""
    for art in event['sources_json']:
        color = "#3b82f6" if art['bias'] == "Left" else "#eab308" if art['bias'] == "Center" else "#ef4444"
        sources_html += f"<div style='margin-bottom:8px;'><small><b><span style='color:{color}'>▪ {art['bias']}</span> | {art['source']}</b>: <a href='{art['link']}' style='color:#333; text-decoration:none;'>{art['title']}</a></small></div>"

    html = f"""
    <div class="custom-card">
        {topics_html}
        <h3 style="margin-top: 12px; color: #111827;">{event['title']}</h3>
        <p style="color: #374151; font-size: 15px; line-height: 1.5;">{event['summary']}</p>
        
        <div class="insight-tag">Key Fact</div>
        <p style="margin: 4px 0 0 0; font-size: 14px; color: #4b5563;">{event['key_fact']}</p>
        
        <div class="insight-tag">Media Discrepancy</div>
        <p style="margin: 4px 0 0 0; font-size: 14px; color: #4b5563;">{event['discrepancy']}</p>
        
        <div class="bias-bar-container">
            <div style="width: {(event['left_count']/event['total_articles'])*100}%; background-color: #3b82f6;"></div>
            <div style="width: {(event['center_count']/event['total_articles'])*100}%; background-color: #eab308;"></div>
            <div style="width: {(event['right_count']/event['total_articles'])*100}%; background-color: #ef4444;"></div>
        </div>
        
        <details>
            <summary style="cursor:pointer; color:#6b7280; font-size:14px; font-weight:500;">View {event['total_articles']} Sources</summary>
            <div class="sources-box">{sources_html}</div>
        </details>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- UI APP LAYOUT ---
st.title("News Aggregator")
st.markdown("▪️ Bias tracking and AI event clustering.")

# 1. Region Tabs
tab_global, tab_india, tab_tn = st.tabs(["🌍 Global", "🇮🇳 India", "🐅 Tamil Nadu"])

def build_view(region):
    # 2. Timeframe Toggle
    view_mode = st.radio("Timeframe", ["Top News Today (Live)", "Weekly Radar (Database)"], horizontal=True, label_visibility="collapsed")
    
    with st.spinner(f"Loading {region} news..."):
        if "Live" in view_mode:
            events = process_live_news(region)
        else:
            events = fetch_weekly_radar(region)

    if not events:
        st.info("No heavily covered stories found for this filter right now.")
        return

    # 3. Topic Cloud Extraction & Filtering
    all_topics = []
    for ev in events:
        all_topics.extend(ev.get('topics', []))
    unique_topics = sorted(list(set(all_topics)))
    
    selected_topics = st.multiselect("☁️ Filter by Topic", unique_topics, placeholder="Select a topic to filter...")
    st.divider()

    # 4. Render the Cards
    for event in events:
        # If the user selected topics, only show cards that match
        if selected_topics:
            event_topics = event.get('topics', [])
            if not any(topic in selected_topics for topic in event_topics):
                continue # Skip this card if it doesn't match the filter
                
        render_event_card(event)

with tab_global: build_view("Global")
with tab_india: build_view("India")
with tab_tn: build_view("Tamil Nadu")
