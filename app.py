import re
import json
import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
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
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SUPABASE INIT
# ---------------------------------------------------------------------------
try:
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        supabase: Client = create_client(
            st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
        )
    else:
        supabase = None
except Exception:
    st.warning("Supabase not connected. Weekly Radar will be disabled.")
    supabase = None

# ---------------------------------------------------------------------------
# GEMINI INIT  (dynamic model selection with graceful fallback)
# ---------------------------------------------------------------------------
gemini_model = None

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    available_models = [
        m.name
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]

    target_model = "gemini-1.0-pro"
    for m in available_models:
        if "gemini-1.5-flash" in m:
            target_model = m
            break
        elif "1.5-pro" in m:
            target_model = m
        elif "1.0-pro" in m:
            target_model = m

    gemini_model = genai.GenerativeModel(
        target_model,
        generation_config={"response_mime_type": "application/json"},
    )
except Exception as e:
    st.error(f"Failed to initialise AI model: {e}")

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
        "CNN": {
            "url": "http://rss.cnn.com/rss/cnn_topstories.rss",
            "bias": "Left",
        },
        "BBC": {
            "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
            "bias": "Center",
        },
        "Fox": {
            "url": "http://feeds.foxnews.com/foxnews/world",
            "bias": "Right",
        },
    },
    "India": {
        "The Hindu": {
            "url": "https://news.google.com/rss/search?q=site:thehindu.com+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            "bias": "Left",
        },
        "NDTV": {
            "url": "https://feeds.feedburner.com/ndtvnews-india-news",
            "bias": "Center",
        },
        "Times of India": {
            "url": "https://news.google.com/rss/search?q=site:timesofindia.indiatimes.com+india+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            "bias": "Right",
        },
    },
    "Tamil Nadu": {
        "The Hindu TN": {
            "url": "https://news.google.com/rss/search?q=site:thehindu.com+tamil+nadu+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            "bias": "Left",
        },
        "Indian Express TN": {
            "url": "https://news.google.com/rss/search?q=site:indianexpress.com+chennai+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            "bias": "Center",
        },
        "TOI Chennai": {
            "url": "https://news.google.com/rss/search?q=site:timesofindia.indiatimes.com+chennai+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            "bias": "Right",
        },
    },
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _safe_parse_json(raw: str) -> dict:
    """Strip markdown fences robustly, then parse JSON."""
    raw = raw.strip()
    # Remove opening ```json or ``` fence
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    # Remove closing ``` fence
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


def _deserialize_sources(sources) -> list:
    """Supabase JSONB columns can come back as a string in some client versions."""
    if isinstance(sources, str):
        return json.loads(sources)
    if isinstance(sources, list):
        return sources
    return []


# ---------------------------------------------------------------------------
# AI METADATA GENERATION  (kept outside cache so errors surface every run)
# ---------------------------------------------------------------------------

def generate_ai_metadata(titles: list) -> dict:
    """Call Gemini to produce summary, insights, and topic tags for a cluster."""
    if not gemini_model:
        return {"error": "AI offline"}

    headlines = "\n".join(titles)
    json_schema = """{
    "summary": "Factual 2-sentence summary of the event.",
    "insights": {
        "key_fact": "One verified fact.",
        "discrepancy": "Media differences (or 'Single source narrative' if only one article)."
    },
    "topics": ["Tag1", "Tag2"]
}"""
    prompt = (
        f"Analyse these headlines:\n{headlines}\n\n"
        "Return JSON strictly following this structure. "
        "For topics provide 1–3 broad categories "
        "(e.g. Politics, Tech, Business, Crime, Election):\n"
        f"{json_schema}"
    )

    try:
        resp = gemini_model.generate_content(prompt)
        return _safe_parse_json(resp.text)
    except json.JSONDecodeError as e:
        st.warning(f"JSON parse error from AI: {e} — raw: {resp.text[:120]}")
        return {"error": f"json_decode: {e}"}
    except Exception as e:
        st.warning(f"AI call failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# STEP 1 — fetch RSS + cluster  (cached; no AI calls inside)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_and_cluster(region: str) -> list[list[dict]]:
    """
    Fetch RSS articles for the given region, embed titles with the NLP model,
    and group semantically similar articles into clusters.

    Returns a list of clusters sorted by size (largest first), capped at 15.
    """
    articles = []
    feedparser.USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    for source_name, data in RSS_FEEDS[region].items():
        try:
            feed = feedparser.parse(data["url"])
            for entry in feed.entries[:35]:
                articles.append(
                    {
                        "title": entry.title,
                        "link": entry.link,
                        "source": source_name,
                        "bias": data["bias"],
                    }
                )
        except Exception:
            continue

    if not articles:
        return []

    titles = [a["title"] for a in articles]
    embeddings = nlp_model.encode(titles, show_progress_bar=False)

    clusters: list[list[dict]] = []
    used: set[int] = set()

    for i in range(len(articles)):
        if i in used:
            continue
        cluster = [articles[i]]
        used.add(i)
        for j in range(i + 1, len(articles)):
            if j not in used:
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > 0.45:
                    cluster.append(articles[j])
                    used.add(j)
        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters[:15]


# ---------------------------------------------------------------------------
# STEP 2 — enrich clusters with AI + persist to Supabase
# ---------------------------------------------------------------------------

def process_live_news(region: str) -> list[dict]:
    """
    Takes raw clusters from the cache layer, calls Gemini per cluster,
    and returns processed event dicts.  AI calls are intentionally outside
    the cache so transient failures don't get frozen into stale data.
    """
    clusters = fetch_and_cluster(region)
    if not clusters:
        return []

    processed_events: list[dict] = []

    for cluster in clusters:
        ai_data = generate_ai_metadata([a["title"] for a in cluster])

        if not isinstance(ai_data, dict) or "error" in ai_data:
            continue

        event_title = cluster[0]["title"]
        left_c   = sum(1 for a in cluster if a["bias"] == "Left")
        center_c = sum(1 for a in cluster if a["bias"] == "Center")
        right_c  = sum(1 for a in cluster if a["bias"] == "Right")

        event_record = {
            "region":         region,
            "title":          event_title,
            "summary":        ai_data.get("summary", "Summary not available."),
            "key_fact":       ai_data.get("insights", {}).get("key_fact", "N/A"),
            "discrepancy":    ai_data.get("insights", {}).get("discrepancy", "N/A"),
            "topics":         ai_data.get("topics", []),
            "left_count":     left_c,
            "center_count":   center_c,
            "right_count":    right_c,
            "total_articles": len(cluster),
            "sources_json":   cluster,
        }

        processed_events.append(event_record)

        # Persist to Supabase for the Weekly Radar
        if supabase:
            try:
                existing = (
                    supabase.table("news_events")
                    .select("id")
                    .eq("title", event_title)
                    .execute()
                )
                if not existing.data:
                    # Supabase needs sources_json serialised
                    record_for_db = {**event_record, "sources_json": json.dumps(cluster)}
                    supabase.table("news_events").insert(record_for_db).execute()
            except Exception as db_err:
                # Non-fatal — don't block the UI
                st.warning(f"Supabase write failed: {db_err}")

    return processed_events


# ---------------------------------------------------------------------------
# WEEKLY RADAR  (reads from Supabase)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def fetch_weekly_radar(region: str) -> list[dict]:
    if not supabase:
        return []
    try:
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        res = (
            supabase.table("news_events")
            .select("*")
            .eq("region", region)
            .gte("created_at", week_ago)
            .order("total_articles", desc=True)
            .limit(20)
            .execute()
        )
        return res.data or []
    except Exception as e:
        st.warning(f"Supabase read failed: {e}")
        return []


# ---------------------------------------------------------------------------
# CARD RENDERER
# ---------------------------------------------------------------------------

def render_event_card(event: dict) -> None:
    # Deserialise sources safely (Supabase may return a JSON string)
    sources = _deserialize_sources(event.get("sources_json", []))
    total   = max(event.get("total_articles", len(sources)), 1)  # guard /0

    # Topic pills
    topics_html = "".join(
        f"<span style='display:inline-block;background:#f3f4f6;color:#374151;"
        f"padding:4px 10px;border-radius:16px;font-size:12px;"
        f"margin-right:6px;font-weight:500;'>{t}</span>"
        for t in event.get("topics", [])
    )

    # Source list
    sources_html = ""
    for art in sources:
        color = (
            "#3b82f6" if art.get("bias") == "Left"
            else "#eab308" if art.get("bias") == "Center"
            else "#ef4444"
        )
        sources_html += (
            f"<div style='margin-bottom:8px;'>"
            f"<small><b><span style='color:{color}'>▪ {art.get('bias','?')}</span>"
            f" | {art.get('source','?')}</b>: "
            f"<a href='{art.get('link','#')}' style='color:#333;text-decoration:none;'>"
            f"{art.get('title','')}</a></small></div>"
        )

    # Bias bar / single-source badge
    if total > 1:
        left_pct   = (event.get("left_count",   0) / total) * 100
        center_pct = (event.get("center_count", 0) / total) * 100
        right_pct  = (event.get("right_count",  0) / total) * 100
        bias_ui = f"""
        <div class="bias-bar-container">
            <div style="width:{left_pct:.1f}%;background-color:#3b82f6;"></div>
            <div style="width:{center_pct:.1f}%;background-color:#eab308;"></div>
            <div style="width:{right_pct:.1f}%;background-color:#ef4444;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#9ca3af;margin-bottom:12px;">
            <span>▪ Left ({event.get('left_count',0)})</span>
            <span>▪ Center ({event.get('center_count',0)})</span>
            <span>▪ Right ({event.get('right_count',0)})</span>
        </div>
        """
    else:
        bias = sources[0].get("bias", "Unknown") if sources else "Unknown"
        bias_color = (
            "#3b82f6" if bias == "Left"
            else "#eab308" if bias == "Center"
            else "#ef4444"
        )
        bias_ui = (
            f"<p style='font-size:13px;color:{bias_color};font-weight:bold;"
            f"margin:16px 0;'>▪ Single Source Narrative ({bias} Leaning)</p>"
        )

    html_card = f"""
<style>
.custom-card {{
    background:white; padding:20px; border-radius:8px;
    border:1px solid #e5e7eb; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    margin-bottom:20px;
}}
.insight-tag {{
    color:#4b5563; font-size:11px; font-weight:bold;
    text-transform:uppercase; letter-spacing:0.5px; margin-top:12px;
}}
.bias-bar-container {{
    display:flex; height:6px; width:100%;
    border-radius:3px; overflow:hidden; margin-top:16px; margin-bottom:4px;
}}
.sources-box {{
    background:#f9fafb; padding:12px; border-radius:6px;
    margin-top:16px; border:1px solid #f3f4f6;
}}
</style>

<div class="custom-card">
    {topics_html}
    <h3 style="margin-top:12px;color:#111827;">{event.get("title","")}</h3>
    <p style="color:#374151;font-size:15px;line-height:1.5;">{event.get("summary","")}</p>

    <div class="insight-tag">Key Fact</div>
    <p style="margin:4px 0 0 0;font-size:14px;color:#4b5563;">{event.get("key_fact","N/A")}</p>

    <div class="insight-tag">Media Discrepancy</div>
    <p style="margin:4px 0 0 0;font-size:14px;color:#4b5563;">{event.get("discrepancy","N/A")}</p>

    {bias_ui}

    <details>
        <summary style="cursor:pointer;color:#6b7280;font-size:14px;font-weight:500;">
            View {total} Source(s)
        </summary>
        <div class="sources-box">{sources_html}</div>
    </details>
</div>
"""
    st.markdown(html_card, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VIEW BUILDER
# ---------------------------------------------------------------------------

def build_view(region: str) -> None:
    view_mode = st.radio(
        "Timeframe",
        ["Top News Today (Live)", "Weekly Radar (Database)"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"radio_{region}",
    )

    with st.spinner(f"Loading {region} news…"):
        if "Live" in view_mode:
            events = process_live_news(region)
        else:
            events = fetch_weekly_radar(region)

    if not events:
        st.info("No stories found right now. Try again in a moment.")
        return

    # Collect all unique topics for the filter pills
    all_topics: list[str] = []
    for ev in events:
        all_topics.extend(ev.get("topics", []))
    unique_topics = sorted(set(all_topics))

    selected_topics = st.pills(
        "▪ Filter by Topic",
        unique_topics,
        selection_mode="multi",
        key=f"pills_{region}",
    )
    st.divider()

    shown = 0
    for event in events:
        if selected_topics:
            if not any(t in selected_topics for t in event.get("topics", [])):
                continue
        render_event_card(event)
        shown += 1

    if shown == 0:
        st.info("No stories match the selected filters.")


# ---------------------------------------------------------------------------
# APP LAYOUT
# ---------------------------------------------------------------------------

st.title("News Aggregator")
st.markdown("▪ Bias tracking · AI event clustering · Multi-region coverage")

tab_global, tab_india, tab_tn = st.tabs(["▪ Global", "▪ India", "▪ Tamil Nadu"])

with tab_global:
    build_view("Global")

with tab_india:
    build_view("India")

with tab_tn:
    build_view("Tamil Nadu")
