raw = re.sub(r"```\s*$",          "", raw, flags=re.MULTILINE)
    try:
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": f"JSON parse error: {e}"}


def _deserialize_sources(sources) -> list:
    """Supabase JSONB columns can come back as a string in some client versions."""
    if isinstance(sources, str):
        try:
            return json.loads(sources)
        except:
            return []
    if isinstance(sources, list):
        return sources
    return []


# ---------------------------------------------------------------------------
# CORE AI CALL (TENACITY + MULTI-MODEL FALLBACK)
# ---------------------------------------------------------------------------

# 1. THE RETRY ENGINE (Exponential Backoff for the active model)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ResourceExhausted, TooManyRequests))
)
def _fetch_ai_content_with_retry(model_name: str, prompt: str) -> str:
    """Raw function to call the API using Tenacity's exponential backoff."""
    model = _get_gemini_model(model_name)
    response = model.generate_content(prompt)
    return response.text

# 2. THE MULTI-MODEL FALLBACK MANAGER
def _call_gemini_with_fallback(prompt: str) -> str:
    if not GEMINI_FALLBACKS:
        raise RuntimeError("No Gemini models configured.")

    last_error = None

    for model_name in GEMINI_FALLBACKS:
        try:
            # Attempt generation with exponential backoff on the current model
            return _fetch_ai_content_with_retry(model_name, prompt)

        except (ResourceExhausted, TooManyRequests) as e:
            # If it fails 3 times, move to the NEXT available model
            st.toast(
                f"⚠️ `{model_name}` quota exhausted after retries. Trying next...", 
                icon="🔄"
            )
            last_error = e
            continue
        
        except Exception as e:
            err_str = str(e).lower()
            
            if "404" in err_str or "not found" in err_str:
                last_error = e
                continue
                
            if "429" in err_str or "quota" in err_str:
                st.toast(f"⚠️ `{model_name}` quota hit. Trying next...", icon="🔄")
                last_error = e
                continue
                
            raise # Unhandled generic error, bubble up

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# BATCH AI CALL
# Sends all clusters in ONE prompt instead of N individual calls.
# ---------------------------------------------------------------------------
def generate_all_metadata(clusters: list[list[dict]]) -> list[dict]:
    if not GEMINI_FALLBACKS:
        return [{"error": "AI offline"}] * len(clusters)

    numbered_groups = ""
    for i, cluster in enumerate(clusters):
        titles = "\n  ".join(a["title"] for a in cluster)
        numbered_groups += f"\nGroup {i + 1}:\n  {titles}\n"

    json_schema = """[
  {
    "group": 1,
    "summary": "Factual 2-sentence summary.",
    "insights": {
      "key_fact": "One verified fact.",
      "discrepancy": "Media differences or 'Single source narrative'."
    },
    "topics": ["Tag1", "Tag2"]
  }
]"""

    prompt = (
        f"You will receive {len(clusters)} groups of news headlines. "
        "For EACH group return exactly one JSON object inside a JSON array, "
        "strictly following this structure. "
        "topics: 1–3 broad categories (e.g. Politics, Tech, Business, Crime, Election).\n"
        f"{json_schema}\n\nHere are the groups:{numbered_groups}"
    )

    try:
        raw    = _call_gemini_with_fallback(prompt)
        parsed = _safe_parse_json(raw)

        if isinstance(parsed, list) and len(parsed) == len(clusters):
            return parsed

        # Alignment fallback in case lengths don't match
        result: list[dict | None] = [None] * len(clusters)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("group", 0) - 1
                    if 0 <= idx < len(clusters):
                        result[idx] = item
        return [r if r else {"error": "missing"} for r in result]

    except Exception:
        st.toast("Batch AI call failed — falling back to per-cluster mode.", icon="ℹ️")
        return [
            _single_cluster_metadata([a["title"] for a in c])
            for c in clusters
        ]


def _single_cluster_metadata(titles: list) -> dict:
    """Fallback per-cluster caller if batch fails."""
    json_schema = """{
    "summary": "Factual 2-sentence summary.",
    "insights": {
        "key_fact": "One verified fact.",
        "discrepancy": "Media differences or 'Single source narrative'."
    },
    "topics": ["Tag1", "Tag2"]
}"""
    prompt = (
        f"Analyse these headlines:\n" + "\n".join(titles) + "\n\n"
        "Return JSON strictly following this structure. "
        "topics: 1–3 broad categories (Politics, Tech, Business, Crime, Election):\n"
        f"{json_schema}"
    )
    try:
        raw = _call_gemini_with_fallback(prompt)
        return _safe_parse_json(raw)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# STEP 1 — fetch RSS + cluster
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_and_cluster(region: str) -> list[list[dict]]:
    articles = []
    # Disguise the scraper to avoid being blocked
    feedparser.USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    for source_name, data in RSS_FEEDS[region].items():
        try:
            feed = feedparser.parse(data["url"])
            for entry in feed.entries[:35]:
                articles.append({
                    "title":  entry.title,
                    "link":   entry.link,
                    "source": source_name,
                    "bias":   data["bias"],
                })
        except Exception:
            continue

    if not articles:
        return []

    titles     = [a["title"] for a in articles]
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
                if util.cos_sim(embeddings[i], embeddings[j]).item() > 0.45:
                    cluster.append(articles[j])
                    used.add(j)
        
        # We allow single-source stories
        if len(cluster) >= 1:
            clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters[:12]


# ---------------------------------------------------------------------------
# STEP 2 — enrich clusters with AI + persist to Supabase
# ---------------------------------------------------------------------------
def process_live_news(region: str) -> list[dict]:
    clusters = fetch_and_cluster(region)
    if not clusters:
        return []

    all_metadata = generate_all_metadata(clusters)
    processed_events: list[dict] = []

    for cluster, ai_data in zip(clusters, all_metadata):
        if not isinstance(ai_data, dict) or "error" in ai_data:
            continue

        event_title = cluster[0]["title"]
        left_c      = sum(1 for a in cluster if a["bias"] == "Left")
        center_c    = sum(1 for a in cluster if a["bias"] == "Center")
        right_c     = sum(1 for a in cluster if a["bias"] == "Right")

        event_record = {
            "region":         region,
            "title":          event_title,
            "summary":        ai_data.get("summary",  "Summary not available."),
            "key_fact":       ai_data.get("insights", {}).get("key_fact",    "N/A"),
            "discrepancy":    ai_data.get("insights", {}).get("discrepancy", "N/A"),
            "topics":         ai_data.get("topics",   []),
            "left_count":     left_c,
            "center_count":   center_c,
            "right_count":    right_c,
            "total_articles": len(cluster),
            "sources_json":   cluster,
        }

        processed_events.append(event_record)

        if supabase:
            try:
                existing = (
                    supabase.table("news_events")
                    .select("id")
                    .eq("title", event_title)
                    .execute()
                )
                if not existing.data:
                    record_for_db = {
                        **event_record,
                        "sources_json": json.dumps(cluster),
                    }
                    supabase.table("news_events").insert(record_for_db).execute()
            except Exception as db_err:
                pass

    return processed_events


# ---------------------------------------------------------------------------
# WEEKLY RADAR
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
    sources = _deserialize_sources(event.get("sources_json", []))
    total   = max(event.get("total_articles", len(sources)), 1)

    topics_html = "".join(
        f"<span style='display:inline-block;background:#f3f4f6;color:#374151;"
        f"padding:4px 10px;border-radius:16px;font-size:12px;"
        f"margin-right:6px;font-weight:500;'>{t}</span>"
        for t in event.get("topics", [])
    )

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

    # Note: These f-strings are pushed all the way to the left so Streamlit Markdown doesn't block them.
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
        bias       = sources[0].get("bias", "Unknown") if sources else "Unknown"
        bias_color = (
            "#3b82f6" if bias == "Left"
            else "#eab308" if bias == "Center"
            else "#ef4444"
        )
        bias_ui = f"""
<p style='font-size:13px;color:{bias_color};font-weight:bold;margin:16px 0;'>▪ Single Source Narrative ({bias} Leaning)</p>
"""

    st.markdown(f"""
<style>
.custom-card {{ background:white; padding:20px; border-radius:8px; border:1px solid #e5e7eb; box-shadow:0 1px 3px rgba(0,0,0,0.05); margin-bottom:20px; }}
.insight-tag {{ color:#4b5563; font-size:11px; font-weight:bold; text-transform:uppercase; letter-spacing:0.5px; margin-top:12px; }}
.bias-bar-container {{ display:flex; height:6px; width:100%; border-radius:3px; overflow:hidden; margin-top:16px; margin-bottom:4px; }}
.sources-box {{ background:#f9fafb; padding:12px; border-radius:6px; margin-top:16px; border:1px solid #f3f4f6; }}
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
""", unsafe_allow_html=True)


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
        events = (
            process_live_news(region)
            if "Live" in view_mode
            else fetch_weekly_radar(region)
        )

    if not events:
        st.info("No stories found right now. Try again in a moment.")
        return

    all_topics: list[str] = []
    for ev in events:
        all_topics.extend(ev.get("topics", []))

    selected_topics = st.pills(
        "▪ Filter by Topic",
        sorted(set(all_topics)),
        selection_mode="multi",
        key=f"pills_{region}",
    )
    st.divider()

    shown = 0
    for event in events:
        if selected_topics and not any(
            t in selected_topics for t in event.get("topics", [])
        ):
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
