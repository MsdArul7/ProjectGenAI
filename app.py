import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import re

# ================= CONFIG =================
DATA_DIR = Path("data")
DEFAULT_GEOAPIFY = "0ace8c8462a943b982df4fd2750d3407"  # Replace if needed
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

# ================= UTILS =================
def clean_string(text):
    if not isinstance(text, str): return ""
    return re.sub(r"^\s*\d+[\.\)\-]*\s*", "", text).strip()

# ================= 1. LOAD ALL 4 DATA SOURCES =================
@st.cache_data(ttl=3600)
def load_all_data():
    data = {}
    
    # A. Load Places.csv
    try:
        data["places_df"] = pd.read_csv(DATA_DIR / "Places.csv")
    except:
        try:
            data["places_df"] = pd.read_csv(DATA_DIR / "Places.csv", encoding="utf8", errors="replace")
        except:
            data["places_df"] = pd.DataFrame()

    # B. Load City.csv
    try:
        data["city_df"] = pd.read_csv(DATA_DIR / "City.csv")
    except:
        try:
            data["city_df"] = pd.read_csv(DATA_DIR / "City.csv", encoding="utf8", errors="replace")
        except:
            data["city_df"] = pd.DataFrame()

    # C. Load India Tourism JSON
    try:
        with open(DATA_DIR / "india_tourism_big.json", "r", encoding="utf8") as f:
            data["tourism_json"] = json.load(f)
    except:
        data["tourism_json"] = []

    # D. Load FAQs JSON
    try:
        with open(DATA_DIR / "faqs_big.json", "r", encoding="utf8") as f:
            data["faqs_json"] = json.load(f)
    except:
        data["faqs_json"] = []
        
    return data

DATA = load_all_data()

# ================= 2. BUILD SEARCH INDEX (RAG) =================
# We combine text from ALL sources into one search engine
local_docs = []

# Index Places.csv
if not DATA["places_df"].empty:
    for i, row in DATA["places_df"].fillna("").iterrows():
        name = clean_string(str(row.get("Place", "")))
        city = str(row.get("City", ""))
        desc = str(row.get("Place_desc", ""))
        txt = f"{name} {city} {desc}"
        local_docs.append({"id": f"csv_{i}", "name": name, "city": city, "text": txt, "type": "place_csv", "meta": row.to_dict()})

# Index Tourism JSON
for i, entry in enumerate(DATA["tourism_json"]):
    name = entry.get("place", "")
    city = entry.get("city", "")
    desc = entry.get("description", "")
    txt = f"{name} {city} {desc}"
    local_docs.append({"id": f"json_{i}", "name": name, "city": city, "text": txt, "type": "place_json", "meta": entry})

# Index FAQs JSON (To find answers like "How to reach...")
for i, entry in enumerate(DATA["faqs_json"]):
    q = entry.get("question", "")
    a = entry.get("answer", "")
    local_docs.append({"id": f"faq_{i}", "name": "FAQ", "city": "", "text": q + " " + a, "type": "faq", "meta": entry})

# Index City.csv (For city-level info)
if not DATA["city_df"].empty:
    for i, row in DATA["city_df"].fillna("").iterrows():
        c_name = str(row.get("City", ""))
        c_desc = str(row.get("City_desc", ""))
        best_time = str(row.get("Best_time_to_visit", ""))
        local_docs.append({"id": f"city_{i}", "name": c_name, "city": c_name, "text": f"{c_name} {c_desc} {best_time}", "type": "city_info", "meta": row.to_dict()})

# Vectorize
@st.cache_resource
def get_vectorizer(docs):
    texts = [d["text"] for d in docs] if docs else [""]
    vect = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vect.fit_transform(texts)
    return vect, X

vectorizer, docs_X = get_vectorizer(local_docs)

# ================= 3. CONTEXT GATHERING =================
def get_context_for_place(place_name, city_name):
    """
    Searches ALL 4 files to find relevant info for the AI.
    """
    query = f"{place_name} {city_name}"
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, docs_X).flatten()
    
    # Get top matches
    idxs = sims.argsort()[::-1][:8] # Top 8 snippets
    
    context_parts = []
    
    for i in idxs:
        if sims[i] < 0.1: continue # Filter weak matches
        doc = local_docs[i]
        
        if doc["type"] == "place_csv":
            context_parts.append(f"[CSV Details]: {doc['meta'].get('Place_desc')}")
        elif doc["type"] == "place_json":
            context_parts.append(f"[JSON Details]: {doc['meta'].get('description')}")
        elif doc["type"] == "faq":
            # Only include FAQ if it relates to the city or place
            if city_name.lower() in doc['text'].lower() or place_name.lower() in doc['text'].lower():
                context_parts.append(f"[FAQ]: Q: {doc['meta'].get('question')} A: {doc['meta'].get('answer')}")
        elif doc["type"] == "city_info":
            # Only include if city matches
            if city_name.lower() in doc['name'].lower():
                context_parts.append(f"[City Info]: Best time: {doc['meta'].get('Best_time_to_visit')}. Desc: {doc['meta'].get('City_desc')}")

    return "\n".join(context_parts)

def get_wiki_summary(name, city):
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(f"{name} {city}")
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return r.json().get("extract")
    except: pass
    return None

# ================= 4. AI & LOGIC =================
def generate_ai_response(name, city, context):
    if not st.session_state.get("ollama_enabled", False):
        return None

    system_prompt = (
        "You are a travel expert. Combine the provided Context (from CSVs, JSONs, FAQs) and your own knowledge.\n"
        "Output ONLY a valid JSON object with keys: description, timings, season, price.\n"
        "Description should be detailed (8-10 lines), mentioning history and specific features found in the context."
    )
    user_prompt = f"Place: {name}, {city}\nContext from files:\n{context}\n\nGenerate JSON."
    
    try:
        payload = {"model": "llama3", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "stream": False, "temperature": 0.3}
        r = requests.post(st.session_state.get("ollama_url", DEFAULT_OLLAMA_URL), json=payload, timeout=25)
        return json.loads(r.json()["message"]["content"])
    except:
        return None

def fetch_image(name, city):
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(f"{name} {city}")
        d = requests.get(url, timeout=3).json()
        if d.get("thumbnail"): return d["thumbnail"]["source"]
    except: pass
    return None

# ================= 5. API SEARCH =================
def get_api_places(city, category):
    # 1. Get City Coords
    lat, lon = None, None
    try:
        u = "https://api.geoapify.com/v1/geocode/search"
        r = requests.get(u, params={"text": city, "apiKey": st.session_state.geo_key, "limit": 1})
        if r.ok:
            props = r.json()["features"][0]["properties"]
            lat, lon = props["lat"], props["lon"]
    except: pass

    if not lat: return []

    # 2. Get Places
    cat_map = {"Tourist Attractions": "tourism", "Temples": "religion", "Hotels": "accommodation", "Restaurants": "catering"}
    api_hits = []
    try:
        u2 = "https://api.geoapify.com/v2/places"
        r2 = requests.get(u2, params={"categories": cat_map.get(category, "tourism"), "filter": f"circle:{lon},{lat},10000", "limit": 20, "apiKey": st.session_state.geo_key})
        if r2.ok: api_hits = r2.json()["features"]
    except: pass
    
    results = []
    seen = set()
    
    # 3. Merge with CSV (Priority to CSV)
    # Filter CSV for this city
    city_csv_data = DATA["places_df"][DATA["places_df"]["City"].str.contains(city, case=False, na=False)]
    
    for _, row in city_csv_data.iterrows():
        name = clean_string(row["Place"])
        results.append({"name": name, "origin": "CSV/File", "meta": row.to_dict(), "lat": lat, "lon": lon})
        seen.add(name.lower())

    # Add API hits if not in CSV
    for hit in api_hits:
        props = hit.get("properties", {})
        name = props.get("name", "")
        if name and name.lower() not in seen:
            results.append({"name": name, "origin": "API", "meta": props, "lat": lat, "lon": lon})
            seen.add(name.lower())
            
    return results

# ================= 6. UI =================
st.set_page_config(layout="wide", page_title="City Explorer Pro")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state["geo_key"] = st.text_input("Geoapify API Key", value=DEFAULT_GEOAPIFY, type="password")
    st.session_state["ollama_enabled"] = st.checkbox("Enable AI (Ollama)", value=False)
    st.session_state["ollama_url"] = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
    st.info(f"Loaded {len(local_docs)} records from 4 files.")

st.title("🏙️ City Explorer Pro")
st.markdown("This tool combines **2 Excel Files, 2 JSON Files, Live API, and AI**.")

c1, c2, c3 = st.columns([2, 2, 1])
city_inp = c1.text_input("City", "Manali")
cat_inp = c2.selectbox("Category", ["Tourist Attractions", "Temples", "Restaurants", "Hotels"])
if c3.button("Search Places"):
    st.session_state["search_results"] = get_api_places(city_inp, cat_inp)
    st.session_state["selected_id"] = None # Reset selection on new search

# RESULT DISPLAY
if "search_results" in st.session_state:
    results = st.session_state["search_results"]
    st.write(f"Found {len(results)} places in **{city_inp}**.")
    
    # Iterate and display
    for i, place in enumerate(results):
        # CARD LAYOUT
        with st.container():
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.subheader(f"{i+1}. {place['name']}")
                origin = place['origin']
                
                # Show quick info from CSV if available
                if origin == "CSV/File":
                    desc_preview = str(place['meta'].get('Place_desc', ''))[:120] + "..."
                    st.caption(f"📂 **Source:** Local Data Files | {desc_preview}")
                else:
                    st.caption(f"🌐 **Source:** Live API")

            with col_b:
                # THE TOGGLE BUTTON
                # We use a unique key for every button. If clicked, we update the session state ID.
                btn_label = "Hide Details" if st.session_state.get("selected_id") == i else "More Details"
                if st.button(btn_label, key=f"btn_{i}"):
                    if st.session_state.get("selected_id") == i:
                        st.session_state["selected_id"] = None # Close if already open
                    else:
                        st.session_state["selected_id"] = i # Open this one
                    st.rerun()

        # DETAILS PANEL (Displayed Immediately Below the Place)
        if st.session_state.get("selected_id") == i:
            with st.container():
                st.markdown("---")
                # Prepare Data
                with st.spinner(" gathering info from all 4 files + AI..."):
                    context = get_context_for_place(place['name'], city_inp)
                    wiki_summ = get_wiki_summary(place['name'], city_inp)
                    
                    full_context = f"{context}\n[Wikipedia]: {wiki_summ}"
                    
                    # Generate AI Response
                    details = generate_ai_response(place['name'], city_inp, full_context)
                    img_url = fetch_image(place['name'], city_inp)
                    
                    # Fallback if AI off or failed
                    if not details:
                        # Use CSV desc or Wiki
                        raw_desc = place['meta'].get('Place_desc') if place['origin'] == "CSV/File" else ""
                        details = {
                            "description": raw_desc if raw_desc else (wiki_summ if wiki_summ else "No detailed description available."),
                            "timings": "9:00 AM - 6:00 PM (Approx)",
                            "season": "Oct - Mar",
                            "price": "Check Locally"
                        }

                # Render Layout
                d_col1, d_col2 = st.columns([2, 1])
                
                with d_col1:
                    st.markdown(f"### 📖 About {place['name']}")
                    st.write(details["description"])
                    
                    st.info(f"""
                    **Visitor Information:**
                    - 🕒 **Timings:** {details.get('timings')}
                    - 📅 **Best Season:** {details.get('season')}
                    - 🎟️ **Entry Fee:** {details.get('price')}
                    """)
                    
                    # Map Link
                    try:
                        # Use coords from API or fallback to city center
                        plat = place['meta'].get('lat', place.get('lat'))
                        plon = place['meta'].get('lon', place.get('lon'))
                        map_link = f"https://www.google.com/maps/dir/?api=1&origin={place['lat']},{place['lon']}&destination={plat},{plon}&travelmode=driving"
                        st.markdown(f"[🗺️ **Click for Google Maps Direction**]({map_link})")
                    except: pass

                with d_col2:
                    if img_url:
                        st.image(img_url, caption="Wikipedia Image", use_container_width=True)
                    else:
                        st.markdown("*(No verified image found)*")
                
                st.markdown("---")