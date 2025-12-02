# app.py — Streamlit app with ChromaDB Vector Search
import streamlit as st
import requests
import pandas as pd
import json
import math
import urllib.parse
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup

# --- NEW IMPORTS FOR CHROMA ---
import chromadb
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
# Create a folder for the vector db if it doesn't exist
CHROMA_DB_PATH = "chroma_db_store"
DEFAULT_GEOAPIFY = "0ace8c8462a943b982df4fd2750d3407"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

INITIAL_SHOW = 15
PAGE_STEP = 10
MAX_GALLERY = 8
MAX_PLACE_GALLERY = 3
MAX_HOTEL_GALLERY = 5

# ---------------- UTIL ----------------
def clean_string(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"^\s*\d+[\.\)\-]*\s*", "", text).strip()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=3600)
def load_all():
    out = {}
    try:
        out["places_df"] = pd.read_csv(DATA_DIR / "Places.csv")
    except Exception:
        out["places_df"] = pd.DataFrame()
    try:
        out["city_df"] = pd.read_csv(DATA_DIR / "City.csv")
    except Exception:
        out["city_df"] = pd.DataFrame()
    try:
        out["tourism_json"] = json.load(open(DATA_DIR / "india_tourism_big.json", "r", encoding="utf8"))
    except Exception:
        out["tourism_json"] = []
    try:
        out["faqs_json"] = json.load(open(DATA_DIR / "faqs_big.json", "r", encoding="utf8"))
    except Exception:
        out["faqs_json"] = []
    return out

DATA = load_all()

# ---------------- PREPARE DOCS FOR DB ----------------
local_docs = []

if not DATA["places_df"].empty:
    for i, row in DATA["places_df"].fillna("").iterrows():
        name = clean_string(str(row.get("Place", "")))
        city = str(row.get("City", ""))
        desc = str(row.get("Place_desc", ""))
        local_docs.append({
            "id": f"csv_{i}",
            "title": name,
            "city": city,
            "text": f"{name} {city} {desc}",
            "type": "place_csv",
            "meta": row.to_dict()
        })

for i, e in enumerate(DATA["tourism_json"]):
    nm = e.get("place", "")
    ct = e.get("city", "")
    desc = e.get("description", "")
    local_docs.append({
        "id": f"json_{i}",
        "title": nm,
        "city": ct,
        "text": f"{nm} {ct} {desc}",
        "type": "place_json",
        "meta": e
    })

for i, e in enumerate(DATA["faqs_json"]):
    q = e.get("question", ""); a = e.get("answer", "")
    local_docs.append({
        "id": f"faq_{i}",
        "title": "FAQ",
        "city": "",
        "text": f"{q} {a}",
        "type": "faq",
        "meta": e
    })

if not DATA["city_df"].empty:
    for i, row in DATA["city_df"].fillna("").iterrows():
        c = str(row.get("City", ""))
        desc = str(row.get("City_desc", ""))
        best = str(row.get("Best_time_to_visit", ""))
        local_docs.append({
            "id": f"city_{i}",
            "title": c,
            "city": c,
            "text": f"{c} {desc} {best}",
            "type": "city_info",
            "meta": row.to_dict()
        })

# ---------------- CHROMA DB SETUP ----------------
@st.cache_resource
def setup_chromadb(docs):
    """
    Initializes ChromaDB, creates a collection, and ingests documents if empty.
    Using 'all-MiniLM-L6-v2' for efficient local embeddings.
    """
    # 1. Initialize Client (Persistent)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 2. Define Embedding Function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 3. Get or Create Collection
    collection = client.get_or_create_collection(name="city_explorer_docs", embedding_function=ef)
    
    # 4. Check if data exists; if not, ingest
    if collection.count() == 0 and docs:
        print("Ingesting documents into ChromaDB...")
        ids = []
        documents = []
        metadatas = []
        
        for d in docs:
            ids.append(d["id"])
            documents.append(d["text"])
            
            # Chroma metadata must be flat and simple types (str, int, float, bool)
            # We clean the 'meta' dict to ensure no nested lists/dicts exist
            clean_meta = {}
            original_meta = d.get("meta", {})
            for k, v in original_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v) # Stringify complex objects
            
            # Add top-level fields to metadata for easier retrieval later
            clean_meta["_type"] = d.get("type", "")
            clean_meta["_city"] = d.get("city", "")
            clean_meta["_title"] = d.get("title", "")
            
            metadatas.append(clean_meta)
            
        # Add in batches to be safe
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        print(f"Ingestion complete. {collection.count()} documents indexed.")
        
    return collection

# Initialize Chroma
collection = setup_chromadb(local_docs)

def retrieve_similar(query, top_k=8):
    """
    Queries ChromaDB for semantic similarity.
    """
    if collection.count() == 0:
        return []
        
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    out = []
    # Chroma structure: results['ids'][0] is a list of ids
    if results and results.get("ids"):
        ids = results["ids"][0]
        distances = results["distances"][0]
        metas = results["metadatas"][0]
        texts = results["documents"][0]
        
        for i in range(len(ids)):
            # Convert metadata back to the structure the app expects
            # We stored original meta fields flattened.
            m = metas[i]
            
            # Reconstruct the 'doc' object
            doc_obj = {
                "id": ids[i],
                "title": m.get("_title", ""),
                "city": m.get("_city", ""),
                "text": texts[i],
                "type": m.get("_type", ""),
                "meta": m 
            }
            
            # Distance to Score (approximate). Chroma default is L2 distance.
            # Lower distance = better match.
            score = 1.0 / (1.0 + distances[i]) 
            
            out.append({"score": score, "doc": doc_obj})
            
    return out

# ---------------- GEOAPIFY helpers ----------------
def geocode(city):
    try:
        r = requests.get("https://api.geoapify.com/v1/geocode/search",
                         params={"text": city, "apiKey": st.session_state.get("geo_key", DEFAULT_GEOAPIFY), "limit": 1}, timeout=8)
        if r.ok and r.json().get("features"):
            return r.json()["features"][0]["properties"]
    except:
        return None

def geo_places(lat, lon, category_key, limit=40):
    try:
        r = requests.get("https://api.geoapify.com/v2/places",
                         params={"categories": category_key, "filter": f"circle:{lon},{lat},9000", "limit": limit, "apiKey": st.session_state.get("geo_key", DEFAULT_GEOAPIFY)},
                         timeout=10)
        if r.ok:
            return r.json().get("features", [])
    except:
        return []
    return []

# ---------------- IMAGE HELPERS ----------------
def fetch_wikipedia_lead(name, city=None):
    try:
        q = f"{name} {city}" if city else name
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(q)
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            d = r.json()
            if d.get("originalimage"): return d["originalimage"]["source"]
            if d.get("thumbnail"): return d["thumbnail"]["source"]
    except:
        pass
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(name)
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            d = r.json()
            if d.get("originalimage"): return d["originalimage"]["source"]
            if d.get("thumbnail"): return d["thumbnail"]["source"]
    except:
        pass
    return None

def fetch_commons_images(name, city=None, limit=6):
    out = []
    try:
        q = f"{name} {city}" if city else name
        api = "https://commons.wikimedia.org/w/api.php"
        params = {"action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 10, "prop": "imageinfo", "iiprop": "url", "format": "json"}
        r = requests.get(api, params=params, timeout=6)
        if r.ok:
            pages = r.json().get("query", {}).get("pages", {})
            for _, p in pages.items():
                ii = p.get("imageinfo")
                if ii and isinstance(ii, list):
                    url = ii[0].get("url")
                    if url:
                        out.append(url)
                        if len(out) >= limit:
                            break
    except:
        pass
    return out

def geoapify_image(props):
    try:
        for k in ("image", "photo", "img", "thumbnail", "image_url"):
            v = props.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
        raw = (props.get("datasource") or {}).get("raw") or {}
        if isinstance(raw, dict):
            for k in ("image", "photo", "thumbnail", "image_url"):
                v = raw.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    return v
    except:
        pass
    return None

def google_images(query, limit=6):
    out = []
    try:
        url = "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote(query)
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.ok:
            soup = BeautifulSoup(r.text, "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src and src.startswith("http"):
                    if src not in out:
                        out.append(src)
                        if len(out) >= limit:
                            break
    except:
        pass
    return out

def build_gallery(name, city, props=None, allow_google=False, limit=MAX_GALLERY):
    gallery = []
    w = fetch_wikipedia_lead(name, city)
    if w and w not in gallery:
        gallery.append(w)
    for u in fetch_commons_images(name, city, limit=limit):
        if u not in gallery:
            gallery.append(u)
            if len(gallery) >= limit: break
    g = geoapify_image(props or {})
    if g and g not in gallery and len(gallery) < limit:
        gallery.append(g)
    if allow_google and len(gallery) < limit:
        for u in google_images(f"{name} {city}", limit=limit - len(gallery)):
            if u not in gallery:
                gallery.append(u)
                if len(gallery) >= limit: break
    return gallery[:limit]

def fetch_more_images(name, city, allow_google=False, needed=MAX_PLACE_GALLERY):
    return build_gallery(name, city, None, allow_google=allow_google, limit=needed)

# ---------------- LLM helpers & JSON extractor ----------------
def ollama_call(system, user):
    if not st.session_state.get("ollama_enabled", False):
        return None
    try:
        payload = {"model": "llama3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.25}
        r = requests.post(st.session_state.get("ollama_url", DEFAULT_OLLAMA_URL), json=payload, timeout=25)
        if r.ok:
            return r.json()
    except:
        return None
    return None

def extract_json_from_text(text):
    if not text: return None
    try: return json.loads(text)
    except:
        s = text.find("{")
        if s == -1: return None
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[s:i+1])
                    except: return None
    return None

# ---------------- City & place helper getters ----------------
def get_city_best_time(city):
    df = DATA["city_df"]
    if df.empty: return "Varies"
    try:
        row = df[df["City"].str.contains(city, case=False, na=False)]
        if row.empty: return "Varies"
        return row.iloc[0].get("Best_time_to_visit") or "Varies"
    except:
        return "Varies"

def get_city_extra(city):
    df = DATA["city_df"]
    if df.empty: return None
    try:
        row = df[df["City"].str.lower() == city.lower()]
        if row.empty: return None
        row = row.iloc[0].to_dict()
        return {"best_time": row.get("Best_time_to_visit"), "rating": row.get("City_rating") or row.get("Rating")}
    except:
        return None

def get_place_rating(place):
    df = DATA["places_df"]
    if df.empty: return None
    try:
        row = df[df["Place"].str.lower() == place.lower()]
        if row.empty: return None
        row = row.iloc[0].to_dict()
        return row.get("Rating") or row.get("Place_rating") or row.get("Stars")
    except:
        return None

# ---------------- WEATHER FUNCTION ----------------
def fetch_weather(lat, lon):
    try:
        u = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_mean&timezone=auto&forecast_days=3"
        r = requests.get(u, timeout=10)
        if r.ok:
            return r.json()
    except:
        return None

# ---------------- Itinerary generator ----------------
def generate_itinerary(city_or_place, places, days=1):
    if not places:
        return "\n".join([f"Day {d+1}: Explore {city_or_place}." for d in range(days)])

    if st.session_state.get("ollama_enabled", False):
        system = "You are a travel planner. Return only JSON with key 'days' mapping to list of {day:int, schedule:[str,...]}."
        user = f"City/Place: {city_or_place}\nPlaces:\n" + "\n".join(places[:30]) + f"\nDays: {days}\nReturn JSON only."
        resp = ollama_call(system, user)
        if resp:
            try:
                for c in resp.get("choices", []):
                    txt = (c.get("message", {}) or {}).get("content") or c.get("content")
                    parsed = extract_json_from_text(txt)
                    if parsed and "days" in parsed:
                        lines = []
                        for d in parsed["days"]:
                            daynum = d.get("day") or d.get("day_num") or (len(lines)+1)
                            schedule = d.get("schedule") or d.get("places") or []
                            if isinstance(schedule, list):
                                lines.append(f"Day {daynum}: " + " | ".join([str(x) for x in schedule]))
                            else:
                                lines.append(f"Day {daynum}: {schedule}")
                        return "\n".join(lines)
            except:
                pass

    per = max(1, math.ceil(len(places) / days))
    idx = 0
    lines = []
    for d in range(days):
        picks = []
        for _ in range(per):
            if idx >= len(places): break
            picks.append(places[idx])
            idx += 1
        lines.append(f"Day {d+1}: " + ", ".join(picks) if picks else f"Day {d+1}: Free / explore {city_or_place}")
    return "\n".join(lines)

# ---------------- Multi-city comparison ----------------
def compare_cities(cities, category="Tourist Attractions"):
    cat_map = {"Tourist Attractions":"tourism.sights,tourism.attraction", "Temples":"religion.place_of_worship", "Restaurants":"catering.restaurant", "Hotels":"accommodation.hotel"}
    out = []
    for city in cities:
        city = city.strip()
        if not city:
            continue
        props = geocode(city)
        if not props:
            out.append({"city": city, "error": "City not found"})
            continue
        lat = props.get("lat"); lon = props.get("lon")
        feats = geo_places(lat, lon, cat_map.get(category, "tourism.sights"), limit=30)
        count = len(feats)
        top_names = []
        ratings = []
        for f in feats[:8]:
            p = f.get("properties", {}) if isinstance(f, dict) else f
            n = p.get("name") or p.get("formatted")
            if n: top_names.append(n)
            r = p.get("rating") or p.get("properties", {}).get("rating") if isinstance(p.get("properties", {}), dict) else None
            if r is None:
                r = p.get("importance") or p.get("popularity")
            if r:
                try: ratings.append(float(r))
                except: pass
        avg_rating = round(sum(ratings)/len(ratings), 2) if ratings else None
        weather = None
        try:
            w = fetch_weather(lat, lon)
            if w and w.get("current_weather"):
                cw = w["current_weather"]
                weather = f"{cw.get('temperature')}°C wind {cw.get('windspeed')} km/h"
        except:
            weather = None
        out.append({"city": city, "count": count, "top": top_names, "avg_rating": avg_rating, "weather": weather})
    return out

# ---------------- CATEGORY MATCHING ----------------
def matches_category(text, category):
    if not text or not isinstance(text, str):
        return False
    t = text.lower()
    keywords = {
        "Tourist Attractions": ["museum", "monument", "attraction", "sight", "fort", "palace", "garden", "park", "beach", "zoo", "tower", "viewpoint", "archaeological", "heritage"],
        "Temples": ["temple", "shrine", "mosque", "church", "gurudwara", "stupa", "mandir", "kovil"],
        "Restaurants": ["restaurant", "cafe", "eatery", "diner", "bistro", "bar", "pub", "food", "canteen"],
        "Hotels": ["hotel", "resort", "lodg", "guesthouse", "inn", "accommodation", "stay", "bnb", "hostel"]
    }
    kwlist = keywords.get(category, [])
    for kw in kwlist:
        if kw in t:
            return True
    if category == "Tourist Attractions":
        place_like = ["tour", "visit", "attraction", "sight", "monument", "museum", "heritage", "park", "garden"]
        for p in place_like:
            if p in t:
                return True
    return False

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide", page_title="City Explorer Pro + ChromaDB")
st.title("🏙️ City Explorer Pro (Powered by ChromaDB RAG)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if "geo_key" not in st.session_state:
        st.session_state["geo_key"] = DEFAULT_GEOAPIFY
    st.session_state["geo_key"] = st.text_input("Geoapify API Key", value=st.session_state.get("geo_key", DEFAULT_GEOAPIFY), type="password")
    st.session_state["ollama_enabled"] = st.checkbox("Enable Ollama (LLaMA-3)", value=st.session_state.get("ollama_enabled", False))
    if "ollama_url" not in st.session_state:
        st.session_state["ollama_url"] = DEFAULT_OLLAMA_URL
    st.session_state["ollama_url"] = st.text_input("Ollama URL", value=st.session_state.get("ollama_url", DEFAULT_OLLAMA_URL))
    st.session_state["allow_google_images"] = st.checkbox("Enable Google Image Scraping", value=st.session_state.get("allow_google_images", False))
    st.markdown("---")
    st.success(f"Chroma Docs: {collection.count()}")

# Search row
c1, c2, c3 = st.columns([3,2,1])
city_inp = c1.text_input("City", "Chennai")
cat_inp = c2.selectbox("Category", ["Tourist Attractions", "Temples", "Restaurants", "Hotels"])
if c3.button("Search"):
    st.session_state["search_city"] = city_inp.strip()
    st.session_state["search_category"] = cat_inp
    st.session_state["offsets"] = {}
    st.session_state["selected_index"] = None

# When searched
if "search_city" in st.session_state:
    city = st.session_state["search_city"]
    category = st.session_state["search_category"]

    coords = geocode(city)
    if not coords:
        st.error("City not found (check Geoapify key / spelling).")
    else:
        city_lat = coords.get("lat"); city_lon = coords.get("lon")
        st.success(f"Located: {coords.get('formatted', city)}")

        cat_map = {
            "Tourist Attractions": "tourism.sights,tourism.attraction",
            "Temples": "religion.place_of_worship",
            "Restaurants": "catering.restaurant",
            "Hotels": "accommodation.hotel"
        }

        api_feats = geo_places(city_lat, city_lon, cat_map[category], limit=80)

        # Merge local CSV/JSON + API
        candidates = []; seen = set()

        # 1. API Results
        for f in api_feats:
            props = f.get("properties", {}) if isinstance(f, dict) else f
            nm = props.get("name") or props.get("formatted")
            if nm and nm.lower() not in seen:
                candidates.append({"name": nm, "source": "api", "meta": props, "text": props.get("formatted","")})
                seen.add(nm.lower())

        # 2. Local CSV (Filtered)
        if not DATA["places_df"].empty:
            try:
                df = DATA["places_df"]
                city_rows = df[df["City"].str.contains(city, case=False, na=False)].fillna("")
            except Exception:
                city_rows = pd.DataFrame()
            for _, r in city_rows.iterrows():
                nm = clean_string(r.get("Place", ""))
                txt = " ".join([str(r.get("Place", "")), str(r.get("Place_desc", "")), str(r.get("Category", ""))]).strip()
                if nm and nm.lower() not in seen:
                    if matches_category(txt, category):
                        candidates.append({"name": nm, "source": "csv", "meta": r.to_dict(), "text": r.get("Place_desc","")})
                        seen.add(nm.lower())

        # 3. Local JSON (Filtered)
        for e in DATA["tourism_json"]:
            nm = e.get("place","")
            txt = " ".join([str(e.get("place","")), str(e.get("description","")), str(e.get("tags",""))])
            if nm and str(e.get("city","")).lower() == city.lower():
                if nm.lower() not in seen and matches_category(txt, category):
                    candidates.append({"name": nm, "source": "json", "meta": e, "text": e.get("description","")})
                    seen.add(nm.lower())

        # Fallback
        if len(candidates) < 6 and category == "Tourist Attractions" and not DATA["places_df"].empty:
            try:
                df = DATA["places_df"]
                city_rows = df[df["City"].str.contains(city, case=False, na=False)].fillna("")
                for _, r in city_rows.iterrows():
                    nm = clean_string(r.get("Place", ""))
                    if nm and nm.lower() not in seen:
                        candidates.append({"name": nm, "source": "csv", "meta": r.to_dict(), "text": r.get("Place_desc","")})
                        seen.add(nm.lower())
                        if len(candidates) >= 12: break
            except Exception: pass

        # Scoring
        def score_item(it):
            s = 0
            if it["source"] == "api": s += 6
            if it["source"] == "csv": s += 4
            if it["source"] == "json": s += 3
            meta = it.get("meta") or {}
            try: s += float(meta.get("rating") or meta.get("Rating") or 0)
            except: pass
            name = (it.get("name") or "").lower()
            if category == "Restaurants" and ("restaurant" in name or "cafe" in name): s += 2
            return s

        scored = sorted(candidates, key=lambda x: score_item(x), reverse=True)

        # Pagination
        key = city.lower() + "_" + category.replace(" ","_").lower()
        if "offsets" not in st.session_state:
            st.session_state["offsets"] = {}
        if key not in st.session_state["offsets"]:
            st.session_state["offsets"][key] = 0
        offset = st.session_state["offsets"][key]
        show_until = offset + (INITIAL_SHOW if offset == 0 else PAGE_STEP)
        shown = scored[:show_until]
        total = len(scored)

        st.markdown("---")
        st.subheader(f"Top {len(shown)} (of {total}) — {category}")

        for i, place in enumerate(shown):
            with st.container():
                num = i+1
                col_a, col_b = st.columns([4,1])
                with col_a:
                    st.markdown(f"**{num}. {clean_string(place['name'])}**")
                    if place.get("text"):
                        st.caption(place["text"][:120] + "...")
                with col_b:
                    if st.button("More Details", key=f"btn_{key}_{i}"):
                        st.session_state["selected_index"] = i
                        st.session_state["selected_item"] = place
                        st.session_state["city_lat"] = city_lat
                        st.session_state["city_lon"] = city_lon
                        st.rerun()

                # DETAILS PANEL
                if st.session_state.get("selected_index") == i:
                    st.markdown("")
                    with st.container():
                        st.markdown("---")
                        left, right = st.columns([3,1])

                        with left:
                            pname = clean_string(place["name"])
                            meta = place.get("meta") or {}
                            
                            # RAG: RETRIEVE FROM CHROMA
                            sims = retrieve_similar(pname + " " + city, top_k=8)
                            ctx = [s["doc"]["text"] for s in sims]
                            wiki = fetch_wikipedia_lead(pname, city)
                            full_ctx = "\n".join([c for c in ctx if c]) + "\n" + (wiki or "")

                            # AI call
                            details = None
                            if st.session_state.get("ollama_enabled", False):
                                system = "You are a travel expert. Return only JSON {description, timings, season, price, highlights}."
                                user = f"Place: {pname}\nCity: {city}\nContext:\n{full_ctx}\nReturn JSON only."
                                resp = ollama_call(system, user)
                                if resp:
                                    try:
                                        for c in resp.get("choices", []):
                                            txt = c.get("message", {}).get("content") or c.get("content")
                                            parsed = extract_json_from_text(txt)
                                            if parsed:
                                                details = parsed
                                                break
                                    except: details = None

                            if not details:
                                raw_desc = ""
                                try:
                                    if place["source"] == "csv": raw_desc = meta.get("Place_desc", "")
                                except: pass
                                details = {
                                    "description": raw_desc or wiki or (place.get("text") or f"{pname} in {city}."),
                                    "timings": "Varies",
                                    "season": "Varies",
                                    "price": "Varies",
                                    "highlights": []
                                }

                            st.markdown(f"### 📖 About {pname}")
                            st.write(details.get("description", ""))

                            city_best = get_city_best_time(city)
                            st.info(f"""
**Visitor Information**
- 🕒 **Timings:** {details.get('timings')}
- 🎟️ **Entry Fee:** {details.get('price')}
- 🌤️ **City Best Time:** {city_best}
""")

                            # Gallery
                            allow_google = bool(st.session_state.get("allow_google_images", False))
                            place_imgs = fetch_more_images(pname, city, allow_google=allow_google, needed=MAX_PLACE_GALLERY)
                            if len(place_imgs) < MAX_PLACE_GALLERY:
                                lead = fetch_wikipedia_lead(pname, city)
                                if lead and lead not in place_imgs:
                                    place_imgs.insert(0, lead)
                            place_imgs = place_imgs[:MAX_PLACE_GALLERY]

                            if place_imgs:
                                st.markdown("### 📸 Place Photos")
                                cols_imgs = st.columns(MAX_PLACE_GALLERY)
                                for idx_img, img_url in enumerate(place_imgs):
                                    if idx_img < len(cols_imgs):
                                        with cols_imgs[idx_img]:
                                            st.image(img_url, use_container_width=True)
                            else:
                                st.caption("No verified images found.")

                            # Map & Weather
                            plat = None; plon = None
                            try:
                                plat = meta.get("lat") or meta.get("latitude")
                                plon = meta.get("lon") or meta.get("longitude")
                            except: pass
                            
                            if plat and plon:
                                map_url = f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}"
                                st.markdown(f"[🗺️ Open place in Google Maps]({map_url})")
                                
                                place_weather = fetch_weather(float(plat), float(plon))
                                if place_weather:
                                    st.markdown("### 🌤️ Weather at this place")
                                    cw = place_weather.get("current_weather", {})
                                    st.write(f"**Current:** {cw.get('temperature')}°C  |  wind {cw.get('windspeed')} km/h")

                            # Itinerary
                            st.markdown("### 🗓️ Itinerary generator")
                            days_local = st.number_input(f"Days to plan for {pname}", min_value=1, max_value=7, value=1, key=f"days_{i}")
                            if st.button(f"Generate itinerary for {pname}", key=f"itin_{i}"):
                                top_places = [clean_string(x["name"]) for x in scored[:12]] if 'scored' in locals() and scored else [pname]
                                itin = generate_itinerary(pname if len(top_places)==1 else city, top_places, days=int(days_local))
                                st.text(itin)

                            # Compare
                            st.markdown("### 🔁 Multi-city comparison")
                            compare_input = st.text_input(f"Cities to compare for {pname}", value=f"{city}, Mumbai, Delhi", key=f"cmp_{i}")
                            if st.button(f"Compare cities", key=f"cmp_btn_{i}"):
                                clist = [c.strip() for c in compare_input.split(",")][:4]
                                comp = compare_cities(clist, category=category)
                                st.table(pd.DataFrame(comp))
                            
                            st.markdown("---")
                            
                            # Nearby Hotels
                            st.markdown("### 🏨 Nearby Hotels")
                            hotels_hits = geo_places(city_lat, city_lon, "accommodation.hotel,accommodation", limit=20)
                            if not hotels_hits:
                                st.write("No nearby hotels found.")
                            else:
                                for hh in hotels_hits[:5]:
                                    hp = hh.get("properties", {})
                                    st.markdown(f"**{hp.get('name', 'Unnamed Hotel')}** - ⭐ {hp.get('rating', 'NR')}")

                        with right:
                            allow_google = bool(st.session_state.get("allow_google_images", False))
                            quick_gallery = fetch_more_images(clean_string(place["name"]), city, allow_google=allow_google, needed=MAX_PLACE_GALLERY)
                            if quick_gallery:
                                st.markdown("**Gallery (small)**")
                                for u in quick_gallery[:MAX_PLACE_GALLERY]:
                                    st.image(u, use_container_width=True)
                        st.markdown("---")

        st.markdown("---")
        if show_until < total:
            if st.button(f"Show more (next {PAGE_STEP})"):
                st.session_state["offsets"][key] = show_until
                st.rerun()