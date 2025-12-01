# ================= PART 1/3 =================
# app.py — Part 1: imports, config, utils, data loading, RAG, image helpers

import streamlit as st
import requests
import pandas as pd
import json
import math
import time
import urllib.parse
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
DEFAULT_GEOAPIFY = "0ace8c8462a943b982df4fd2750d3407"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

INITIAL_SHOW = 15
PAGE_STEP = 10
MAX_GALLERY = 8
MAX_PLACE_GALLERY = 3  # show 3 images under info box
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
    except:
        out["places_df"] = pd.DataFrame()
    try:
        out["city_df"] = pd.read_csv(DATA_DIR / "City.csv")
    except:
        out["city_df"] = pd.DataFrame()
    try:
        out["tourism_json"] = json.load(open(DATA_DIR / "india_tourism_big.json", "r", encoding="utf8"))
    except:
        out["tourism_json"] = []
    try:
        out["faqs_json"] = json.load(open(DATA_DIR / "faqs_big.json", "r", encoding="utf8"))
    except:
        out["faqs_json"] = []
    return out

DATA = load_all()

# ---------------- RAG: build local_docs ----------------
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

# ---------------- VECTORIZE ----------------
@st.cache_resource
def build_vectorizer(docs):
    texts = [d["text"] for d in docs] if docs else [""]
    vect = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vect.fit_transform(texts)
    return vect, X

vectorizer, docs_X = build_vectorizer(local_docs)

def retrieve_similar(query, top_k=8):
    if not local_docs:
        return []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, docs_X).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    out = []
    for i in idxs:
        if sims[i] <= 0: continue
        out.append({"score": float(sims[i]), "doc": local_docs[i]})
    return out

# ---------------- GEOAPIFY helpers ----------------
def geocode(city):
    try:
        r = requests.get("https://api.geoapify.com/v1/geocode/search",
                         params={"text": city, "apiKey": st.session_state["geo_key"], "limit": 1}, timeout=8)
        if r.ok and r.json().get("features"):
            return r.json()["features"][0]["properties"]
    except:
        return None

def geo_places(lat, lon, category_key, limit=40):
    try:
        r = requests.get("https://api.geoapify.com/v2/places",
                         params={"categories": category_key, "filter": f"circle:{lon},{lat},9000", "limit": limit, "apiKey": st.session_state["geo_key"]},
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
    # try just name
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
    # 1. Wikipedia lead
    w = fetch_wikipedia_lead(name, city)
    if w and w not in gallery:
        gallery.append(w)
    # 2. Commons
    for u in fetch_commons_images(name, city, limit=limit):
        if u not in gallery:
            gallery.append(u)
            if len(gallery) >= limit: break
    # 3. Geoapify
    g = geoapify_image(props or {})
    if g and g not in gallery and len(gallery) < limit:
        gallery.append(g)
    # 4. Google images (optional)
    if allow_google and len(gallery) < limit:
        for u in google_images(f"{name} {city}", limit=limit - len(gallery)):
            if u not in gallery:
                gallery.append(u)
                if len(gallery) >= limit: break
    return gallery[:limit]

# helper to fetch extra images (used for place gallery)
def fetch_more_images(name, city, allow_google=False, needed=MAX_PLACE_GALLERY):
    return build_gallery(name, city, None, allow_google=allow_google, limit=needed)
# ================= PART 2/3 =================
# app.py — Part 2: main UI, search, listing, more-details block (place gallery under info), hotels text-only
# includes place map link, place weather under info, itinerary generator under weather, and multi-city comparison under itinerary

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

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide", page_title="City Explorer Pro")
st.title("🏙️ City Explorer Pro — Final")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.session_state["geo_key"] = st.text_input("Geoapify API Key", value=st.session_state.get("geo_key", DEFAULT_GEOAPIFY), type="password")
    st.session_state["ollama_enabled"] = st.checkbox("Enable Ollama (LLaMA-3)", value=st.session_state.get("ollama_enabled", False))
    st.session_state["ollama_url"] = st.text_input("Ollama URL", value=st.session_state.get("ollama_url", DEFAULT_OLLAMA_URL))
    st.session_state["allow_google_images"] = st.checkbox("Enable Google Image Scraping (optional)", value=False)
    st.markdown("---")
    st.write(f"Indexed docs: {len(local_docs)}")

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

        if not DATA["places_df"].empty:
            try:
                df = DATA["places_df"]
                city_rows = df[df["City"].str.contains(city, case=False, na=False)].fillna("")
            except:
                city_rows = pd.DataFrame()
            for _, r in city_rows.iterrows():
                nm = clean_string(r.get("Place", ""))
                if nm and nm.lower() not in seen:
                    candidates.append({"name": nm, "source": "csv", "meta": r.to_dict(), "text": r.get("Place_desc","")})
                    seen.add(nm.lower())

        for e in DATA["tourism_json"]:
            nm = e.get("place","")
            if nm and str(e.get("city","")).lower() == city.lower():
                if nm.lower() not in seen:
                    candidates.append({"name": nm, "source": "json", "meta": e, "text": e.get("description","")})
                    seen.add(nm.lower())

        for f in api_feats:
            props = f.get("properties", {}) if isinstance(f, dict) else f
            nm = props.get("name") or props.get("formatted")
            if nm and nm.lower() not in seen:
                candidates.append({"name": nm, "source": "api", "meta": props, "text": props.get("formatted","")})
                seen.add(nm.lower())

        def score_item(it):
            s = 0
            if it["source"] == "csv": s += 5
            if it["source"] == "json": s += 4
            meta = it.get("meta") or {}
            try: s += float(meta.get("rating") or 0)
            except: pass
            try: s += float(meta.get("popularity") or meta.get("importance") or 0) * 0.2
            except: pass
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

        # LIST DISPLAY
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

                # DETAILS PANEL (inline, Option A — Indented card)
                if st.session_state.get("selected_index") == i:
                    st.markdown("")  # spacer
                    with st.container():
                        st.markdown("---")
                        left, right = st.columns([3,1])

                        # LEFT: Text details + Info box + Place images (3-across)
                        with left:
                            pname = clean_string(place["name"])
                            meta = place.get("meta") or {}
                            # gather context
                            sims = retrieve_similar(pname + " " + city, top_k=8)
                            ctx = [s["doc"]["text"] for s in sims]
                            wiki = fetch_wikipedia_lead(pname, city)
                            full_ctx = "\n".join([c for c in ctx if c]) + "\n" + (wiki or "")

                            # AI (if enabled)
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
                                    except:
                                        details = None

                            if not details:
                                raw_desc = ""
                                try:
                                    if place["source"] == "csv":
                                        raw_desc = meta.get("Place_desc", "")
                                except:
                                    raw_desc = ""
                                details = {
                                    "description": raw_desc or wiki or (place.get("text") or f"{pname} in {city}."),
                                    "timings": "Varies",
                                    "season": "Varies",
                                    "price": "Varies",
                                    "highlights": []
                                }

                            st.markdown(f"### 📖 About {pname}")
                            st.write(details.get("description", ""))

                            # Info Box
                            city_best = get_city_best_time(city)
                            st.info(f"""
**Visitor Information**
- 🕒 **Timings:** {details.get('timings')}
- 📅 **Best Season:** {details.get('season')}
- 🎟️ **Entry Fee:** {details.get('price')}
- 🌤️ **City Best Time:** {city_best}
""")

                            # 3-column place image gallery (exactly under the info box)
                            allow_google = bool(st.session_state.get("allow_google_images", False))
                            place_imgs = fetch_more_images(pname, city, allow_google=allow_google, needed=MAX_PLACE_GALLERY)
                            # ensure we have up to 3 images (try wiki lead first)
                            if len(place_imgs) < MAX_PLACE_GALLERY:
                                lead = fetch_wikipedia_lead(pname, city)
                                if lead and lead not in place_imgs:
                                    place_imgs.insert(0, lead)
                            # limit to 3
                            place_imgs = place_imgs[:MAX_PLACE_GALLERY]

                            if place_imgs:
                                st.markdown("### 📸 Place Photos")
                                cols_imgs = st.columns(MAX_PLACE_GALLERY)
                                for idx_img, img_url in enumerate(place_imgs):
                                    if idx_img < len(cols_imgs):
                                        with cols_imgs[idx_img]:
                                            st.image(img_url, use_container_width=True)
                            else:
                                st.caption("No verified images found for this place.")

                            # PLACE MAP LINK (place-specific)
                            plat = None; plon = None
                            try:
                                # meta may contain lat/lon for API items
                                plat = meta.get("lat") or meta.get("latitude") or meta.get("LAT") or None
                                plon = meta.get("lon") or meta.get("longitude") or meta.get("LON") or None
                            except:
                                pass
                            try:
                                if plat and plon:
                                    map_url = f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}"
                                    st.markdown(f"[🗺️ Open place in Google Maps]({map_url})")
                            except:
                                pass

                            # PLACE WEATHER (for the place coordinates if available)
                            place_weather = None
                            try:
                                if plat and plon:
                                    place_weather = fetch_weather(float(plat), float(plon))
                            except:
                                place_weather = None

                            if place_weather:
                                st.markdown("### 🌤️ Weather at this place")
                                cw = place_weather.get("current_weather", {})
                                st.write(f"**Current:** {cw.get('temperature')}°C  |  wind {cw.get('windspeed')} km/h")
                                daily = place_weather.get("daily", {})
                                if daily:
                                    df = pd.DataFrame({
                                        "date": daily.get("time", []),
                                        "temp_max": daily.get("temperature_2m_max", []),
                                        "temp_min": daily.get("temperature_2m_min", []),
                                        "precip_prob": daily.get("precipitation_probability_mean", [])
                                    })
                                    st.table(df)
                            else:
                                st.info("Place-level weather not available (no coordinates).")

                            # ITINERARY GENERATOR (inside details, under weather)
                            st.markdown("### 🗓️ Itinerary generator (for this place / city)")
                            days_local = st.number_input(f"Days to plan for {pname}", min_value=1, max_value=7, value=1, key=f"days_{i}")
                            if st.button(f"Generate itinerary for {pname}", key=f"itin_{i}"):
                                # choose top places for city as input
                                top_places = [clean_string(x["name"]) for x in scored[:12]] if 'scored' in locals() else [pname]
                                with st.spinner("Generating itinerary..."):
                                    from_part = top_places
                                    # call LLM-based or fallback itinerary
                                    itin = generate_itinerary(pname if len(top_places)==1 else city, from_part, days=days_local)
                                    st.json(itin)

                            # MULTI-CITY COMPARISON (right under itinerary)
                            st.markdown("### 🔁 Multi-city comparison")
                            compare_input = st.text_input(f"Enter up to 4 cities to compare (comma separated) for {pname}", value=f"{city}, Mumbai, Delhi", key=f"cmp_{i}")
                            if st.button(f"Compare cities (for {pname})", key=f"cmp_btn_{i}"):
                                clist = [c.strip() for c in compare_input.split(",")][:4]
                                with st.spinner("Comparing..."):
                                    comp = compare_cities(clist, category=category)
                                    st.table(pd.DataFrame(comp))

                            st.markdown("---")

                            # Nearby hotels — text only (no images)
                            st.markdown("### 🏨 Nearby Hotels & Accommodation (text-only)")
                            hotels_hits = geo_places(city_lat, city_lon, "accommodation.hotel,accommodation", limit=20)
                            if not hotels_hits:
                                st.write("No nearby hotels found.")
                            else:
                                for hh in hotels_hits[:7]:
                                    hp = hh.get("properties", {})
                                    hname = hp.get("name") or hp.get("formatted") or "Unnamed Hotel"
                                    addr = hp.get("address_line1") or hp.get("address_line2") or hp.get("formatted") or "Address not available"
                                    rating = hp.get("rating") or "Not rated"
                                    st.markdown(f"**{hname}**  \n📍 {addr}  \n⭐ {rating}")
                                    if hp.get("lat") and hp.get("lon"):
                                        map_link = f"https://www.google.com/maps/dir/?api=1&destination={hp['lat']},{hp['lon']}"
                                        st.markdown(f"[View on Google Maps]({map_link})")
                                    st.markdown("---")

                        # RIGHT: small place gallery / quick facts
                        with right:
                            allow_google = bool(st.session_state.get("allow_google_images", False))
                            quick_gallery = fetch_more_images(clean_string(place["name"]), city, allow_google=allow_google, needed=MAX_PLACE_GALLERY)
                            if quick_gallery:
                                st.markdown("**Gallery (small)**")
                                for u in quick_gallery[:MAX_PLACE_GALLERY]:
                                    st.image(u, use_container_width=True)
                            else:
                                st.caption("No images.")

                        st.markdown("---")

        # Pagination controls
        st.markdown("---")
        if show_until < total:
            if st.button(f"Show more (next {PAGE_STEP})"):
                st.session_state["offsets"][key] = show_until
                st.rerun()
        else:
            st.info("End of results.")
# ================= PART 3/3 =================
# app.py — Part 3: Weather helper, itinerary + compare functions, (NO Notes & Tips footer)

# ---------------- WEATHER FUNCTION (used earlier) ----------------
def fetch_weather(lat, lon):
    try:
        u = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_mean&timezone=auto&forecast_days=3"
        r = requests.get(u, timeout=10)
        if r.ok:
            return r.json()
    except:
        return None

# ---------------- Itinerary generator (LLM with fallback) ----------------
def generate_itinerary(city_or_place, places, days=1):
    """
    Try Ollama LLM to create an itinerary JSON.
    Fallback to a simple greedy split if LLM not available or fails.
    """
    # If no places, return empty structure
    if not places:
        return {"days": [{"day": d+1, "schedule": []} for d in range(days)]}

    # LLM attempt
    if st.session_state.get("ollama_enabled", False):
        system = "You are a helpful travel planner. Return ONLY valid JSON with key 'days' containing list of days and schedules."
        user = f"City/Place: {city_or_place}\nPlaces:\n" + "\n".join(places[:30]) + f"\nDays: {days}\nReturn JSON only."
        resp = ollama_call(system, user)
        if resp:
            try:
                for c in resp.get("choices", []):
                    msg = c.get("message") or {}
                    text = msg.get("content") or c.get("content")
                    j = extract_json_from_text(text)
                    if j:
                        return j
            except:
                pass

    # Fallback simple planner: split places across days
    per = max(1, math.ceil(len(places) / days))
    itinerary = {"days": []}
    idx = 0
    for d in range(days):
        schedule = []
        hour = 9
        for _ in range(per):
            if idx >= len(places): break
            schedule.append({"time": f"{hour:02d}:00", "place": places[idx], "note": "Suggested visit 1-2 hours"})
            hour += 2
            idx += 1
        itinerary["days"].append({"day": d+1, "schedule": schedule})
    return itinerary

# ---------------- Multi-city comparison (used earlier) ----------------
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
            p = f.get("properties", {})
            n = p.get("name") or p.get("formatted")
            if n: top_names.append(n)
            r = p.get("rating") or p.get("properties", {}).get("rating")
            if r:
                try: ratings.append(float(r))
                except: pass
        avg_rating = round(sum(ratings)/len(ratings), 2) if ratings else None
        out.append({"city": city, "count": count, "top": top_names, "avg_rating": avg_rating})
    return out

# ---------------- END: no Notes & Tips footer (removed as requested) ----------------

# small confirmation caption
st.caption("App ready — place map links, place-level weather, itinerary and multi-city comparison are available inside the More Details card.")
