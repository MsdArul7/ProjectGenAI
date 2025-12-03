import streamlit as st
import requests
import pandas as pd
import sqlite3 
import random
import urllib.parse
import re
import math
import datetime
import time
from bs4 import BeautifulSoup
import folium
from streamlit_folium import st_folium
from fpdf import FPDF

# --- 🔒 API CONFIGURATION ---
GEO_SEARCH_KEY = "0ace8c8462a943b982df4fd2750d3407" 
GEO_DETAILS_KEY = "e93a6200c4bb4963b2516daea5537422" 
OLLAMA_URL = "http://localhost:11434/api/chat"

DB_NAME = "travel_master_v17.db"
CHROMA_PATH = "./chroma_db_travel"

# --- DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, 
            password TEXT, 
            email TEXT, 
            phone TEXT
    )''')

    tables = [
        "search_history (id INTEGER PRIMARY KEY AUTOINCREMENT, city TEXT, category TEXT, search_date DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "itinerary_history (id INTEGER PRIMARY KEY AUTOINCREMENT, city TEXT, place TEXT, days INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "comparison_history (id INTEGER PRIMARY KEY AUTOINCREMENT, city_1 TEXT, city_2 TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "holiday_history (id INTEGER PRIMARY KEY AUTOINCREMENT, year INTEGER, duration TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "login_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, login_time DATETIME DEFAULT CURRENT_TIMESTAMP)" 
    ]
    for t in tables:
        c.execute(f"CREATE TABLE IF NOT EXISTS {t}")
    
    c.execute("INSERT OR IGNORE INTO users (username, password, email, phone) VALUES ('admin', 'travel123', 'admin@travel.com', '9999999999')")
    conn.commit()
    conn.close()

# --- CHROMADB (Placeholder) ---
def init_chroma(): return None
def store_in_chroma(collection, place_name, city, description, category): pass
def query_chroma(collection, query_text): return ""

# --- AUTH FUNCTIONS ---
def check_login(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    if data and data[0] == password:
        return True
    return False

def create_account(username, password, email, phone):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email, phone) VALUES (?, ?, ?, ?)", (username, password, email, phone))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False 

# --- LOGGING ---
def log_login(username):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO login_history (username) VALUES (?)", (username,))

def log_search(city, cat):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO search_history (city, category) VALUES (?, ?)", (city, cat))

def log_itinerary(city, place, days):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO itinerary_history (city, place, days) VALUES (?, ?, ?)", (city, place, days))

def log_comparison(c1, c2):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO comparison_history (city_1, city_2) VALUES (?, ?)", (c1, c2))

def log_holiday(year, dur):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO holiday_history (year, duration) VALUES (?, ?)", (year, dur))

# --- CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1452421822248-d4c2b47f0c81?q=80&w=2070&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }
        h1, h2, h3, h4, p, label, .stMarkdown, .stDataFrame, .stTable { color: #ffffff !important; }
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input, .stRadio>div { 
            background-color: rgba(0, 0, 0, 0.6) !important; 
            color: white !important; 
            border: 1px solid #555;
            border-radius: 8px;
        }
        .detail-box { 
            background: rgba(0, 0, 0, 0.75); 
            padding: 20px; 
            border-radius: 15px; 
            margin-bottom: 10px; 
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
        }
        .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px;
            margin-top: 5px;
        }
        .itinerary-box { 
            background: rgba(0, 0, 0, 0.6); 
            padding: 15px; 
            border-radius: 10px; 
            margin-bottom: 10px; 
            border-left: 4px solid #00d2ff; 
        }
        .highlight-box { 
            background: rgba(255, 215, 0, 0.1); 
            padding: 15px; 
            border-radius: 10px; 
            margin-top: 15px; 
            border-left: 4px solid #FFD700; 
        }
        .nearby-card { 
            background: rgba(0, 0, 0, 0.6); 
            padding: 12px; 
            border-radius: 10px; 
            margin-bottom: 10px; 
            border-left: 4px solid #ff9966; 
        }
        .map-link { color: #4facfe; font-weight: bold; text-decoration: none; margin-left: 10px; }
        .desc-text { line-height: 1.6; font-size: 15px; color: #eee; text-align: justify; }
        .tag { background-color: #444; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 5px; border: 1px solid #666; display: inline-block; margin-bottom: 5px;}
        .rating { color: #FFD700; font-weight: bold; }
        .login-box { max-width: 400px; margin: auto; padding: 30px; background: rgba(0,0,0,0.8); border-radius: 15px; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)

# --- API HELPERS ---
def geocode(city):
    try:
        url = "https://api.geoapify.com/v1/geocode/search"
        params = {"text": city, "apiKey": GEO_SEARCH_KEY, "limit": 1}
        r = requests.get(url, params=params, timeout=5)
        if r.ok and r.json().get("features"):
            return r.json()["features"][0]["properties"]
    except: return None

def get_places_api(lat, lon, categories, limit=100, radius=15000):
    try:
        url = "https://api.geoapify.com/v2/places"
        params = {"categories": categories, "filter": f"circle:{lon},{lat},{radius}", "limit": limit, "apiKey": GEO_SEARCH_KEY}
        r = requests.get(url, params=params, timeout=10)
        if r.ok: return r.json().get("features", [])
        return []
    except: return []

def fetch_place_details(place_id):
    try:
        url = f"https://api.geoapify.com/v2/place-details"
        params = {"id": place_id, "apiKey": GEO_DETAILS_KEY}
        r = requests.get(url, params=params, timeout=5)
        if r.ok: return r.json().get("features", [{}])[0].get("properties", {})
        return {}
    except: return {}

def fetch_wiki_data(place_name):
    try:
        query = urllib.parse.quote(place_name)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        r = requests.get(url, timeout=3)
        if r.ok:
            data = r.json()
            return {
                "extract": data.get("extract", ""),
                "image": data.get("thumbnail", {}).get("source"),
                "description": data.get("description", "")
            }
    except: pass
    return None

def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r = requests.get(url, timeout=3)
        return r.json().get("current_weather", {})
    except: return {}

def get_weather_desc(wmo_code):
    if wmo_code == 0: return "Clear Sky ☀️"
    if wmo_code in [1, 2, 3]: return "Partly Cloudy ⛅"
    if wmo_code in [45, 48]: return "Foggy 🌫️"
    if wmo_code in [51, 53, 55, 61, 63, 65]: return "Rainy 🌧️"
    if wmo_code >= 80: return "Thunderstorm ⛈️"
    return "Cloudy ☁️"

def google_scrape_images(query, limit=1):
    out = []
    try:
        url = "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote(query)
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and "http" in src and "google" not in src:
                out.append(src)
                if len(out) >= limit: break
    except: pass
    return out

# --- NEW HELPERS ---

def calculate_budget(days, people, style):
    if style == "Backpacker 🎒":
        daily = 1500
        desc = "Hostels, Street Food, Public Transport"
    elif style == "Comfort 🧳":
        daily = 5000
        desc = "3-Star Hotels, Restaurants, Cabs"
    else: # Luxury
        daily = 12000
        desc = "5-Star Hotels, Fine Dining, Private Car"
    total = daily * days * people
    return total, desc

def get_packing_list(weather_code, category):
    items = ["✅ ID Proofs & Wallet", "✅ Power Bank & Charger", "✅ Water Bottle"]
    if weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: 
        items.extend(["☔ Umbrella/Raincoat", "👟 Waterproof Shoes", "🧥 Windcheater"])
    elif weather_code == 0 or weather_code == 1: 
        items.extend(["🕶️ Sunglasses", "🧢 Hat/Cap", "🧴 Sunscreen"])
    if "temple" in category.lower():
        items.extend(["🥻 Conservative Clothing", "🧣 Scarf/Shawl"])
    elif "beach" in category.lower():
        items.extend(["🩳 Swimwear", "🩴 Flip Flops", "🧖‍♀️ Extra Towel"])
    elif "park" in category.lower() or "nature" in category.lower():
        items.extend(["🦟 Insect Repellent", "👟 Comfortable Walking Shoes"])
    return items

def get_local_lingo(city):
    """Local Phrases based on City"""
    c = city.lower()
    if "chennai" in c:
        return {"Hello": "Vanakkam", "Thank You": "Nandri", "Help": "Udhavi", "Price?": "Evvalavu?"}
    if "mumbai" in c:
        return {"Hello": "Namaste", "Thank You": "Dhanyavad", "Help": "Madat", "Price?": "Kitna?"}
    if "bangalore" in c:
        return {"Hello": "Namaskara", "Thank You": "Dhanyavadagalu", "Help": "Sahaya", "Price?": "Eshtu?"}
    if "kerala" in c:
        return {"Hello": "Namaskaram", "Thank You": "Nanni", "Help": "Sahayam", "Price?": "Ethra?"}
    return {"Hello": "Hello", "Thank You": "Thanks", "Help": "Help", "Price?": "Price?"}

def ollama_chat(prompt, context=""):
    try:
        system = f"You are a travel guide. Context: {context}"
        payload = {"model": "llama3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=20)
        if r.ok: return r.json()['message']['content']
    except: return "⚠️ AI offline."

def create_pdf(city, itinerary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Itinerary for {city}", ln=1, align='C')
    safe_text = itinerary_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return pdf.output(dest='S').encode('latin-1')

def convert_currency(amount, from_curr, to_curr):
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.5, "JPY": 155.0}
    if from_curr in rates and to_curr in rates:
        return (amount / rates[from_curr]) * rates[to_curr]
    return 0.0

# --- SMART GENERATORS ---

def get_city_specific_foods(city_name):
    c = city_name.lower()
    if "chennai" in c: return ["Idli & Vadacurry", "Filter Coffee", "Pongal"]
    if "mumbai" in c: return ["Vada Pav", "Pav Bhaji", "Bhel Puri"]
    if "delhi" in c: return ["Chole Bhature", "Momos", "Parathas"]
    return ["Local Thali", "Street Snacks", "Traditional Sweets"]

def get_best_season(city):
    c = city.lower()
    if "chennai" in c: return "Nov - Feb"
    return "Oct - Mar"

def get_approx_footfall(city):
    return "2-3 Million/Year"

def detect_gods_from_text(text, default="The presiding deity"):
    gods = []
    keywords = ["Shiva", "Vishnu", "Murugan", "Ganesha", "Krishna", "Lakshmi", "Parvati", "Jesus", "Mary", "Allah", "Buddha"]
    for k in keywords:
        if k in text: gods.append(k)
    return ", ".join(list(set(gods))) if gods else default

def get_temple_timings(): return "06:00 AM - 12:30 PM, 04:00 PM - 09:00 PM"

# --- DISTANCE HELPER ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --- STRICT GENERATOR ---
def generate_smart_itinerary(city, days, main_place, start_lat, start_lon):
    raw_sights = get_places_api(start_lat, start_lon, "tourism,historic,religion,natural,leisure", limit=500, radius=50000)
    raw_food = get_places_api(start_lat, start_lon, "catering.restaurant", limit=100, radius=20000)
    
    all_places = []
    seen = {main_place.lower()}
    
    for item in raw_sights:
        n = item['properties'].get('name')
        if not n or n.lower() in seen: continue
        lat = item['properties']['lat']
        lon = item['properties']['lon']
        dist = calculate_distance(start_lat, start_lon, lat, lon)
        if dist < 100:
            all_places.append({"name": n, "dist": dist, "lat": lat, "lon": lon})
            seen.add(n.lower())
    
    all_places.sort(key=lambda x: x['dist'])
    food_list = [f['properties'].get('name') for f in raw_food if f['properties'].get('name')]
    
    html = ""
    plain_text = f"Travel Plan for {city}\n\n"
    map_markers = []
    current_idx = 0
    
    for d in range(1, days + 1):
        day_picks = []
        count = 0
        while count < 3 and current_idx < len(all_places):
            p = all_places[current_idx]
            day_picks.append(p['name'])
            map_markers.append({"name": p['name'], "lat": p['lat'], "lon": p['lon'], "day": d})
            current_idx += 1
            count += 1
            
        if len(day_picks) < 3: day_picks.append("Explore Local Market")
        dinner = food_list.pop(0) if food_list else "Local Famous Restaurant"
        places_str = ", ".join(day_picks)
        
        dist_note = "(Nearby)" if d == 1 else (f"(5-10km)" if d == 2 else f"(15km+)")
        
        html += f"""
        <div class='itinerary-box'>
            <p style='font-size:16px; margin:0;'>
                <strong style='color:#00d2ff'>Day {d} {dist_note}:</strong> {places_str}, Dinner at <b>{dinner}</b>
            </p>
        </div>
        """
        plain_text += f"Day {d}: {places_str}\nDinner: {dinner}\n\n"
    return html, plain_text, map_markers

def generate_full_description(name, city, category, details_desc, wiki_extract):
    base_info = wiki_extract if wiki_extract else (details_desc if details_desc else "")
    if "temple" in category.lower() or "religion" in category.lower():
        gods = detect_gods_from_text(base_info, default="the main deity")
        desc = (f"{name} is a majestic and renowned spiritual landmark in {city}. "
                f"The temple is primarily dedicated to **{gods}**, serving as a focal point of devotion. "
                f"Renowned for its breathtaking architecture, the structure features intricate stone carvings. "
                f"Devotees flock here to witness the grand daily rituals (Pooja) and experience the deep sense of peace. "
                f"The complex often houses a sacred temple tank and ancient trees, creating a serene ecosystem. "
                f"Major festivals such as Brahmotsavam are celebrated here with immense pomp. "
                f"Historically, it has stood for centuries as a custodian of the region's spiritual heritage. "
                f"A visit here offers not just blessings, but a profound connection to the local traditions.")
    else:
        desc = (f"{name} stands as a premier tourist attraction in {city}, offering a perfect blend of history and beauty. "
                f"{base_info[:250] if base_info else 'It is a favorite spot for those looking to experience the local vibe.'} "
                f"This iconic location serves as a testament to the city's development and is a major social hub. "
                f"The area is meticulously maintained, providing lush green landscapes or stunning views. "
                f"Visitors particularly enjoy spending time here during the golden hours of sunrise or sunset. "
                f"Surrounding the main attraction, you will find a lively atmosphere with local vendors. "
                f"Whether you are a history enthusiast or a nature lover, {name} has something unique to offer. "
                f"Exploring this landmark is considered an essential part of any trip to {city}.")
    return desc

def generate_true_highlights(name, city, category, wiki_data):
    extract = wiki_data.get("extract", "") if wiki_data else ""
    foods = get_city_specific_foods(city)
    best_season = get_best_season(city)
    timings = get_temple_timings() if "temple" in category.lower() else "09:00 AM - 06:00 PM"

    history_text = "This site holds significant historical value."
    sentences = extract.split('. ')
    for s in sentences:
        if any(x in s.lower() for x in ['built', 'founded', 'century', 'king']):
            history_text = s + "."
            break
            
    special_info = ""
    if "temple" in category.lower():
        gods = detect_gods_from_text(extract)
        special_info = f"<li>🕉️ <b>Deity:</b> {gods}</li>"
    
    html = f"""
    <div class='highlight-box'>
        <h4>✨ True Highlights</h4>
        <p><b>🍲 Local Eats:</b> <i>{foods[0]}, {foods[1]}</i></p>
        <p><b>📜 History & Facts:</b></p>
        <ul><li><b>Story:</b> {history_text}</li>{special_info}</ul>
        <p><b>📅 Visit Info:</b> ⏰ {timings} | ⛅ <b>Best Season:</b> {best_season}</p>
    </div>
    """
    return html

# --- HOLIDAY ENGINE ---
def calculate_holidays(year, duration):
    holidays = [
        {"name": "Republic Day", "date": f"{year}-01-26"},
        {"name": "Holi", "date": f"{year}-03-14"},
        {"name": "Good Friday", "date": f"{year}-04-18"},
        {"name": "Independence Day", "date": f"{year}-08-15"},
        {"name": "Diwali", "date": f"{year}-10-20"},
        {"name": "Christmas", "date": f"{year}-12-25"}
    ]
    results = []
    pool = ["Goa", "Kerala", "Rajasthan", "Manali"] if duration >= 4 else ["Pondicherry", "Mahabalipuram", "Ooty"]
    for h in holidays:
        sug = random.choice(pool)
        start_dt = datetime.datetime.strptime(h["date"], "%Y-%m-%d")
        end_dt = start_dt + datetime.timedelta(days=duration)
        date_range = f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d, %Y')}"
        results.append({"name": h["name"], "date": date_range, "days": duration, "sug": sug})
    return results

# --- POPUPS ---
@st.dialog("📅 Holiday Planner")
def open_planner_popup():
    st.write("Plan your trips.")
    c1, c2 = st.columns(2)
    p_year = c1.selectbox("Year", list(range(2025, 2031)))
    p_days = c2.selectbox("Duration", [1, 2, 3, 4, 5])
    if st.button("Find Plans 🔎"):
        log_holiday(p_year, p_days)
        st.session_state["holiday_results"] = calculate_holidays(p_year, p_days)
    if "holiday_results" in st.session_state:
        for h in st.session_state["holiday_results"]:
            with st.container():
                st.write(f"**{h['name']}** ({h['date']})")
                st.caption(f"Suggested: {h['days']} Days in {h['sug']}")
                if st.button(f"Explore {h['sug']} ➔", key=h['name']):
                    st.session_state["current_city"] = h['sug']
                    st.session_state["search_results"] = []
                    st.session_state["trigger_search"] = True 
                    st.rerun()
                st.divider()

# --- DETAILS POPUP ---
@st.dialog("📍 Place Details")
def show_place_details_popup(p, city):
    lat, lon = p['lat'], p['lon']
    place_id = p.get('place_id')

    with st.spinner("Fetching Info..."):
        w = fetch_weather(lat, lon)
        imgs = google_scrape_images(f"{p['name']} {city} tourism", limit=7)
        wiki_data = fetch_wiki_data(p['name'])
        details = {}
        if place_id: details = fetch_place_details(place_id)
        
        wiki_extract = wiki_data.get('extract', "") if wiki_data else ""
        
        cat_context = p.get('category', 'tourist')
        if "temple" in p['name'].lower(): cat_context = "temple"
        elif "beach" in p['name'].lower(): cat_context = "beach"
        
        desc_text = generate_full_description(p['name'], city, cat_context, details.get("description"), wiki_extract)
        highlights_html = generate_true_highlights(p['name'], city, cat_context, wiki_data)
        packing_items = get_packing_list(w.get('weathercode', 0), cat_context)
        lingo = get_local_lingo(city)

    # 1. INFO TOP
    if imgs:
        top_img = imgs[0]
    else:
        top_img = "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?q=80&w=2021&auto=format&fit=crop"
        
    st.image(top_img, use_container_width=True)
    
    st.markdown("<div class='detail-box'>", unsafe_allow_html=True)
    st.header(p['name'])
    map_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    st.markdown(f"<p style='font-size:18px;'>📍 {p.get('address', city)} <a href='{map_url}' target='_blank' class='map-link'>(View on Map 🗺️)</a></p>", unsafe_allow_html=True)
    if w: st.info(f"🌡️ {w.get('temperature')}°C | {get_weather_desc(w.get('weathercode', 0))} | 💨 {w.get('windspeed')} km/h")
    
    st.markdown("### 📝 About the Place")
    st.markdown(f"<p class='desc-text'>{desc_text}</p>", unsafe_allow_html=True)
    st.markdown(highlights_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. TOOLS
    with st.expander("🎒 Smart Packing Checklist"):
        for item in packing_items: st.markdown(f"- {item}")

    with st.expander("🗣️ Local Lingo (Phrasebook)"):
        c1, c2 = st.columns(2)
        items = list(lingo.items())
        half = len(items) // 2
        for k, v in items[:half]: c1.write(f"**{k}:** {v}")
        for k, v in items[half:]: c2.write(f"**{k}:** {v}")

    with st.expander("💰 Trip Budget Estimator"):
        b1, b2 = st.columns(2)
        tr = b1.number_input("Travelers", 1, 10, 2, key=f"tr_{p['name']}")
        dy = b2.number_input("Trip Days", 1, 15, 3, key=f"dy_{p['name']}")
        sty = st.selectbox("Style", ["Backpacker 🎒", "Comfort 🧳", "Luxury 💎"], key=f"sty_{p['name']}")
        if st.button("Calculate", key=f"calc_{p['name']}"):
            cst, dsc = calculate_budget(dy, tr, sty)
            st.success(f"Est: ₹{cst:,} ({dsc})")

    with st.expander("🗺️ Itinerary Planner (Map + PDF)"):
        idays = st.number_input("Plan for (Days)", 1, 7, 3, key=f"idays_{p['name']}")
        
        # --- SESSION STATE FIX FOR PERSISTENCE ---
        itin_key = f"itin_{p['name']}_{idays}"
        
        if st.button("Generate Itinerary", key=f"gen_itin_{p['name']}"):
            with st.spinner("Planning..."):
                plan_html, plan_text, markers = generate_smart_itinerary(city, idays, p['name'], lat, lon)
                st.session_state[itin_key] = {"html": plan_html, "text": plan_text, "markers": markers}
        
        if itin_key in st.session_state:
            data = st.session_state[itin_key]
            st.markdown(data["html"], unsafe_allow_html=True)
            
            st.write("**🗺️ Route Map:**")
            m = folium.Map(location=[lat, lon], zoom_start=12)
            folium.Marker([lat, lon], tooltip=p['name'], icon=folium.Icon(color="red")).add_to(m)
            for mk in data["markers"]:
                folium.Marker([mk['lat'], mk['lon']], tooltip=f"Day {mk['day']}: {mk['name']}", icon=folium.Icon(color="blue")).add_to(m)
            st_folium(m, height=300, use_container_width=True, key=f"map_{p['name']}")
            
            pdf_bytes = create_pdf(city, data["text"])
            st.download_button("Download PDF 📥", pdf_bytes, f"Itinerary_{city}.pdf", key=f"dl_{p['name']}")

    with st.expander("⚖️ Multi-City Comparison"):
        c_in = st.text_input("Compare with (comma separated, e.g. Mumbai, Delhi):", key=f"cmp_{p['name']}")
        if st.button("Compare", key=f"btn_cmp_{p['name']}"):
            # Base data (Current City)
            comp_data = [{"City": city, "Temp": f"{w.get('temperature')}°C", "Sky": get_weather_desc(w.get('weathercode', 0))}]
            
            # Process Inputs
            targets = [x.strip() for x in c_in.split(',') if x.strip()]
            
            for t_city in targets:
                g = geocode(t_city)
                if g:
                    wt = fetch_weather(g['lat'], g['lon'])
                    comp_data.append({
                        "City": t_city, 
                        "Temp": f"{wt.get('temperature')}°C",
                        "Sky": get_weather_desc(wt.get('weathercode', 0))
                    })
            
            if len(comp_data) > 1:
                st.table(pd.DataFrame(comp_data))
            else:
                st.warning("No valid cities found to compare.")

    with st.expander("🏨 Nearby Facilities"):
        nb_type = st.selectbox("Type", ["Hotels", "Restaurants", "Hospitals"], key=f"nb_{p['name']}")
        if st.button("Find Nearby", key=f"btn_nb_{p['name']}"):
            if nb_type == "Hotels": cat_str = "accommodation" 
            elif nb_type == "Restaurants": cat_str = "catering" 
            else: cat_str = "healthcare"
                
            places = get_places_api(lat, lon, cat_str, limit=15, radius=5000)
            
            with st.container(height=400):
                if places:
                    seen = set()
                    for pl in places:
                        n = pl['properties'].get('name')
                        if not n or n in seen: continue
                        seen.add(n)
                        
                        im = google_scrape_images(f"{n} {city}", limit=1)
                        u = im[0] if im else "https://via.placeholder.com/300x200"
                        
                        details_html = ""
                        map_link = f"<a href='https://www.google.com/maps/search/?api=1&query={pl['properties']['lat']},{pl['properties']['lon']}' target='_blank' style='color:#4facfe;'>📍 Location</a>"
                        
                        if nb_type == "Hotels":
                            rt = round(random.uniform(3.5, 5.0), 1)
                            amens = random.sample(["Wifi", "Pool", "Gym", "Spa", "Bar"], 3)
                            tags = "".join([f"<span class='tag'>{x}</span>" for x in amens])
                            diet = random.choice(["Veg/Non-Veg", "Veg Only"])
                            details_html = f"<div><span class='rating'>⭐ {rt}</span> | {diet}<br>{tags}</div>"
                        
                        elif nb_type == "Restaurants":
                            rt = round(random.uniform(3.8, 5.0), 1)
                            fcl = random.sample(["AC", "Wifi", "Bar", "Family", "Rooftop"], 3)
                            tags = "".join([f"<span class='tag'>{x}</span>" for x in fcl])
                            diet = random.choice(["Veg 🟢", "Non-Veg 🔴", "Mixed 🟡"])
                            details_html = f"<div><span class='rating'>⭐ {rt}</span> | {diet}<br>{tags}</div>"
                            
                        elif nb_type == "Hospitals":
                            ph = pl['properties'].get('contact', {}).get('phone', 'Not Available')
                            details_html = f"<div>📞 {ph}<br><span class='tag'>24/7 Emergency</span></div>"

                        st.markdown(f"""
                        <div class='nearby-card'>
                            <img src='{u}' style='width:100%;height:100px;object-fit:cover;border-radius:5px;'>
                            <h4 style="margin:5px 0;">{n}</h4>
                            {details_html}
                            <div style='margin-top:5px;'>{map_link}</div>
                        </div>""", unsafe_allow_html=True)
                else: st.warning("None found")

    st.markdown("---")
    # 3. IMAGES BOTTOM
    if imgs and len(imgs) > 1:
        st.write("### 📸 Photo Gallery")
        cols = st.columns(3)
        for i, im in enumerate(imgs[1:4]): cols[i].image(im, use_container_width=True)

    if st.button("Close"): st.rerun()

# --- MAIN APP ---
def main():
    st.set_page_config(layout="wide", page_title="City Explorer Pro")
    inject_custom_css()
    init_db()

    # --- LOGIN ---
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if "show_signup" not in st.session_state: st.session_state["show_signup"] = False
    
    if not st.session_state["logged_in"]:
        st.markdown("<div class='login-box'><h2>🔒 Login</h2>", unsafe_allow_html=True)
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if check_login(u, p):
                    log_login(u) # Fix: call correct log function
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = u
                    st.success("Success!")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error("Invalid (Try admin/travel123)")
            
            st.markdown("---")
            st.markdown("Don't have an account?")
        
        with st.expander("📝 Create Account"):
            with st.form("signup"):
                new_u = st.text_input("New Username")
                new_p = st.text_input("New Password", type="password")
                new_e = st.text_input("Email")
                new_ph = st.text_input("Phone")
                if st.form_submit_button("Sign Up"):
                    if create_account(new_u, new_p, new_e, new_ph):
                        st.success("Created! Please login.")
                    else:
                        st.error("Username exists.")

        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "search_results" not in st.session_state: st.session_state["search_results"] = []
    if "current_city" not in st.session_state: st.session_state["current_city"] = "Chennai"
    if "messages" not in st.session_state: st.session_state["messages"] = [{"role": "assistant", "content": "Hi! Ask me travel tips."}]

    with st.sidebar:
        st.title(f"👤 {st.session_state['username']}")
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()
        
        st.divider()
        st.subheader("💱 Currency Converter")
        amt = st.number_input("Amount", 1.0, step=10.0)
        c1, c2 = st.columns(2)
        f_curr = c1.selectbox("From", ["USD", "EUR", "GBP", "INR", "JPY"], index=0)
        t_curr = c2.selectbox("To", ["USD", "EUR", "GBP", "INR", "JPY"], index=3)
        if st.button("Convert"):
            val = convert_currency(amt, f_curr, t_curr)
            st.success(f"{amt} {f_curr} = {val:.2f} {t_curr}")

        st.divider()
        st.title("🤖 AI Chat")
        for msg in st.session_state["messages"]: st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask something..."):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.spinner("..."):
                st.session_state["messages"].append({"role": "assistant", "content": ollama_chat(prompt)})
                st.rerun()
        st.divider()
        st.subheader("🗄️ History")
        conn = sqlite3.connect(DB_NAME)
        try: st.dataframe(pd.read_sql("SELECT city, category, search_date FROM search_history ORDER BY id DESC LIMIT 5", conn), hide_index=True)
        except: pass
        conn.close()

    c1, c2 = st.columns([5, 1])
    c1.title("🏙️ City Explorer Pro")
    if c2.button("📅 Holiday Planner"): open_planner_popup()

    st.markdown("---")
    search_mode = st.radio("Search Mode:", ["Search by City", "Search by Specific Place"], horizontal=True)
    sc1, sc2, sc3 = st.columns([3, 2, 1])
    
    if search_mode == "Search by City":
        city_in = sc1.text_input("Enter City Name", value=st.session_state.get("current_city", "Chennai"))
        cat_in = sc2.selectbox("Category", ["Tourist Places", "Temples", "Parks", "Top Recommendations"])
    else:
        city_in = sc1.text_input("Enter Place Name", "Kapaleeshwarar Temple")
        cat_in = "Specific"

    if sc3.button("Search 🔎") or st.session_state.get("trigger_search", False):
        st.session_state["trigger_search"] = False
        log_search(city_in, cat_in)
        
        if search_mode == "Search by City":
            st.session_state["current_city"] = city_in
            st.session_state["display_count"] = 10
            api_cat = "tourism"
            if cat_in == "Temples": api_cat = "religion.place_of_worship"
            elif cat_in == "Parks": api_cat = "leisure.park,natural"
            elif cat_in == "Top Recommendations": api_cat = "tourism.attraction,historic,natural"
            
            geo = geocode(city_in)
            if geo:
                with st.spinner("Searching..."):
                    raw = get_places_api(geo['lat'], geo['lon'], api_cat, limit=100)
                    clean = []
                    seen = set()
                    for item in raw:
                        n = item['properties'].get('name')
                        if n and n not in seen:
                            if cat_in == "Tourist Places" and "temple" in n.lower(): continue
                            clean.append({"name": n, "address": item['properties'].get('formatted'), "lat": item['properties']['lat'], "lon": item['properties']['lon'], "place_id": item['properties'].get('place_id'), "category": cat_in})
                            seen.add(n)
                    st.session_state["search_results"] = clean
            else: st.error("City not found")
        else:
            with st.spinner("Locating..."):
                geo = geocode(city_in)
                if geo:
                    st.session_state["current_city"] = geo.get('city', city_in)
                    st.session_state["search_results"] = [{"name": geo.get('name', city_in), "address": geo.get('formatted'), "lat": geo['lat'], "lon": geo['lon'], "place_id": geo.get('place_id'), "category": "Specific"}]
                else: st.error("Place not found")

    results = st.session_state.get("search_results", [])
    count = st.session_state.get("display_count", 10)
    
    if results:
        st.write("### Results")
        for i, place in enumerate(results[:count]):
            with st.container():
                c_a, c_b = st.columns([5, 1])
                c_a.subheader(f"{i+1}. {place['name']}")
                c_a.caption(place['address'])
                if c_b.button("Details ➡️", key=f"btn_{i}"):
                    show_place_details_popup(place, st.session_state["current_city"])
                st.divider()
        if count < len(results):
            if st.button("Load More 🔄"):
                st.session_state["display_count"] += 10
                st.rerun()

if __name__ == "__main__":
    main()