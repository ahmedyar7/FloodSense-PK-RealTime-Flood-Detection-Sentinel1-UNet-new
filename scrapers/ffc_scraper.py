import requests
import re
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://ffd.pmd.gov.pk"
URL = f"{BASE_URL}/river-state"
CACHE_DIR = os.path.join("data", "json")
CACHE_FILE = os.path.join(CACHE_DIR, "pmd_data_latest.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": f"{BASE_URL}/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
def clean_num(text):
    if not text:
        return 0
    match = re.findall(r"[\d,]+", text)
    return int(match[0].replace(",", "")) if match else 0

def clean_num(text):
    if not text:
        return 0
    match = re.findall(r"[\d,]+", text)
    return int(match[0].replace(",", "")) if match else 0

def save_cache(data):
    """Saves the scraped data to a JSON file for fallback usage."""
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Cache updated: {CACHE_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

def load_cache():
    """Loads data from the JSON cache file."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            print(f"ℹ️ Loaded fallback data from {CACHE_FILE}")
            return data
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
    return []

def get_ffc_data():
    print(f"[FFD] Fetching high-fidelity data from {URL}...", flush=True)

    data = []
    try:
        # ✅ Use a session so cookies persist between requests
        session = requests.Session()
        session.headers.update(HEADERS)

        # ✅ Visit homepage first to get cookies (mimics real browser)
        session.get(BASE_URL, timeout=15)

        # ✅ Now fetch the actual data page
        res = session.get(URL, timeout=30)
        res.raise_for_status()
        html = res.text

        # Extract all station blocks: var s = { ... };
        pattern = r'var\s+s\s*=\s*\{(.*?)\};'
        matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)

        for m in matches:
            def field(key):
                patterns = [
                    rf'["\']?{key}["\']?\s*:\s*["\']([^"\']*)["\']', # quoted value
                    rf'["\']?{key}["\']?\s*:\s*([^,\n\}}\s]*)'        # unquoted value
                ]
                for p in patterns:
                    r = re.search(p, m)
                    if r and r.group(1): return r.group(1).strip()
                return None

            name      = field("name") or "Unknown Station"
            status    = field("status") or "NORMAL"
            river     = field("area_name") or "Unknown River"
            recorded  = field("recording_time") or "N/A"

            # Robustly find all gauge objects { type: "...", discharge: "...", trend: "..." }
            gauge_objects = re.findall(r'\{([^{}]+)\}', m)

            inflow = 0
            outflow = 0
            inflow_trend = "Steady"
            outflow_trend = "Steady"

            for obj_str in gauge_objects:
                def obj_field(key):
                    pat = rf'["\']?{key}["\']?\s*:\s*["\']([^"\']*)["\']'
                    res = re.search(pat, obj_str)
                    return res.group(1) if res else None

                g_type = obj_field("type")
                g_disc = obj_field("discharge")
                g_trend = obj_field("trend")

                if not g_type: continue

                val = clean_num(g_disc)
                if g_type.upper() == "INFLOW":
                    inflow = val
                    inflow_trend = g_trend or "Steady"
                elif g_type.upper() == "OUTFLOW":
                    outflow = val
                    outflow_trend = g_trend or "Steady"

            # Normalize status
            std_status = "UNKNOWN"
            if status:
                s_up = status.upper()
                if "NORMAL" in s_up: std_status = "NORMAL"
                elif "HIGH" in s_up: std_status = "HIGH"
                elif "EXTREME" in s_up or "EX" in s_up: std_status = "EXTREME"

            data.append({
                "station": name,
                "river": river,
                "inflow": inflow,
                "outflow": outflow,
                "status": std_status,
                "inflow_trend": inflow_trend,
                "outflow_trend": outflow_trend,
                "recorded": recorded
            })

        if data:
            save_cache(data)
            return data

    except Exception as e:
        print(f"❌ FFD Fetch Error: {e}")

    # Fallback to cache if live fetch fails or returns no data
    print("⚠️ Live fetch failed. Attempting to use cached data...")
    return load_cache()


def main():
    data = get_ffc_data()
    print(f"\n🌊 FFD High-Fidelity River Data ({len(data)} stations)\n")
    for d in data:
        print(
            f"{d['station']} ({d['river']}) → "
            f"Inflow: {d['inflow']} ({d['inflow_trend']}) | "
            f"Outflow: {d['outflow']} ({d['outflow_trend']}) | "
            f"Status: {d['status']}"
        )
    return data

if __name__ == "__main__":
    main()