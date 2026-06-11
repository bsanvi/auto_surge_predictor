from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os
import httpx
import pandas as pd
import re
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model and BMRC data at startup ---
model = joblib.load('commute_ai_model.pkl')

with open('bmrc_hourly_ridership.json') as f:
    bmrc_data = json.load(f)

HOURLY_RIDERSHIP: dict[str, float] = bmrc_data["hourly_avg"]   # "0"–"22" → avg riders
MAX_RIDERSHIP: float = bmrc_data["max_ridership"]

METRO_OPEN_MINUTES = 5 * 60
METRO_CLOSE_MINUTES = 23 * 60
DEFAULT_BANGALORE_COORDS = (12.9716, 77.5946)

KNOWN_LOCATIONS: dict[str, tuple[float, float]] = {
    "current location": DEFAULT_BANGALORE_COORDS,
    "majestic": (12.9763, 77.5713),
    "kempegowda bus station": (12.9763, 77.5713),
    "mg road": (12.9759, 77.6051),
    "indiranagar": (12.9784, 77.6408),
    "koramangala": (12.9352, 77.6245),
    "whitefield": (12.9698, 77.7500),
    "electronic city": (12.8456, 77.6603),
    "hebbal": (13.0358, 77.5913),
    "silk board": (12.9179, 77.6234),
    "jayanagar": (12.9250, 77.5938),
    "malleshwaram": (13.0031, 77.5640),
    "banashankari": (12.9250, 77.5467),
    "airport": (13.1986, 77.7066),
    "yeshwantpur": (13.0250, 77.5500),
}

GEOCODE_CACHE: dict[str, tuple[float, float] | None] = {}
OSRM_CACHE: dict[tuple[str, str, int], int | None] = {}

# Open-Meteo: free, no API key, covers Bangalore
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=12.9716&longitude=77.5946"
    "&current=temperature_2m,weathercode,precipitation"
    "&timezone=Asia%2FKolkata"
)

# WMO weather code → human label + our internal code (0=clear,1=cloudy,2=rain)
def wmo_to_label_and_code(wmo: int) -> tuple[str, int]:
    if wmo == 0:
        return "Clear sky", 0
    elif wmo in range(1, 4):
        return "Partly cloudy", 1
    elif wmo in range(4, 50):
        return "Overcast / foggy", 1
    elif wmo in range(50, 70):
        return "Drizzle", 2
    elif wmo in range(70, 80):
        return "Snow", 2
    elif wmo in range(80, 100):
        return "Rain / thunderstorm", 2
    return "Unknown", 0


class CommuteRequest(BaseModel):
    source: str
    destination: str
    time: str
    weather_condition: str   # kept for fallback / manual override
    source_lat: float | None = None
    source_lng: float | None = None


def parse_time(time_str: str) -> int:
    """Return hour (0–23) from '8:00 AM' style string."""
    try:
        clean = time_str.replace("✎", "").strip()
        parts = clean.split()
        hour = int(parts[0].split(":")[0])
        am_pm = parts[1].upper()
        if am_pm == "PM" and hour != 12:
            hour += 12
        if am_pm == "AM" and hour == 12:
            hour = 0
        return hour
    except Exception:
        return 14


def parse_time_minutes(time_str: str) -> int:
    """Return total minutes since midnight from a '8:00 AM' style string."""
    try:
        clean = time_str.replace("✎", "").strip()
        parts = clean.split()
        hour_part, minute_part = parts[0].split(":")
        hour = int(hour_part)
        minute = int(minute_part)
        am_pm = parts[1].upper()
        if am_pm == "PM" and hour != 12:
            hour += 12
        if am_pm == "AM" and hour == 12:
            hour = 0
        return hour * 60 + minute
    except Exception:
        return 14 * 60


def normalize_location(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * radius_km * asin(sqrt(a))


async def resolve_location(location: str) -> tuple[float, float] | None:
    normalized = normalize_location(location)
    if not normalized:
        return None

    if normalized in GEOCODE_CACHE:
        return GEOCODE_CACHE[normalized]

    if normalized in KNOWN_LOCATIONS:
        GEOCODE_CACHE[normalized] = KNOWN_LOCATIONS[normalized]
        return KNOWN_LOCATIONS[normalized]

    try:
        async with httpx.AsyncClient(timeout=5, headers={"User-Agent": "commute-ai/1.0"}) as client:
            # FIX: Force the search to stay inside Bengaluru!
            search_query = location if "bengaluru" in location.lower() or "bangalore" in location.lower() else f"{location}, Bengaluru"
            
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": search_query, "format": "jsonv2", "limit": 1},
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                GEOCODE_CACHE[normalized] = None
                return None
            coords = (float(results[0]["lat"]), float(results[0]["lon"]))
            GEOCODE_CACHE[normalized] = coords
            return coords
    except Exception:
        GEOCODE_CACHE[normalized] = None
        return None


async def estimate_distance_km(source: str, destination: str) -> float:
    source_coords = await resolve_location(source) or DEFAULT_BANGALORE_COORDS
    destination_coords = await resolve_location(destination)

    if destination_coords is None:
        return 10.0

    if source_coords == destination_coords:
        return 1.5

    direct_km = haversine_km(
        source_coords[0],
        source_coords[1],
        destination_coords[0],
        destination_coords[1],
    )
    return round(max(1.0, direct_km * 1.3), 1)


async def estimate_distance_km_from_coords(
    source_coords: tuple[float, float],
    destination: str,
) -> float:
    destination_coords = await resolve_location(destination)

    if destination_coords is None:
        return 10.0

    if source_coords == destination_coords:
        return 1.5

    direct_km = haversine_km(
        source_coords[0],
        source_coords[1],
        destination_coords[0],
        destination_coords[1],
    )
    return round(max(1.0, direct_km * 1.3), 1)


def metro_travel_minutes(distance_km: float) -> int:
    return max(4, int(round(distance_km * 1.3 + 2)))


def auto_wait_cap(distance_km: float, hour: int, weather_code: int) -> int:
    if distance_km <= 2:
        cap = 6
    elif distance_km <= 5:
        cap = 9
    elif distance_km <= 10:
        cap = 12
    else:
        cap = 15

    if hour in {8, 9, 10, 17, 18, 19}:
        cap += 2
    if weather_code == 1:
        cap += 1
    elif weather_code == 2:
        cap += 3

    return cap


def is_metro_open(total_minutes: int) -> bool:
    return METRO_OPEN_MINUTES <= total_minutes < METRO_CLOSE_MINUTES


def bangalore_datetime_for_minutes(total_minutes: int) -> datetime:
    bangalore_tz = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(bangalore_tz)
    hour = max(0, min(23, total_minutes // 60))
    minute = max(0, min(59, total_minutes % 60))
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


async def fetch_osrm_route_minutes(
    origin_coords: tuple[float, float],
    destination: str,
    total_minutes: int,
) -> int | None:
    cache_key = (
        f"{origin_coords[0]:.5f},{origin_coords[1]:.5f}",
        normalize_location(destination),
        total_minutes,
    )
    if cache_key in OSRM_CACHE:
        return OSRM_CACHE[cache_key]

    destination_coords = await resolve_location(destination)
    if destination_coords is None:
        OSRM_CACHE[cache_key] = None
        return None

    origin_lon_lat = f"{origin_coords[1]},{origin_coords[0]}"
    destination_lon_lat = f"{destination_coords[1]},{destination_coords[0]}"
    url = f"https://router.project-osrm.org/route/v1/driving/{origin_lon_lat};{destination_lon_lat}"
    params = {
        "overview": "false",
        "steps": "false",
        "annotations": "false",
    }

    try:
        async with httpx.AsyncClient(timeout=8, headers={"User-Agent": "commute-ai/1.0"}) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                OSRM_CACHE[cache_key] = None
                return None

            minutes = max(1, int(round(data["routes"][0]["duration"] / 60)))
            OSRM_CACHE[cache_key] = minutes
            return minutes
    except Exception:
        OSRM_CACHE[cache_key] = None
        return None


def auto_wait_minutes(distance_km: float, hour: int, weather_code: int, metro_load_percent: int) -> int:
    base_wait = 3 if distance_km <= 3 else 4 if distance_km <= 8 else 5
    rush_hour_penalty = 2 if hour in {7, 8, 9, 17, 18, 19} else 0
    weather_penalty = 0 if weather_code == 0 else 1 if weather_code == 1 else 3
    load_penalty = 0 if metro_load_percent < 20 else 1 if metro_load_percent < 50 else 2 if metro_load_percent < 80 else 3
    return min(12, base_wait + rush_hour_penalty + weather_penalty + load_penalty)


def parse_weather_override(weather_str: str) -> int:
    """Manual override fallback if live weather fetch fails."""
    w = weather_str.lower()
    if "rain" in w or "storm" in w:
        return 2
    elif "cloud" in w or "fog" in w or "overcast" in w:
        return 1
    return 0


async def fetch_live_weather() -> dict:
    """
    Returns:
        {
            "weather_code": int,           # 0/1/2
            "weather_label": str,
            "temperature_c": float,
            "precipitation_mm": float,
            "source": "live" | "fallback"
        }
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(OPEN_METEO_URL)
            resp.raise_for_status()
            data = resp.json()["current"]
            wmo = int(data["weathercode"])
            label, code = wmo_to_label_and_code(wmo)
            return {
                "weather_code": code,
                "weather_label": label,
                "temperature_c": data["temperature_2m"],
                "precipitation_mm": data.get("precipitation", 0.0),
                "source": "live",
            }
    except Exception:
        return {
            "weather_code": -1,          # signals caller to use override
            "weather_label": "Unknown",
            "temperature_c": None,
            "precipitation_mm": None,
            "source": "fallback",
        }


@app.get("/api/weather")
async def get_weather():
    """Frontend polls this to display live Bangalore weather."""
    return await fetch_live_weather()


@app.post("/api/predict")
async def predict_surge(request: CommuteRequest):
    hour = parse_time(request.time)
    total_minutes = parse_time_minutes(request.time)
    
    # FIX: Properly resolve the typed location and save it!
    source_coords = None
    if request.source_lat is not None and request.source_lng is not None:
        source_coords = (request.source_lat, request.source_lng)
    else:
        source_coords = await resolve_location(request.source)
        
    # Final safety fallback 
    if source_coords is None:
        source_coords = DEFAULT_BANGALORE_COORDS

    estimated_distance_km = await estimate_distance_km_from_coords(source_coords, request.destination)


    # --- Live weather (preferred) ---
    live = await fetch_live_weather()
    if live["source"] == "live":
        weather_code = live["weather_code"]
        weather_label = live["weather_label"]
        temperature_c = live["temperature_c"]
    else:
        weather_code = parse_weather_override(request.weather_condition)
        weather_label = request.weather_condition
        temperature_c = None

    # --- BMRC metro ridership at this hour ---
    metro_ridership = HOURLY_RIDERSHIP.get(str(hour), 0.0)
    load_pct = round((metro_ridership / MAX_RIDERSHIP) * 100)

    osrm_drive_minutes = await fetch_osrm_route_minutes(
        source_coords, # Look how clean this is now!
        request.destination,
        total_minutes,
    )

    # --- ML prediction (now uses 4 features incl. real ridership) ---
    import pandas as pd
    X_input = pd.DataFrame([{
        "Hour": hour,
        "Weather_Code": weather_code,
        "Distance_km": estimated_distance_km,
        "Metro_Ridership": metro_ridership,
    }])
    pred = model.predict(X_input)[0]
    predicted_travel_time = max(4, int(pred[0]))
    predicted_wait_time = max(1, int(pred[1]))

    # Prefer free OSRM routing when coordinates are available.
    baseline_drive_time = max(predicted_travel_time, max(4, int(round(estimated_distance_km * 1.4 + 2))))

# We take the MAXIMUM of the empty-road OSRM time OR your ML model's traffic prediction!
    if osrm_drive_minutes:
        driving_travel_time = max(osrm_drive_minutes, predicted_travel_time)
    else:
        driving_travel_time = baseline_drive_time
    metro_travel_time = metro_travel_minutes(estimated_distance_km)

    # Keep auto waits short for short trips, but still react to BMRC load and weather.
    predicted_wait_time = min(
        auto_wait_minutes(estimated_distance_km, hour, weather_code, load_pct),
        max(1, predicted_wait_time),
    )

    # Metro congestion level derived from real BMRC data
    if load_pct >= 80:
        metro_congestion = "Very crowded"
    elif load_pct >= 50:
        metro_congestion = "Moderate"
    elif load_pct >= 20:
        metro_congestion = "Light"
    else:
        metro_congestion = "Empty / closed"

    # --- Metro availability ---
    is_metro_closed = not is_metro_open(total_minutes)
    metro_wait_display = "CLOSED" if is_metro_closed else max(3, min(10, 4 + load_pct // 20))
    metro_probability   = 0 if is_metro_closed else min(99, max(50, int(98 - load_pct * 0.3)))

    # --- Build response ---
    response_data = {
        "live_weather": {
            "label": weather_label,
            "temperature_c": temperature_c,
            "source": live["source"],
        },
        "metro_stats": {
            "ridership_this_hour": int(metro_ridership),
            "load_percent": load_pct,
            "congestion": metro_congestion,
        },
        "auto_apps": [
            {"name": "Namma Yatri",  "url": "https://nammayatri.in/",          "wait": predicted_wait_time,     "travel": driving_travel_time,      "probability": 84, "source": "OSRM" if osrm_drive_minutes else "BMRC model"},
            {"name": "Rapido Auto",  "url": "https://www.rapido.bike/",         "wait": predicted_wait_time + 2, "travel": driving_travel_time,      "probability": 79, "source": "OSRM" if osrm_drive_minutes else "BMRC model"},
            {"name": "Ola Auto",     "url": "https://www.olacabs.com/",         "wait": predicted_wait_time + 4, "travel": driving_travel_time + 1,  "probability": 69, "source": "OSRM" if osrm_drive_minutes else "BMRC model"},
            {"name": "Uber Auto",    "url": "https://m.uber.com/ul",            "wait": predicted_wait_time + 6, "travel": driving_travel_time + 1,  "probability": 64, "source": "OSRM" if osrm_drive_minutes else "BMRC model"},
        ],
        "alternative_modes": [
            {"name": "Namma Metro", "url": "https://play.google.com/store/search?q=Namma%20Metro%20ticket%20booking&c=apps",         "wait": metro_wait_display,              "travel": metro_travel_time,                 "probability": metro_probability, "congestion": metro_congestion, "source": "BMRC load model"},
            {"name": "BMTC Bus",    "url": "https://mybmtc.karnataka.gov.in/",    "wait": 15,                              "travel": driving_travel_time + 15,        "probability": 75,                "congestion": None, "source": "Estimated"},
            {"name": "Bike Taxi",   "url": "https://www.rapido.bike/",            "wait": 3,                               "travel": max(4, int(driving_travel_time * 0.8)),  "probability": 92,                "congestion": None, "source": "Estimated"},
        ],
        "recommendation": "",
    }

    # --- Smart recommendation ---
    if is_metro_closed:
        response_data["recommendation"] = (
            "Namma Metro is currently closed (runs 05:00–23:00). "
            "Bike Taxi or Namma Yatri are your best options at this hour."
        )
    elif weather_code == 2:
        response_data["recommendation"] = (
            f"{'Rain' if 'rain' in weather_label.lower() else 'Bad weather'} detected in Bangalore. "
            "Auto cancellations will be high. Namma Metro is your most reliable option."
        )
    elif load_pct >= 80:
        response_data["recommendation"] = (
            f"Metro is very crowded right now ({int(metro_ridership):,} passengers/hr). "
            f"Auto travel is {predicted_travel_time} mins. Consider BMTC or travel in 30 mins."
        )
    elif estimated_distance_km <= 3:
        response_data["recommendation"] = (
            f"This is a short trip ({estimated_distance_km:.1f} km). "
            f"Namma Yatri or Bike Taxi should keep the wait low, and metro travel is about {metro_travel_time} mins."
        )
    elif predicted_travel_time > 50:
        response_data["recommendation"] = (
            f"Heavy road traffic ({predicted_travel_time} mins by auto). "
            "Namma Metro is strongly recommended."
        )
    else:
        response_data["recommendation"] = (
            f"Traffic is manageable. Bike Taxi ({int(predicted_travel_time * 0.8)} mins) "
            "or Namma Yatri are your fastest options right now."
        )

    return response_data