from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os
import httpx
import pandas as pd

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

    # --- ML prediction (now uses 4 features incl. real ridership) ---
    import pandas as pd
    X_input = pd.DataFrame([{
        "Hour": hour,
        "Weather_Code": weather_code,
        "Distance_km": 10,
        "Metro_Ridership": metro_ridership,
    }])
    pred = model.predict(X_input)[0]
    predicted_travel_time = max(5, int(pred[0]))
    predicted_wait_time   = max(1, int(pred[1]))

    # Metro congestion level derived from real BMRC data
    load_pct = round((metro_ridership / MAX_RIDERSHIP) * 100)
    if load_pct >= 80:
        metro_congestion = "Very crowded"
    elif load_pct >= 50:
        metro_congestion = "Moderate"
    elif load_pct >= 20:
        metro_congestion = "Light"
    else:
        metro_congestion = "Empty / closed"

    # --- Metro availability ---
    is_metro_closed = (hour >= 0 and hour < 5)
    metro_wait_display = "CLOSED" if is_metro_closed else 5
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
            {"name": "Namma Yatri",  "url": "https://nammayatri.in/",          "wait": predicted_wait_time,     "travel": predicted_travel_time,     "probability": 84},
            {"name": "Rapido Auto",  "url": "https://www.rapido.bike/",         "wait": predicted_wait_time + 2, "travel": predicted_travel_time,     "probability": 79},
            {"name": "Ola Auto",     "url": "https://www.olacabs.com/",         "wait": predicted_wait_time + 5, "travel": predicted_travel_time + 2, "probability": 69},
            {"name": "Uber Auto",    "url": "https://m.uber.com/ul",            "wait": predicted_wait_time + 8, "travel": predicted_travel_time + 2, "probability": 64},
        ],
        "alternative_modes": [
            {"name": "Namma Metro", "url": "https://english.bmrc.co.in/",         "wait": metro_wait_display,              "travel": 35,                                "probability": metro_probability, "congestion": metro_congestion},
            {"name": "BMTC Bus",    "url": "https://mybmtc.karnataka.gov.in/",    "wait": 15,                              "travel": predicted_travel_time + 15,        "probability": 75,                "congestion": None},
            {"name": "Bike Taxi",   "url": "https://www.rapido.bike/",            "wait": 3,                               "travel": int(predicted_travel_time * 0.8),  "probability": 92,                "congestion": None},
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