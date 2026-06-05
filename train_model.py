import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading BMRC real data...")

df = pd.read_excel('bmrlc_data.xlsx')

# Extract hourly ridership columns (Hour 0–22)
hour_cols = [c for c in df.columns if 'Hrs' in c and c != '23:00 Hrs To Last train']

# Build hourly average ridership per hour slot
hourly_avg = {}
for i, col in enumerate(hour_cols):
    hourly_avg[i] = df[col].mean()

MAX_RIDERS = max(hourly_avg.values())  # ~850 at peak hour 18

def riders_to_wait(riders, weather_code):
    """Convert ridership count to estimated auto wait time in minutes."""
    load_factor = riders / MAX_RIDERS  # 0.0 to 1.0
    base_wait = 3 + load_factor * 30   # 3 min off-peak → 33 min at peak
    weather_penalty = {0: 0, 1: 2, 2: 8}.get(weather_code, 0)
    return round(base_wait + weather_penalty, 1)

def riders_to_travel(riders, weather_code, distance_km):
    """Convert ridership + conditions to road travel time."""
    load_factor = riders / MAX_RIDERS
    speed_kmph = 40 - (load_factor * 25)  # 40 kmph free flow → 15 kmph at peak
    weather_speed_penalty = {0: 0, 1: 3, 2: 8}.get(weather_code, 0)
    effective_speed = max(speed_kmph - weather_speed_penalty, 8)
    return round((distance_km / effective_speed) * 60, 1)

# Generate training rows from all 24 hours × 3 weather codes × realistic distances
records = []
for hour in range(24):
    riders = hourly_avg.get(hour, 0)
    for weather_code in [0, 1, 2]:
        for distance_km in [3, 5, 7, 10, 15, 20]:
            travel = riders_to_travel(riders, weather_code, distance_km)
            wait = riders_to_wait(riders, weather_code)
            # Add slight noise for model generalization
            for _ in range(5):
                noise_t = np.random.normal(0, travel * 0.05)
                noise_w = np.random.normal(0, wait * 0.05)
                records.append({
                    'Hour': hour,
                    'Weather_Code': weather_code,
                    'Distance_km': distance_km,
                    'Metro_Ridership': round(riders, 1),
                    'Travel_Time_Mins': max(2, round(travel + noise_t, 1)),
                    'Wait_Time_Mins': max(1, round(wait + noise_w, 1)),
                })

df_train = pd.DataFrame(records)
print(f"Training on {len(df_train)} samples derived from {len(hour_cols)} hourly BMRC slots across {df['STATION'].nunique()} stations")

X = df_train[['Hour', 'Weather_Code', 'Distance_km', 'Metro_Ridership']]
y = df_train[['Travel_Time_Mins', 'Wait_Time_Mins']]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, 'commute_ai_model.pkl')

# Also export hourly ridership map for the API to use at runtime
import json
ridership_map = {str(h): round(v, 1) for h, v in hourly_avg.items()}
with open('bmrc_hourly_ridership.json', 'w') as f:
    json.dump({"hourly_avg": ridership_map, "max_ridership": round(MAX_RIDERS, 1)}, f, indent=2)

print("✅ BMRC-trained AI Model saved: commute_ai_model.pkl")
print("✅ Hourly ridership map saved: bmrc_hourly_ridership.json")
print("\nSample predictions:")
for hour, label in [(8, "Morning peak"), (14, "Afternoon"), (18, "Evening peak"), (23, "Night")]:
    riders = hourly_avg.get(hour, 0)
    pred = model.predict([[hour, 0, 10, riders]])[0]
    print(f"  {label} (Hour {hour}): travel={pred[0]:.0f} min, wait={pred[1]:.0f} min")