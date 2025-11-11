import csv
import random
import numpy as np

# --- Configuration ---
NUM_DAYS = 1000
OUTPUT_FILE = "renewable_energy.csv"

# --- Generate Synthetic Data ---
print(f"Generating {NUM_DAYS} days of synthetic energy data...")
data = []

# Base values
base_temp = 15.0  # Celsius
base_humidity = 60.0 # %
base_wind_speed = 10.0 # km/h
base_power = 50.0 # MW

for day in range(NUM_DAYS):
    # Add some noise and seasonality (simple sine wave for seasonality)
    season_factor = np.sin((day / 365.0) * 2 * np.pi) # Simple yearly cycle
    
    # Generate weather features
    temp = base_temp + (season_factor * 10) + random.uniform(-5, 5)
    humidity = base_humidity - (season_factor * 20) + random.uniform(-10, 10)
    wind_speed = base_wind_speed + (season_factor * 5) + random.uniform(-3, 3)
    
    # Clamp values to realistic ranges
    temp = max(-10, min(40, temp))
    humidity = max(20, min(100, humidity))
    wind_speed = max(0, min(50, wind_speed))
    
    # Simulate power output (target variable)
    # Let's assume power is strongly correlated with wind speed and moderately with temp
    power = (wind_speed * 4) + (temp * 1.5) + random.uniform(-10, 10)
    
    # Add some missing values
    if random.random() < 0.05:
        temp = None
    if random.random() < 0.05:
        humidity = None
    if random.random() < 0.05:
        wind_speed = None
    if random.random() < 0.03:
        power = None
        
    # Clamp power output
    if power is not None:
        power = max(0, min(200, power))
    
    data.append({
        "day": day,
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "power_output": power
    })

# --- Write to CSV ---
with open(OUTPUT_FILE, 'w', newline='') as f:
    fieldnames = ["day", "temperature", "humidity", "wind_speed", "power_output"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"Generated {OUTPUT_FILE} successfully.")