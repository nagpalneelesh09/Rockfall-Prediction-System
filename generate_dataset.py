
import numpy as np
import pandas as pd
import os

# Step 1: Generate 5000 observations
np.random.seed(42)

data_size = 5000

# Step 2: Generate environmental parameters

rainfall = np.random.uniform(0, 100, data_size)

slope_angle = np.random.uniform(15, 85, data_size)

soil_moisture = np.random.uniform(5, 100, data_size)

ground_vibration = np.random.uniform(0, 10, data_size)

rock_strength = np.random.uniform(0, 100, data_size)

# Step 3: Calculate a synthetic risk score

risk_score = (
    rainfall * 0.30
    + ((slope_angle - 15) / 70) * 25
    + soil_moisture * 0.20
    + ground_vibration * 2.5
    + (100 - rock_strength) * 0.10
)

# Add small random variation
noise = np.random.normal(0, 5, data_size)

risk_score = np.clip(risk_score + noise, 0, 100)

# Step 4: Classify risk levels

risk_level = []

for score in risk_score:

    if score < 35:
        risk_level.append("Low")

    elif score < 65:
        risk_level.append("Medium")

    else:
        risk_level.append("High")

# Step 5: Create a DataFrame

df = pd.DataFrame({
    "Rainfall": rainfall,
    "Slope_Angle": slope_angle,
    "Soil_Moisture": soil_moisture,
    "Ground_Vibration": ground_vibration,
    "Rock_Strength": rock_strength,
    "Risk_Score": risk_score,
    "Risk_Level": risk_level
})

# Step 6: Save dataset

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/rockfall_data.csv", index=False)

# Step 7: Display results

print("Rockfall Dataset Generated Successfully!")

print("\nFirst 5 Records:")
print(df.head())

print("\nRisk Distribution:")
print(df["Risk_Level"].value_counts())

print("\nTotal Records:", len(df))
