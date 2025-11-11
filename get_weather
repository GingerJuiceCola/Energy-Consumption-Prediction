import pandas as pd
import requests

# set up parameters
params = {
    "latitude": -2.148,
    "longitude": -79.964,
    "start_date": "2021-05-01",
    "end_date": "2021-12-31",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "timezone": "America/Guayaquil"
}

# get the data
response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
data = response.json()

# Convert to DataFrame and format the time
df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]).strftime("%Y-%m-%d %H:%M:%S"),
    "temperature(°C)": data["hourly"]["temperature_2m"],
    "humidity(%)": data["hourly"]["relative_humidity_2m"],
    "wind_speed(m/s)": data["hourly"]["wind_speed_10m"]
})

# save as CSV
df.to_csv("weather2021_05to12.csv", index=False)
