# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 09:27:47 2026

@author: sarah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, f, s

files = [
    "2018 consumption data.xlsx",
    "2019 consumption data.xlsx",
    "2020 consumption data.xlsx",
    "2021 consumption data.xlsx",
    "2022 consumption data.xlsx"
]

all_years = []

files_weather = [
    "2022 weather data.xlsx",
    "2021 weather data.xlsx",
    "2020 weather data.xlsx",
    "2019 weather data.xlsx",
    "2018 weather data.xlsx"
    ]

all_weather = []

#Build it into one big data set for consumption over time
for path in files:
    usecols1 = ["Consommation", "Date", "Heures"]
    df = pd.read_excel(path, usecols = usecols1)
    df["Consommation"] = pd.to_numeric(df["Consommation"], errors="coerce")
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Heures"].astype(str),
        dayfirst=True,errors="coerce", utc = True    )
    #Clean data 
    df = df.dropna(subset=["Consommation", "datetime"])
    
    #Extract only datetime and consommation data and add to all years list
    all_years.append(df[["datetime", "Consommation"]])

for path in files_weather:
    usecols = ["ID OMM station", "Date", "Température (°C)"]
    df_weather = pd.read_excel(path, usecols=usecols)
    df_weather["Temperature"] = pd.to_numeric(df_weather["Température (°C)"], errors = 'coerce')
    df_weather["datetime"] = pd.to_datetime(df_weather["Date"], errors="coerce", utc=True)
    df_weather = df_weather.dropna(subset=["datetime", "Temperature"])
    df_weather = df_weather.groupby("datetime", as_index=False)["Temperature"].mean()
    all_weather.append(df_weather)


#Take lists of data and stacks them in one data frame, sorting by date time and resetting the index
df_all = pd.concat(all_years).sort_values("datetime").reset_index(drop=True)
weather_all = pd.concat(all_weather).sort_values("datetime").reset_index(drop=True)

#Interpolate the missing weather temperatures
weather_interp = (weather_all.set_index("datetime").sort_index().resample("15min")
              .interpolate(method="time").reset_index())

#Merge weather and consumption data
df_merged = pd.merge(df_all, weather_interp, on="datetime", how="left")

#Extract variables
df_merged["hour"]= df_merged["datetime"].dt.hour+ df_merged["datetime"].dt.minute/60
df_merged["minute"]= df_merged["datetime"].dt.minute
df_merged["day_week"]= df_merged["datetime"].dt.dayofweek
df_merged["day_year"]= df_merged["datetime"].dt.dayofyear
df_merged["year"]= df_merged["datetime"].dt.year


#Split into predicting and influencing variables
Variables = df_merged[["hour", "day_week", "day_year", "Temperature"]].to_numpy()
Prediction = df_merged["Consommation"].to_numpy()

#Build the GAM model 
gam = LinearGAM(s(0)+f(1)+s(2, basis = 'cp')+s(3)).fit(Variables, Prediction)

#Apply prediction 
df_merged["GAM_pred"] = gam.predict(Variables)
df_merged["error"] = df_merged["Consommation"] - df_merged["GAM_pred"]


#Plot to compare 
plt.scatter(df_merged["datetime"], df_merged["Consommation"], color = 'green', label = 'Actual')
plt.plot(df_merged["datetime"], df_merged["GAM_pred"], color = 'red', label = "Prediction")
plt.xlabel("Year")
plt.ylabel("Consumption [MW?]")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df_all["datetime"], df_all["error"])
plt.title("GAM Residuals Over Time")
plt.show()

