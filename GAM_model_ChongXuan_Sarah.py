# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 09:27:47 2026

@author: Sarah Fenner and Chong Xuan Tan
"""

import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, f, s, te
import holidays
import numpy as np
from statsmodels.stats.stattools import durbin_watson


def prepare_variables(df):
    """
    Function to take imported files and convert the columns into a dataframe
    Parameters
    ----------
    df : DATAFRAME
        The read path of the dataframe 

    Returns
    -------
    df : DATAFRAME
        The full dataframe including all varaibles extracted from the raw data
    Features : LIST
        List of all variables to be considered in the model

    """
    
    df["datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize("Europe/Paris",nonexistent="NaT",ambiguous="NaT")
    
    #Convert all data to numeric values
    cols_convert = ["IsWeekend", "Lag_1", "Lag_4", "Lag_48", "Lag_336", "Temp_CC", "Temp_CE", "Temp_CW", "Temp_NC",
                    "Temp_NE", "Temp_NW", "Temp_SC", "Temp_SE", "Temp_SW", "Consumption_MW"]
    for col in cols_convert:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.dropna(subset=["datetime", "Consumption_MW"] + cols_convert)
    
    #Extract variables
    df["hour"]= df["datetime"].dt.hour+ df["datetime"].dt.minute/60
    df["minute"]= df["datetime"].dt.minute
    df["day_week"]= df["datetime"].dt.dayofweek
    df["day_year"]= df["datetime"].dt.dayofyear
    df["year"]= df["datetime"].dt.year


    #Add Holidays
    fr_holidays = holidays.country_holidays("FR", years=[2018, 2019, 2020, 2021, 2022]) 
    df["is_holiday"] = df["datetime"].dt.date.isin(fr_holidays).astype(int)
    df["is_pre_holiday"]  = (df["datetime"] + pd.Timedelta(days=1)).dt.date.isin(fr_holidays).astype(int)
    df["is_post_holiday"] = (df["datetime"] - pd.Timedelta(days=1)).dt.date.isin(fr_holidays).astype(int)
    
    #Group Temperature into North, South and Central
    df["Temp_C_mean"] = df[["Temp_CC","Temp_CE","Temp_CW"]].mean(axis=1)
    df["Temp_C_pred_mean"] = df["Temp_C_mean"].shift(-48)
    df["Temp_N_mean"] = df[["Temp_NC","Temp_NE","Temp_NW"]].mean(axis=1)
    df["Temp_N_pred_mean"] = df["Temp_N_mean"].shift(-48)
    df["Temp_S_mean"] = df[["Temp_SC","Temp_SE","Temp_SW"]].mean(axis=1)
    df["Temp_S_pred_mean"] = df["Temp_S_mean"].shift(-48)
   
    #Apply a 24hour prediction period
    df["Target_24h"] = df["Consumption_MW"].shift(-48)
    df = df.dropna(subset=["Target_24h"]).copy()
    
    #Build features list
    Features = ["hour", "day_week", "day_year", "year", "minute", "IsWeekend", 
                    "is_holiday", "is_pre_holiday", "is_post_holiday", "Temp_C_mean",
                   "Temp_N_mean", "Temp_S_mean",
                    "Lag_1", "Lag_4", "Lag_48", "Lag_336", "Temp_C_pred_mean",
                    "Temp_N_pred_mean", "Temp_S_pred_mean"]  

    return df,Features

def convert_to_numpy(dataframe, Features):
    """
    Converts the pandas dataframe to numpy so that it reduces errors in the GAM fit 
    Parameters
    ----------
    dataframe : DATAFRAME
        Dataframe with all variables for the particular period indicated 
    Features : LIST
        List of all variables to be considered in the model

    Returns
    -------
    result_x : ARRAY
        returns a numpy array of the features
    result_y : ARRAY
        returns a numpy array of the consumption 

    """
    result_x = dataframe[Features].to_numpy()
    result_y = dataframe["Target_24h"].to_numpy()
    return result_x, result_y

def build_periods(df, Features):
    """
    Defines the periods for training, validation and prediction
    Parameters
    ----------
    df : DATAFRAME
        Dataframe with all variables for the particular period indicated 
    Features : LIST
        List of all variables to be considered in the model

    Returns
    -------
    results_list : LIST
        List of all results arrays for the different periods .
    dataframe_list : LIST
        List of all the dataframes for the correct periods

    """
    #Define test/validation/predict periods
    start = df["datetime"].min()
    train_end = start+pd.DateOffset(years=4)
    validate_end = train_end+pd.DateOffset(months=8)
    test_end = validate_end+pd.DateOffset(months=16)
    
    #Build seperate data sets for each
    train_df = df[df["datetime"] < train_end].copy()
    validate_df = df[(df["datetime"] >= train_end) & (df["datetime"] <validate_end)].copy()
    test_df = df[(df["datetime"] >= validate_end) & (df["datetime"]< test_end)].copy()
    
    x_train, y_train =convert_to_numpy(train_df, Features)
    x_val, y_val = convert_to_numpy(validate_df, Features)
    x_test, y_test = convert_to_numpy(test_df, Features)
    results_list = [x_train,y_train,x_val,y_val,x_test,y_test]
    dataframe_list = [train_df, validate_df, test_df]
    return results_list, dataframe_list
    
def fit_gam(results_list, dataframe_list):
    """
    Builds the GAM model to train and predict 
    Parameters
    ----------
   results_list : LIST
       List of all results arrays for the different periods .
     dataframe_list : LIST
         List of all the dataframes for the correct periods

    Returns
    -------
    gam : TYPE
        The linear gam model 
    dataframe_list : LIST
        List of updated dataframes for the correct periods

    """
    #Build the GAM model 
    gam = LinearGAM(
        s(0,n_splines=8) +           # hour
        f(1) +                       # day_week
        s(2, n_splines=12, basis='cp') +   # day_year
        s(3) +                       #day 
        s(4) +                       #minute
        f(5) +                       #IsWeekend
        f(6) +                       # is_holiday
        f(7) +                       # is_pre_holiday
        f(8) +                       # is_post_holiday
        s(9) +                       #Temp_C_mean
        s(10) +                      #Temp_N_mean
        s(11) +                      #Temp_S_mean
        s(12) +                      #Lag_1
        s(13) +                      #Lag_4
        s(14) +                      #Lag_48
        s(15) +                      #Lag_336
        s(16) +                      #predict_Temp_C_mean
        s(17) +                      #Predict_Temp_N_mean
        s(18) +                      #Predict_Temp_S_mean
        te(0,9)+te(0,10)+te(0,11)+te(0,16)+te(0,17)+te(0,18)
    ).gridsearch(results_list[0], results_list[1])
    
    #Make predictions
    dataframe_list[1]["GAM_pred"] = gam.predict(results_list[2])
    dataframe_list[2]["GAM_pred"] = gam.predict(results_list[4])
    return gam, dataframe_list


def calc_errors(dataframe, x, gam, Name):
    """
    Calculates the error parameters of the model
    Parameters
    ----------
    dataframe : DATAFRAME
        Dataframe with all variables for the particular period indicated 
    x : ARRAY
        Numpy array of the features at that time
    gam : TYPE
        The Gam model 
    Name : STRING
        A label for which period the error is for 

    Returns
    -------
    dataframe : DATAFRAME
        Updated dataframe for that period

    """
    
    # Confidence Interval
    intervals = gam.prediction_intervals(x, width=0.95)
    dataframe["lower_ci"] = intervals[:, 0]
    dataframe["upper_ci"] = intervals[:, 1]

    # Error Analysis
    dataframe["error"] = dataframe["Target_24h"] - dataframe["GAM_pred"]
    dataframe["abs_error"] = np.abs(dataframe["error"])
    dataframe["squared_error"] = dataframe["error"] ** 2
    mbe = dataframe["error"].mean()

    # Metrics
    mae  = dataframe["abs_error"].mean()
    rmse = np.sqrt(dataframe["squared_error"].mean())
    mape = (dataframe["abs_error"] / dataframe["Target_24h"].replace(0, np.nan)).mean()

    # Statistical CI 
    stat_ci_coverage = (
        (dataframe["Target_24h"] >= dataframe["lower_ci"]) &
        (dataframe["Target_24h"] <= dataframe["upper_ci"])
    ).mean()

    # Residual Autocorrelation
    dw_stat = durbin_watson(dataframe["error"].dropna())

    # Peak vs Off Peak MAE
    peak_mask    = dataframe["hour"].between(8, 12) | dataframe["hour"].between(17, 21)
    peak_mae     = dataframe.loc[peak_mask,  "abs_error"].mean()
    offpeak_mae  = dataframe.loc[~peak_mask, "abs_error"].mean()

    # Ramp Error
    dataframe = dataframe.sort_values("datetime")
    dataframe["actual_ramp"] = dataframe["Target_24h"].diff()
    dataframe["pred_ramp"]   = dataframe["GAM_pred"].diff()
    dataframe["ramp_error"]  = dataframe["actual_ramp"] - dataframe["pred_ramp"]
    ramp_mae = dataframe["ramp_error"].abs().mean()

    #Check if within 200MW of actual consumption
    #Set bands
    tolerance = 400
    dataframe["elec_lower"]=dataframe["GAM_pred"]-tolerance
    dataframe["elec_upper"]=dataframe["GAM_pred"]+tolerance
    
    #Cut to bounds within tolerance
    dataframe["in_tolerance"]=((dataframe["Target_24h"] >= dataframe["elec_lower"]) & (dataframe["Target_24h"] <= dataframe["elec_upper"]))
    
    #See how much % within tolerance
    electrical_coverage = dataframe["in_tolerance"].mean()

    # Summary
    print(f"\n{'='*45}")
    print(f"  {Name}")
    print(f"{'='*45}")

    print(f"\nPoint Forecast")
    print(f"  MAE:               {mae:.1f} MW")
    print(f"  RMSE:              {rmse:.1f} MW")
    print(f"  MAPE:              {mape:.3%}")
    print(f"  MBE (bias):        {mbe:+.1f} MW  ")

    print(f"\nStatistical CI")
    print(f"  Coverage @ 95%:    {stat_ci_coverage:.2%}  ")
    print(f"  Durbin-Watson:     {dw_stat:.3f}  ")

    print(f"\nElectrical / Operational")
    print(f"  Peak MAE (8-12, 17-21h):  {peak_mae:.1f} MW")
    print(f"  Off-peak MAE:             {offpeak_mae:.1f} MW")
    print(f"  Ramp MAE (per 30 min):    {ramp_mae:.1f} MW")
    print(f"Share within tolerance for {Name}:{electrical_coverage:.2%}")
    

    return dataframe

    
def plot(dataframe, Name):   
    """
    Plots the actual vs modelled consumption over the period
    Parameters
    ----------
    dataframe : DATAFRAME
        Dataframe with all variables for the particular period indicated .
    Name : STRING
        Description for which period is being plotted 

    Returns
    -------
    None.

    """
    #Plot validation
    plt.figure(figsize=(12,5))
    plt.plot(dataframe["datetime"], dataframe["Target_24h"], label="Actual", color = 'hotpink')
    plt.plot(dataframe["datetime"], dataframe["GAM_pred"], label="Prediction", color = 'purple')
    plt.xlabel("Time")
    plt.ylabel("Consumption MW")
    plt.title(Name)
    
    # Confidence interval shading
    plt.fill_between(
        dataframe["datetime"],
        dataframe["lower_ci"],
        dataframe["upper_ci"],
        alpha=0.4,
        label="95% CI"
    )
    plt.legend()
    plt.show()
    
    #Plot of Error over time
    plt.figure()
    plt.plot(dataframe["datetime"], dataframe["error"], color = 'mediumvioletred')
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title(f"Error over time {Name}")
    plt.xticks(rotation=60)
    plt.show()
    
    #Plot for one 24 hour prediction
    plt.figure()
    start = dataframe["datetime"].min()+pd.Timedelta(weeks = 13)
    end = start + pd.Timedelta(days=1)
    win = dataframe[(dataframe["datetime"] >= start) & (dataframe["datetime"] < end)]
    plt.figure(figsize=(12,5))
    plt.plot(win["datetime"], win["Target_24h"], label="Actual", color = 'darkviolet')
    plt.plot(win["datetime"], win["GAM_pred"], label="Prediction", color = 'deepskyblue')
    plt.xlabel("Time")
    plt.ylabel("Consumption MW")
    plt.title(f"Prediction across one timestep {Name}")
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    file_path = r"Merged_Energy_Weather_30min_Final.csv"
    df = pd.read_csv(file_path)
    df,Features = prepare_variables(df)
    results_list, dataframe_list = build_periods(df, Features)
    gam, updated_dataframe_list=fit_gam(results_list, dataframe_list)
    validate_df = calc_errors(updated_dataframe_list[1], results_list[2], gam, "Validate")
    predict_df = calc_errors(updated_dataframe_list[2], results_list[4], gam, "Prediction")
    plot(validate_df, "Validation of 8 months")
    plot(predict_df, "Prediction period of 16 months")
    
    


