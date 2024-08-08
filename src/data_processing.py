# *********************************************************************************************** #
#                                                                                                 #
#    data_processing.py                          For: None                                    #
#                                                                                    __           #
#    By: Myosin <hans95.bourgeois@gmail.com>         .--------..--.--..-----..-----.|__|.-----.   #
#                                                    |        ||  |  ||  _  ||__ --||  ||     |   #
#    Created: 2024/08/05 10:53:51 by Myosin          |__|__|__||___  ||_____||_____||__||__|__|   #
#    Updated: 2024/08/05 10:53:51 by Myosin                    |_____|                            #
#                                                                                                 #
# *********************************************************************************************** #

import pandas as pd
import numpy as np

from typing import Dict, Any, List
from decimal import Decimal


def exclude_playoff(df : pd) -> pd :
    """
    Exclusion from the last eight matches of each season, corresponding to the play-offs.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with playoff match data excluded.
    """
    # sort data in ascending order by "season" and in descending order by "match date", 
    # to ensure that for each season, the first match corresponds to the last match of the season.    
    sorted_df = df.sort_values(["season", "matchDate"],  ascending=[True, False], ignore_index=True)
    # Group match by season, in each group the first row corresponding to the last match of the season (playoff).
    grouped_df = sorted_df.groupby("season")
    # select all lines after the first 8 in each group. Lines with indices from 0 to 7 are excluded.
    new_df = grouped_df.apply(lambda x: x.iloc[8:].reset_index(drop=True), include_groups=False).reset_index()
    return new_df

def calculate_relative_stats(df : pd, kpi_cols : List[str], descriptive_cols : List[str]) :
    """
    Calculate relative statistics from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        kpi_cols (List[str]): List of KPI columns.
        descriptive_cols (List[str]): List of descriptive columns.

    Returns:
        pd.DataFrame: DataFrame containing calculated relative statistics.
    """
    
    sorted_df = df.sort_values(["matchEspnId", "teamScore"],  ascending=[True, False], ignore_index=True)
    grouped_df = sorted_df.groupby("matchEspnId")

    sub_kpi_df = grouped_df[kpi_cols]
    sub_descriptive_df = grouped_df[descriptive_cols]

    # calculate relative
    relative_kpi_df1 = sub_kpi_df.diff().dropna().reset_index()
    relative_kpi_df2 = sub_kpi_df.diff(-1).dropna().reset_index()

    # Separate descriptive
    relative_descriptive_df1 = sub_descriptive_df.nth(1).reset_index()
    relative_descriptive_df2 = sub_descriptive_df.nth(0).reset_index()

    # Concate kpi and descripitve
    relative_df1 = pd.concat([relative_descriptive_df1, relative_kpi_df1], axis = 1)
    relative_df2 = pd.concat([relative_descriptive_df2, relative_kpi_df2], axis = 1)

    #concate two relative df
    relative_df = pd.concat([relative_df1, relative_df2]).reset_index(drop=True)
    relative_df = relative_df.sort_values(["matchEspnId"], ignore_index=True)

    # Calculate score difference
    relative_df["scoreDiff"] = relative_df["teamScore"] - relative_df["opponentTeamScore"]
    return relative_df

def convert_decimal_to_float(x):
    """
    Convert a Decimal value to float if necessary.

    Args:
        x: Value to convert.

    Returns:
        float or the original value.
    """
    if isinstance(x, Decimal):
        return float(x)
    return x

def mean_threshold_detection(y, flat_curve_start_idx, flat_curve_end_idx, window_size):
    flat_curve =  y[flat_curve_start_idx: flat_curve_end_idx]
    mean = np.mean(abs(flat_curve))
    std = np.std(abs(flat_curve))
    thr = mean - 3 * std
    for i in range(len(y) - window_size + 1) :
        if all(np.mean(abs(points)) > thr  for points in y[i : i + window_size] ):
            break

    if i is not None :
        stable_point = i + 1
    else :
        stable_point = len(y)
    return stable_point, thr

def get_predicted_matches(row, df, selected_features, descriptive_features, n_matches):
    """
    Prepare data for prediction based on previous matches.
    
    Args:
    row (pd.Series): Current match data
    df (pd.DataFrame): Full dataset
    selected_features (list): Features to use for prediction
    descriptive_features (list): Descriptive columns to include
    n_matches (int): Number of previous matches to consider
    
    Returns:
    pd.DataFrame: Prepared data for the current match
    """
    # Filter previous matches of the team before the current match date
    previous_matches = df[(df["teamName"] == row["teamName"]) & (df["matchDate"] < row["matchDate"]) & (df["isHome"] == row["isHome"])]
    
    # Sort previous matches by date and select the last n_matches
    previous_matches = previous_matches.sort_values(["matchDate"]).tail(n_matches)
    
    # Calculate the mean of selected features
    features_df = previous_matches[selected_features].mean()
    
    # Add the 'isWinner' column with the current match result
    features_df['isWinner'] = row["isWinner"]
    features_df['isHome'] = row["isHome"]
    
    # Select descriptive features from the current match
    descriptive_df = row[descriptive_features]
    
    # Concatenate descriptive data and calculated features
    predicted_match = pd.concat([descriptive_df, features_df])
    return predicted_match