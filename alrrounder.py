import pandas as pd
import numpy as np

# Load the datasets
try:
    allrounder_df = pd.read_csv('allrounder_performance.csv')
    batsman_df = pd.read_csv('ipl_batsman_stats_fixed.csv')
    bowler_df = pd.read_csv('ipl_bowler_stats_output.csv')
except FileNotFoundError:
    print("One or more CSV files not found. Please check the file names.")
    exit()

# --- Data Cleaning and Preparation ---

# First rename the column in bowler_df
bowler_df.rename(columns={'Unnamed: 0': 'Player'}, inplace=True)

# Then strip whitespace from player names
allrounder_df['Player'] = allrounder_df['Player'].str.strip()
batsman_df['fullName'] = batsman_df['fullName'].str.strip()
bowler_df['Player'] = bowler_df['Player'].str.strip()

# Handle missing values *before* merging
allrounder_df = allrounder_df.replace('N/A', np.nan)
allrounder_df = allrounder_df.dropna(subset=['Team'])  # Drop rows with missing team

# --- Identify All-Rounders ---

# Define all-rounders as those who have both batted (appear in batsman_df)
# AND bowled (have taken wickets in allrounder_df)
allrounders_list = allrounder_df[(allrounder_df['T.Wic'] > 0)]['Player'].tolist()
allrounders_df = allrounder_df[allrounder_df['Player'].isin(allrounders_list)]

# --- Merge Data ---

# INNER join with batsman_df to get batting stats. Only keep players present in BOTH.
merged_df = pd.merge(allrounders_df, batsman_df, left_on='Player', right_on='fullName', how='inner')
# LEFT join with bowler_df to get bowling stats. Keep all all-rounders, even if no bowling data.
merged_df = pd.merge(merged_df, bowler_df, on='Player', how='left')

# --- Feature Engineering and Selection ---

# Select initial columns, handling missing values after the merge
allrounder_stats_df = merged_df[[
    'Player', 'Team', 'Matches', 'T.RAB', 'x(Bowl)', 'T(50)', 'T(100)',
    'T(4s)', 'T(6s)', 'T.over', 'T.Run.Given', 'T.Wic'
]].copy()

allrounder_stats_df.rename(columns={
    'T.RAB': 'Runs',
    'x(Bowl)': 'Balls Faced',
    'T(50)': 'Batting_50s',
    'T(100)': 'Batting_100s',
    'T.over': 'Overs',
    'T.Run.Given': 'Runs_Given',
    'T.Wic': 'Wickets'
}, inplace=True)

# Calculate Batting Average, avoiding division by zero
allrounder_stats_df['Batting_Average'] = allrounder_stats_df['Runs'] / (allrounder_stats_df['Wickets'].replace(0, np.nan))
allrounder_stats_df['Batting_Average'] = allrounder_stats_df['Batting_Average'].replace([np.inf, -np.inf], 0).fillna(0)

# Calculate Batting Strike Rate (from batsman_df - more reliable)
allrounder_stats_df['Batting_SR'] = merged_df['StrikeRate'].fillna(0)

# Calculate Bowling Average, handling cases where Wickets = 0
allrounder_stats_df['Bowling_Average'] = allrounder_stats_df['Runs_Given'] / allrounder_stats_df['Wickets'].replace(0, np.nan)
allrounder_stats_df['Bowling_Average'] = allrounder_stats_df['Bowling_Average'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate Bowling Strike Rate
allrounder_stats_df['Bowling_SR'] = (allrounder_stats_df['Overs'] * 6) / allrounder_stats_df['Wickets'].replace(0, np.nan)
allrounder_stats_df['Bowling_SR'] = allrounder_stats_df['Bowling_SR'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate Economy Rate
allrounder_stats_df['Economy'] = merged_df['Eco'].fillna(0)

# Fill remaining NaN values with 0
allrounder_stats_df.fillna(0, inplace=True)

# Save the new dataframe to a CSV file
allrounder_stats_df.to_csv('allrounder_statistics_final.csv', index=False)

print("All-rounder statistics saved to 'allrounder_statistics_final.csv'")
print(allrounder_stats_df.head())