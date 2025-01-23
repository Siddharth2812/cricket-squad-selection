import pandas as pd
from collections import defaultdict
import numpy as np

class BowlerDataProcessor:
    def __init__(self):
        self.bowler_stats = defaultdict(lambda: {
            'runs_given': 0,
            'balls_bowled': 0,
            'wickets': 0,
            'matches': set(),
            'innings': set(),
            'opponents': set(),
            'economy': 0,
            'strike_rate': 0,
            'average': 0,
            'five_wickets': 0,
            'three_wickets': 0,
            'maiden_overs': 0,
            'total_overs': 0,
            'dot_balls': 0,
            'extras': 0,
            'wickets_in_innings': defaultdict(int),
            'overs': defaultdict(lambda: {'runs': 0, 'balls': 0}),
            'valid_dismissals': {'bowled', 'caught', 'lbw', 'stumped', 'hit wicket', 'caught and bowled'}
        })
        
    def process_ball(self, row):
        bowler = row['bowler']
        if pd.isna(bowler) or bowler == 'NA':
            return
            
        bowler_stat = self.bowler_stats[bowler]
        match_id = row['match_id']
        innings = f"{match_id}_{row['inning']}"
        over_number = int(row['over'])
        
        # Track basic match info
        bowler_stat['matches'].add(match_id)
        bowler_stat['innings'].add(innings)
        bowler_stat['opponents'].add(row['batting_team'])
        
        # Extract ball data
        extras_type = str(row['extras_type']) if pd.notna(row['extras_type']) else ''
        batsman_runs = int(row['batsman_runs']) if pd.notna(row['batsman_runs']) else 0
        extra_runs = int(row['extra_runs']) if pd.notna(row['extra_runs']) else 0
        
        # Calculate valid runs against bowler
        bowler_runs = batsman_runs
        if extras_type in ['wides', 'noballs']:
            bowler_runs += extra_runs
        bowler_stat['runs_given'] += bowler_runs
        
        # Count valid deliveries (exclude wides)
        if extras_type != 'wides':
            bowler_stat['balls_bowled'] += 1
            
        # Track dot balls (only valid deliveries with 0 runs)
        if batsman_runs == 0 and extras_type not in ['wides', 'noballs']:
            bowler_stat['dot_balls'] += 1
            
        # Process wickets with valid dismissal types
        if row['is_wicket'] == 1 and pd.notna(row['player_dismissed']):
            dismissal = str(row['dismissal_kind']).lower()
            if dismissal in bowler_stat['valid_dismissals']:
                bowler_stat['wickets'] += 1
                bowler_stat['wickets_in_innings'][innings] += 1
                
        # Track extras (only wides and noballs)
        if extras_type in ['wides', 'noballs']:
            bowler_stat['extras'] += extra_runs
            
        # Track over-wise statistics for maidens
        over_key = (innings, over_number)
        bowler_stat['overs'][over_key]['runs'] += bowler_runs
        if extras_type != 'wides':
            bowler_stat['overs'][over_key]['balls'] += 1

    def calculate_final_stats(self):
        final_stats = []
        for bowler, stats in self.bowler_stats.items():
            matches = len(stats['matches'])
            if matches == 0:
                continue
                
            # Calculate core metrics
            overs_bowled = stats['balls_bowled'] / 6
            economy = (stats['runs_given'] / overs_bowled) if overs_bowled > 0 else 0
            strike_rate = (stats['balls_bowled'] / stats['wickets']) if stats['wickets'] > 0 else 0
            average = (stats['runs_given'] / stats['wickets']) if stats['wickets'] > 0 else 0
            
            # Calculate maiden overs (full overs with 0 runs)
            maidens = sum(1 for o in stats['overs'].values() 
                         if o['balls'] >= 6 and o['runs'] == 0)
            
            # Calculate wicket hauls
            three_wickets = five_wickets = 0
            for count in stats['wickets_in_innings'].values():
                if count >= 5:
                    five_wickets += 1
                if count >= 3:
                    three_wickets += 1
                    
            final_stats.append({
                'Bowler': bowler,
                'Matches': matches,
                'Innings': len(stats['innings']),
                'Overs': round(overs_bowled, 1),
                'RunsGiven': stats['runs_given'],
                'Wickets': stats['wickets'],
                'Economy': round(economy, 2),
                'StrikeRate': round(strike_rate, 2),
                'Average': round(average, 2),
                '5W': five_wickets,
                '3W': three_wickets,
                'MaidenOvers': maidens,
                'DotBalls': stats['dot_balls'],
                'Extras': stats['extras'],
                'UniqueOpponents': len(stats['opponents']),
                'OpponentsList': '|'.join(sorted(stats['opponents']))
            })
            
        return final_stats

def process_bowler_data(input_file, output_file):
    """Process bowler data from input CSV file and save results to output CSV"""
    try:
        # Read the CSV file using pandas
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        print("Processing data...")
        processor = BowlerDataProcessor()
        
        # Sort dataframe by match_id, inning, over, and ball to ensure correct order
        df = df.sort_values(['match_id', 'inning', 'over', 'ball'])
        
        # Process each ball
        for _, row in df.iterrows():
            processor.process_ball(row)
        
        # Calculate final statistics
        final_stats = processor.calculate_final_stats()
        
        # Save to CSV
        if final_stats:
            stats_df = pd.DataFrame(final_stats)
            # Sort by wickets in descending order
            stats_df = stats_df.sort_values('Wickets', ascending=False)
            stats_df.to_csv(output_file, index=False)
            print(f"\nSuccessfully processed data and saved to {output_file}")
            print(f"Processed statistics for {len(final_stats)} bowlers")
            
            # Print sample of the data
            print("\nFirst few rows of the processed data:")
            print(stats_df[['Bowler', 'Wickets', 'Economy', 'StrikeRate', '5W', '3W']].head())
            
            # Print some summary statistics
            print("\nSummary Statistics:")
            print(f"Total matches: {len(df['match_id'].unique())}")
            print(f"Total bowlers: {len(stats_df)}")
            print(f"Most wickets: {stats_df.sort_values('Wickets', ascending=False).iloc[0]['Bowler']} "
                  f"({int(stats_df.sort_values('Wickets', ascending=False).iloc[0]['Wickets'])})")
            print(f"Best economy: {stats_df.sort_values('Economy').iloc[0]['Bowler']} "
                  f"({stats_df.sort_values('Economy').iloc[0]['Economy']})")
            print(f"Most 5-wicket hauls: {stats_df.sort_values('5W', ascending=False).iloc[0]['Bowler']} "
                  f"({int(stats_df.sort_values('5W', ascending=False).iloc[0]['5W'])})")
            print(f"Most 3-wicket hauls: {stats_df.sort_values('3W', ascending=False).iloc[0]['Bowler']} "
                  f"({int(stats_df.sort_values('3W', ascending=False).iloc[0]['3W'])})")
        else:
            print("No valid statistics were generated")
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_file = "/Users/dog/Documents/CricketSquadSelection/deliveries.csv"  # Replace with your input file path
    output_file = "bowler_statistics.csv"  # Replace with your desired output file name
    
    process_bowler_data(input_file, output_file)