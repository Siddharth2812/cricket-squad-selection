import pandas as pd
from collections import defaultdict
import numpy as np

class CricketDataProcessor:
    def __init__(self):
        self.player_stats = defaultdict(lambda: {
            'runs': 0,
            'balls_faced': 0,
            'fours': 0,
            'sixes': 0,
            'matches': set(),
            'fifties': 0,
            'hundreds': 0,
            'current_inning_runs': 0,
            'opponents': set(),
            'innings': set(),
            'dismissals': 0,
            'duck_outs': 0,
            'batting_positions': [],
            'position_counts': defaultdict(int)
        })
        self.innings_batters = {}
        self.current_innings = None
        
    def process_ball(self, row):
        batter = row['batter']
        if pd.isna(batter) or batter == 'NA':
            return
            
        # Track innings changes
        innings_key = f"{row['match_id']}_{row['inning']}"
        if innings_key != self.current_innings:
            self._finalize_innings()
            self.current_innings = innings_key
            
        # Track batting position
        if innings_key not in self.innings_batters:
            self.innings_batters[innings_key] = []
            
        if batter not in self.innings_batters[innings_key]:
            position = len(self.innings_batters[innings_key]) + 1
            self.innings_batters[innings_key].append(batter)
            self.player_stats[batter]['batting_positions'].append(position)
            self.player_stats[batter]['position_counts'][position] += 1
            
        # Initialize player stats
        player = self.player_stats[batter]
        
        # Track matches and innings
        player['matches'].add(row['match_id'])
        player['innings'].add(innings_key)
        player['opponents'].add(row['bowling_team'])
        
        # Process runs and balls faced
        batsman_runs = int(row['batsman_runs']) if pd.notna(row['batsman_runs']) else 0
        extras_type = str(row['extras_type']) if pd.notna(row['extras_type']) else ''
        
        # Count valid deliveries (exclude wides)
        if extras_type in ['', 'noballs']:
            player['balls_faced'] += 1
            
        # Count valid runs (include no balls, exclude byes/legbyes)
        if extras_type in ['', 'noballs']:
            player['runs'] += batsman_runs
            player['current_inning_runs'] += batsman_runs
            
            # Count boundaries
            if batsman_runs == 4:
                player['fours'] += 1
            elif batsman_runs == 6:
                player['sixes'] += 1
                
        # Handle dismissals
        if row['is_wicket'] == 1 and row['player_dismissed'] == batter:
            # Record milestones
            current_runs = player['current_inning_runs']
            if current_runs >= 100:
                player['hundreds'] += 1
            elif current_runs >= 50:
                player['fifties'] += 1
                
            # Track duck outs
            if current_runs == 0:
                player['duck_outs'] += 1
                
            player['dismissals'] += 1
            player['current_inning_runs'] = 0

    def _finalize_innings(self):
        """Handle end of innings - count not out scores"""
        if self.current_innings:
            for batter in self.innings_batters.get(self.current_innings, []):
                player = self.player_stats[batter]
                current_runs = player['current_inning_runs']
                
                # Count not-out scores for milestones
                if current_runs >= 100:
                    player['hundreds'] += 1
                elif current_runs >= 50:
                    player['fifties'] += 1
                
                # Reset for next innings
                player['current_inning_runs'] = 0

    def calculate_final_stats(self):
        final_stats = []
        for player, stats in self.player_stats.items():
            matches_played = len(stats['matches'])
            if matches_played == 0:
                continue
                
            # Calculate averages
            avg = stats['runs'] / stats['dismissals'] if stats['dismissals'] > 0 else stats['runs']
            strike_rate = (stats['runs'] / stats['balls_faced'] * 100) if stats['balls_faced'] > 0 else 0
            duck_pct = (stats['duck_outs'] / stats['dismissals'] * 100) if stats['dismissals'] > 0 else 0
            
            # Position analysis
            positions = stats['batting_positions']
            if positions:
                most_common_pos = max(stats['position_counts'].items(), key=lambda x: x[1])[0]
                avg_pos = sum(positions) / len(positions)
                pos_consistency = (stats['position_counts'][most_common_pos] / len(positions)) * 100
                pos_range = f"{min(positions)}-{max(positions)}"
            else:
                most_common_pos = avg_pos = pos_consistency = pos_range = 0
                
            final_stats.append({
                'Player': player,
                'TotalRuns': stats['runs'],
                'BallsFaced': stats['balls_faced'],
                'StrikeRate': round(strike_rate, 2),
                'Fours': stats['fours'],
                'Sixes': stats['sixes'],
                'Matches': matches_played,
                'Innings': len(stats['innings']),
                'Fifties': stats['fifties'],
                'Hundreds': stats['hundreds'],
                'AverageRun': round(avg, 2),
                'RF_50s': round(stats['fifties'] / matches_played, 3),
                'RF_100s': round(stats['hundreds'] / matches_played, 3),
                'UniqueOpponents': len(stats['opponents']),
                'OpponentsList': '|'.join(sorted(stats['opponents'])),
                'Dismissals': stats['dismissals'],
                'DuckOuts': stats['duck_outs'],
                'DuckOutPercentage': round(duck_pct, 2),
                'MostCommonPosition': most_common_pos,
                'AveragePosition': round(avg_pos, 2),
                'PositionConsistency': round(pos_consistency, 2),
                'PositionRange': pos_range,
                'OpeningInnings': stats['position_counts'].get(1, 0),
                'TopOrderInnings': sum(stats['position_counts'][i] for i in [1, 2, 3]),
            })
            
        return final_stats

def process_cricket_data(input_file, output_file):
    """Process cricket data from input CSV file and save results to output CSV"""
    try:
        # Read the CSV file using pandas
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        print("Processing data...")
        processor = CricketDataProcessor()
        
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
            # Sort by total runs in descending order
            stats_df = stats_df.sort_values('TotalRuns', ascending=False)
            stats_df.to_csv(output_file, index=False)
            print(f"\nSuccessfully processed data and saved to {output_file}")
            print(f"Processed statistics for {len(final_stats)} players")
            
            # Print sample of the data
            print("\nFirst few rows of the processed data:")
            print(stats_df[['Player', 'TotalRuns', 'MostCommonPosition', 'DuckOuts']].head())
            
            # Print some summary statistics
            print("\nSummary Statistics:")
            print(f"Total matches: {len(df['match_id'].unique())}")
            print(f"Total players: {len(stats_df)}")
            print(f"Most duck outs: {stats_df.sort_values('DuckOuts', ascending=False).iloc[0]['Player']} "
                  f"({int(stats_df.sort_values('DuckOuts', ascending=False).iloc[0]['DuckOuts'])})")
            print(f"Most consistent opener: {stats_df[stats_df['OpeningInnings'] > 5].sort_values('AverageRun', ascending=False).iloc[0]['Player']}")
        else:
            print("No valid statistics were generated")
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_file = "/Users/dog/Documents/CricketSquadSelection/deliveries.csv"
    output_file = "cricket_statistics_fixed.csv"
    
    process_cricket_data(input_file, output_file)