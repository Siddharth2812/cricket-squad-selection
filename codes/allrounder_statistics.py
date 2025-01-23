import pandas as pd
from collections import defaultdict
import numpy as np

class AllRounderProcessor:
    def __init__(self):
        self.players = defaultdict(lambda: {
            # Batting stats
            'bat_matches': set(),
            'bat_innings': set(),
            'total_runs': 0,
            'total_balls_faced': 0,
            'total_fours': 0,
            'total_sixes': 0,
            'fifties': 0,
            'hundreds': 0,
            'dismissals': 0,
            'bat_opponents': set(),
            'venues': set(),
            'bat_teams': defaultdict(set),
            
            # Bowling stats
            'bowl_matches': set(),
            'bowl_innings': set(),
            'total_overs': 0,
            'total_runs_given': 0,
            'total_wickets': 0,
            'total_maidens': 0,
            'total_dot_balls': 0,
            'bowl_opponents': set(),
            'bowl_teams': defaultdict(set),
            
            # Match results
            'wins': 0,
            'losses': 0,
            'draws': 0,
            
            # Per-match tracking
            'match_stats': defaultdict(lambda: {
                'runs': 0,
                'balls_faced': 0,
                'fours': 0,
                'sixes': 0,
                'overs': 0,
                'runs_given': 0,
                'wickets': 0,
                'maidens': 0,
                'result': None
            })
        })
        
        self.match_info = defaultdict(lambda: {
            'teams': set(),
            'winner': None,
            'venue': 'Unknown'
        })

    def preprocess_matches(self, df):
        """Determine match results and venues"""
        match_groups = df.groupby('match_id')
        for match_id, match_df in match_groups:
            teams = set(match_df['batting_team'].unique())
            inning_scores = {}
            
            # Get scores for each inning
            for inning, inning_df in match_df.groupby('inning'):
                total_runs = inning_df['total_runs'].sum()
                batting_team = inning_df['batting_team'].iloc[0]
                inning_scores[inning] = {'team': batting_team, 'runs': total_runs}
            
            # Determine match result
            winner = None
            if len(inning_scores) >= 2:
                team1 = inning_scores[1]['team']
                score1 = inning_scores[1]['runs']
                team2 = inning_scores[2]['team']
                score2 = inning_scores[2]['runs']
                winner = team1 if score1 > score2 else team2 if score2 > score1 else None
            
            self.match_info[match_id] = {
                'teams': teams,
                'winner': winner,
                'venue': 'Unknown'  # Add venue extraction if available
            }

    def process_ball(self, row):
        """Process each ball for both batting and bowling stats"""
        match_id = row['match_id']
        batter = row['batter']
        bowler = row['bowler']
        player_team = row['batting_team']
        opponent_team = row['bowling_team']
        
        # Batting processing
        if batter != 'NA':
            p = self.players[batter]
            p['bat_matches'].add(match_id)
            p['bat_innings'].add(f"{match_id}_{row['inning']}")
            p['bat_opponents'].add(opponent_team)
            p['venues'].add(self.match_info[match_id]['venue'])
            p['bat_teams'][match_id].add(player_team)
            
            # Update batting stats
            runs = int(row['batsman_runs'])
            extras_type = row['extras_type'] if pd.notna(row['extras_type']) else ''
            
            if extras_type in ['', 'noballs']:
                p['total_runs'] += runs
                p['total_balls_faced'] += 1
                p['match_stats'][match_id]['runs'] += runs
                p['match_stats'][match_id]['balls_faced'] += 1
                
                if runs == 4:
                    p['total_fours'] += 1
                    p['match_stats'][match_id]['fours'] += 1
                if runs == 6:
                    p['total_sixes'] += 1
                    p['match_stats'][match_id]['sixes'] += 1

            # Handle dismissals
            if row['is_wicket'] == 1 and row['player_dismissed'] == batter:
                p['dismissals'] += 1
                match_runs = p['match_stats'][match_id]['runs']
                if match_runs >= 100:
                    p['hundreds'] += 1
                elif match_runs >= 50:
                    p['fifties'] += 1

        # Bowling processing
        if bowler != 'NA':
            p = self.players[bowler]
            p['bowl_matches'].add(match_id)
            p['bowl_innings'].add(f"{match_id}_{row['inning']}")
            p['bowl_opponents'].add(player_team)
            p['bowl_teams'][match_id].add(opponent_team)
            
            # Update bowling stats
            total_runs = int(row['total_runs'])
            extras_type = row['extras_type'] if pd.notna(row['extras_type']) else ''
            
            p['total_runs_given'] += total_runs
            p['match_stats'][match_id]['runs_given'] += total_runs
            
            if extras_type != 'wides':
                p['total_overs'] += 1/6
                p['match_stats'][match_id]['overs'] += 1/6
                
            # Track wickets
            if row['is_wicket'] == 1 and pd.notna(row['player_dismissed']):
                p['total_wickets'] += 1
                p['match_stats'][match_id]['wickets'] += 1

            # Track maidens (simplified implementation)
            if total_runs == 0 and extras_type not in ['wides', 'noballs']:
                p['total_dot_balls'] += 1
                if p['match_stats'][match_id]['overs'] % 1 == 0:  # Complete over
                    p['total_maidens'] += 1
                    p['match_stats'][match_id]['maidens'] += 1

    def calculate_final_stats(self):
        """Calculate all required parameters"""
        results = []
        
        for player, stats in self.players.items():
            # Skip players with no data
            if not stats['bat_matches'] and not stats['bowl_matches']:
                continue
                
            # Match results calculation
            total_matches = len(stats['bat_matches'].union(stats['bowl_matches']))
            wins = sum(1 for match_id in stats['bat_matches'].union(stats['bowl_matches']) 
                      if self.match_info[match_id]['winner'] in 
                      (stats['bat_teams'][match_id].union(stats['bowl_teams'][match_id])))
            
            losses = total_matches - wins - stats['draws']
            
            # Batting parameters
            batting_stats = {
                'Player': player,
                'T.RAB': stats['total_runs'],
                'TBF': stats['total_balls_faced'],
                'T.50': stats['fifties'],
                'T.100': stats['hundreds'],
                'T.4': stats['total_fours'],
                'T.6': stats['total_sixes'],
                'Avg.RAB': stats['total_runs'] / len(stats['bat_matches']) if stats['bat_matches'] else 0,
                'Avg.BF': stats['total_balls_faced'] / len(stats['bat_matches']) if stats['bat_matches'] else 0,
                'Avg.SR': (stats['total_runs'] / stats['total_balls_faced'] * 100) if stats['total_balls_faced'] > 0 else 0,
                'RF.50': stats['fifties'] / len(stats['bat_matches']) if stats['bat_matches'] else 0,
                'RF.100': stats['hundreds'] / len(stats['bat_matches']) if stats['bat_matches'] else 0,
            }

            # Bowling parameters
            bowling_stats = {
                'T.over': round(stats['total_overs'], 1),
                'T.Run.Given': stats['total_runs_given'],
                'T.Wic': stats['total_wickets'],
                'T.Mdn': stats['total_maidens'],
                'T.ECN': stats['total_runs_given'] / stats['total_overs'] if stats['total_overs'] > 0 else 0,
                'Avg.over': stats['total_overs'] / len(stats['bowl_matches']) if stats['bowl_matches'] else 0,
                'Avg.Run.Given': stats['total_runs_given'] / len(stats['bowl_matches']) if stats['bowl_matches'] else 0,
                'Avg.Wic': stats['total_wickets'] / len(stats['bowl_matches']) if stats['bowl_matches'] else 0,
                'Avg.Mdn': stats['total_maidens'] / len(stats['bowl_matches']) if stats['bowl_matches'] else 0,
                'Avg.ECN': (stats['total_runs_given'] / stats['total_overs']) / len(stats['bowl_matches']) 
                           if stats['total_overs'] > 0 and stats['bowl_matches'] else 0,
            }

            # Combined parameters
            combined_stats = {
                'T.Win': wins,
                'T.Loss': losses,
                'Win%': (wins / total_matches * 100) if total_matches > 0 else 0,
                'Loss%': (losses / total_matches * 100) if total_matches > 0 else 0,
                'Unique.OP': len(stats['bat_opponents'].union(stats['bowl_opponents'])),
                'Venues': len(stats['venues']),
                'Matches': total_matches
            }

            # Merge all stats
            final_stats = {**batting_stats, **bowling_stats, **combined_stats}
            results.append(final_stats)

        return pd.DataFrame(results)

def process_allrounder_data(input_file, output_file):
    """Process data and generate output file"""
    try:
        df = pd.read_csv(input_file)
        processor = AllRounderProcessor()
        
        # Preprocess match information
        processor.preprocess_matches(df)
        
        # Process each ball
        df = df.sort_values(['match_id', 'inning', 'over', 'ball'])
        for _, row in df.iterrows():
            processor.process_ball(row)
        
        # Calculate and save results
        result_df = processor.calculate_final_stats()
        result_df.to_csv(output_file, index=False)
        
        print(f"All-rounder stats saved to {output_file}")
        return result_df

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "../deliveries.csv"
    output_file = "allrounder_stats.csv"
    
    df = process_allrounder_data(input_file, output_file)
    if df is not None:
        print("\nSample output:")
        print(df.head()[['Player', 'T.RAB', 'T.Wic', 'Avg.RAB', 'Avg.Wic', 'Win%']])