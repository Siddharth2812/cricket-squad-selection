import pandas as pd
from collections import defaultdict
import numpy as np

class CricketAllRounderAnalyzer:
    def __init__(self):
        self.players = defaultdict(lambda: {
            # Batting stats
            'matches': set(),
            'innings': defaultdict(lambda: defaultdict(int)),
            'total_runs': 0,
            'total_balls': 0,
            'fours': 0,
            'sixes': 0,
            'fifties': 0,
            'hundreds': 0,
            'dismissals': 0,
            'opponents': set(),
            'venues': set(),
            'bat_teams': defaultdict(set),
            
            # Bowling stats
            'bowl_matches': set(),
            'total_overs': 0.0,
            'total_runs_given': 0,
            'total_wickets': 0,
            'total_maidens': 0,
            'current_over': defaultdict(lambda: {'balls': 0, 'runs': 0}),
            'bowl_opponents': set(),
            'bowl_teams': defaultdict(set),
            
            # Match results
            'wins': 0,
            'losses': 0,
            'draws': 0,
            
            # Per-match tracking
            'match_stats': defaultdict(lambda: {
                'bat': {'runs': 0, 'balls': 0, '4s': 0, '6s': 0, 'sr': 0},
                'bowl': {'overs': 0, 'runs': 0, 'wickets': 0, 'maidens': 0, 'econ': 0},
                'result': None
            })
        })
        
        self.matches = defaultdict(lambda: {
            'teams': set(),
            'winner': None,
            'venue': None,
            'innings': defaultdict(dict)
        })

    def _preprocess_matches(self, df):
        """Analyze match outcomes and store context"""
        for match_id, match_df in df.groupby('match_id'):
            innings_data = {}
            teams = set()
            
            # Process innings
            for inning, inning_df in match_df.groupby('inning'):
                batting_team = inning_df['batting_team'].iloc[0]
                total_runs = inning_df['total_runs'].sum()
                teams.add(batting_team)
                innings_data[inning] = {'team': batting_team, 'runs': total_runs}
            
            # Determine match result
            winner = None
            if len(innings_data) >= 2:
                first_inn = innings_data[1]['runs']
                second_inn = innings_data[2]['runs']
                winner = innings_data[1]['team'] if first_inn > second_inn else innings_data[2]['team'] if second_inn > first_inn else None
            
            # Store match info
            self.matches[match_id] = {
                'teams': teams,
                'winner': winner,
                'venue': match_df['venue'].iloc[0] if 'venue' in match_df.columns else 'Unknown',
                'innings': innings_data
            }

    def _process_batting(self, row):
        batter = row['batter']
        match_id = row['match_id']
        inning = row['inning']
        player = self.players[batter]
        
        # Update basic info
        player['matches'].add(match_id)
        player['opponents'].add(row['bowling_team'])
        player['venues'].add(self.matches[match_id]['venue'])
        player['bat_teams'][match_id].add(row['batting_team'])
        
        # Update batting stats
        runs = int(row['batsman_runs'])
        extras_type = str(row['extras_type']).lower()
        
        if extras_type in {'', 'noballs'}:
            player['total_runs'] += runs
            player['total_balls'] += 1
            player['innings'][match_id][inning] += runs
            
            # Update per-match stats
            player['match_stats'][match_id]['bat']['runs'] += runs
            player['match_stats'][match_id]['bat']['balls'] += 1
            
            # Update boundaries
            if runs == 4:
                player['fours'] += 1
                player['match_stats'][match_id]['bat']['4s'] += 1
            if runs == 6:
                player['sixes'] += 1
                player['match_stats'][match_id]['bat']['6s'] += 1
        
        # Handle dismissals
        if row['is_wicket'] == 1 and row['player_dismissed'] == batter:
            player['dismissals'] += 1
            inning_runs = player['innings'][match_id][inning]
            if inning_runs >= 100:
                player['hundreds'] += 1
            elif inning_runs >= 50:
                player['fifties'] += 1

    def _process_bowling(self, row):
        bowler = row['bowler']
        match_id = row['match_id']
        over = int(row['over'])
        player = self.players[bowler]
        
        # Update basic info
        player['bowl_matches'].add(match_id)
        player['bowl_opponents'].add(row['batting_team'])
        player['bowl_teams'][match_id].add(row['bowling_team'])
        
        # Process bowling figures
        total_runs = int(row['total_runs'])
        extras_type = str(row['extras_type']).lower()
        
        player['total_runs_given'] += total_runs
        over_key = (match_id, over)
        
        # Track valid deliveries
        if extras_type != 'wides':
            player['current_over'][over_key]['balls'] += 1
            player['current_over'][over_key]['runs'] += total_runs
            
            # Check for completed over
            if player['current_over'][over_key]['balls'] == 6:
                player['total_overs'] += 1
                if player['current_over'][over_key]['runs'] == 0:
                    player['total_maidens'] += 1
                del player['current_over'][over_key]
        
        # Track wickets
        if row['is_wicket'] == 1 and row['dismissal_kind'] in {'bowled', 'caught', 'lbw', 'stumped'}:
            player['total_wickets'] += 1
        
        # Update per-match stats
        player['match_stats'][match_id]['bowl']['runs'] += total_runs
        player['match_stats'][match_id]['bowl']['wickets'] += 1 if row['is_wicket'] == 1 else 0

    def _calculate_results(self):
        """Determine match outcomes for each player"""
        for player, data in self.players.items():
            for match_id in data['matches'].union(data['bowl_matches']):
                teams = data['bat_teams'][match_id].union(data['bowl_teams'][match_id])
                winner = self.matches[match_id]['winner']
                
                if winner in teams:
                    data['wins'] += 1
                elif winner is None:
                    data['draws'] += 1
                else:
                    data['losses'] += 1

    def generate_stats(self):
        """Generate final dataframe with all 45 columns including player details"""
        stats = []
        
        for player_name, data in self.players.items():
            # Skip players with no data
            if not data['matches'] and not data['bowl_matches']:
                continue
                
            # Calculate totals
            total_matches = len(data['matches'].union(data['bowl_matches']))
            total_innings = sum(len(innings) for innings in data['innings'].values())
            
            # Get last match IDs safely
            last_bat_match = max(data['matches']) if data['matches'] else None
            last_bowl_match = max(data['bowl_matches']) if data['bowl_matches'] else None

            # Batting parameters with player name
            batting_stats = {
                'Player': player_name,
                'X(RAB)': data['total_runs'],
                'x(Bowl)': data['total_balls'],
                'T.RAB': data['total_runs'],
                'TBF': data['total_balls'],
                'T(50)': data['fifties'],
                'T(100)': data['hundreds'],
                'T(4s)': data['fours'],
                'T(6s)': data['sixes'],
                'Avg.Run': data['total_runs'] / max(1, data['dismissals']),
                'Avg.SR': (data['total_runs'] / data['total_balls'] * 100) if data['total_balls'] else 0,
                'RF(50s)': data['fifties'] / max(1, total_innings),
                'RF(100s)': data['hundreds'] / max(1, total_innings),
                'Avg.BF': data['total_balls'] / max(1, len(data['matches'])),
                'Avg.RAB': data['total_runs'] / max(1, len(data['matches'])),
            }

            # Get team info safely
            team = 'N/A'
            if data['bat_teams']:
                bat_teams_values = next(iter(data['bat_teams'].values()), set())
                if bat_teams_values:
                    team = next(iter(bat_teams_values))
            elif data['bowl_teams']:
                bowl_teams_values = next(iter(data['bowl_teams'].values()), set())
                if bowl_teams_values:
                    team = next(iter(bowl_teams_values))

            # Bowling parameters
            bowling_stats = {
                'T.over': round(data['total_overs'], 1),
                'T.Run.Given': data['total_runs_given'],
                'T.Wic': data['total_wickets'],
                'T.Mdn': data['total_maidens'],
                'T.ECN': data['total_runs_given'] / data['total_overs'] if data['total_overs'] else 0,
                'Avg.over': data['total_overs'] / max(1, len(data['bowl_matches'])),
                'Avg.Run.Given': data['total_runs_given'] / max(1, len(data['bowl_matches'])),
                'Avg.Wic': data['total_wickets'] / max(1, len(data['bowl_matches'])),
                'Avg.Mdn': data['total_maidens'] / max(1, len(data['bowl_matches'])),
                'Avg.ECN': (data['total_runs_given'] / data['total_overs']) if data['total_overs'] else 0,
            }

            # Match results and context
            result_stats = {
                'Team': team,
                'W': data['wins'],
                'L': data['losses'],
                'D': data['draws'],
                'T.Win': data['wins'],
                'T.Loss': data['losses'],
                'Win(in percent)': (data['wins'] / total_matches * 100) if total_matches else 0,
                'Loss(in percent)': (data['losses'] / total_matches * 100) if total_matches else 0,
                'Avg.Win': data['wins'] / max(1, total_matches),
                'Avg.loss': data['losses'] / max(1, total_matches),
                'x(OP)': list(data['opponents'])[-1] if data['opponents'] else 'N/A',
                'x(VNU)': list(data['venues'])[-1] if data['venues'] else 'N/A',
                'x(W/L)': 'W' if data['wins'] > data['losses'] else 'L' if data['losses'] > 0 else 'D'
            }

            # Combine all stats
            final_stats = {
                **batting_stats,
                **bowling_stats,
                **result_stats,
                # Per-match stats
                'x(SR)': (data['match_stats'][last_bat_match]['bat']['runs'] / 
                         data['match_stats'][last_bat_match]['bat']['balls'] * 100) 
                         if last_bat_match and data['match_stats'][last_bat_match]['bat']['balls'] else 0,
                'x(4)': data['match_stats'][last_bat_match]['bat']['4s'] if last_bat_match else 0,
                'x(6)': data['match_stats'][last_bat_match]['bat']['6s'] if last_bat_match else 0,
                'x(Over)': data['match_stats'][last_bowl_match]['bowl']['overs'] 
                          if last_bowl_match else 0,
                'x(Run)': data['match_stats'][last_bowl_match]['bowl']['runs'] 
                         if last_bowl_match else 0,
                'X(Wic)': data['match_stats'][last_bowl_match]['bowl']['wickets'] 
                         if last_bowl_match else 0,
                'x(Mdn)': data['match_stats'][last_bowl_match]['bowl']['maidens'] 
                         if last_bowl_match else 0,
                'x(ECN)': (data['match_stats'][last_bowl_match]['bowl']['runs'] / 
                          data['match_stats'][last_bowl_match]['bowl']['overs']) 
                          if last_bowl_match and data['match_stats'][last_bowl_match]['bowl']['overs'] else 0
            }

            stats.append(final_stats)

        return pd.DataFrame(stats)

    def process_data(self, df):
        """Main processing pipeline"""
        self._preprocess_matches(df)
        
        for _, row in df.iterrows():
            if row['batter'] != 'NA':
                self._process_batting(row)
            if row['bowler'] != 'NA':
                self._process_bowling(row)
        
        self._calculate_results()
        return self.generate_stats()

# Usage
if __name__ == "__main__":
    analyzer = CricketAllRounderAnalyzer()
    df = pd.read_csv("../deliveries.csv")
    result_df = analyzer.process_data(df)
    
    # Ensure all columns exist
    required_columns = [
        'Player', 'Team', 'X(RAB)', 'x(Bowl)', 'x(SR)', 'x(4)', 'x(6)', 'x(OP)', 'x(VNU)', 'x(W/L)',
        'T(50)', 'T(100)', 'T.RAB', 'TBF', 'Avg.Run', 'Avg.SR', 'RF(50s)', 'RF(100s)',
        'L', 'W', 'D', 'x(Over)', 'x(Run)', 'X(Wic)', 'x(Mdn)', 'x(ECN)', 'T(4s)',
        'T(6s)', 'T.over', 'T.Run.Given', 'T.Wic', 'T.Mdn', 'T.ECN', 'T.Win', 'T.Loss',
        'Avg.RAB', 'Avg.BF', 'Avg.SR', 'Avg.over', 'Avg.Run.Given', 'Avg.Wic', 'Avg.Mdn',
        'Avg.ECN', 'Avg.Win', 'Avg.loss', 'Win(in percent)', 'Loss(in percent)'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in result_df.columns:
            result_df[col] = np.nan
            
    # Order columns properly
    result_df = result_df[required_columns]
    
    # Save with player details
    result_df.to_csv("allrounder_performance.csv", index=False)
    print(result_df[['Player', 'Team', 'X(RAB)', 'T.Wic', 'Win(in percent)']].head())