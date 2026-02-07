"""
ICC 2026 World Cup Match Predictor
Predict match outcomes using the trained Logistic Regression model
Made by: Chanitha Abeygunawardena 
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load model and scaler
print("Loading...")
with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler_lr.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load current data
results = pd.read_csv('datasets/results.csv')
results['date'] = pd.to_datetime(results['date'])
rankings = pd.read_csv('datasets/icc_rankings.csv')

print("Model and data loaded successfully\n")

def calculate_team_statistics(results_df, team_name, reference_date=None):
    """Calculate team statistics up to a reference date"""
    if reference_date:
        team_matches = results_df[
            (results_df['date'] <= reference_date) &
            ((results_df['Team_1'] == team_name) | (results_df['Team_2'] == team_name))
        ]
    else:
        team_matches = results_df[
            (results_df['Team_1'] == team_name) | (results_df['Team_2'] == team_name)
        ]
    
    total_matches = len(team_matches)
    wins = len(team_matches[team_matches['Winner'] == team_name])
    
    if total_matches == 0:
        return {
            'matches': 0,
            'wins': 0,
            'win_rate': 0.0,
            'recent_form': 0.0
        }
    
    # Recent form (last 20 matches)
    recent_matches = team_matches.tail(20)
    recent_wins = len(recent_matches[recent_matches['Winner'] == team_name])
    
    return {
        'matches': total_matches,
        'wins': wins,
        'win_rate': wins / total_matches if total_matches > 0 else 0,
        'recent_form': recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0
    }

def create_head_to_head_features(results_df, team1, team2):
    """Calculate head-to-head statistics between two teams"""
    h2h = results_df[
        ((results_df['Team_1'] == team1) & (results_df['Team_2'] == team2)) |
        ((results_df['Team_1'] == team2) & (results_df['Team_2'] == team1))
    ]
    
    if len(h2h) == 0:
        return {'h2h_total': 0, 'h2h_team1_wins': 0, 'h2h_win_rate': 0.5}
    
    team1_wins = len(h2h[h2h['Winner'] == team1])
    
    return {
        'h2h_total': len(h2h),
        'h2h_team1_wins': team1_wins,
        'h2h_win_rate': team1_wins / len(h2h)
    }

def predict_match(team1, team2, verbose=True):
    """Predict the outcome of a match between two teams"""
    
    # Get statistics for both teams
    team1_stats = calculate_team_statistics(results, team1)
    team2_stats = calculate_team_statistics(results, team2)
    h2h = create_head_to_head_features(results, team1, team2)
    
    # Create feature vector
    features = np.array([[
        team1_stats['matches'],
        team1_stats['wins'],
        team1_stats['win_rate'],
        team1_stats['recent_form'],
        team2_stats['matches'],
        team2_stats['wins'],
        team2_stats['win_rate'],
        team2_stats['recent_form'],
        h2h['h2h_total'],
        h2h['h2h_team1_wins'],
        h2h['h2h_win_rate']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    winner = team1 if prediction == 1 else team2
    confidence = probability[1] if prediction == 1 else probability[0]
    
    if verbose:
        print(f"\nMATCH PREDICTION: {team1} vs {team2}")
        print(f"\n{team1} Statistics:")
        print(f"  Matches: {team1_stats['matches']}")
        print(f"  Win Rate: {team1_stats['win_rate']:.1%}")
        print(f"  Recent Form: {team1_stats['recent_form']:.1%} (last 20 matches)")
        
        print(f"\n{team2} Statistics:")
        print(f"  Matches: {team2_stats['matches']}")
        print(f"  Win Rate: {team2_stats['win_rate']:.1%}")
        print(f"  Recent Form: {team2_stats['recent_form']:.1%} (last 20 matches)")
        
        print(f"\nHead-to-Head:")
        print(f"  Total matches: {h2h['h2h_total']}")
        print(f"  {team1} wins: {h2h['h2h_team1_wins']}")
        print(f"  {team1} win rate: {h2h['h2h_win_rate']:.1%}")
        
        print(f"  PREDICTION: {winner} wins")
        print(f"  Confidence: {confidence:.1%}")
    
    return {
        'team1': team1,
        'team2': team2,
        'predicted_winner': winner,
        'confidence': confidence,
        'team1_win_prob': probability[1],
        'team2_win_prob': probability[0]
    }

print("\n" + "="*36)
print("ICC T20 WORLD CUP 2026 - PREDICTIONS")
print("="*36)

groups = {
    'Group A': ['India', 'Namibia', 'Pakistan', 'Netherlands', 'United States Of America'],
    'Group B': ['Australia', 'Ireland', 'Oman', 'Sri Lanka', 'Zimbabwe'],
    'Group C': ['England', 'Italy', 'Nepal', 'Scotland', 'West Indies'],
    'Group D': ['Afghanistan','Canada','New Zealand','South Africa', 'United Arab Emirates']
}

print("\nGroup Compositions:")
for group, teams in groups.items():
    print(f"\n{group}:")
    for i, team in enumerate(teams, 1):
        rank_info = rankings[rankings['Team'] == team]
        if not rank_info.empty:
            rank = rank_info.iloc[0]['Position']
            rating = rank_info.iloc[0]['Rating']
            print(f"  {i}. {team:20s} Rank: {rank:2.0f} | Rating: {rating:.0f}")
        else:
            print(f"  {i}. {team}")

# Predict all group stage matches
print("\n\n" + "="*29)
print("GROUP STAGE MATCH PREDICTIONS")
print("="*29)

all_predictions = []

for group_name, teams in groups.items():
    print(f"\n{group_name} Matches:")
    print("-" * 70)
    
    # Generate all match combinations in the group
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1, team2 = teams[i], teams[j]
            result = predict_match(team1, team2, verbose=False)
            all_predictions.append(result)
            
            print(f"{team1} vs {team2} >>> "
                  f"{result['predicted_winner']} wins"
                  f" with {result['confidence']:.1%} confidence")

# Calculate group standings based on 
print("\n\n" + "="*25)
print("PREDICTED GROUP STANDINGS")
print("="*25)

for group_name, teams in groups.items():
    standings = {team: {'played': 0, 'won': 0, 'lost': 0, 'points': 0} for team in teams}
    
    # Calculate wins/losses based on predictions
    for pred in all_predictions:
        if pred['team1'] in teams and pred['team2'] in teams:
            standings[pred['team1']]['played'] += 1
            standings[pred['team2']]['played'] += 1
            
            if pred['predicted_winner'] == pred['team1']:
                standings[pred['team1']]['won'] += 1
                standings[pred['team1']]['points'] += 2
                standings[pred['team2']]['lost'] += 1
            else:
                standings[pred['team2']]['won'] += 1
                standings[pred['team2']]['points'] += 2
                standings[pred['team1']]['lost'] += 1
    
    # Sort by points
    standings_sorted = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
    
    print(f"\n{group_name}:")
    print(f"{'Team':<20} {'Played':>10} {'Won':>10} {'Lost':>10} {'Points':>10}")
    print("-" * 70)
    for team, stats in standings_sorted:
        print(f"{team:<20} {stats['played']:>10} {stats['won']:>10} "
              f"{stats['lost']:>10} {stats['points']:>10}")

# Get top 2 from each group for Super 8
print("\n\n" + "="*27)
print("QUALIFIED FOR SUPER 8 STAGE")
print("="*27)

super8_teams = []
for group_name, teams in groups.items():
    standings = {team: {'played': 0, 'won': 0, 'lost': 0, 'points': 0} for team in teams}
    
    # Calculate standings
    for pred in all_predictions:
        if pred['team1'] in teams and pred['team2'] in teams:
            standings[pred['team1']]['played'] += 1
            standings[pred['team2']]['played'] += 1
            
            if pred['predicted_winner'] == pred['team1']:
                standings[pred['team1']]['won'] += 1
                standings[pred['team1']]['points'] += 2
                standings[pred['team2']]['lost'] += 1
            else:
                standings[pred['team2']]['won'] += 1
                standings[pred['team2']]['points'] += 2
                standings[pred['team1']]['lost'] += 1
    
    # Get top 2
    standings_sorted = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
    top2 = [team for team, stats in standings_sorted[:2]]
    super8_teams.extend(top2)
    print(f"{group_name}: {top2[0]} (1st), {top2[1]} (2nd)")

# Create Super 8 groups
# Group 1: A1, B2, C1, D2
# Group 2: B1, A2, D1, C2
super8_group1 = [super8_teams[0], super8_teams[3], super8_teams[4], super8_teams[7]]  # A1, B2, C1, D2
super8_group2 = [super8_teams[2], super8_teams[1], super8_teams[6], super8_teams[5]]  # B1, A2, D1, C2

print("\n" + "="*14)
print("SUPER 8 GROUPS")
print("="*14)
print(f"\nSuper 8 Group 1: {', '.join(super8_group1)}")
print(f"Super 8 Group 2: {', '.join(super8_group2)}")

# Predict Super 8 matches
print("\n" + "="*25)
print("SUPER 8 STAGE PREDICTIONS")
print("="*25)

super8_predictions = []

for group_num, super8_group in enumerate([super8_group1, super8_group2], 1):
    print(f"\nSuper 8 Group {group_num}:")
    print("-" * 70)
    
    for i in range(len(super8_group)):
        for j in range(i + 1, len(super8_group)):
            team1, team2 = super8_group[i], super8_group[j]
            result = predict_match(team1, team2, verbose=False)
            super8_predictions.append(result)
            
            print(f"{team1} vs {team2} >>> "
                  f"{result['predicted_winner']} wins"
                  f" with {result['confidence']:.1%} confidence")

# Calculate Super 8 standings
print("\n" + "="*17)
print("SUPER 8 STANDINGS")
print("="*17)

semifinalists = []

for group_num, super8_group in enumerate([super8_group1, super8_group2], 1):
    standings = {team: {'played': 0, 'won': 0, 'lost': 0, 'points': 0} for team in super8_group}
    
    # Calculate standings
    for pred in super8_predictions:
        if pred['team1'] in super8_group and pred['team2'] in super8_group:
            standings[pred['team1']]['played'] += 1
            standings[pred['team2']]['played'] += 1
            
            if pred['predicted_winner'] == pred['team1']:
                standings[pred['team1']]['won'] += 1
                standings[pred['team1']]['points'] += 2
                standings[pred['team2']]['lost'] += 1
            else:
                standings[pred['team2']]['won'] += 1
                standings[pred['team2']]['points'] += 2
                standings[pred['team1']]['lost'] += 1
    
    # Sort by points
    standings_sorted = sorted(standings.items(), key=lambda x: x[1]['points'], reverse=True)
    
    print(f"\nSuper 8 Group {group_num}:")
    print(f"{'Team':<25} {'Played':>10} {'Won':>10} {'Lost':>10} {'Points':>10}")
    print("-" * 70)
    for team, stats in standings_sorted:
        print(f"{team:<25} {stats['played']:>10} {stats['won']:>10} "
              f"{stats['lost']:>10} {stats['points']:>10}")
    
    # Top 2 advance to semi-finals
    top2 = [team for team, stats in standings_sorted[:2]]
    semifinalists.extend(top2)
    print(f"\nQualified for Semi-Finals: {top2[0]}, {top2[1]}")

# Predict Semi-Finals
print("\n\n" + "="*23)
print("SEMI-FINALS PREDICTIONS")
print("="*23)

# Semi-Final 1: Super 8 Group 1 winner vs Super 8 Group 2 runner-up
sf1 = predict_match(semifinalists[0], semifinalists[3], verbose=True)

# Semi-Final 2: Super 8 Group 2 winner vs Super 8 Group 1 runner-up  
sf2 = predict_match(semifinalists[2], semifinalists[1], verbose=True)

# Final
print("\n" + "="*16)
print("FINAL PREDICTION")
print("="*16)

final = predict_match(sf1['predicted_winner'], sf2['predicted_winner'], verbose=True)

print("\n" + "="*46)
print(f"ICC T20 WORLD CUP 2026 PREDICTED WINNER: {final['predicted_winner'].upper()}")
print("="*46)

# Save all predictions
all_tournament_predictions = all_predictions + super8_predictions
predictions_df = pd.DataFrame(all_tournament_predictions)
predictions_df.to_csv('datasets/tournament_predictions.csv', index=False)
print(f"\nPredictions saved to datasets/tournament_predictions.csv")
