"""
ICC 2026 World Cup Prediction - Data Preprocessing
This script processes T20I match data from 2005-2024 to prepare features for prediction
Made by: Chanitha Abeygunawardena 
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(matches_path):
    """Load and clean T20I matches data"""
    print("Loading matches data...")
    matches = pd.read_csv(matches_path)
    
    # Convert date
    matches['Match Date'] = pd.to_datetime(matches['Match Date'])
    
    # Filter for international T20I matches (exclude associate nation matches for main model)
    main_teams = ['India', 'Namibia', 'Pakistan', 'Netherlands', 'United States',
              'Australia', 'Ireland', 'Oman', 'Sri Lanka', 'Zimbabwe',
              'England', 'Italy', 'Nepal', 'Scotland', 'West Indies',
              'Afghanistan', 'Canada', 'New Zealand', 'South Africa', 'United Arab Emirates', 'Bangladesh']
    
    # Create a filtered dataset for main teams
    matches_main = matches[
        (matches['Team1 Name'].isin(main_teams)) & 
        (matches['Team2 Name'].isin(main_teams))
    ].copy()
    
    print(f"Total matches: {len(matches)}")
    print(f"Main teams matches: {len(matches_main)}")
    
    return matches, matches_main

def create_results_dataset(matches_df):
    """
    Transform matches data into results format similar to original project
    Columns: date, Team_1, Team_2, Winner, Margin, Ground
    """
    results = []
    
    for idx, match in matches_df.iterrows():
        result = {
            'date': match['Match Date'],
            'Team_1': match['Team1 Name'],
            'Team_2': match['Team2 Name'],
            'Winner': match['Match Winner'],
            'Margin': match['Match Result Text'],
            'Ground': f"{match['Match Venue (City)']}, {match['Match Venue (Country)']}"
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Clean winner (remove 'no result' matches)
    results_df = results_df[results_df['Winner'].notna()].copy()
    
    return results_df

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
            'losses': 0,
            'win_rate': 0.0,
            'recent_form': 0.0
        }
    
    # Recent form (last 20 matches)
    recent_matches = team_matches.tail(20)
    recent_wins = len(recent_matches[recent_matches['Winner'] == team_name])
    
    return {
        'matches': total_matches,
        'wins': wins,
        'losses': total_matches - wins,
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

def create_training_dataset(results_df):
    """
    Create training dataset with features for each match
    Features: team statistics, head-to-head, recent form
    Target: match winner (1 if Team_1 wins, 0 if Team_2 wins)
    """
    training_data = []
    
    print("Creating training dataset...")
    for idx, match in results_df.iterrows():
        team1 = match['Team_1']
        team2 = match['Team_2']
        winner = match['Winner']
        match_date = match['date']
        
        # Get statistics for both teams UP TO this match date
        team1_stats = calculate_team_statistics(results_df, team1, match_date)
        team2_stats = calculate_team_statistics(results_df, team2, match_date)
        
        # Head-to-head
        h2h = create_head_to_head_features(
            results_df[results_df['date'] < match_date], team1, team2
        )
        
        # Create feature row
        features = {
            'Team_1': team1,
            'Team_2': team2,
            'date': match_date,
            
            # Team 1 features
            'team1_matches': team1_stats['matches'],
            'team1_wins': team1_stats['wins'],
            'team1_win_rate': team1_stats['win_rate'],
            'team1_recent_form': team1_stats['recent_form'],
            
            # Team 2 features
            'team2_matches': team2_stats['matches'],
            'team2_wins': team2_stats['wins'],
            'team2_win_rate': team2_stats['win_rate'],
            'team2_recent_form': team2_stats['recent_form'],
            
            # Head-to-head features
            'h2h_total': h2h['h2h_total'],
            'h2h_team1_wins': h2h['h2h_team1_wins'],
            'h2h_win_rate': h2h['h2h_win_rate'],
            
            # Target
            'team1_wins_match': 1 if winner == team1 else 0
        }
        
        training_data.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(results_df)} matches")
    
    training_df = pd.DataFrame(training_data)
    return training_df

def create_icc_rankings(results_df, reference_date=None):
    """
    Create ICC rankings based on performance
    Similar to the original rankings dataset
    """
    if reference_date is None:
        reference_date = results_df['date'].max()
    
    main_teams = ['India', 'Namibia', 'Pakistan', 'Netherlands', 'United States',
              'Australia', 'Ireland', 'Oman', 'Sri Lanka', 'Zimbabwe',
              'England', 'Italy', 'Nepal', 'Scotland', 'West Indies',
              'Afghanistan', 'Canada', 'New Zealand', 'South Africa', 'United Arab Emirates', 'Bangladesh']
    
    rankings = []
    for team in main_teams:
        stats = calculate_team_statistics(results_df, team, reference_date)
        
        # Calculate rating (similar to ICC rating system - simplified)
        # Weight recent form more heavily
        rating = (stats['win_rate'] * 0.4 + stats['recent_form'] * 0.6) * 1000
        
        rankings.append({
            'Team': team,
            'Matches': stats['matches'],
            'Wins': stats['wins'],
            'Win_Rate': stats['win_rate'],
            'Recent_Form': stats['recent_form'],
            'Rating': rating,
            'Position': 0  # Will be filled after sorting
        })
    
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
    rankings_df['Position'] = range(1, len(rankings_df) + 1)
    
    return rankings_df

if __name__ == "__main__":
    # Paths
    matches_path = r'datasets\t20i_Matches_Data.csv'
    
    # Load data
    matches_all, matches_main = load_and_clean_data(matches_path)
    
    # Create results dataset
    results = create_results_dataset(matches_main)
    print(f"\nResults dataset shape: {results.shape}")
    print(results.head())
    
    # Save results
    results.to_csv(r'datasets\results.csv', index=False)
    print(f"\nSaved results to datasets/results.csv")
    
    # Filter for recent data (2010 onwards for better relevance)
    results_filtered = results[results['date'] >= '2010-01-01'].copy()
    print(f"\nFiltered results (2010+): {len(results_filtered)} matches")
    
    # Create training dataset
    training_df = create_training_dataset(results_filtered)
    print(f"\nTraining dataset shape: {training_df.shape}")
    print(training_df.head())
    
    # Save training data
    training_df.to_csv(r'datasets\training_data.csv', index=False)
    print(f"\nSaved training data to datasets/training_data.csv")
    
    # Create current ICC rankings (as of latest data)
    rankings = create_icc_rankings(results_filtered)
    print(f"\n=== Current T20I Rankings (Based on Data) ===")
    print(rankings[['Position', 'Team', 'Matches', 'Win_Rate', 'Recent_Form', 'Rating']].head(20))
    
    # Save rankings
    rankings.to_csv(r'datasets\icc_rankings.csv', index=False)
    print(f"\nSaved rankings to datasets/icc_rankings.csv")
    
    print("\nData preprocessing complete!")
