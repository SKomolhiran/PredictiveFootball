import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FootballFeatureEngineer:
    """
    Feature engineering for football match predictions
    
    Creates pre-match features avoiding any data leakage:
    - Elo ratings
    - Rolling form metrics  
    - Head-to-head records
    - Rest days and situational factors
    """
    
    def __init__(self, elo_k: float = 20, home_advantage: float = 70):
        """
        Args:
            elo_k: Elo update rate (10-30 typical)
            home_advantage: Home advantage in Elo points (60-80 typical)
        """
        self.elo_k = elo_k
        self.home_advantage = home_advantage
        self.team_elos = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    def _initialize_elo_ratings(self, teams: List[str], initial_rating: float = 1500) -> Dict[str, float]:
        """Initialize Elo ratings for all teams"""
        return {team: initial_rating for team in teams}
    
    def _update_elo(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> Tuple[float, float]:
        """
        Update Elo ratings after a match
        
        Returns:
            (new_home_elo, new_away_elo)
        """
        # Get current ratings
        home_elo = self.team_elos.get(home_team, 1500)
        away_elo = self.team_elos.get(away_team, 1500)
        
        # Expected scores with home advantage
        expected_home = 1 / (1 + 10**(-((home_elo - away_elo) + self.home_advantage) / 400))
        expected_away = 1 - expected_home
        
        # Actual scores
        if home_goals > away_goals:
            actual_home, actual_away = 1, 0
        elif home_goals < away_goals:
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Update ratings
        new_home_elo = home_elo + self.elo_k * (actual_home - expected_home)
        new_away_elo = away_elo + self.elo_k * (actual_away - expected_away)
        
        self.team_elos[home_team] = new_home_elo
        self.team_elos[away_team] = new_away_elo
        
        return new_home_elo, new_away_elo
    
    def _calculate_rolling_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate rolling statistics for each team
        
        Args:
            df: Match dataframe sorted by date
            window: Number of matches for rolling window
            
        Returns:
            DataFrame with rolling stats
        """
        df = df.copy()
        
        # Initialize rolling stats columns
        rolling_cols = [
            f'home_rolling_points_{window}', f'away_rolling_points_{window}',
            f'home_rolling_goals_for_{window}', f'away_rolling_goals_for_{window}',
            f'home_rolling_goals_against_{window}', f'away_rolling_goals_against_{window}',
            f'home_rolling_shots_{window}', f'away_rolling_shots_{window}',
            f'home_rolling_sot_{window}', f'away_rolling_sot_{window}',
            f'home_rolling_corners_{window}', f'away_rolling_corners_{window}'
        ]
        
        for col in rolling_cols:
            df[col] = np.nan
        
        # Calculate for each team
        all_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
        
        for team in all_teams:
            # Get all matches for this team (chronologically ordered)
            home_matches = df[df['home'] == team].copy()
            away_matches = df[df['away'] == team].copy()
            
            # Add team perspective columns
            home_matches['team_goals_for'] = home_matches['home_goals']
            home_matches['team_goals_against'] = home_matches['away_goals']
            home_matches['team_shots'] = home_matches['HS'].fillna(0)
            home_matches['team_sot'] = home_matches['HST'].fillna(0)
            home_matches['team_corners'] = home_matches['HC'].fillna(0)
            home_matches['team_points'] = home_matches['fulltime_result'].map({'H': 3, 'D': 1, 'A': 0})
            home_matches['is_home'] = True
            
            away_matches['team_goals_for'] = away_matches['away_goals']
            away_matches['team_goals_against'] = away_matches['home_goals']
            away_matches['team_shots'] = away_matches['AS'].fillna(0)
            away_matches['team_sot'] = away_matches['AST'].fillna(0)
            away_matches['team_corners'] = away_matches['AC'].fillna(0)
            away_matches['team_points'] = away_matches['fulltime_result'].map({'A': 3, 'D': 1, 'H': 0})
            away_matches['is_home'] = False
            
            # Combine all matches for this team
            team_matches = pd.concat([home_matches, away_matches])
            team_matches = team_matches.sort_values('date')
            
            # Calculate rolling averages (excluding current match)
            team_matches['rolling_points'] = team_matches['team_points'].rolling(window=window, min_periods=1).mean().shift(1)
            team_matches['rolling_goals_for'] = team_matches['team_goals_for'].rolling(window=window, min_periods=1).mean().shift(1)
            team_matches['rolling_goals_against'] = team_matches['team_goals_against'].rolling(window=window, min_periods=1).mean().shift(1)
            team_matches['rolling_shots'] = team_matches['team_shots'].rolling(window=window, min_periods=1).mean().shift(1)
            team_matches['rolling_sot'] = team_matches['team_sot'].rolling(window=window, min_periods=1).mean().shift(1)
            team_matches['rolling_corners'] = team_matches['team_corners'].rolling(window=window, min_periods=1).mean().shift(1)
            
            # Map back to original dataframe
            for _, match in team_matches.iterrows():
                match_idx = match.name
                if match['is_home']:
                    df.loc[match_idx, f'home_rolling_points_{window}'] = match['rolling_points']
                    df.loc[match_idx, f'home_rolling_goals_for_{window}'] = match['rolling_goals_for']
                    df.loc[match_idx, f'home_rolling_goals_against_{window}'] = match['rolling_goals_against']
                    df.loc[match_idx, f'home_rolling_shots_{window}'] = match['rolling_shots']
                    df.loc[match_idx, f'home_rolling_sot_{window}'] = match['rolling_sot']
                    df.loc[match_idx, f'home_rolling_corners_{window}'] = match['rolling_corners']
                else:
                    df.loc[match_idx, f'away_rolling_points_{window}'] = match['rolling_points']
                    df.loc[match_idx, f'away_rolling_goals_for_{window}'] = match['rolling_goals_for']
                    df.loc[match_idx, f'away_rolling_goals_against_{window}'] = match['rolling_goals_against']
                    df.loc[match_idx, f'away_rolling_shots_{window}'] = match['rolling_shots']
                    df.loc[match_idx, f'away_rolling_sot_{window}'] = match['rolling_sot']
                    df.loc[match_idx, f'away_rolling_corners_{window}'] = match['rolling_corners']
        
        return df
    
    def _calculate_head_to_head(self, df: pd.DataFrame, h2h_window: int = 3) -> pd.DataFrame:
        """Calculate head-to-head statistics"""
        df = df.copy()
        df['h2h_home_wins'] = np.nan
        df['h2h_draws'] = np.nan
        df['h2h_away_wins'] = np.nan
        
        for idx, match in df.iterrows():
            home_team, away_team = match['home'], match['away']
            match_date = match['date']
            
            # Find previous meetings between these teams
            h2h_home = df[(df['home'] == home_team) & (df['away'] == away_team) & (df['date'] < match_date)]
            h2h_away = df[(df['home'] == away_team) & (df['away'] == home_team) & (df['date'] < match_date)]
            
            # Combine and take most recent matches
            all_h2h = pd.concat([h2h_home, h2h_away]).sort_values('date', ascending=False).head(h2h_window)
            
            if len(all_h2h) > 0:
                # Count results from current home team's perspective
                home_wins = len(all_h2h[(all_h2h['home'] == home_team) & (all_h2h['fulltime_result'] == 'H')]) + \
                           len(all_h2h[(all_h2h['away'] == home_team) & (all_h2h['fulltime_result'] == 'A')])
                
                away_wins = len(all_h2h[(all_h2h['home'] == away_team) & (all_h2h['fulltime_result'] == 'H')]) + \
                           len(all_h2h[(all_h2h['away'] == away_team) & (all_h2h['fulltime_result'] == 'A')])
                
                draws = len(all_h2h[all_h2h['fulltime_result'] == 'D'])
                
                df.loc[idx, 'h2h_home_wins'] = home_wins
                df.loc[idx, 'h2h_draws'] = draws
                df.loc[idx, 'h2h_away_wins'] = away_wins
            else:
                df.loc[idx, 'h2h_home_wins'] = 0
                df.loc[idx, 'h2h_draws'] = 0
                df.loc[idx, 'h2h_away_wins'] = 0
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit feature engineering pipeline and transform data
        
        Args:
            df: Match dataframe with required columns
            
        Returns:
            (feature_matrix, feature_names)
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Initialize Elo ratings
        all_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
        self.team_elos = self._initialize_elo_ratings(all_teams)
        
        # Calculate pre-match Elo ratings
        df['home_elo_pre'] = np.nan
        df['away_elo_pre'] = np.nan
        
        for idx, match in df.iterrows():
            home_team, away_team = match['home'], match['away']
            
            # Store pre-match Elo ratings
            df.loc[idx, 'home_elo_pre'] = self.team_elos.get(home_team, 1500)
            df.loc[idx, 'away_elo_pre'] = self.team_elos.get(away_team, 1500)
            
            # Update Elo ratings after the match (for future matches)
            if not pd.isna(match['home_goals']):
                self._update_elo(home_team, away_team, match['home_goals'], match['away_goals'])
        
        # Calculate rolling statistics
        df = self._calculate_rolling_stats(df, window=5)
        
        # Calculate head-to-head
        df = self._calculate_head_to_head(df)
        
        # Create final feature matrix
        features = []
        feature_names = []
        
        # Basic features
        features.append(np.ones(len(df)))  # Home advantage indicator
        feature_names.append('home_advantage')
        
        # Elo difference
        elo_diff = df['home_elo_pre'] - df['away_elo_pre']
        features.append(elo_diff.fillna(0))
        feature_names.append('elo_difference')
        
        # Rolling form differences
        rolling_features = [
            ('rolling_points_5', 'points_diff_5'),
            ('rolling_goals_for_5', 'goals_for_diff_5'),
            ('rolling_goals_against_5', 'goals_against_diff_5'),
            ('rolling_shots_5', 'shots_diff_5'),
            ('rolling_sot_5', 'sot_diff_5'),
            ('rolling_corners_5', 'corners_diff_5')
        ]
        
        for base_name, feat_name in rolling_features:
            home_col = f'home_{base_name}'
            away_col = f'away_{base_name}'
            if home_col in df.columns and away_col in df.columns:
                diff = df[home_col] - df[away_col]
                features.append(diff.fillna(0))
                feature_names.append(feat_name)
        
        # Market signal features (using pre-closing odds)
        if 'feature_prob_H' in df.columns:
            # Pre-closing probabilities as features (available at betting time)
            features.append(df['feature_prob_H'].fillna(0.33))
            feature_names.append('market_prob_home')
            
            features.append(df['feature_prob_D'].fillna(0.33))
            feature_names.append('market_prob_draw')
            
            features.append(df['feature_prob_A'].fillna(0.33))
            feature_names.append('market_prob_away')
            
            # Market logits (for stacking approaches)
            features.append(np.log(df['feature_prob_H'].fillna(0.33) / (1 - df['feature_prob_H'].fillna(0.33))))
            feature_names.append('market_logit_home')
            
        # Market movement features (when both pre and closing available)
        if 'market_move_H' in df.columns:
            features.append(df['market_move_H'].fillna(0))
            feature_names.append('market_move_home')
            
            features.append(df['market_move_D'].fillna(0))
            feature_names.append('market_move_draw')
            
            features.append(df['market_move_A'].fillna(0))
            feature_names.append('market_move_away')
            
            # Market movement magnitude (strong signal when > 0.02)
            market_move_magnitude = np.abs(df[['market_move_H', 'market_move_D', 'market_move_A']].fillna(0)).max(axis=1)
            features.append(market_move_magnitude)
            feature_names.append('market_move_magnitude')
        
        # Head-to-head features
        h2h_features = ['h2h_home_wins', 'h2h_draws', 'h2h_away_wins']
        for feat in h2h_features:
            if feat in df.columns:
                features.append(df[feat].fillna(0))
                feature_names.append(feat)
        
        # Stack all features
        X = np.column_stack(features)
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = feature_names
        self.is_fitted = True
        
        return X_scaled, feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
        
        # This is a simplified version - in practice you'd need to 
        # handle the rolling calculations properly for new data
        # For now, return placeholder
        return np.zeros((len(df), len(self.feature_names)))

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

class FootballLogisticRegression:
    """
    Logistic regression baseline for football predictions
    """
    
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.binary_model = None
        self.multiclass_model = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y_binary: np.ndarray, y_multiclass: np.ndarray):
        """
        Fit both binary and multiclass models
        
        Args:
            X: Feature matrix
            y_binary: Binary target (home win vs not)
            y_multiclass: Multiclass target (0=away, 1=draw, 2=home)
        """
        # Binary model (home win vs not)
        self.binary_model = LogisticRegression(
            C=self.C, 
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Use calibrated classifier for better probabilities
        self.binary_model = CalibratedClassifierCV(
            self.binary_model, 
            method='isotonic', 
            cv=3
        )
        self.binary_model.fit(X, y_binary)
        
        # Multiclass model (H/D/A)
        self.multiclass_model = LogisticRegression(
            C=self.C,
            random_state=self.random_state,
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000
        )
        
        self.multiclass_model = CalibratedClassifierCV(
            self.multiclass_model,
            method='isotonic',
            cv=3
        )
        self.multiclass_model.fit(X, y_multiclass)
        
        self.is_fitted = True
    
    def predict_proba_binary(self, X: np.ndarray) -> np.ndarray:
        """Predict home win probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.binary_model.predict_proba(X)[:, 1]
    
    def predict_proba_multiclass(self, X: np.ndarray) -> np.ndarray:
        """Predict H/D/A probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.multiclass_model.predict_proba(X)

# Example usage
if __name__ == "__main__":
    # This would be used with your actual data
    print("Feature engineering and logistic regression baseline ready!")
    print("Use with your Premier League dataset to create the first baseline model.")