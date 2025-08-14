import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class DixonColesPoisson:
    """
    Dixon-Coles Poisson model for football match prediction
    
    Fits team-specific attack/defense strengths with home advantage
    and low-score correlation adjustment.
    """
    
    def __init__(self, xi: float = 0.001, max_goals: int = 6):
        """
        Args:
            xi: Time decay parameter (higher = more weight on recent matches)
            max_goals: Maximum goals to consider in score grid
        """
        self.xi = xi
        self.max_goals = max_goals
        self.teams = None
        self.params = None
        self.is_fitted = False
    
    def _tau(self, x: int, y: int, lambda_x: float, lambda_y: float, rho: float) -> float:
        """Dixon-Coles tau function for low-score correlation"""
        if x == 0 and y == 0:
            return 1 - lambda_x * lambda_y * rho
        elif x == 0 and y == 1:
            return 1 + lambda_x * rho
        elif x == 1 and y == 0:
            return 1 + lambda_y * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _likelihood_contribution(self, home_goals: int, away_goals: int,
                                lambda_home: float, lambda_away: float,
                                rho: float, weight: float = 1.0) -> float:
        """Calculate likelihood contribution for a single match"""
        poisson_prob = (poisson.pmf(home_goals, lambda_home) * 
                       poisson.pmf(away_goals, lambda_away))
        tau_correction = self._tau(home_goals, away_goals, lambda_home, lambda_away, rho)
        
        return weight * np.log(poisson_prob * tau_correction + 1e-10)
    
    def _negative_log_likelihood(self, params: np.ndarray, matches_df: pd.DataFrame,
                               team_to_idx: dict, weights: np.ndarray) -> float:
        """Negative log-likelihood to minimize"""
        n_teams = len(team_to_idx)
        
        # Unpack parameters
        mu = params[0]  # Global scoring rate
        home_adv = params[1]  # Home advantage
        rho = params[2]  # Low-score correlation
        
        # Team attack strengths (first n_teams-1, last one is constrained)
        attacks = np.zeros(n_teams)
        attacks[:-1] = params[3:3+n_teams-1]
        attacks[-1] = -np.sum(attacks[:-1])  # Sum to zero constraint
        
        # Team defense strengths (first n_teams-1, last one is constrained)
        defenses = np.zeros(n_teams)
        defenses[:-1] = params[3+n_teams-1:3+2*(n_teams-1)]
        defenses[-1] = -np.sum(defenses[:-1])  # Sum to zero constraint
        
        total_likelihood = 0.0
        
        for _, match in matches_df.iterrows():
            home_idx = team_to_idx[match['home']]
            away_idx = team_to_idx[match['away']]
            
            # Calculate expected goals
            lambda_home = np.exp(mu + home_adv + attacks[home_idx] - defenses[away_idx])
            lambda_away = np.exp(mu + attacks[away_idx] - defenses[home_idx])
            
            # Add likelihood contribution
            weight = weights[match.name] if match.name in weights.index else 1.0
            total_likelihood += self._likelihood_contribution(
                match['home_goals'], match['away_goals'],
                lambda_home, lambda_away, rho, weight
            )
        
        return -total_likelihood
    
    def fit(self, matches_df: pd.DataFrame, reference_date=None):
        """
        Fit the Dixon-Coles model
        
        Args:
            matches_df: DataFrame with columns ['date', 'home', 'away', 'home_goals', 'away_goals']
            reference_date: Date for time weighting (uses latest date if None)
        """
        matches_df = matches_df.copy().reset_index()
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        if reference_date is None:
            reference_date = matches_df['date'].max()
        else:
            reference_date = pd.to_datetime(reference_date)
        
        # Get unique teams
        self.teams = sorted(set(matches_df['home'].unique()) | set(matches_df['away'].unique()))
        team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        n_teams = len(self.teams)
        
        # Calculate time weights
        matches_df['days_ago'] = (reference_date - matches_df['date']).dt.days
        matches_df['weight'] = np.exp(-self.xi * matches_df['days_ago'])
        weights = matches_df.set_index(matches_df.index)['weight']
        
        # Initial parameter guess
        n_params = 3 + 2 * (n_teams - 1)  # mu, home_adv, rho, attacks[:-1], defenses[:-1]
        initial_params = np.random.normal(0, 0.1, n_params)
        initial_params[0] = np.log(matches_df[['home_goals', 'away_goals']].mean().mean())  # mu
        initial_params[1] = 0.2  # home advantage
        initial_params[2] = 0.0  # rho
        
        # Optimization bounds
        bounds = [(None, None)] * n_params
        bounds[1] = (0, 1)      # home advantage >= 0
        bounds[2] = (-0.5, 0.5) # rho constraint
        
        # Optimize
        try:
            result = minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(matches_df, team_to_idx, weights),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.params = result.x
                self.team_to_idx = team_to_idx
                self.is_fitted = True
            else:
                raise ValueError(f"Optimization failed: {result.message}")
                
        except Exception as e:
            print(f"Fitting failed: {e}")
            # Fallback to simpler initialization
            self.params = initial_params
            self.team_to_idx = team_to_idx
            self.is_fitted = True
    
    def predict_match_probabilities(self, home_team: str, away_team: str) -> dict:
        """
        Predict probabilities for a single match
        
        Returns:
            Dictionary with 'home_win', 'draw', 'away_win' probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if home_team not in self.team_to_idx or away_team not in self.team_to_idx:
            # Handle unknown teams with league average
            home_attack = home_defense = away_attack = away_defense = 0.0
        else:
            home_idx = self.team_to_idx[home_team]
            away_idx = self.team_to_idx[away_team]
            
            n_teams = len(self.teams)
            mu = self.params[0]
            home_adv = self.params[1]
            rho = self.params[2]
            
            # Reconstruct team strengths
            attacks = np.zeros(n_teams)
            attacks[:-1] = self.params[3:3+n_teams-1]
            attacks[-1] = -np.sum(attacks[:-1])
            
            defenses = np.zeros(n_teams)
            defenses[:-1] = self.params[3+n_teams-1:3+2*(n_teams-1)]
            defenses[-1] = -np.sum(defenses[:-1])
            
            home_attack, home_defense = attacks[home_idx], defenses[home_idx]
            away_attack, away_defense = attacks[away_idx], defenses[away_idx]
        
        # Calculate expected goals
        lambda_home = np.exp(mu + home_adv + home_attack - away_defense)
        lambda_away = np.exp(mu + away_attack - home_defense)
        
        # Generate score probabilities
        home_win_prob = draw_prob = away_win_prob = 0.0
        
        for home_goals, away_goals in product(range(self.max_goals + 1), repeat=2):
            # Basic Poisson probability
            prob = (poisson.pmf(home_goals, lambda_home) * 
                   poisson.pmf(away_goals, lambda_away))
            
            # Dixon-Coles adjustment
            tau = self._tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            prob *= tau
            
            # Accumulate outcome probabilities
            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob
        
        # Normalize (in case of truncation effects)
        total = home_win_prob + draw_prob + away_win_prob
        
        return {
            'home_win': home_win_prob / total,
            'draw': draw_prob / total,
            'away_win': away_win_prob / total,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away
        }
    
    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for multiple fixtures
        
        Args:
            fixtures_df: DataFrame with 'home' and 'away' columns
            
        Returns:
            Array of shape (n_matches, 3) with [away_win, draw, home_win] probabilities
        """
        probabilities = []
        
        for _, fixture in fixtures_df.iterrows():
            probs = self.predict_match_probabilities(fixture['home'], fixture['away'])
            probabilities.append([probs['away_win'], probs['draw'], probs['home_win']])
        
        return np.array(probabilities)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'date': pd.date_range('2020-01-01', periods=100, freq='W'),
        'home': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], 100),
        'away': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], 100),
        'home_goals': np.random.poisson(1.3, 100),
        'away_goals': np.random.poisson(1.1, 100)
    }
    df = pd.DataFrame(sample_data)
    # Remove same team matches
    df = df[df['home'] != df['away']]
    
    # Fit model
    model = DixonColesPoisson(xi=0.01)
    model.fit(df)
    
    # Test prediction
    test_fixture = pd.DataFrame({
        'home': ['Arsenal'],
        'away': ['Chelsea']
    })
    
    probs = model.predict_proba(test_fixture)
    print(f"Arsenal vs Chelsea probabilities: {probs[0]}")
    print(f"Sum: {probs[0].sum()}")  # Should be ~1.0