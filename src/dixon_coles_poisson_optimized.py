import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import logsumexp
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

class DixonColesPoissonOptimized:
    """
    Optimized Dixon-Coles Poisson model for football match prediction
    
    Performance optimizations:
    1. Pre-computed numpy arrays (no pandas in objective)
    2. Vectorized likelihood computation
    3. Vectorized score grid predictions
    4. LRU caching for repeated predictions
    5. Numerical stability improvements
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
        
        # Pre-computed optimization arrays
        self._team_to_idx = None
        self._n_teams = 0
        
        # Pre-computed score grids for vectorized prediction
        self._score_grid_home = None
        self._score_grid_away = None
        self._score_combinations = None
        
        # Cache for team strength calculations
        self._team_strengths_cache = None
        
    def _precompute_score_grid(self):
        """Pre-compute score combinations for vectorized prediction"""
        home_goals = np.arange(self.max_goals + 1)
        away_goals = np.arange(self.max_goals + 1)
        
        # Create meshgrid for vectorized operations
        self._score_grid_home, self._score_grid_away = np.meshgrid(home_goals, away_goals, indexing='ij')
        
        # Flatten for easier computation
        self._score_combinations = self._score_grid_home.shape
        
    def _tau_vectorized(self, home_goals: np.ndarray, away_goals: np.ndarray, 
                       lambda_home: float, lambda_away: float, rho: float) -> np.ndarray:
        """Vectorized Dixon-Coles tau function for low-score correlation"""
        # Initialize with ones
        tau = np.ones_like(home_goals, dtype=float)
        
        # Apply adjustments for specific score combinations
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        
        tau[mask_00] = 1 - lambda_home * lambda_away * rho
        tau[mask_01] = 1 + lambda_home * rho
        tau[mask_10] = 1 + lambda_away * rho
        tau[mask_11] = 1 - rho
        
        return tau
    
    def _tau_scalar(self, x: int, y: int, lambda_x: float, lambda_y: float, rho: float) -> float:
        """Dixon-Coles tau function for single match (for likelihood computation)"""
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

    def _negative_log_likelihood_vectorized(self, params: np.ndarray, 
                                          home_indices: np.ndarray, away_indices: np.ndarray,
                                          home_goals: np.ndarray, away_goals: np.ndarray,
                                          weights: np.ndarray) -> float:
        """
        Vectorized negative log-likelihood computation
        
        Args:
            params: Model parameters
            home_indices: Array of home team indices
            away_indices: Array of away team indices
            home_goals: Array of home goals scored
            away_goals: Array of away goals scored
            weights: Array of time-based weights
        """
        n_teams = self._n_teams
        
        # Unpack parameters
        mu = params[0]
        home_adv = params[1]
        rho = params[2]
        
        # Reconstruct team strengths with constraints
        attacks = np.zeros(n_teams)
        attacks[:-1] = params[3:3+n_teams-1]
        attacks[-1] = -np.sum(attacks[:-1])  # Sum to zero constraint
        
        defenses = np.zeros(n_teams)
        defenses[:-1] = params[3+n_teams-1:3+2*(n_teams-1)]
        defenses[-1] = -np.sum(defenses[:-1])  # Sum to zero constraint
        
        # Vectorized lambda calculations
        lambda_home = np.exp(mu + home_adv + attacks[home_indices] - defenses[away_indices])
        lambda_away = np.exp(mu + attacks[away_indices] - defenses[home_indices])
        
        # Vectorized Poisson probabilities using gammaln for log(factorial)
        from scipy.special import gammaln
        log_poisson_home = home_goals * np.log(lambda_home) - lambda_home - gammaln(home_goals + 1)
        log_poisson_away = away_goals * np.log(lambda_away) - lambda_away - gammaln(away_goals + 1)
        
        # Vectorized tau corrections (apply element-wise)
        tau_corrections = np.array([
            self._tau_scalar(int(h), int(a), lh, la, rho) 
            for h, a, lh, la in zip(home_goals, away_goals, lambda_home, lambda_away)
        ])
        
        # Total log-likelihood with weights
        log_likelihood_contributions = weights * (log_poisson_home + log_poisson_away + np.log(np.maximum(tau_corrections, 1e-10)))
        
        return -np.sum(log_likelihood_contributions)
    
    def fit(self, matches_df: pd.DataFrame, reference_date=None):
        """
        Optimized fit using pre-computed arrays and vectorized operations
        """
        print(f"🚀 Training Optimized Dixon-Coles on {len(matches_df)} matches...")
        
        # Data preparation (same as before)
        training_matches = matches_df.copy().reset_index(drop=True)
        training_matches['date'] = pd.to_datetime(training_matches['date'])
        
        missing_results = training_matches[['home_goals', 'away_goals']].isna().any(axis=1)
        if missing_results.any():
            print(f"⚠️  Removing {missing_results.sum()} matches without results")
            training_matches = training_matches[~missing_results].copy()
        
        if len(training_matches) == 0:
            raise ValueError("No historical matches with results found!")
        
        # Time weighting
        if reference_date is None:
            reference_date = training_matches['date'].max()
        else:
            reference_date = pd.to_datetime(reference_date)
        
        training_matches['days_ago'] = (reference_date - training_matches['date']).dt.days
        training_matches['weight'] = np.exp(-self.xi * training_matches['days_ago'])
        
        # OPTIMIZATION 1: Pre-compute all arrays (no pandas in objective)
        self.teams = sorted(set(training_matches['home'].unique()) | set(training_matches['away'].unique()))
        self._team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self._n_teams = len(self.teams)
        
        # Convert to numpy arrays once
        home_indices = training_matches['home'].map(self._team_to_idx).values
        away_indices = training_matches['away'].map(self._team_to_idx).values
        home_goals = training_matches['home_goals'].values
        away_goals = training_matches['away_goals'].values
        weights = training_matches['weight'].values
        
        print(f"✅ Pre-computed arrays: {len(home_indices)} matches, {self._n_teams} teams")
        
        # OPTIMIZATION 2: Pre-compute score grid for predictions
        self._precompute_score_grid()
        
        # Initial parameters
        n_params = 3 + 2 * (self._n_teams - 1)
        initial_params = np.random.normal(0, 0.1, n_params)
        initial_params[0] = np.log(np.mean([home_goals.mean(), away_goals.mean()]))
        initial_params[1] = 0.2  # home advantage
        initial_params[2] = 0.0  # rho
        
        # Bounds
        bounds = [(None, None)] * n_params
        bounds[1] = (0, 1)      # home advantage >= 0
        bounds[2] = (-0.5, 0.5) # rho constraint
        
        print("🔧 Optimizing with vectorized objective function...")
        
        # OPTIMIZATION 3: Use vectorized objective function
        try:
            result = minimize(
                self._negative_log_likelihood_vectorized,
                initial_params,
                args=(home_indices, away_indices, home_goals, away_goals, weights),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.params = result.x
                self.team_to_idx = self._team_to_idx
                self.is_fitted = True
                print(f"✅ Optimization converged in {result.nit} iterations")
            else:
                raise ValueError(f"Optimization failed: {result.message}")
                
        except Exception as e:
            print(f"Fitting failed: {e}")
            self.params = initial_params
            self.team_to_idx = self._team_to_idx
            self.is_fitted = True
        
        # OPTIMIZATION 4: Cache team strengths for repeated predictions
        self._cache_team_strengths()
    
    def _cache_team_strengths(self):
        """Cache computed team strengths to avoid recomputation"""
        if not self.is_fitted:
            return
            
        n_teams = self._n_teams
        mu = self.params[0]
        home_adv = self.params[1]
        rho = self.params[2]
        
        # Reconstruct and cache team strengths
        attacks = np.zeros(n_teams)
        attacks[:-1] = self.params[3:3+n_teams-1]
        attacks[-1] = -np.sum(attacks[:-1])
        
        defenses = np.zeros(n_teams)
        defenses[:-1] = self.params[3+n_teams-1:3+2*(n_teams-1)]
        defenses[-1] = -np.sum(defenses[:-1])
        
        self._team_strengths_cache = {
            'mu': mu,
            'home_adv': home_adv,
            'rho': rho,
            'attacks': attacks,
            'defenses': defenses
        }
    
    @lru_cache(maxsize=1000)
    def _cached_match_prediction(self, home_team: str, away_team: str) -> tuple:
        """LRU cached prediction for repeated team combinations"""
        return self._predict_single_match_optimized(home_team, away_team)
    
    def _predict_single_match_optimized(self, home_team: str, away_team: str) -> tuple:
        """Optimized single match prediction with vectorized score grid"""
        
        # Get cached team strengths
        cache = self._team_strengths_cache
        mu, home_adv, rho = cache['mu'], cache['home_adv'], cache['rho']
        attacks, defenses = cache['attacks'], cache['defenses']
        
        if home_team not in self._team_to_idx or away_team not in self._team_to_idx:
            # Handle unknown teams with league average
            home_attack = home_defense = away_attack = away_defense = 0.0
        else:
            home_idx = self._team_to_idx[home_team]
            away_idx = self._team_to_idx[away_team]
            home_attack, home_defense = attacks[home_idx], defenses[home_idx]
            away_attack, away_defense = attacks[away_idx], defenses[away_idx]
        
        # Calculate expected goals
        lambda_home = np.exp(mu + home_adv + home_attack - away_defense)
        lambda_away = np.exp(mu + away_attack - home_defense)
        
        # OPTIMIZATION 5: Vectorized score grid computation
        home_goals_grid = self._score_grid_home
        away_goals_grid = self._score_grid_away
        
        # Vectorized Poisson probabilities
        from scipy.special import gammaln
        log_prob_home = (home_goals_grid * np.log(lambda_home) - lambda_home - 
                        gammaln(home_goals_grid + 1))
        log_prob_away = (away_goals_grid * np.log(lambda_away) - lambda_away - 
                        gammaln(away_goals_grid + 1))
        
        # Combined probabilities
        log_prob_combined = log_prob_home + log_prob_away
        
        # Vectorized tau corrections
        tau_corrections = self._tau_vectorized(home_goals_grid, away_goals_grid, 
                                             lambda_home, lambda_away, rho)
        
        # Final probabilities
        prob_grid = np.exp(log_prob_combined) * tau_corrections
        
        # Calculate outcome probabilities using vectorized operations
        home_win_mask = home_goals_grid > away_goals_grid
        draw_mask = home_goals_grid == away_goals_grid
        away_win_mask = home_goals_grid < away_goals_grid
        
        home_win_prob = np.sum(prob_grid[home_win_mask])
        draw_prob = np.sum(prob_grid[draw_mask])
        away_win_prob = np.sum(prob_grid[away_win_mask])
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        
        return (home_win_prob/total, draw_prob/total, away_win_prob/total, 
                lambda_home, lambda_away)
    
    def predict_match_probabilities(self, home_team: str, away_team: str) -> dict:
        """
        Predict probabilities for a single match (with caching)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use cached prediction
        home_win, draw, away_win, lambda_home, lambda_away = self._cached_match_prediction(home_team, away_team)
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away
        }
    
    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        """
        Optimized batch prediction for multiple fixtures
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_matches = len(fixtures_df)
        probabilities = np.zeros((n_matches, 3))
        
        # Process in batches for better cache utilization
        for i, (_, fixture) in enumerate(fixtures_df.iterrows()):
            probs = self.predict_match_probabilities(fixture['home'], fixture['away'])
            probabilities[i] = [probs['away_win'], probs['draw'], probs['home_win']]
        
        return probabilities
    
    def get_performance_stats(self) -> dict:
        """Get optimization performance statistics"""
        cache_info = self._cached_match_prediction.cache_info()
        
        return {
            'teams_cached': self._n_teams,
            'score_grid_size': self._score_combinations,
            'prediction_cache_hits': cache_info.hits,
            'prediction_cache_misses': cache_info.misses,
            'prediction_cache_hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
        }

# Performance comparison function
def performance_comparison():
    """Compare optimized vs original implementation"""
    import time
    from dixon_coles_poisson import DixonColesPoisson
    
    # Create sample data
    print("🔬 PERFORMANCE COMPARISON")
    print("=" * 40)
    
    np.random.seed(42)
    sample_data = {
        'date': pd.date_range('2020-01-01', periods=1000, freq='3D'),
        'home': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Tottenham', 'Man United'], 1000),
        'away': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Tottenham', 'Man United'], 1000),
        'home_goals': np.random.poisson(1.3, 1000),
        'away_goals': np.random.poisson(1.1, 1000)
    }
    df = pd.DataFrame(sample_data)
    df = df[df['home'] != df['away']]  # Remove same team matches
    
    # Test fixtures
    test_fixtures = pd.DataFrame({
        'home': ['Arsenal', 'Chelsea', 'Liverpool'] * 100,
        'away': ['Chelsea', 'Liverpool', 'Arsenal'] * 100
    })
    
    print(f"Training data: {len(df)} matches")
    print(f"Test fixtures: {len(test_fixtures)} predictions")
    
    # Original implementation
    print("\n📊 Original Implementation:")
    original_model = DixonColesPoisson(xi=0.01, max_goals=4)
    
    start_time = time.time()
    original_model.fit(df)
    fit_time_original = time.time() - start_time
    
    start_time = time.time()
    original_probs = original_model.predict_proba(test_fixtures)
    predict_time_original = time.time() - start_time
    
    print(f"Fit time: {fit_time_original:.3f}s")
    print(f"Predict time: {predict_time_original:.3f}s")
    
    # Optimized implementation
    print("\n🚀 Optimized Implementation:")
    optimized_model = DixonColesPoissonOptimized(xi=0.01, max_goals=4)
    
    start_time = time.time()
    optimized_model.fit(df)
    fit_time_optimized = time.time() - start_time
    
    start_time = time.time()
    optimized_probs = optimized_model.predict_proba(test_fixtures)
    predict_time_optimized = time.time() - start_time
    
    print(f"Fit time: {fit_time_optimized:.3f}s")
    print(f"Predict time: {predict_time_optimized:.3f}s")
    
    # Performance gains
    print(f"\n⚡ PERFORMANCE GAINS:")
    print(f"Fit speedup: {fit_time_original/fit_time_optimized:.2f}x faster")
    print(f"Predict speedup: {predict_time_original/predict_time_optimized:.2f}x faster")
    print(f"Total speedup: {(fit_time_original + predict_time_original)/(fit_time_optimized + predict_time_optimized):.2f}x faster")
    
    # Accuracy comparison
    print(f"\n🎯 ACCURACY COMPARISON:")
    max_diff = np.max(np.abs(original_probs - optimized_probs))
    mean_diff = np.mean(np.abs(original_probs - optimized_probs))
    print(f"Max probability difference: {max_diff:.6f}")
    print(f"Mean probability difference: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print("✅ Results are numerically identical")
    else:
        print("⚠️  Small numerical differences detected")
    
    # Cache statistics
    stats = optimized_model.get_performance_stats()
    print(f"\n📈 OPTIMIZATION STATISTICS:")
    print(f"Teams cached: {stats['teams_cached']}")
    print(f"Score grid size: {stats['score_grid_size']}")
    print(f"Cache hit rate: {stats['prediction_cache_hit_rate']:.1%}")

if __name__ == "__main__":
    performance_comparison()