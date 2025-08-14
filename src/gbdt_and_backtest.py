import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FootballGBDT:
    """
    Gradient Boosted Decision Trees for football prediction
    with probability calibration
    """
    
    def __init__(self, 
                 learning_rate: float = 0.05,
                 max_depth: int = 6,
                 n_estimators: int = 1000,
                 min_data_in_leaf: int = 100,
                 random_state: int = 42):
        
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state = random_state
        
        self.binary_model = None
        self.multiclass_model = None
        self.binary_calibrator = None
        self.multiclass_calibrator = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y_binary: np.ndarray, y_multiclass: np.ndarray,
            X_val: Optional[np.ndarray] = None, 
            y_binary_val: Optional[np.ndarray] = None,
            y_multiclass_val: Optional[np.ndarray] = None):
        """
        Fit GBDT models with early stopping
        
        Args:
            X: Training features
            y_binary: Binary target (home win vs not)
            y_multiclass: Multiclass target (0=away, 1=draw, 2=home)
            X_val, y_binary_val, y_multiclass_val: Validation data for early stopping
        """
        
        # Configure LightGBM parameters
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_data_in_leaf': self.min_data_in_leaf,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # Prepare datasets
        train_data_binary = lgb.Dataset(X, label=y_binary)
        train_data_multi = lgb.Dataset(X, label=y_multiclass)
        
        valid_sets_binary = [train_data_binary]
        valid_sets_multi = [train_data_multi]
        
        if X_val is not None:
            valid_data_binary = lgb.Dataset(X_val, label=y_binary_val)
            valid_data_multi = lgb.Dataset(X_val, label=y_multiclass_val)
            valid_sets_binary.append(valid_data_binary)
            valid_sets_multi.append(valid_data_multi)
        
        # Train binary model
        print("Training binary model (home win vs not)...")
        self.binary_model = lgb.train(
            params,
            train_data_binary,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_binary,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Train multiclass model
        print("Training multiclass model (H/D/A)...")
        params_multi = params.copy()
        params_multi['objective'] = 'multiclass'
        params_multi['num_class'] = 3
        
        self.multiclass_model = lgb.train(
            params_multi,
            train_data_multi,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets_multi,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Calibrate probabilities
        print("Calibrating probabilities...")
        binary_probs = self.binary_model.predict(X)
        multi_probs = self.multiclass_model.predict(X)
        
        # Create calibrators
        from sklearn.linear_model import LogisticRegression
        
        self.binary_calibrator = LogisticRegression()
        self.binary_calibrator.fit(binary_probs.reshape(-1, 1), y_binary)
        
        self.multiclass_calibrator = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.multiclass_calibrator.fit(multi_probs, y_multiclass)
        
        self.is_fitted = True
    
    def predict_proba_binary(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Predict home win probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        raw_probs = self.binary_model.predict(X)
        
        if calibrated and self.binary_calibrator is not None:
            calibrated_probs = self.binary_calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            return calibrated_probs
        else:
            return raw_probs
    
    def predict_proba_multiclass(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Predict H/D/A probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        raw_probs = self.multiclass_model.predict(X)
        
        if calibrated and self.multiclass_calibrator is not None:
            calibrated_probs = self.multiclass_calibrator.predict_proba(raw_probs)
            return calibrated_probs
        else:
            return raw_probs
    
    def plot_reliability_curve(self, X: np.ndarray, y_true: np.ndarray, task: str = 'binary'):
        """Plot calibration reliability curve"""
        if task == 'binary':
            y_prob = self.predict_proba_binary(X, calibrated=False)
            y_prob_cal = self.predict_proba_binary(X, calibrated=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Raw probabilities
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            ax1.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title("Calibration Plot (Raw)")
            ax1.legend()
            
            # Calibrated probabilities
            fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                y_true, y_prob_cal, n_bins=10
            )
            ax2.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrated")
            ax2.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
            ax2.set_xlabel("Mean Predicted Probability")
            ax2.set_ylabel("Fraction of Positives")
            ax2.set_title("Calibration Plot (Calibrated)")
            ax2.legend()
            
            plt.tight_layout()
            plt.show()

class FootballBacktester:
    """
    Simple backtester for football betting strategies
    """
    
    def __init__(self, initial_bankroll: float = 1000):
        self.initial_bankroll = initial_bankroll
        self.results = []
    
    def kelly_fraction(self, prob: float, odds: float, fraction: float = 0.5) -> float:
        """
        Calculate Kelly fraction for betting
        
        Args:
            prob: Model probability
            odds: Bookmaker odds
            fraction: Fraction of full Kelly (0.5 = half Kelly)
        """
        if prob <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f* = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        p = prob
        q = 1 - p
        
        kelly_full = (b * p - q) / b
        return max(0, kelly_full * fraction)
    
    def calculate_edge(self, model_prob: float, book_odds: float) -> float:
        """Calculate expected value edge"""
        if model_prob <= 0 or book_odds <= 1:
            return -1
        
        return (book_odds * model_prob) - 1
    
    def backtest_strategy(self, 
                         predictions_df: pd.DataFrame,
                         ev_threshold: float = 0.02,
                         kelly_fraction: float = 0.5,
                         max_bet_fraction: float = 0.05) -> Dict:
        """
        Run backtest simulation
        
        Args:
            predictions_df: DataFrame with columns:
                - date, home, away, result (H/D/A)
                - model_prob_H/D/A, book_odds_H/D/A
            ev_threshold: Minimum edge to place bet
            kelly_fraction: Fraction of Kelly to bet
            max_bet_fraction: Maximum fraction of bankroll per bet
        """
        
        df = predictions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        total_staked = 0
        n_bets = 0
        winning_bets = 0
        
        bet_log = []
        
        for idx, match in df.iterrows():
            result = match['result']
            
            # Check each outcome for betting opportunities
            outcomes = ['H', 'D', 'A']
            for outcome in outcomes:
                model_prob = match[f'model_prob_{outcome}']
                book_odds = match[f'book_odds_{outcome}']
                
                # Calculate edge
                edge = self.calculate_edge(model_prob, book_odds)
                
                if edge > ev_threshold:
                    # Calculate stake using Kelly
                    kelly_stake = self.kelly_fraction(model_prob, book_odds, kelly_fraction)
                    stake = min(
                        kelly_stake * bankroll,
                        max_bet_fraction * bankroll,
                        bankroll * 0.1  # Never bet more than 10%
                    )
                    
                    if stake > 1:  # Minimum bet size
                        # Place bet
                        is_winner = (result == outcome)
                        payout = stake * book_odds if is_winner else 0
                        profit = payout - stake
                        
                        bankroll += profit
                        total_staked += stake
                        n_bets += 1
                        
                        if is_winner:
                            winning_bets += 1
                        
                        bet_log.append({
                            'date': match['date'],
                            'match': f"{match['home']} vs {match['away']}",
                            'outcome': outcome,
                            'stake': stake,
                            'odds': book_odds,
                            'model_prob': model_prob,
                            'edge': edge,
                            'result': result,
                            'won': is_winner,
                            'profit': profit,
                            'bankroll': bankroll
                        })
            
            bankroll_history.append(bankroll)
        
        # Calculate statistics
        final_bankroll = bankroll
        total_return = final_bankroll - self.initial_bankroll
        roi = (total_return / self.initial_bankroll) * 100
        
        win_rate = (winning_bets / n_bets) * 100 if n_bets > 0 else 0
        turnover = total_staked / self.initial_bankroll if n_bets > 0 else 0
        
        # Calculate maximum drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for br in bankroll_history:
            if br > peak:
                peak = br
            dd = (peak - br) / peak
            if dd > max_dd:
                max_dd = dd
        
        results = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': final_bankroll,
            'total_return': total_return,
            'roi_percent': roi,
            'n_bets': n_bets,
            'winning_bets': winning_bets,
            'win_rate_percent': win_rate,
            'total_staked': total_staked,
            'turnover': turnover,
            'max_drawdown_percent': max_dd * 100,
            'bankroll_history': bankroll_history,
            'bet_log': bet_log
        }
        
        return results
    
    def plot_equity_curve(self, results: Dict):
        """Plot bankroll evolution over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(results['bankroll_history'])
        plt.title('Bankroll Evolution')
        plt.xlabel('Matches')
        plt.ylabel('Bankroll')
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', alpha=0.5, label='Initial')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print backtest summary"""
        print("=" * 50)
        print("BACKTEST SUMMARY")
        print("=" * 50)
        print(f"Initial Bankroll: £{results['initial_bankroll']:.2f}")
        print(f"Final Bankroll: £{results['final_bankroll']:.2f}")
        print(f"Total Return: £{results['total_return']:.2f}")
        print(f"ROI: {results['roi_percent']:.2f}%")
        print(f"Number of Bets: {results['n_bets']}")
        print(f"Winning Bets: {results['winning_bets']}")
        print(f"Win Rate: {results['win_rate_percent']:.1f}%")
        print(f"Turnover: {results['turnover']:.2f}x")
        print(f"Max Drawdown: {results['max_drawdown_percent']:.2f}%")
        print("=" * 50)

# Complete training pipeline
def train_week1_models(df: pd.DataFrame) -> Dict:
    """
    Complete Week 1 training pipeline
    
    Args:
        df: Premier League match data
        
    Returns:
        Dictionary containing all trained models and results
    """
    from time_series_cv import TimeSeriesCV, prepare_target_variables, calculate_metrics
    from feature_engineering import FootballFeatureEngineer
    from dixon_coles_poisson import DixonColesPoisson
    
    print("Starting Week 1 model training pipeline...")
    
    # Prepare data
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Prepare targets
    y_binary, y_multiclass = prepare_target_variables(df)
    
    # Initialize models
    dc_model = DixonColesPoisson()
    fe = FootballFeatureEngineer()
    lr_model = FootballLogisticRegression()
    gbdt_model = FootballGBDT()
    
    # Time series CV
    tscv = TimeSeriesCV(min_train_seasons=3)
    results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\nFold {fold_idx + 1}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Train Dixon-Coles
        print("Training Dixon-Coles...")
        dc_model.fit(train_df)
        dc_probs = dc_model.predict_proba(val_df[['home', 'away']])
        
        # Feature engineering and logistic regression
        print("Training Logistic Regression...")
        X_train, feature_names = fe.fit_transform(train_df)
        # Note: For proper implementation, you'd need to implement fe.transform() properly
        # For now, we'll skip LR in this demo
        
        # Evaluate Dixon-Coles
        dc_metrics_binary = calculate_metrics(
            y_binary[val_idx], 
            dc_probs[:, 2],  # Home win probability
            'binary'
        )
        
        dc_metrics_multi = calculate_metrics(
            y_multiclass[val_idx],
            dc_probs,
            'multiclass'
        )
        
        results.append({
            'fold': fold_idx,
            'model': 'dixon_coles',
            'task': 'binary',
            **dc_metrics_binary
        })
        
        results.append({
            'fold': fold_idx,
            'model': 'dixon_coles', 
            'task': 'multiclass',
            **dc_metrics_multi
        })
    
    print("\nWeek 1 training complete!")
    print("Next steps:")
    print("1. Implement proper feature engineering transform() method")
    print("2. Add GBDT training to the pipeline")
    print("3. Run backtesting on held-out test set")
    print("4. Create baseline report")
    
    return {
        'models': {'dixon_coles': dc_model},
        'results': results,
        'feature_names': feature_names if 'feature_names' in locals() else []
    }

if __name__ == "__main__":
    print("GBDT model, calibration, and backtesting framework ready!")
    print("Use train_week1_models() with your Premier League data to complete Week 1")