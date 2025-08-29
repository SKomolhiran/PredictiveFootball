import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from typing import List, Tuple, Iterator
from datetime import datetime
from typing import Tuple

class TimeSeriesCV:
    """Time-aware cross-validation for football predictions"""
    
    def __init__(self, min_train_seasons: int = 3, validation_seasons: int = 1):
        self.min_train_seasons = min_train_seasons
        self.validation_seasons = validation_seasons
    
    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-based train/validation splits
        
        Args:
            df: DataFrame with 'date' and 'season' columns
            
        Yields:
            (train_indices, val_indices) tuples
        """
        df = df.sort_values('date').reset_index(drop=True)
        seasons = sorted(df['season'].unique())
        
        for i in range(self.min_train_seasons, len(seasons)):
            # Define validation season(s)
            val_seasons = seasons[i:i+self.validation_seasons]
            train_seasons = seasons[:i]
            
            train_mask = df['season'].isin(train_seasons)
            val_mask = df['season'].isin(val_seasons)
            
            train_idx = df[train_mask].index.values # Array of data's index which belongs in train ex. [0,1,2,,...,299]
            val_idx = df[val_mask].index.values # Array of data's index which belongs in validate ex. [300,301,302,..., 399]
            
            if len(val_idx) > 0:
                yield train_idx, val_idx

def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                     task: str = 'binary') -> dict:
    """
    Calculate comprehensive probability-based metrics
    
    Args:
        y_true: True labels (binary: 0/1, multiclass: 0/1/2 for A/D/H) ex. [[0],[2],[1],[1],....], got from prepare_target_variables
        y_pred_proba: Predicted probabilities: probabilities of that match ex. [[0.2,0.3,0.5],[0.3,0.3,0.4],....]
        task: 'binary' or 'multiclass'
    
    Returns:
        Dictionary of metrics
    """
    if task == 'binary':
        # For binary win/not-win prediction
        logloss = log_loss(y_true, y_pred_proba)
        brier = brier_score_loss(y_true, y_pred_proba)
        
        return {
            'logloss': logloss,
            'brier': brier,
            'rps': None  # RPS only for multiclass
        }
    
    elif task == 'multiclass':
        # For H/D/A prediction
        logloss = log_loss(y_true, y_pred_proba)
        
        # Calculate RPS (Ranked Probability Score)
        rps_scores = []
        for i in range(len(y_true)):
            true_outcome = np.zeros(3) # [0,0,0]
            true_outcome[int(y_true[i])] = 1 # Set true outcome as 1, ex. [0,1,0]
            
            cumulative_pred = np.cumsum(y_pred_proba[i]) # ex. from [0.2,0.3,0.5] -> [0.2,0.5,1.0]
            cumulative_true = np.cumsum(true_outcome) # ex. from [0,1,0] -> [0,1,1]
            
            rps = np.sum((cumulative_pred - cumulative_true) ** 2) / 2
            rps_scores.append(rps)
        
        avg_rps = np.mean(rps_scores)
        
        # Average Brier score across classes
        brier_scores = []
        for class_idx in range(3):
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = y_pred_proba[:, class_idx]
            brier_scores.append(brier_score_loss(y_true_binary, y_pred_binary))
        
        avg_brier = np.mean(brier_scores)
        
        return {
            'logloss': logloss,
            'brier': avg_brier,
            'rps': avg_rps
        }
    else:
        print(f"{task} is neither binary or multiclass")
        return {}
        

def prepare_target_variables(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare target variables for both binary and multiclass tasks
    
    Args:
        df: DataFrame with 'fulltime_result' column (H/D/A)
    
    Returns:
        (binary_target, multiclass_target) - home win vs not, H/D/A encoded as 2/1/0
    """
    # Binary: home win (1) vs not home win (0)
    binary_target = (df['fulltime_result'] == 'H').astype(int) # Array with all binary result of each row ex. [1,0,0,1,0,0..]
    
    # Multiclass: Away=0, Draw=1, Home=2 (for ordered probabilities in RPS)
    multiclass_mapping = {'A': 0, 'D': 1, 'H': 2}
    multiclass_target = df['fulltime_result'].map(multiclass_mapping).astype(int) # Array with all multiclass result of each row ex. [2,1,0,1,0,0]
    
    return (np.array(binary_target), np.array(multiclass_target))

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/processed/matches.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Prepare targets
    y_binary, y_multiclass = prepare_target_variables(df)
    
    # Initialize CV
    tscv = TimeSeriesCV(min_train_seasons=3, validation_seasons=1)
    
    # Create results tracking
    results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"Fold {fold_idx}")
        print(f"Train seasons: {sorted(df.iloc[train_idx]['season'].unique())}")
        print(f"Val seasons: {sorted(df.iloc[val_idx]['season'].unique())}")
        print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        print("-" * 50)
        
        # Here you would train your models and collect predictions
        # results.append({
        #     'fold': fold_idx,
        #     'model': 'baseline',
        #     'task': 'binary',
        #     'logloss': ...,
        #     'brier': ...,
        #     'rps': None
        # })