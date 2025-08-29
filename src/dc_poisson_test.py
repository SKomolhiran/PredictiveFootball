# test_dixon_coles.py
from dixon_coles_poisson_optimized import DixonColesPoissonOptimized
from time_series_cv import TimeSeriesCV, calculate_metrics, prepare_target_variables
import pandas as pd
import numpy as np
from datetime import datetime

def effective_sample_size(weights: np.ndarray) -> float:
    s1 = weights.sum()
    s2 = (weights**2).sum()
    return (s1*s1) / s2 if s2 > 0 else 0.0

def compute_train_weights(train_df: pd.DataFrame, xi: float) -> np.ndarray:
    ref = train_df['date'].max()
    days_ago = (ref - train_df['date']).dt.days.to_numpy()
    return np.exp(-xi * days_ago)

def uniform_baseline(n: int) -> np.ndarray:
    return np.tile(np.array([1/3, 1/3, 1/3]), (n, 1))

def frequency_baseline(train_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> np.ndarray:
    # Compute global outcome frequencies on TRAIN ONLY
    # H/D/A mapping must match your metric order: [away, draw, home]
    home_wins = (train_df['home_goals'] > train_df['away_goals']).mean()
    draws     = (train_df['home_goals'] == train_df['away_goals']).mean()
    away_wins = 1.0 - home_wins - draws
    p = np.array([away_wins, draws, home_wins])
    return np.tile(p, (len(fixtures_df), 1))


if __name__ == "__main__":
    # Load verified data
    df = pd.read_csv('../data/processed/matches.csv')
    df['date'] = pd.to_datetime(df['date'])

    # # Limit only 4 seasons of data first
    # included_seasons = ['2024-25','2023-24','2022-23','2021-22','2020-21']
    # df = df[df['season'].isin(included_seasons)]

    # Prepare targets
    y_binary, y_multiclass = prepare_target_variables(df)

    # Initialize CV
    tscv = TimeSeriesCV(min_train_seasons=3, validation_seasons=1)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        # Train Dixon-Coles on historical data
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        current_time_3 = datetime.now()
        print(f"Before training for fold {fold_idx} time: {current_time_3}")
        model = DixonColesPoissonOptimized(xi=0.002)
        model.fit(train_df)  # Only uses historical results

        current_time_4 = datetime.now()
        print(f"After training for fold {fold_idx} time: {current_time_4}")

        # 1) Convergence
        res = getattr(model, 'last_result_', None)
        if res is not None:
            print(f"Convergence: success={res.success}, nit={res.nit}, nfev={res.nfev}, fun(NLL)={res.fun:.3f}")
            if not res.success:
                print("Message:", res.message)

        # 2) Effective sample size under decay
        w = compute_train_weights(train_df, xi=0.002)
        ess = effective_sample_size(w)
        print(f"Train N={len(train_df)}, ESS≈{ess:.1f} (xi=0.002)")

        # 3) Parameter snapshot
        theta = model.params
        print(f"Params: mu={theta[0]:.3f}, home_adv={theta[1]:.3f}, rho={theta[2]:.6f}, ||theta||={np.linalg.norm(theta):.6f}")


        # Predict on validation fixtures
        val_proba = model.predict_proba(val_df[['home', 'away']])

        current_time_5 = datetime.now()
        print(f"Predicted prob for fold {fold_idx} time: {current_time_5}")

        # Get optimization stats
        stats = model.get_performance_stats()
        print(f"Cache hit rate: {stats['prediction_cache_hit_rate']:.1%}")

        # Baselines
        u_proba = uniform_baseline(len(val_df))
        f_proba = frequency_baseline(train_df, val_df)

        # Compute metrics (reuse your calculate_metrics)
        m_model = calculate_metrics(y_multiclass[val_idx], val_proba, task='multiclass')
        m_uni   = calculate_metrics(y_multiclass[val_idx], u_proba, task='multiclass')
        m_freq  = calculate_metrics(y_multiclass[val_idx], f_proba, task='multiclass')

        print(f"Fold {fold_idx} — Model:    LogLoss={m_model['logloss']:.6f}, RPS={m_model['rps']:.6f}")
        print(f"Fold {fold_idx} — Uniform:  LogLoss={m_uni['logloss']:.6f}, RPS={m_uni['rps']:.6f}")
        print(f"Fold {fold_idx} — Freq:     LogLoss={m_freq['logloss']:.6f}, RPS={m_freq['rps']:.6f}")
        print(f"Uplift vs uniform: ΔLogLoss={(m_uni['logloss']-m_model['logloss']):.6f}, ΔRPS={(m_uni['rps']-m_model['rps']):.6f}")
        print(f"Uplift vs freq:    ΔLogLoss={(m_freq['logloss']-m_model['logloss']):.6f}, ΔRPS={(m_freq['rps']-m_model['rps']):.6f}")


        # Calculate metrics
        metrics = calculate_metrics(y_multiclass[val_idx], val_proba, task='multiclass')

        current_time_6 = datetime.now()
        print(f"Calculate metrics for fold {fold_idx} time: {current_time_6}")

        results.append({
            'fold': fold_idx,
            'model': 'dixon_coles',
            **metrics
        })

        print(f"Fold {fold_idx}: LogLoss={metrics['logloss']:.6f}, RPS={metrics['rps']:.6f}")

    # Average performance
    avg_metrics = {
        'logloss': np.mean([r['logloss'] for r in results]),
        'rps': np.mean([r['rps'] for r in results])
    }
    print(f"\nAverage: LogLoss={avg_metrics['logloss']:.6f}, RPS={avg_metrics['rps']:.6f}")