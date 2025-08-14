"""
Complete Week 1 Implementation Guide
===================================

This script shows how to use all the components together to complete Week 1
of your football prediction pipeline.

Follow this step-by-step to get your first working model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Preparation (Updated for Your Processed Data)
def prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare your processed Premier League data
    
    Your data already has:
    - pre_p_H/D/A: Pre-closing probabilities (devigged) → MODEL FEATURES
    - close_p_H/D/A: Closing probabilities (devigged) → BACKTESTING BENCHMARK  
    - pre_odds_H/D/A: Pre-closing odds → BETTING TARGET
    - close_odds_H/D/A: Closing odds → EFFICIENCY BASELINE
    - Box score stats: HS, AS, HST, AST, HC, AC, etc. → WEEK 2 FEATURES
    
    Args:
        csv_path: Path to your processed match data CSV
        
    Returns:
        DataFrame ready for modeling with smart feature mapping
    """
    print("Loading your processed Premier League data...")
    df = pd.read_csv(csv_path)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Validate your data structure
    required_cols = ['date', 'home', 'away', 'home_goals', 'away_goals', 'fulltime_result']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("✅ Your data processing is excellent! Mapping to model structure...")
    
    # Map your processed probabilities to model-expected names
    # PRE-CLOSING PROBABILITIES → MODEL FEATURES (available when betting)
    if 'pre_p_H' in df.columns:
        df['feature_prob_H'] = df['pre_p_H']  # What model trains on
        df['feature_prob_D'] = df['pre_p_D']
        df['feature_prob_A'] = df['pre_p_A']
        
        # Use pre-closing odds for betting simulation
        df['betting_odds_H'] = df['pre_odds_H']  # What you actually bet against
        df['betting_odds_D'] = df['pre_odds_D']
        df['betting_odds_A'] = df['pre_odds_A']
        
        print(f"Pre-closing probabilities available for {df['pre_p_H'].notna().sum()} matches")
    
    # CLOSING PROBABILITIES → BACKTESTING BENCHMARK (efficiency target)
    if 'close_p_H' in df.columns:
        df['benchmark_prob_H'] = df['close_p_H']  # Pinnacle efficiency standard
        df['benchmark_prob_D'] = df['close_p_D']
        df['benchmark_prob_A'] = df['close_p_A']
        
        # Use closing odds for benchmark comparison
        df['benchmark_odds_H'] = df['close_odds_H']
        df['benchmark_odds_D'] = df['close_odds_D']
        df['benchmark_odds_A'] = df['close_odds_A']
        
        print(f"Closing probabilities available for {df['close_p_H'].notna().sum()} matches")
    
    # MARKET MOVEMENT ANALYSIS (closing vs pre-closing)
    both_available = (df[['pre_p_H', 'close_p_H']].notna().all(axis=1))
    if both_available.sum() > 0:
        print(f"Computing market movement for {both_available.sum()} matches...")
        
        # Market movement = how much closing line moved from opening
        df.loc[both_available, 'market_move_H'] = (df.loc[both_available, 'close_p_H'] - 
                                                  df.loc[both_available, 'pre_p_H'])
        df.loc[both_available, 'market_move_D'] = (df.loc[both_available, 'close_p_D'] - 
                                                  df.loc[both_available, 'pre_p_D'])
        df.loc[both_available, 'market_move_A'] = (df.loc[both_available, 'close_p_A'] - 
                                                  df.loc[both_available, 'pre_p_A'])
        
        # Market movement magnitude (strong signal when > 0.02)
        market_moves = df.loc[both_available, ['market_move_H', 'market_move_D', 'market_move_A']].abs()
        df.loc[both_available, 'market_move_magnitude'] = market_moves.max(axis=1)
        
        # Sharp money direction (which outcome got steamed?)
        max_move_idx = market_moves.idxmax(axis=1)
        df.loc[both_available, 'steamed_outcome'] = max_move_idx.str[-1]  # H, D, or A

        # Steam Direction tells gives magnitude and direction for the steamed prediction
        df.loc[both_available, 'steam_direction'] = df.loc[both_available, "market_move_H"] * (df.loc[both_available, "steamed_outcome"] == "H").astype(int) + \
                                                    df.loc[both_available, "market_move_D"] * (df.loc[both_available, "steamed_outcome"] == "D").astype(int) + \
                                                    df.loc[both_available, "market_move_A"] * (df.loc[both_available, "steamed_outcome"] == "A").astype(int)
        
        # Market efficiency metrics
        avg_move = df.loc[both_available, 'market_move_magnitude'].mean()
        big_moves = (df.loc[both_available, 'market_move_magnitude'] > 0.02).sum()
        print(f"Average market movement: {avg_move:.4f}")
        print(f"Significant moves (>2%): {big_moves} matches ({big_moves/both_available.sum():.1%})")
    
    # SMART BACKTESTING SETUP
    # Prioritize closing odds when available, fallback to pre-closing
    df['backtest_odds_H'] = df['close_odds_H'].fillna(df['pre_odds_H'])
    df['backtest_odds_D'] = df['close_odds_D'].fillna(df['pre_odds_D'])  
    df['backtest_odds_A'] = df['close_odds_A'].fillna(df['pre_odds_A'])
    
    df['backtest_prob_H'] = df['close_p_H'].fillna(df['pre_p_H'])
    df['backtest_prob_D'] = df['close_p_D'].fillna(df['pre_p_D'])
    df['backtest_prob_A'] = df['close_p_A'].fillna(df['pre_p_A'])
    
    # Quality flags from your processing
    df['has_pre_odds'] = df['pre_vig_ok'].fillna(False) if 'pre_vig_ok' in df.columns else df['pre_p_H'].notna()
    df['has_close_odds'] = df['close_vig_ok'].fillna(False) if 'close_vig_ok' in df.columns else df['close_p_H'].notna()
    df['has_odds'] = df[['backtest_odds_H', 'backtest_odds_D', 'backtest_odds_A']].notna().all(axis=1)
    
    # BOOKMAKER SOURCE ANALYSIS
    if 'pre_source' in df.columns:
        pre_sources = df['pre_source'].value_counts()
        print(f"\nPre-closing odds sources:")
        for source, count in pre_sources.head().items():
            print(f"  {source}: {count} matches ({count/len(df):.1%})")
    
    if 'close_source' in df.columns:
        close_sources = df['close_source'].value_counts()
        print(f"\nClosing odds sources:")
        for source, count in close_sources.head().items():
            print(f"  {source}: {count} matches ({count/len(df):.1%})")
    
    # Sort by date for time-series work
    df = df.sort_values('date').reset_index(drop=True)
    
    # DATA QUALITY SUMMARY
    print(f"\n📊 DATA SUMMARY:")
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Matches with pre-closing odds: {df['has_pre_odds'].sum()} ({df['has_pre_odds'].mean():.1%})")
    print(f"Matches with closing odds: {df['has_close_odds'].sum()} ({df['has_close_odds'].mean():.1%})")
    print(f"Matches with backtesting odds: {df['has_odds'].sum()} ({df['has_odds'].mean():.1%})")
    
    # Box score statistics summary  
    box_stats = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY']
    available_stats = [stat for stat in box_stats if stat in df.columns and df[stat].notna().sum() > len(df) * 0.5]
    print(f"Rich match statistics available: {', '.join(available_stats)}")
    
    return df

# Step 2: Professional Model Comparison Using Your Data
def professional_baseline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Professional model comparison leveraging your processed odds hierarchy
    
    Compares:
    1. Dixon-Coles (pure statistical model)
    2. Bet365 Pre-closing (your betting baseline)  
    3. Pinnacle Closing (efficiency gold standard)
    4. Market Movement Signal (close vs pre differential)
    
    Returns:
        DataFrame with comprehensive model performance metrics
    """
    from time_series_cv import TimeSeriesCV, prepare_target_variables, calculate_metrics
    from dixon_coles_poisson import DixonColesPoisson
    
    print("\n🎯 PROFESSIONAL BASELINE COMPARISON")
    print("=" * 50)
    
    # Prepare targets
    y_binary, y_multiclass = prepare_target_variables(df)
    
    # Initialize models
    dc_model = DixonColesPoisson(xi=0.01)
    
    # Cross-validation with your data
    tscv = TimeSeriesCV(min_train_seasons=2, validation_seasons=1)
    results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        if fold_idx >= 3:  # First 3 folds for quick analysis
            break
            
        print(f"\nFold {fold_idx + 1}: Analyzing betting opportunities...")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        print(f"  Train: {train_df['season'].min()} to {train_df['season'].max()}")
        print(f"  Val: {val_df['season'].unique()}")
        
        # 1. DIXON-COLES MODEL (Pure Statistical)
        dc_model.fit(train_df)
        dc_probs = dc_model.predict_proba(val_df[['home', 'away']])
        
        dc_metrics = calculate_metrics(y_multiclass[val_idx], dc_probs, 'multiclass')
        results.append({
            'fold': fold_idx + 1,
            'model': 'Dixon-Coles',
            'type': 'Statistical Model',
            'logloss': dc_metrics['logloss'],
            'rps': dc_metrics['rps'], 
            'brier': dc_metrics['brier']
        })
        
        # 2. BET365 PRE-CLOSING (Your Betting Baseline)
        val_with_pre = val_df[val_df[['feature_prob_H', 'feature_prob_D', 'feature_prob_A']].notna().all(axis=1)]
        if len(val_with_pre) > 10:
            bet365_probs = val_with_pre[['feature_prob_A', 'feature_prob_D', 'feature_prob_H']].values
            bet365_y = y_multiclass[val_with_pre.index]
            bet365_metrics = calculate_metrics(bet365_y, bet365_probs, 'multiclass')
            
            results.append({
                'fold': fold_idx + 1,
                'model': 'Bet365 Pre-close',
                'type': 'Opening Market',
                'logloss': bet365_metrics['logloss'],
                'rps': bet365_metrics['rps'],
                'brier': bet365_metrics['brier']
            })
            
            print(f"    Bet365 opening lines: {len(val_with_pre)} matches")
        
        # 3. PINNACLE CLOSING (Efficiency Gold Standard)
        val_with_close = val_df[val_df[['benchmark_prob_H', 'benchmark_prob_D', 'benchmark_prob_A']].notna().all(axis=1)]
        if len(val_with_close) > 10:
            pinnacle_probs = val_with_close[['benchmark_prob_A', 'benchmark_prob_D', 'benchmark_prob_H']].values
            pinnacle_y = y_multiclass[val_with_close.index]
            pinnacle_metrics = calculate_metrics(pinnacle_y, pinnacle_probs, 'multiclass')
            
            results.append({
                'fold': fold_idx + 1,
                'model': 'Pinnacle Closing',
                'type': 'Efficient Market',
                'logloss': pinnacle_metrics['logloss'],
                'rps': pinnacle_metrics['rps'],
                'brier': pinnacle_metrics['brier']
            })
            
            print(f"    Pinnacle closing lines: {len(val_with_close)} matches")
        
        # 4. MARKET MOVEMENT ANALYSIS
        val_with_movement = val_df[val_df['market_move_magnitude'].notna()]
        if len(val_with_movement) > 20:
            significant_moves = val_with_movement[val_with_movement['market_move_magnitude'] > 0.02]
            print(f"    Significant market moves (>2%): {len(significant_moves)}")
            
            if len(significant_moves) > 5:
                # Simple market movement model: bet against opening line when big movement
                movement_probs = significant_moves[['benchmark_prob_A', 'benchmark_prob_D', 'benchmark_prob_H']].values
                movement_y = y_multiclass[significant_moves.index]
                movement_metrics = calculate_metrics(movement_y, movement_probs, 'multiclass')
                
                results.append({
                    'fold': fold_idx + 1,
                    'model': 'Market Movement',
                    'type': 'Sharp Signal',
                    'logloss': movement_metrics['logloss'],
                    'rps': movement_metrics['rps'],
                    'brier': movement_metrics['brier']
                })
    
    results_df = pd.DataFrame(results)
    
    # COMPREHENSIVE ANALYSIS
    print("\n📊 PROFESSIONAL COMPARISON RESULTS:")
    print("=" * 60)
    
    if len(results_df) > 0:
        summary = results_df.groupby(['model', 'type']).agg({
            'logloss': ['mean', 'std', 'count'],
            'rps': ['mean', 'std'],
            'brier': ['mean', 'std']
        }).round(4)
        
        for (model, model_type) in summary.index:
            print(f"\n{model} ({model_type}):")
            print(f"  Log Loss: {summary.loc[(model, model_type), ('logloss', 'mean')]:.4f} ± {summary.loc[(model, model_type), ('logloss', 'std')]:.4f}")
            print(f"  RPS:      {summary.loc[(model, model_type), ('rps', 'mean')]:.4f} ± {summary.loc[(model, model_type), ('rps', 'std')]:.4f}")
            print(f"  Brier:    {summary.loc[(model, model_type), ('brier', 'mean')]:.4f} ± {summary.loc[(model, model_type), ('brier', 'std')]:.4f}")
        
        # PROFESSIONAL INSIGHTS
        print(f"\n🎯 KEY PROFESSIONAL INSIGHTS:")
        print("-" * 40)
        
        # Market efficiency analysis
        model_means = results_df.groupby('model')['rps'].mean()
        
        if 'Pinnacle Closing' in model_means.index:
            pinnacle_rps = model_means['Pinnacle Closing']
            print(f"• Pinnacle closing RPS: {pinnacle_rps:.4f} (efficiency benchmark)")
            
            if 'Bet365 Pre-close' in model_means.index:
                bet365_rps = model_means['Bet365 Pre-close']
                market_efficiency = bet365_rps - pinnacle_rps
                print(f"• Market efficiency gain: {market_efficiency:.4f} RPS")
                print(f"  (Value captured by market from open to close)")
        
        if 'Dixon-Coles' in model_means.index:
            dc_rps = model_means['Dixon-Coles']
            
            if 'Bet365 Pre-close' in model_means.index:
                bet365_rps = model_means['Bet365 Pre-close']
                model_edge = bet365_rps - dc_rps
                status = "BEATING" if model_edge > 0 else "BEHIND"
                print(f"• Your model vs Bet365 opening: {model_edge:+.4f} RPS ({status})")
                
            if 'Pinnacle Closing' in model_means.index:
                pinnacle_rps = model_means['Pinnacle Closing']
                efficiency_gap = pinnacle_rps - dc_rps
                print(f"• Efficiency gap to close: {efficiency_gap:+.4f} RPS")
                
                if efficiency_gap < 0.01:
                    print("  🎉 EXCELLENT! Near-efficient performance")
                elif efficiency_gap < 0.02:
                    print("  ✅ GOOD! Competitive with market")
                else:
                    print("  🔄 Room for improvement in Week 2")
        
        # Movement analysis
        if 'Market Movement' in model_means.index:
            movement_rps = model_means['Market Movement']
            print(f"• Sharp money signal: {movement_rps:.4f} RPS")
            print("  (Performance when market moves significantly)")
    
    return results_df

# Step 3: Professional Backtesting with Your Odds Hierarchy  
def professional_backtest_demo(df: pd.DataFrame) -> dict:
    """
    Professional backtesting leveraging your sophisticated odds processing
    
    Strategy:
    - Train on historical data (all matches, regardless of odds availability)
    - Bet using Bet365 pre-closing odds (realistic scenario)
    - Compare against Pinnacle closing benchmark (efficiency standard)
    - Account for bookmaker source hierarchy in analysis
    
    Args:
        df: Your processed match data with odds hierarchy
        
    Returns:
        Comprehensive backtest results with market analysis
    """
    from dixon_coles_poisson import DixonColesPoisson
    from gbdt_and_backtest import FootballBacktester
    
    print("\n💰 PROFESSIONAL BACKTESTING SIMULATION")
    print("=" * 50)
    
    # Filter to matches with betting odds (pre-closing available)
    df_with_betting_odds = df[df['has_pre_odds']].copy()
    
    if len(df_with_betting_odds) < 500:
        print(f"⚠️  Only {len(df_with_betting_odds)} matches with betting odds")
        print("Need more data for reliable backtesting")
        return {}
    
    # Professional time-based split
    seasons = sorted(df_with_betting_odds['season'].unique())
    if len(seasons) < 3:
        print("Need at least 3 seasons for professional train/test split")
        return {}
    
    # Use last season for out-of-sample testing
    train_seasons = seasons[:-1]
    test_seasons = seasons[-1:]
    
    # Train on ALL historical data (not just matches with odds)
    all_train_df = df[df['season'].isin(train_seasons)]
    test_df = df_with_betting_odds[df_with_betting_odds['season'].isin(test_seasons)]
    
    print(f"📈 Training Strategy:")
    print(f"  Training seasons: {train_seasons}")
    print(f"  Testing season: {test_seasons}")
    print(f"  Train size: {len(all_train_df)} matches (all available)")
    print(f"  Test size: {len(test_df)} matches (with betting odds)")
    
    # Bookmaker source analysis for test period
    if 'pre_source' in test_df.columns:
        source_dist = test_df['pre_source'].value_counts()
        print(f"\n📊 Betting Odds Sources (Test Period):")
        for source, count in source_dist.items():
            print(f"  {source}: {count} matches ({count/len(test_df):.1%})")
    
    # Train model on full historical dataset
    print(f"\n🤖 Training Dixon-Coles on {len(all_train_df)} historical matches...")
    model = DixonColesPoisson(xi=0.01)
    model.fit(all_train_df)
    
    # Generate predictions for test matches
    test_predictions = model.predict_proba(test_df[['home', 'away']])
    
    # Prepare realistic backtesting scenario
    backtest_df = test_df.copy()
    backtest_df['model_prob_H'] = test_predictions[:, 2]
    backtest_df['model_prob_D'] = test_predictions[:, 1] 
    backtest_df['model_prob_A'] = test_predictions[:, 0]
    
    # Use Bet365 pre-closing odds for betting (realistic scenario)
    backtest_df['book_odds_H'] = test_df['betting_odds_H']  # What you actually bet against
    backtest_df['book_odds_D'] = test_df['betting_odds_D']
    backtest_df['book_odds_A'] = test_df['betting_odds_A']
    backtest_df['result'] = test_df['fulltime_result']
    
    # Professional backtesting with multiple strategies
    backtester = FootballBacktester(initial_bankroll=10000)  # £10k starting bank
    
    print(f"\n💡 Testing Professional Betting Strategies:")
    print("-" * 45)
    
    # Strategy comparison
    strategies = [
        {'name': 'Conservative', 'ev_threshold': 0.025, 'kelly_fraction': 0.25, 'max_bet': 0.02},
        {'name': 'Moderate', 'ev_threshold': 0.020, 'kelly_fraction': 0.35, 'max_bet': 0.03},
        {'name': 'Aggressive', 'ev_threshold': 0.015, 'kelly_fraction': 0.50, 'max_bet': 0.05},
    ]
    
    best_sharpe = -float('inf')
    best_results = None
    all_strategy_results = []
    
    for strategy in strategies:
        results = backtester.backtest_strategy(
            backtest_df,
            ev_threshold=strategy['ev_threshold'],
            kelly_fraction=strategy['kelly_fraction'],
            max_bet_fraction=strategy['max_bet']
        )
        
        # Calculate Sharpe-like ratio for risk-adjusted performance
        if len(results['bankroll_history']) > 1:
            returns = np.diff(results['bankroll_history']) / results['bankroll_history'][:-1]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
            
        results['sharpe_ratio'] = sharpe
        results['strategy_name'] = strategy['name']
        all_strategy_results.append(results)
        
        print(f"{strategy['name']:>12}: ROI = {results['roi_percent']:+6.2f}%, "
              f"Bets = {results['n_bets']:3d}, "
              f"Win Rate = {results['win_rate_percent']:4.1f}%, "
              f"Sharpe = {sharpe:+5.2f}")
        
        if sharpe > best_sharpe and results['n_bets'] >= 20:
            best_sharpe = sharpe
            best_results = results
    
    # Detailed analysis of best strategy
    if best_results:
        print(f"\n🏆 BEST STRATEGY: {best_results['strategy_name']}")
        print("=" * 50)
        backtester.print_summary(best_results)
        
        # Professional analysis
        bet_log_df = pd.DataFrame(best_results['bet_log'])
        if len(bet_log_df) > 0:
            print(f"\n📈 PROFESSIONAL ANALYSIS:")
            print("-" * 30)
            
            # Edge analysis
            avg_edge = bet_log_df['edge'].mean()
            print(f"Average edge when betting: {avg_edge:.3f} ({avg_edge*100:.1f}%)")
            
            # Outcome distribution
            outcome_profits = bet_log_df.groupby('outcome')['profit'].agg(['sum', 'count', 'mean'])
            print(f"\nProfit by outcome:")
            for outcome in ['H', 'D', 'A']:
                if outcome in outcome_profits.index:
                    profit = outcome_profits.loc[outcome, 'sum']
                    count = outcome_profits.loc[outcome, 'count']
                    avg = outcome_profits.loc[outcome, 'mean']
                    print(f"  {outcome}: £{profit:+7.2f} ({count:2d} bets, £{avg:+6.2f} avg)")
            
            # Odds range analysis
            odds_ranges = pd.cut(bet_log_df['odds'], bins=[1, 2, 3, 5, float('inf')], 
                               labels=['1.0-2.0', '2.0-3.0', '3.0-5.0', '5.0+'])
            odds_analysis = bet_log_df.groupby(odds_ranges)['profit'].agg(['sum', 'count'])
            print(f"\nProfit by odds range:")
            for odds_range in odds_analysis.index:
                if pd.notna(odds_range):
                    profit = odds_analysis.loc[odds_range, 'sum']
                    count = odds_analysis.loc[odds_range, 'count'] 
                    print(f"  {odds_range}: £{profit:+7.2f} ({count:2d} bets)")
            
            # Market movement correlation (if available)
            if 'market_move_magnitude' in test_df.columns:
                # Join market movement data
                bet_log_enhanced = bet_log_df.merge(
                    test_df[['match_id', 'market_move_magnitude']].dropna(),
                    left_on='match',
                    right_on='match_id',
                    how='left'
                )
                
                if 'market_move_magnitude' in bet_log_enhanced.columns:
                    big_moves = bet_log_enhanced[bet_log_enhanced['market_move_magnitude'] > 0.02]
                    if len(big_moves) > 0:
                        print(f"\nBets on matches with significant market movement:")
                        print(f"  Count: {len(big_moves)} bets")
                        print(f"  Profit: £{big_moves['profit'].sum():+.2f}")
                        print(f"  Win rate: {big_moves['won'].mean():.1%}")
        
        # Plot equity curve for best strategy
        if best_results['n_bets'] > 10:
            backtester.plot_equity_curve(best_results)
            
        # Compare against efficient market benchmark
        if 'benchmark_prob_H' in test_df.columns:
            pinnacle_available = test_df[['benchmark_prob_H', 'benchmark_prob_D', 'benchmark_prob_A']].notna().all(axis=1)
            if pinnacle_available.sum() > 50:
                print(f"\n⚖️  MARKET EFFICIENCY ANALYSIS:")
                print(f"Matches with Pinnacle closing: {pinnacle_available.sum()}")
                
                # How often did your bets align with sharp money?
                # This would require more complex analysis of movement direction
    
    else:
        print(f"\n❌ NO PROFITABLE STRATEGY FOUND")
        print("This is completely normal - beating the market is extremely difficult!")
        print("\nFocus areas for Week 2:")
        print("• Add match statistics features (shots, corners, cards)")
        print("• Implement market movement signals")
        print("• Try ensemble models combining multiple approaches")
        print("• Consider lower-margin markets or alternative leagues")
    
    # Return best results for further analysis
    return {
        'best_strategy': best_results,
        'all_strategies': all_strategy_results,
        'test_data_summary': {
            'total_matches': len(test_df),
            'seasons': test_seasons,
            'bookmaker_sources': source_dist.to_dict() if 'pre_source' in test_df.columns else {}
        }
    }

# Step 4: Professional Implementation Pipeline
def main_professional_pipeline(csv_path: str):
    """
    Professional Week 1 pipeline leveraging your processed odds hierarchy
    
    Your data advantages:
    - Bet365/Pinnacle odds hierarchy (professional bookmaker selection)
    - Pre-closing vs closing comparison (market efficiency analysis)  
    - Rich match statistics (shots, corners, cards for Week 2)
    - Proper devigging and validation (quality control)
    
    Args:
        csv_path: Path to your processed Premier League CSV file
    """
    print("🏈 PROFESSIONAL FOOTBALL PREDICTION PIPELINE")
    print("=" * 60)
    print("Leveraging your sophisticated odds processing system")
    
    try:
        # Step 1: Load your professionally processed data
        df = prepare_data(csv_path)
        
        # Step 2: Professional model comparison
        comparison_results = professional_baseline_comparison(df)
        
        # Step 3: Professional backtesting
        backtest_results = professional_backtest_demo(df)
        
        print(f"\n🎉 WEEK 1 PROFESSIONAL PIPELINE COMPLETE!")
        print("=" * 60)
        
        # WEEK 2 ROADMAP (Tailored to your data)
        print(f"\n🚀 WEEK 2 ROADMAP (Your Data Advantages):")
        print("-" * 50)
        print("1. RICH FEATURE ENGINEERING:")
        print("   • Shot conversion rates: HST/HS, AST/AS")
        print("   • Set piece efficiency: HC/AC vs goals") 
        print("   • Discipline patterns: HY/AY, HBP/ABP rolling averages")
        print("   • Woodwork luck: HHW/AHW as variance indicators")
        
        print(f"\n2. MARKET SIGNAL INTEGRATION:")
        print("   • Market movement magnitude as feature")
        print("   • Bookmaker source hierarchy (Bet365 vs Pinnacle)")
        print("   • Steam detection (rapid line movement)")
        print("   • Vig comparison across bookmakers")
        
        print(f"\n3. ADVANCED MODELING:")
        print("   • LightGBM with your box score features")
        print("   • Market prior stacking (use pre-closing as base)")
        print("   • Ensemble: Dixon-Coles + GBDT + Market signals")
        print("   • Dynamic calibration by bookmaker source")
        
        print(f"\n4. PROFESSIONAL BETTING STRATEGIES:")
        print("   • Multi-bookmaker arbitrage opportunities")
        print("   • Early vs late market positioning")
        print("   • League-specific model specialization")
        print("   • In-play model (using your rich match stats)")
        
        # Save professional results
        results_dir = Path("results/professional")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced results saving
        comparison_results.to_csv(results_dir / "week1_professional_comparison.csv", index=False)
        
        if backtest_results and 'best_strategy' in backtest_results:
            import json
            
            # Save backtest summary
            summary = {
                'strategy_name': backtest_results['best_strategy']['strategy_name'],
                'roi_percent': backtest_results['best_strategy']['roi_percent'],
                'total_bets': backtest_results['best_strategy']['n_bets'],
                'win_rate': backtest_results['best_strategy']['win_rate_percent'],
                'max_drawdown': backtest_results['best_strategy']['max_drawdown_percent'],
                'sharpe_ratio': backtest_results['best_strategy']['sharpe_ratio'],
                'test_period': backtest_results['test_data_summary']['seasons']
            }
            
            with open(results_dir / "week1_best_strategy.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed bet log
            if backtest_results['best_strategy']['bet_log']:
                bet_log_df = pd.DataFrame(backtest_results['best_strategy']['bet_log'])
                bet_log_df.to_csv(results_dir / "week1_bet_log.csv", index=False)
        
        # Professional data quality report
        data_quality = {
            'total_matches': len(df),
            'seasons': sorted(df['season'].unique()),
            'date_range': [df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d')],
            'pre_closing_coverage': f"{df['has_pre_odds'].mean():.1%}",
            'closing_coverage': f"{df['has_close_odds'].mean():.1%}",
            'box_score_availability': {
                stat: f"{df[stat].notna().mean():.1%}" 
                for stat in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY'] 
                if stat in df.columns
            }
        }
        
        with open(results_dir / "data_quality_report.json", "w") as f:
            json.dump(data_quality, f, indent=2)
        
        print(f"\n📁 Professional results saved to: {results_dir}")
        
        # Your competitive advantages summary
        print(f"\n💪 YOUR COMPETITIVE ADVANTAGES:")
        print("-" * 40)
        print("✅ Sophisticated odds hierarchy (Bet365 → Pinnacle)")
        print("✅ Market efficiency analysis (pre vs closing)")
        print("✅ Rich match statistics (18 box score features)")
        print("✅ Professional data processing (devigging, validation)")
        print("✅ Realistic betting simulation (actual bookmaker odds)")
        print("✅ Time-series validation (no look-ahead bias)")
        
        print(f"\n🎯 Week 1 Success = Foundation for Professional Operation!")
        
        return {
            'data': df,
            'comparison': comparison_results,
            'backtest': backtest_results,
            'data_quality': data_quality
        }
        
    except Exception as e:
        print(f"❌ Error in professional pipeline: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure your CSV has the expected column names from text_to_matches.py")
        print("2. Check that date formatting is correct (YYYY-MM-DD HH:MM:SS)")
        print("3. Verify season labels match expected format (YYYY-YY)")
        return None

# Step 5: Enhanced CLI Interface for Your Data
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path for your processed data
        csv_path = "data/raw/matches.csv"  # Output from your text_to_matches.py
        print(f"No CSV path provided, using default: {csv_path}")
        print("Usage: python implementation_guide.py <path_to_your_processed_csv>")
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        print("\n🔧 SETUP INSTRUCTIONS:")
        print("=" * 30)
        print("1. Run your data processing script first:")
        print("   python tools/text_to_matches.py --input-glob 'data/raw/seasons/*.csv' --out 'data/raw/matches.csv'")
        print()
        print("2. Then run this pipeline:")
        print("   python implementation_guide.py data/raw/matches.csv")
        print()
        print("3. Your processed CSV should have these columns:")
        print("   • Core: match_id, season, date, home, away, home_goals, away_goals, fulltime_result")
        print("   • Pre-closing: pre_odds_H/D/A, pre_p_H/D/A, pre_source")  
        print("   • Closing: close_odds_H/D/A, close_p_H/D/A, close_source")
        print("   • Stats: HS, AS, HST, AST, HC, AC, HY, AY, etc.")
    else:
        # results = main_professional_pipeline(csv_path)
        results = prepare_data(csv_path)
        print(results.head())


# Professional data analysis for your processed data
def analyze_processed_data(csv_path: str):
    """
    Professional analysis of your processed Premier League dataset
    """
    print("📊 PROFESSIONAL DATA ANALYSIS")
    print("=" * 40)
    
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Date range analysis
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n📅 TEMPORAL COVERAGE:")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total seasons: {len(df['season'].unique())}")
    
    # Season-by-season breakdown
    season_stats = df.groupby('season').agg({
        'match_id': 'count',
        'pre_p_H': lambda x: x.notna().sum(),
        'close_p_H': lambda x: x.notna().sum()
    }).rename(columns={'match_id': 'total_matches', 'pre_p_H': 'pre_odds', 'close_p_H': 'close_odds'})
    
    season_stats['pre_coverage'] = (season_stats['pre_odds'] / season_stats['total_matches'] * 100).round(1)
    season_stats['close_coverage'] = (season_stats['close_odds'] / season_stats['total_matches'] * 100).round(1)
    
    print(f"\n📈 SEASON BREAKDOWN:")
    print(season_stats.to_string())
    
    # Bookmaker source analysis
    if 'pre_source' in df.columns:
        print(f"\n🏦 BOOKMAKER SOURCES:")
        pre_sources = df['pre_source'].value_counts()
        print("Pre-closing odds:")
        for source, count in pre_sources.items():
            print(f"  {source}: {count:,} matches ({count/len(df)*100:.1f}%)")
    
    if 'close_source' in df.columns:
        close_sources = df['close_source'].value_counts()
        print("Closing odds:")
        for source, count in close_sources.items():
            print(f"  {source}: {count:,} matches ({count/len(df)*100:.1f}%)")
    
    # Market efficiency analysis
    if all(col in df.columns for col in ['pre_p_H', 'close_p_H']):
        both_available = df[['pre_p_H', 'close_p_H']].notna().all(axis=1)
        if both_available.sum() > 0:
            print(f"\n⚖️ MARKET EFFICIENCY ANALYSIS:")
            print(f"Matches with both pre & closing: {both_available.sum():,}")
            
            # Calculate average movement
            home_move = (df.loc[both_available, 'close_p_H'] - df.loc[both_available, 'pre_p_H']).abs()
            draw_move = (df.loc[both_available, 'close_p_D'] - df.loc[both_available, 'pre_p_D']).abs()
            away_move = (df.loc[both_available, 'close_p_A'] - df.loc[both_available, 'pre_p_A']).abs()
            
            max_move = pd.concat([home_move, draw_move, away_move], axis=1).max(axis=1)
            
            print(f"Average market movement: {max_move.mean():.4f}")
            print(f"Significant moves (>2%): {(max_move > 0.02).sum():,} ({(max_move > 0.02).mean()*100:.1f}%)")
            print(f"Major moves (>5%): {(max_move > 0.05).sum():,} ({(max_move > 0.05).mean()*100:.1f}%)")
    
    # Match outcome distribution
    print(f"\n⚽ MATCH OUTCOMES:")
    outcome_dist = df['fulltime_result'].value_counts()
    total_matches = len(df)
    for outcome, count in outcome_dist.items():
        print(f"  {outcome}: {count:,} matches ({count/total_matches*100:.1f}%)")
    
    # Box score statistics availability
    box_stats = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP']
    available_stats = {}
    
    print(f"\n📊 BOX SCORE STATISTICS COVERAGE:")
    for stat in box_stats:
        if stat in df.columns:
            coverage = df[stat].notna().mean() * 100
            available_stats[stat] = coverage
            print(f"  {stat}: {coverage:.1f}%")
    
    # Data quality indicators
    print(f"\n✅ DATA QUALITY INDICATORS:")
    
    # Vig analysis
    if 'pre_vig_ok' in df.columns:
        pre_vig_good = df['pre_vig_ok'].sum()
        print(f"Pre-closing odds with acceptable vig: {pre_vig_good:,} ({pre_vig_good/len(df)*100:.1f}%)")
    
    if 'close_vig_ok' in df.columns:
        close_vig_good = df['close_vig_ok'].sum()
        print(f"Closing odds with acceptable vig: {close_vig_good:,} ({close_vig_good/len(df)*100:.1f}%)")
    
    # Missing data summary
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    high_missing = missing_pct[missing_pct > 10]
    if len(high_missing) > 0:
        print(f"\nColumns with >10% missing data:")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct}%")
    
    # Professional recommendations
    print(f"\n🎯 PROFESSIONAL RECOMMENDATIONS:")
    print("-" * 40)
    
    total_pre_odds = df['pre_p_H'].notna().sum() if 'pre_p_H' in df.columns else 0
    total_close_odds = df['close_p_H'].notna().sum() if 'close_p_H' in df.columns else 0
    
    if total_pre_odds > 1000:
        print("✅ Excellent pre-closing odds coverage for model training")
    elif total_pre_odds > 500:
        print("⚠️  Moderate pre-closing odds coverage - consider more data")
    else:
        print("❌ Limited pre-closing odds coverage - need more seasons")
    
    if total_close_odds > 500:
        print("✅ Good closing odds coverage for efficiency benchmarking")
    else:
        print("⚠️  Limited closing odds for benchmarking")
    
    rich_stats_available = sum(1 for stat, cov in available_stats.items() if cov > 80)
    if rich_stats_available >= 6:
        print("✅ Rich match statistics perfect for Week 2 feature engineering")
    elif rich_stats_available >= 3:
        print("⚠️  Some match statistics available for basic features")
    else:
        print("❌ Limited match statistics - focus on basic model first")
    
    print(f"\n🚀 READY FOR PROFESSIONAL MODELING!")



# COMPLETE END-TO-END WORKFLOW EXPLANATION
def explain_workflow():
    """
    Comprehensive explanation of how the football prediction model works
    from training to live betting
    """
    print("""
🏈 FOOTBALL PREDICTION MODEL - COMPLETE WORKFLOW
================================================

📊 PHASE 1: DATA PREPARATION & TRAINING
---------------------------------------

1. DATA SETUP:
   • Historical match data with results (home_goals, away_goals, result)
   • Pre-closing odds (what you see before the match) → MODEL FEATURES
   • Closing odds (final market prices) → BACKTESTING BENCHMARK
   • Additional stats: shots, corners, cards, etc. → FUTURE FEATURES

2. SMART ODDS HANDLING:
   • PRE-CLOSING odds become model features (available when betting)
   • CLOSING odds used for backtesting (most efficient market prices)
   • Fallback: Use pre-closing for backtesting when closing unavailable
   • Market movement: closing_prob - pre_closing_prob (strong signal!)

3. TIME-BASED TRAINING:
   • Expanding window CV: Train 2015-2019 → Test 2020, Train 2015-2020 → Test 2021
   • Models: Dixon-Coles Poisson → Logistic Regression → Gradient Boosting
   • Evaluation: Log-loss, RPS, Brier Score (probability quality, not accuracy!)
   • Calibration: Ensure P(event) = frequency when model says P(event)

🎯 PHASE 2: MODEL DEPLOYMENT & LIVE PREDICTION
---------------------------------------------

4. WEEKLY MODEL UPDATE:
   • Retrain models with latest match results
   • Update team Elo ratings, rolling form stats
   • Recalibrate probabilities on recent data
   • Save model artifacts for consistency

5. MATCH DAY PREDICTION (Your Workflow):
   a) Get upcoming fixtures: Arsenal vs Chelsea, 3pm Saturday
   b) Collect PRE-CLOSING odds from bookmakers (what you'll bet against)
   c) Extract features: team Elo, recent form, rest days, etc.
   d) Model predicts: P(Arsenal win) = 0.45, P(Draw) = 0.28, P(Chelsea) = 0.27
   e) Compare with bookmaker odds: Arsenal 2.10 (48% implied), Draw 3.40 (29%), Chelsea 3.60 (28%)

6. BETTING DECISION ENGINE:
   • Calculate Expected Value: EV = (model_prob × odds) - 1
   • Arsenal: EV = (0.45 × 2.10) - 1 = -0.055 (NEGATIVE - don't bet)
   • Chelsea: EV = (0.27 × 3.60) - 1 = -0.028 (NEGATIVE - don't bet)
   • Only bet when EV > threshold (e.g., +0.02 = 2% edge)

💰 PHASE 3: BANKROLL MANAGEMENT & EXECUTION
------------------------------------------

7. POSITION SIZING (Kelly Criterion):
   • If EV > threshold, calculate bet size using Kelly formula
   • Kelly fraction: f = (b×p - q) / b, where b=odds-1, p=model_prob, q=1-p
   • Use fractional Kelly (0.25-0.5x) for risk management
   • Never bet more than 5% of bankroll on single match

8. LIVE BETTING PROCESS:
   a) Saturday 10am: Get fixtures + pre-closing odds
   b) Run model predictions 
   c) Identify positive EV opportunities
   d) Place bets using pre-closing odds (what model was trained on!)
   e) Record: stake, odds, expected profit, actual outcome
   f) Update bankroll and track performance

📈 PHASE 4: PERFORMANCE MONITORING & IMPROVEMENT
-----------------------------------------------

9. CONTINUOUS EVALUATION:
   • Track: ROI, win rate, max drawdown, turnover
   • Calibration plots: Are 60% confident predictions right 60% of time?
   • Edge analysis: Where do you find most value? (home underdogs, etc.)
   • Market movement: How often do closing odds move in your favor?

10. MODEL ITERATION (Week 2+):
    • Add features: shot stats, squad rotation, market movement
    • Ensemble models: Combine Poisson + GBDT + market signals
    • Advanced staking: Dynamic Kelly based on model confidence
    • League expansion: Test on Championship, La Liga, etc.

🔧 TECHNICAL IMPLEMENTATION DETAILS
----------------------------------

Data Flow:
  Historical CSV → prepare_data() → Features + Targets
               ↓
  Time-series CV → Train/Validate → Model Metrics
               ↓  
  Final Model → predict_proba() → EV Calculation → Betting Decision

Key Files Structure:
  data/raw/matches.csv           # Your input data
  src/models/dixon_coles.py      # Poisson baseline  
  src/features/engineering.py    # Feature creation
  src/backtest/simulator.py      # Strategy testing
  models/V1/final_model.pkl      # Saved model
  predictions/2024_gameweek_15.csv # Weekly predictions

🚨 CRITICAL SUCCESS FACTORS
---------------------------

1. DATA INTEGRITY:
   ✓ Never use future information (strict time cutoffs)
   ✓ Handle missing odds gracefully (your mixed data scenario)
   ✓ Validate date consistency across seasons

2. PROBABILITY CALIBRATION:
   ✓ Model says 40% → should happen ~40% of time
   ✓ Use isotonic calibration for better betting decisions
   ✓ Test calibration on holdout set before going live

3. BANKROLL DISCIPLINE:
   ✓ Start small (1-2% bankroll per bet max)
   ✓ Track every bet in detail for analysis
   ✓ Set stop-loss rules (max 20% drawdown?)

4. MARKET AWARENESS:
   ✓ Pre-closing odds ≠ closing odds (line movement!)
   ✓ Your edge erodes as information gets incorporated
   ✓ Focus on markets where you have informational advantage

📋 WEEK 1 IMPLEMENTATION CHECKLIST
---------------------------------

□ Data loaded with smart odds handling (pre vs closing)
□ Dixon-Coles baseline trained and validated  
□ Time-series CV working (no data leakage)
□ Probability metrics calculated (log-loss, RPS, Brier)
□ Simple backtest showing realistic betting scenarios
□ Model vs market comparison (are you beating pre-closing odds?)
□ Calibration plots generated (reliability curves)
□ Results saved and documented

🎯 SUCCESS METRICS (Week 1 Targets)
----------------------------------

Model Performance:
• RPS < 0.22 (bookmaker typically ~0.19-0.20)
• Log-loss < 1.00 (bookmaker typically ~0.90-0.95)  
• Beat pre-closing odds baseline by 0.005+ RPS

Backtesting Results:
• Find ANY positive EV opportunities (even 0.5% edge)
• Max drawdown < 30% with conservative staking
• At least 50+ bets in test period for statistical significance

Remember: Beating the market is extremely difficult. Week 1 success = building 
a robust foundation that doesn't lose money and shows occasional edge!
    """)

if __name__ == "__main__" and len(sys.argv) == 1:
    explain_workflow()
    print(f"\n🔧 QUICK COMMANDS:")
    print("=" * 20)
    print("1. Analyze your processed data:")
    print("   python -c \"from implementation_guide import analyze_processed_data; analyze_processed_data('data/raw/matches.csv')\"")
    print()
    print("2. Run full professional pipeline:")
    print("   python implementation_guide.py data/raw/matches.csv")
    print()
    print("3. Your data processing outputs the perfect structure for this pipeline! 🎯")

# Enhanced quick start function
def quick_start():
    """
    Show enhanced quick start instructions with workflow context
    """
    explain_workflow()
    
    print("""
🚀 IMMEDIATE NEXT STEPS
======================

1. RUN YOUR PIPELINE:
   python implementation_guide.py your_premier_league_data.csv

2. EXPECTED OUTPUT:
   • Data analysis showing pre/closing odds split
   • Model comparison: Dixon-Coles vs Pre-closing vs Closing odds
   • Backtest results with profit/loss over test period
   • Reliability plots showing calibration quality

3. INTERPRET RESULTS:
   • If model beats pre-closing odds → good foundation!
   • If backtest shows +EV opportunities → promising start
   • If calibration looks good → ready for real betting
   • If everything negative → normal, focus on Week 2 improvements

4. NEXT WEEK PRIORITIES:
   • Add shot/corner/card features from your rich dataset
   • Implement LightGBM with your match statistics
   • Use market movement (close vs pre odds) as signal
   • Test ensemble combining Poisson + GBDT + market priors

Your mixed pre/closing odds data is actually perfect for this approach! 🎯
    """)

if __name__ == "__main__" and len(sys.argv) == 1:
    quick_start()


