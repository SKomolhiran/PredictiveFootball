"""
Data verification script for your processed Premier League data
Run this first to ensure your data is ready for the professional pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def verify_your_data(csv_path: str) -> bool:
    """
    Comprehensive verification of your processed data structure
    
    Checks:
    1. Required columns from text_to_matches.py output
    2. Data quality and completeness
    3. Odds hierarchy and market coverage
    4. Box score statistics availability
    5. Time-series ordering and seasonality
    
    Returns:
        True if data is ready for professional modeling
    """
    
    print("🔍 VERIFYING YOUR PROCESSED DATA")
    print("=" * 50)
    
    # Check file exists
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        print("\n📋 SETUP CHECKLIST:")
        print("1. Run your data processing script:")
        print("   python tools/text_to_matches.py --input-glob 'data/raw/seasons/*.csv' --out 'data/raw/matches.csv'")
        print("2. Ensure the output file is at the correct path")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ File loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # 1. CORE COLUMN VERIFICATION
    print(f"\n1️⃣ CORE COLUMNS VERIFICATION")
    print("-" * 30)
    
    required_core = [
        'match_id', 'season', 'date', 'home', 'away', 
        'home_goals', 'away_goals', 'fulltime_result'
    ]
    
    missing_core = [col for col in required_core if col not in df.columns]
    if missing_core:
        print(f"❌ Missing core columns: {missing_core}")
        return False
    else:
        print("✅ All core columns present")
    
    # Check data types
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    invalid_dates = df['date'].isna().sum()
    if invalid_dates > 0:
        print(f"⚠️  {invalid_dates} matches with invalid dates")
    else:
        print("✅ Date column properly formatted")
    
    # 2. ODDS HIERARCHY VERIFICATION
    print(f"\n2️⃣ ODDS HIERARCHY VERIFICATION")
    print("-" * 30)
    
    # Pre-closing odds (for model features)
    pre_odds_cols = ['pre_odds_H', 'pre_odds_D', 'pre_odds_A', 'pre_p_H', 'pre_p_D', 'pre_p_A']
    pre_available = [col for col in pre_odds_cols if col in df.columns]
    
    if len(pre_available) >= 6:
        pre_coverage = df['pre_p_H'].notna().sum()
        print(f"✅ Pre-closing odds: {pre_coverage:,} matches ({pre_coverage/len(df)*100:.1f}%)")
        
        if 'pre_source' in df.columns:
            top_pre_sources = df['pre_source'].value_counts().head(3)
            print(f"   Top sources: {dict(top_pre_sources)}")
    else:
        print(f"❌ Missing pre-closing odds columns: {set(pre_odds_cols) - set(pre_available)}")
        return False
    
    # Closing odds (for benchmarking)  
    close_odds_cols = ['close_odds_H', 'close_odds_D', 'close_odds_A', 'close_p_H', 'close_p_D', 'close_p_A']
    close_available = [col for col in close_odds_cols if col in df.columns]
    
    if len(close_available) >= 6:
        close_coverage = df['close_p_H'].notna().sum()
        print(f"✅ Closing odds: {close_coverage:,} matches ({close_coverage/len(df)*100:.1f}%)")
        
        if 'close_source' in df.columns:
            top_close_sources = df['close_source'].value_counts().head(3)
            print(f"   Top sources: {dict(top_close_sources)}")
    else:
        print(f"⚠️  Limited closing odds for benchmarking")
    
    # 3. BOX SCORE STATISTICS VERIFICATION
    print(f"\n3️⃣ BOX SCORE STATISTICS VERIFICATION")
    print("-" * 30)
    
    box_score_cols = {
        'Shots': ['HS', 'AS', 'HST', 'AST'],
        'Cards': ['HY', 'AY', 'HR', 'AR', 'HBP', 'ABP'], 
        'Set Pieces': ['HC', 'AC', 'HF', 'AF'],
        'Advanced': ['HO', 'AO', 'HHW', 'AHW', 'HFKC', 'AFKC']
    }
    
    for category, cols in box_score_cols.items():
        available = [col for col in cols if col in df.columns]
        if available:
            avg_coverage = np.mean([df[col].notna().mean() for col in available])
            print(f"✅ {category}: {len(available)}/{len(cols)} cols, {avg_coverage*100:.1f}% coverage")
        else:
            print(f"❌ {category}: No columns available")
    
    # 4. TIME-SERIES VERIFICATION
    print(f"\n4️⃣ TIME-SERIES VERIFICATION")
    print("-" * 30)
    
    # Date range
    date_range = df['date'].max() - df['date'].min()
    print(f"✅ Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Total span: {date_range.days:,} days ({date_range.days/365.25:.1f} years)")
    
    # Season distribution
    if 'season' in df.columns:
        season_counts = df['season'].value_counts().sort_index()
        print(f"✅ Seasons: {len(season_counts)} seasons")
        print(f"   Range: {season_counts.index.min()} to {season_counts.index.max()}")
        
        # Check for reasonable season sizes (EPL = ~380 matches per season)
        unusual_seasons = season_counts[(season_counts < 300) | (season_counts > 500)]
        if len(unusual_seasons) > 0:
            print(f"⚠️  Unusual season sizes: {dict(unusual_seasons)}")
    
    # 5. DATA QUALITY VERIFICATION
    print(f"\n5️⃣ DATA QUALITY VERIFICATION")
    print("-" * 30)
    
    # Check for reasonable match outcomes
    if 'fulltime_result' in df.columns:
        outcome_dist = df['fulltime_result'].value_counts()
        print(f"✅ Match outcomes: {dict(outcome_dist)}")
        
        # Check if outcome distribution is reasonable (EPL: ~46% H, 27% D, 27% A)
        home_rate = outcome_dist.get('H', 0) / len(df)
        if 0.40 <= home_rate <= 0.52:
            print(f"   Home advantage: {home_rate:.1%} (reasonable)")
        else:
            print(f"⚠️  Home advantage: {home_rate:.1%} (unusual)")
    
    # Check for duplicate matches
    if 'match_id' in df.columns:
        duplicates = df['match_id'].duplicated().sum()
        if duplicates == 0:
            print("✅ No duplicate matches")
        else:
            print(f"⚠️  {duplicates} duplicate match_ids found")
    
    # Odds sanity checks
    if 'pre_odds_H' in df.columns:
        odds_cols = ['pre_odds_H', 'pre_odds_D', 'pre_odds_A']
        for col in odds_cols:
            if col in df.columns:
                valid_odds = df[col].between(1.01, 50.0)
                invalid_count = (~valid_odds & df[col].notna()).sum()
                if invalid_count > 0:
                    print(f"⚠️  {col}: {invalid_count} invalid odds values")
    
    # 6. PROFESSIONAL READINESS ASSESSMENT
    print(f"\n6️⃣ PROFESSIONAL READINESS ASSESSMENT")
    print("-" * 30)
    
    readiness_score = 0
    max_score = 6
    
    # Core data completeness
    if len(missing_core) == 0:
        readiness_score += 1
        print("✅ Core data structure complete")
    
    # Sufficient pre-closing odds for training
    if pre_coverage >= 1000:
        readiness_score += 1
        print("✅ Sufficient pre-closing odds for model training")
    elif pre_coverage >= 500:
        print("⚠️  Moderate pre-closing odds coverage")
    
    # Closing odds for benchmarking
    if close_coverage >= 500:
        readiness_score += 1
        print("✅ Sufficient closing odds for benchmarking")
    
    # Multiple seasons for time-series validation
    if len(season_counts) >= 3:
        readiness_score += 1
        print("✅ Multiple seasons for robust validation")
    
    # Rich match statistics
    total_box_stats = sum(len([col for col in cols if col in df.columns]) for cols in box_score_cols.values())
    if total_box_stats >= 8:
        readiness_score += 1
        print("✅ Rich match statistics for advanced features")
    elif total_box_stats >= 4:
        print("⚠️  Basic match statistics available")
    
    # Recent data for testing
    latest_date = df['date'].max()
    if pd.Timestamp.now() - latest_date < pd.Timedelta(days=365):
        readiness_score += 1
        print("✅ Recent data for realistic testing")
    
    # FINAL ASSESSMENT
    print(f"\n🎯 READINESS SCORE: {readiness_score}/{max_score}")
    
    if readiness_score >= 5:
        print("🚀 EXCELLENT! Ready for professional modeling")
        print("\nNext step: python implementation_guide.py " + csv_path)
        return True
    elif readiness_score >= 3:
        print("⚠️  GOOD! Ready for basic modeling with some limitations")
        print("\nRecommendation: Proceed with caution, consider getting more data")
        return True
    else:
        print("❌ NOT READY! Need to address data quality issues first")
        print("\nRecommendations:")
        if pre_coverage < 500:
            print("- Get more seasons with pre-closing odds")
        if len(season_counts) < 3:
            print("- Need at least 3 seasons for time-series validation")
        if total_box_stats < 4:
            print("- Consider adding match statistics data")
        return False

def quick_data_preview(csv_path: str):
    """Show a quick preview of your data structure"""
    print("\n📋 QUICK DATA PREVIEW")
    print("-" * 25)
    
    df = pd.read_csv(csv_path)
    
    # Show sample rows
    print("Sample matches:")
    sample_cols = ['date', 'home', 'away', 'fulltime_result', 'pre_source', 'close_source']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(3).to_string(index=False))
    
    # Show column summary
    print(f"\nAll columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        if i % 4 == 0:
            print()
        print(f"{col:20}", end="")
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data/raw/matches.csv"
        print(f"Using default path: {csv_path}")
        print("Usage: python data_verification.py <path_to_processed_csv>")
    
    success = verify_your_data(csv_path)
    
    if success:
        quick_data_preview(csv_path)
        print(f"\n🎉 Data verification complete! Ready to run the professional pipeline.")
    else:
        print(f"\n🔧 Please address the issues above before proceeding.")