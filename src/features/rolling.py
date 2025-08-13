import pandas as pd

def rolling_features(df, window=5):
    """Compute simple rolling stats per team prior to each match.
    df columns: date, team, goals_for, goals_against, points
    Returns team-date features to merge into matches.
    """
    df = df.sort_values(['team','date'])
    feats = df.groupby('team').rolling(window, on='date')[['goals_for','goals_against','points']].mean()
    feats = feats.reset_index()
    feats.rename(columns={
        'goals_for':'roll_gf', 'goals_against':'roll_ga', 'points':'roll_pts'
    }, inplace=True)
    return feats
