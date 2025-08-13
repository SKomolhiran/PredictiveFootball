import pandas as pd
import numpy as np

def compute_elo(df, k=20, home_adv=70):
    """Compute simple Elo ratings chronologically.
    df columns: date, home, away, home_goals, away_goals
    Returns: dataframe with pre-match Elo for home/away and Elo diff.
    """
    teams = pd.Index(sorted(set(df['home']).union(set(df['away']))))
    ratings = pd.Series(1500, index=teams, dtype=float)
    rows = []
    for _, row in df.sort_values('date').iterrows():
        Rh, Ra = ratings[row['home']], ratings[row['away']]
        Eh = 1 / (1 + 10 ** (-(Rh - Ra + home_adv) / 400))
        result = 1.0 if row['home_goals'] > row['away_goals'] else 0.5 if row['home_goals']==row['away_goals'] else 0.0
        ratings[row['home']] = Rh + k * (result - Eh)
        ratings[row['away']] = Ra + k * ((1-result) - (1-Eh))
        rows.append({
            "date": row['date'], "home": row['home'], "away": row['away'],
            "elo_home": Rh, "elo_away": Ra, "elo_diff": Rh - Ra
        })
    return pd.DataFrame(rows)
