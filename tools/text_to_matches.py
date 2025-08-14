"""
Aggregate Premier League seasons into a single training file focused on **Bet365** odds.

Usage
-----
python tools/text_to_matches.py \
    --input-glob "data/raw/seasons/*.csv" \
    --out "data/processed/matches.csv"

If you have pasted CSV text files, place them in data/raw/seasons/ as .csv and point --input-glob at them.
This script will:
  - parse each season file
  - keep key box-score fields (for future rolling/lag features)
  - pick **Bet365** pre-close odds (B365H/D/A) and closing odds (B365CH/CD/CA) when available
  - compute de-vigged implied probabilities for both snapshots
  - derive a season label
  - validate odds ranges & vig; log dropped rows to reports/dropped_rows.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

# --------------------------------------------
# Helpers
# --------------------------------------------

BOX_FIELDS = [
    ("HS", "int"), ("AS", "int"),
    ("HST","int"), ("AST","int"),
    ("HF","int"), ("AF","int"),
    ("HY","int"), ("AY","int"),
    ("HR","int"), ("AR","int"),
    ("HC","int"), ("AC","int"),
    ("HFKC","int"), ("AFKC","int"),
    ("HO","int"), ("AO","int"),
    ("HHW","int"), ("AHW","int"),
    ("HBP","int"), ("ABP","int"),
]

CORE_MAP = {
    "Date":"date",
    "Time":"time",
    "HomeTeam":"home",
    "AwayTeam":"away",
    "FTHG":"home_goals",
    "FTAG":"away_goals",
    "FTR":"fulltime_result",
    "Referee":"referee",
    "Attendance":"attendance",
}

BOOKS_PRE = [
    ("B365H","B365D","B365A","Bet365"),
    ("PSH","PSD","PSA","Pinnacle"),
    ("WHH","WHD","WHA","WilliamHill"),
    ("1XBH","1XBD","1XBA","1xBet"),
    ("GBH","GBD","GBA","Gamebookers"),
    ("AvgH","AvgD","AvgA","Avg"),
    ("MaxH","MaxD","MaxA","Max"),
]
BOOKS_CLOSE = [   
    ("PSCH","PSCD","PSCA","Pinnacle"),
    ("B365CH","B365CD","B365CA","Bet365"),
    ("WHCH","WHCD","WHCA","WilliamHill"),
    ("1XBCH","1XBCD","1XBCA","1xBet"),
    ("AvgCH","AvgCD","AvgCA","Avg"),
    ("MaxCH","MaxCD","MaxCA","Max"),
]

# Flatten to the full set of column names we should accept
ALL_ODDS_COLS = sorted({
    *[c for triplet in BOOKS_PRE for c in triplet[:3]],
    *[c for triplet in BOOKS_CLOSE for c in triplet[:3]],
})

def parse_season_from_filename(p: Path) -> str:
    """
    Expect filenames like: PREFIX_YY_YY.csv
    Examples:
      EPL_23_24.csv  -> '2023-24'
      PL_19_20.csv   -> '2019-20'
    """
    m = re.search(r'(\d{2})_(\d{2})', p.stem)
    if not m:
        return ""  # fall back to date-based season later
    start_yy = int(m.group(1))
    end_yy   = int(m.group(2))
    start_year = 2000 + start_yy  # assume 20xx seasons
    # end_yy is already the "last two digits" for display
    return f"{start_year}-{end_yy:02d}"

def season_from_date(dt: pd.Timestamp) -> str:
    """
    Determine the match's season
    """
    y = dt.year
    if dt.month >= 7:
        y2 = (y + 1) % 100
        return f"{y}-{y2:02d}"
    else:
        y1 = y - 1
        y2 = y % 100
        return f"{y1}-{y2:02d}"

def devig_three_way(h, d, a):
    """
    Derive probabilities from odds
    """
    try:
        h=float(h); d=float(d); a=float(a)
        qh, qd, qa = 1/h, 1/d, 1/a
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    s = qh + qd + qa
    if s == 0 or not np.isfinite(s):
        return np.nan, np.nan, np.nan, np.nan
    return qh/s, qd/s, qa/s, s

def valid_odds_triplet(h, d, a, lo=1.01, hi=200.0):
    """
    Return if odds are valid
    """
    try:
        h=float(h); d=float(d); a=float(a)
    except Exception:
        return False
    return (lo <= h <= hi) and (lo <= d <= hi) and (lo <= a <= hi)

def preferred_time(t):
    if pd.isna(t) or str(t).strip() == "" or str(t).lower() == "nan":
        return "15:00"
    return str(t)

def choose_odds(row, priority):
    """Return the first available (H,D,A,source_name) from a priority list."""
    for H, D, A, src in priority:
        h, d, a = row.get(H), row.get(D), row.get(A)
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            return pd.Series({"H": float(h), "D": float(d), "A": float(a), "source": src})
    # nothing found
    return pd.Series({"H": np.nan, "D": np.nan, "A": np.nan, "source": None})

def load_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    # Rename core
    keep = {c:CORE_MAP[c] for c in CORE_MAP if c in df.columns}
    df = df.rename(columns=keep)

    # Parse datetime
    if "date" not in df.columns:
        raise ValueError(f"{csv_path} missing 'Date'")
    df["time"] = df["time"].apply(preferred_time) if "time" in df.columns else "15:00"
    df["date"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                                dayfirst=True, errors="coerce")

    # Coerce attendance to numeric
    if "attendance" in df.columns:
        df["attendance"] = (df["attendance"].astype(str).str.replace(",","",regex=False)
                              .replace({"nan":np.nan}).astype(float))

    # Attach season
    season_label = parse_season_from_filename(csv_path)
    if "season" not in df.columns:
        df["season"] = season_label
    df.loc[df["season"]=="", "season"] = df.loc[df["season"]=="", "date"].apply(season_from_date)

    # Ensure required columns exist
    for c in ["home","away","home_goals","away_goals","fulltime_result","referee","attendance"]:
        if c not in df.columns: df[c] = np.nan

    # Box fields
    for col, _ in BOX_FIELDS:
        if col not in df.columns: df[col] = np.nan

    # 👉 Handle odds: create missing cols and coerce present ones to numeric
    for col in ALL_ODDS_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            # convert e.g. strings like '2.5' to float; bad values -> NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Attach source
    df["source_file"] = csv_path.name
    return df

def build_matches(files, out_csv, dropped_csv, vig_low=1.00, vig_high=1.25):
    frames = []
    for f in files:
        try:
            frames.append(load_one(Path(f)))
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    if not frames:
        raise SystemExit("No valid input files found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    # pick best-available pre-close and closing odds per row
    pre_sel = df.apply(lambda r: choose_odds(r, BOOKS_PRE), axis=1)
    pre_sel = pre_sel.rename(columns={"H":"pre_odds_H","D":"pre_odds_D","A":"pre_odds_A","source":"pre_source"})

    close_sel = df.apply(lambda r: choose_odds(r, BOOKS_CLOSE), axis=1)
    close_sel = close_sel.rename(columns={"H":"close_odds_H","D":"close_odds_D","A":"close_odds_A","source":"close_source"})

    df = pd.concat([df, pre_sel, close_sel], axis=1)


    # Compute probs & flags
    def compute_row(row):
        pre_valid = valid_odds_triplet(row["pre_odds_H"], row["pre_odds_D"], row["pre_odds_A"])
        clo_valid = valid_odds_triplet(row["close_odds_H"], row["close_odds_D"], row["close_odds_A"])
        # Start with defaults
        res = {
            "pre_valid": bool(pre_valid),
            "close_valid": bool(clo_valid),
            "pre_p_H": np.nan, "pre_p_D": np.nan, "pre_p_A": np.nan, "pre_qsum": np.nan,
            "close_p_H": np.nan, "close_p_D": np.nan, "close_p_A": np.nan, "close_qsum": np.nan,
        }

        # Pre-close probs
        if pre_valid:
            pH, pD, pA, qs = devig_three_way(row["pre_odds_H"], row["pre_odds_D"], row["pre_odds_A"])
            res["pre_p_H"] = pH
            res["pre_p_D"] = pD
            res["pre_p_A"] = pA
            res["pre_qsum"] = qs

        # Closing probs
        if clo_valid:
            cH, cD, cA, qs2 = devig_three_way(row["close_odds_H"], row["close_odds_D"], row["close_odds_A"])
            res["close_p_H"] = cH
            res["close_p_D"] = cD
            res["close_p_A"] = cA
            res["close_qsum"] = qs2

        return pd.Series(res)

    probs = df.apply(compute_row, axis=1)
    df = pd.concat([df, probs], axis=1)

    df["pre_vig_ok"] = df["pre_qsum"].between(vig_low, vig_high)
    df["close_vig_ok"] = df["close_qsum"].between(vig_low, vig_high)

    # match_id
    df["match_id"] = (df["season"].fillna("")
                        + "_" + df["date"].dt.strftime("%Y%m%d%H%M").fillna("")
                        + "_" + df["home"].astype(str).str.replace(r"\s+","",regex=True)
                        + "_vs_"
                        + df["away"].astype(str).str.replace(r"\s+","",regex=True))

    keep_cols = [
        "match_id","season","date","home","away",
        "home_goals","away_goals","fulltime_result","referee","attendance",
        "pre_odds_H","pre_odds_D","pre_odds_A","pre_source",
        "close_odds_H","close_odds_D","close_odds_A","close_source",
        "pre_p_H","pre_p_D","pre_p_A","pre_qsum","pre_vig_ok",
        "close_p_H","close_p_D","close_p_A","close_qsum","close_vig_ok"
    ]
    for col, _ in BOX_FIELDS:
        keep_cols.append(col)
    keep_cols.append("source_file")

    good_mask = df["pre_valid"] & df["pre_vig_ok"]
    dropped = df.loc[~good_mask, keep_cols].copy()
    good = df.loc[good_mask, keep_cols].copy()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    good.to_csv(out_csv, index=False)

    Path("reports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dropped).to_csv(dropped_csv, index=False)

    n_all = len(df); n_good = len(good); n_drop = len(dropped)
    n_close = int(good["close_odds_H"].notna().sum())
    print(f"✅ Wrote {out_csv} with {n_good}/{n_all} rows (dropped {n_drop}).")
    print(f"   Of kept rows, {n_close} have closing Pinnacle odds (for baseline/CLV).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True, help="Glob for season CSVs, e.g., data/raw/seasons/*.csv")
    ap.add_argument("--out", default="data/processed/matches.csv", help="Output CSV path")
    ap.add_argument("--dropped", default="reports/dropped_rows.csv", help="Dropped rows report path")
    ap.add_argument("--vig-low", type=float, default=1.00, help="Min acceptable qsum (vig+1)")
    ap.add_argument("--vig-high", type=float, default=1.25, help="Max acceptable qsum (vig+1)")
    args = ap.parse_args()

    # Glob files relative to repo root
    base = Path(".")
    files = sorted(base.glob(args.input_glob)) if "*" in args.input_glob else [Path(args.input_glob)]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(f"No files matched {args.input_glob}")

    build_matches(files, args.out, args.dropped, args.vig_low, args.vig_high)

if __name__ == "__main__":
    main()
