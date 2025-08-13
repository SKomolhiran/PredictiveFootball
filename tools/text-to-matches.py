# # save as tools/text_to_matches.py and run: python tools/text_to_matches.py
# from pathlib import Path
# import pandas as pd

# # 1) Paste your raw text between the triple quotes ↓↓↓
# RAW = r"""<PASTE THE WHOLE TEXT HERE>"""
# # Put season
# season = "2024-2025"

# Path("data/raw").mkdir(parents=True, exist_ok=True)
# with open(f"data/raw/epl_raw.csv", "w", encoding="utf-8") as f:
#     f.write(RAW)

# # Read raw, skip broken lines
# df = pd.read_csv("data/raw/epl_raw/{season}.csv", engine="python", on_bad_lines="skip")

# # Combine date+time and rename core columns
# df["date"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
#                             dayfirst=True, errors="coerce")
# df = df.rename(columns={
#     "HomeTeam":"home", "AwayTeam":"away",
#     "FTHG":"home_goals", "FTAG":"away_goals"
# })

# # Pick a bookmaker column set for odds (priority order)
# for H,D,A in [("PSH","PSD","PSA"), ("B365CH","B365CD","B365CA"),
#               ("WHH","WHD","WHA"), ("MaxH","MaxD","MaxA"), ("AvgH","AvgD","AvgA")]:
#     if {H,D,A}.issubset(df.columns):
#         df = df.rename(columns={H:"odds_H", D:"odds_D", A:"odds_A"})
#         break
# else:
#     raise ValueError("No known odds columns found (PSH/PSD/PSA or B365H/B365D/B365A etc.).")

# # Keep only the columns we use; basic sanity filters
# out = df[["date","home","away","home_goals","away_goals","odds_H","odds_D","odds_A"]].dropna()
# out = out[(out["odds_H"]>1.2) & (out["odds_D"]>1.2) & (out["odds_A"]>1.2)]
# out = out.sort_values("date").reset_index(drop=True)

# out.to_csv(f"data/raw/matches/{season}.csv", index=False)
# print(f"Wrote data/raw/matches/{season}.csv", len(out), "rows")

#!/usr/bin/env python3
"""
Aggregate Premier League seasons into a single training file focused on **Bet365** odds.

Usage
-----
python tools/text_to_matches.py \
    --input-glob "data/raw/seasons/*.csv" \
    --out "data/raw/matches.csv"

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

ODDS_PRE = ("B365H","B365D","B365A")
ODDS_CLOSE = ("B365CH","B365CD","B365CA")

def parse_season_from_filename(p: Path) -> str:
    m = re.search(r"(20\d{2})[^\d]?(20)?(\d{2})?", p.stem)
    if m:
        y1 = int(m.group(1))
        if m.group(2) and m.group(3):
            y2 = int(m.group(2) + m.group(3))
        else:
            y2 = y1 + 1
        return f"{y1}-{str(y2)[-2:]}"
    return ""

def season_from_date(dt: pd.Timestamp) -> str:
    y = dt.year
    if dt.month >= 7:
        y2 = (y + 1) % 100
        return f"{y}-{y2:02d}"
    else:
        y1 = y - 1
        y2 = y % 100
        return f"{y1}-{y2:02d}"

def devig_three_way(h, d, a):
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
    try:
        h=float(h); d=float(d); a=float(a)
    except Exception:
        return False
    return (lo <= h <= hi) and (lo <= d <= hi) and (lo <= a <= hi)

def preferred_time(t):
    if pd.isna(t) or str(t).strip() == "" or str(t).lower() == "nan":
        return "15:00"
    return str(t)

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
    for c in ["home","away","home_goals","away_goals","referee","attendance"]:
        if c not in df.columns: df[c] = np.nan

    # Box fields
    for col, _ in BOX_FIELDS:
        if col not in df.columns: df[col] = np.nan

    # Odds fields
    for k in list(ODDS_PRE) + list(ODDS_CLOSE):
        if k not in df.columns: df[k] = np.nan

    df["source_file"] = csv_path.name
    return df

def build_matches(files, out_csv, dropped_csv, vig_low=1.02, vig_high=1.25):
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

    # Map Bet365 odds snapshots
    pre = df[list(ODDS_PRE)].rename(columns={ODDS_PRE[0]:"pre_odds_H", ODDS_PRE[1]:"pre_odds_D", ODDS_PRE[2]:"pre_odds_A"})
    clo = df[list(ODDS_CLOSE)].rename(columns={ODDS_CLOSE[0]:"close_odds_H", ODDS_CLOSE[1]:"close_odds_D", ODDS_CLOSE[2]:"close_odds_A"})
    df = pd.concat([df, pre, clo], axis=1)

    # Compute probs & flags
    def compute_row(row):
        pre_valid = valid_odds_triplet(row["pre_odds_H"], row["pre_odds_D"], row["pre_odds_A"])
        clo_valid = valid_odds_triplet(row["close_odds_H"], row["close_odds_D"], row["close_odds_A"])
        out = {"pre_valid": pre_valid, "close_valid": clo_valid}
        if pre_valid:
            pH,pD,pA,qs = devig_three_way(row["pre_odds_H"], row["pre_odds_D"], row["pre_odds_A"])
            out.update({"pre_p_H":pH, "pre_p_D":pD, "pre_p_A":pA, "pre_qsum":qs})
        else:
            out.update({"pre_p_H":np.nan, "pre_p_D":np.nan, "pre_p_A":np.nan, "pre_qsum":np.nan})
        if clo_valid:
            cH,cD,cA,qs2 = devig_three_way(row["close_odds_H"], row["close_odds_D"], row["close_odds_A"])
            out.update({"close_p_H":cH, "close_p_D":cD, "close_p_A":cA, "close_qsum":qs2})
        else:
            out.update({"close_p_H":np.nan, "close_p_D":np.nan, "close_p_A":np.nan, "close_qsum":np.nan})
        return pd.Series(out)

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
        "home_goals","away_goals","referee","attendance",
        "pre_odds_H","pre_odds_D","pre_odds_A",
        "close_odds_H","close_odds_D","close_odds_A",
        "pre_p_H","pre_p_D","pre_p_A","pre_qsum","pre_vig_ok",
        "close_p_H","close_p_D","close_p_A","close_qsum","close_vig_ok",
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
    print(f"   Of kept rows, {n_close} have closing Bet365 odds (for baseline/CLV).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True, help="Glob for season CSVs, e.g., data/raw/seasons/*.csv")
    ap.add_argument("--out", default="data/raw/matches.csv", help="Output CSV path")
    ap.add_argument("--dropped", default="reports/dropped_rows.csv", help="Dropped rows report path")
    ap.add_argument("--vig-low", type=float, default=1.02, help="Min acceptable qsum (vig+1)")
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
