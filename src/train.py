import argparse, os, yaml, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

# Local utilities
from src.features.elo import compute_elo
from src.eval.metrics import rps

# -------------------------------
# Helpers
# -------------------------------

REQUIRED_COLS = ["date","home","away","home_goals","away_goals","odds_H","odds_D","odds_A"]

def load_matches(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in matches CSV: {missing}")
    df["date"] = pd.to_datetime(df["date"])  # parse
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=REQUIRED_COLS)
    return df

def add_book_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    for k in ["H","D","A"]:
        df[f"q_{k}"] = 1.0 / df[f"odds_{k}"]
    s = df[["q_H","q_D","q_A"]].sum(axis=1)
    for k in ["H","D","A"]:
        df[f"p_book_{k}"] = df[f"q_{k}"] / s
    return df

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    # 0=Home, 1=Draw, 2=Away
    outcome = np.where(df["home_goals"] > df["away_goals"], 0,
               np.where(df["home_goals"] < df["away_goals"], 2, 1))
    df["y_class"] = outcome
    df["y_home_win"] = (outcome == 0).astype(int)  # binary label for home win
    return df

def year_folds(df: pd.DataFrame, years_per_fold: int = 1, min_train_years: int = 3):
    """Generate expanding-window CV folds grouped by calendar year.

    Yields tuples: (train_idx, val_idx, train_years, val_years).
    Keeps at least `min_train_years` in train before first validation.
    """
    years = sorted(df["date"].dt.year.unique().tolist())
    idx_by_year = {y: df.index[df["date"].dt.year == y].to_numpy() for y in years}
    folds = []
    for i in range(min_train_years, len(years)):
        val_years = years[i : i + years_per_fold]
        if not val_years:
            continue
        train_years = years[:i]
        train_idx = np.concatenate([idx_by_year[y] for y in train_years]) if train_years else np.array([], dtype=int)
        val_idx = np.concatenate([idx_by_year[y] for y in val_years])
        folds.append((train_idx, val_idx, train_years, val_years))
    return folds

def summarize_metrics(rows, out_md_path):
    dfm = pd.DataFrame(rows)
    piv = (dfm.pivot_table(index=["task","fold"],
                           values=["logloss","brier","rps"],
                           aggfunc="first")
                  .sort_index())
    # Write markdown report
    lines = []
    lines.append("# Baseline Report — Elo + Logistic Regression")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    for task in sorted(dfm["task"].unique()):
        lines.append(f"## Task: {task}")
        sub = dfm[dfm["task"]==task]
        for _, row in sub.iterrows():
            rps_text = "" if pd.isna(row['rps']) else f" (rps: {row['rps']:.5f})"
            lines.append(f"- **Fold {row['fold']}** — logloss: {row['logloss']:.5f} | brier: {row['brier']:.5f}{rps_text}")
        lines.append("")
    lines.append("### Averages")
    for task in sorted(dfm["task"].unique()):
        sub = dfm[dfm["task"]==task]
        avg_ll = sub['logloss'].mean()
        avg_br = sub['brier'].mean()
        avg_rps = sub['rps'].mean() if not sub['rps'].isna().all() else np.nan
        rps_text = "" if pd.isna(avg_rps) else f" (rps: {avg_rps:.5f})"
        lines.append(f"- **{task}** — logloss: {avg_ll:.5f} | brier: {avg_br:.5f}{rps_text}")
    Path(out_md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return dfm

# -------------------------------
# Main training pipeline
# -------------------------------

def main(cfg):
    matches_csv = cfg["data"]["matches_csv"]
    model_dir = Path(cfg["training"]["model_dir"])
    start_date = pd.to_datetime(cfg["training"]["start_date"]) if cfg["training"].get("start_date") else None
    end_date = pd.to_datetime(cfg["training"]["end_date"]) if cfg["training"].get("end_date") else None
    cv_y_per_fold = int(cfg["training"].get("cv_years_per_fold", 1))

    print(f"Loading matches from {matches_csv} ...")
    df = load_matches(matches_csv)
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    df = df.copy()
    print(f"Rows in date range: {len(df)}")

    # Book probabilities (baseline comparator)
    df = add_book_probabilities(df)

    # Elo features (pre-match)
    print("Computing Elo features ...")
    elo = compute_elo(df[["date","home","away","home_goals","away_goals"]].copy())
    df = df.merge(elo, on=["date","home","away"], how="left")

    # Labels
    df = build_labels(df)

    # Feature matrix (keep it minimal for baseline)
    feat_cols = ["elo_home","elo_away","elo_diff"]
    X_all = df[feat_cols].to_numpy()
    y_bin = df["y_home_win"].to_numpy()
    y_multi = df["y_class"].to_numpy()

    # Pipelines
    pipe_bin = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])
    pipe_multi = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"))
    ])

    # Expanding-window CV
    rows = []
    folds = year_folds(df, years_per_fold=cv_y_per_fold, min_train_years=3)
    if not folds:
        print("Not enough data years for CV folds; training on whole range and skipping CV metrics.")
    else:
        print(f"Running {len(folds)} expanding-window folds ...")
    fold_id = 0
    for train_idx, val_idx, train_years, val_years in folds:
        fold_id += 1
        X_tr, X_va = X_all[train_idx], X_all[val_idx]
        yb_tr, yb_va = y_bin[train_idx], y_bin[val_idx]
        ym_tr, ym_va = y_multi[train_idx], y_multi[val_idx]

        # Binary (home win vs not-win)
        pipe_bin.fit(X_tr, yb_tr)
        proba_b = pipe_bin.predict_proba(X_va)  # [:,1] is P(home win)
        ll_b = log_loss(yb_va, proba_b, labels=[0,1])
        br_b = brier_score_loss(yb_va, proba_b[:,1])
        rows.append({
            "task":"binary_home_win",
            "fold": fold_id,
            "train_years": f"{train_years[0]}–{train_years[-1]}",
            "val_years": f"{val_years[0]}–{val_years[-1]}",
            "logloss": ll_b,
            "brier": br_b,
            "rps": np.nan
        })

        # Multiclass H/D/A
        pipe_multi.fit(X_tr, ym_tr)
        proba_m = pipe_multi.predict_proba(X_va)
        ll_m = log_loss(ym_va, proba_m, labels=[0,1,2])
        o = np.eye(3)[ym_va]
        br_m = np.mean(np.sum((proba_m - o)**2, axis=1))
        rps_m = rps(proba_m, ym_va)
        rows.append({
            "task":"multiclass_HDA",
            "fold": fold_id,
            "train_years": f"{train_years[0]}–{train_years[-1]}",
            "val_years": f"{val_years[0]}–{val_years[-1]}",
            "logloss": ll_m,
            "brier": br_m,
            "rps": rps_m
        })

    # Write report
    report_path = "reports/baseline_report.md"
    if rows:
        print("Writing CV metrics to", report_path)
        summarize_metrics(rows, report_path)

    # Fit final models on all in-range data
    print("Fitting final models on full training range ...")
    pipe_bin.fit(X_all, y_bin)
    pipe_multi.fit(X_all, y_multi)

    # Save artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe_bin, "features": feat_cols}, model_dir / "logreg_binary.pkl")
    joblib.dump({"pipeline": pipe_multi, "features": feat_cols}, model_dir / "logreg_multiclass.pkl")

    # Save a small snapshot for sanity
    snap_cols = ["date","home","away","home_goals","away_goals"] + feat_cols + ["y_home_win","y_class"]
    df[snap_cols].tail(10).to_csv(model_dir / "feature_snapshot.csv", index=False)

    print("✅ Done. Saved models to", model_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    main(cfg)
