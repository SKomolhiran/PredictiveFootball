import argparse, yaml, pandas as pd

def backtest(cfg, date_from, date_to):
    # TODO: load historical predictions + closing odds and simulate bankroll
    print("📈 Backtest placeholder — implement EV gating & Kelly here.")
    # Write a skeleton equity curve CSV
    pd.DataFrame({"date": [], "bankroll": []}).to_csv("reports/equity_curve.csv", index=False)
    print("Wrote reports/equity_curve.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    backtest(cfg, args.date_from, args.date_to)
