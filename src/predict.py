import argparse, yaml, pandas as pd

def main(cfg, fixtures_path):
    # TODO: load saved models/calibrators, build features for fixtures, output P(H/D/A)
    print("🔮 Predict placeholder — implement after training is done.")
    out = pd.DataFrame({
        "match_id": [],
        "P_H": [], "P_D": [], "P_A": []
    })
    out.to_csv("predictions.csv", index=False)
    print("Wrote predictions.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixtures", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.fixtures)
