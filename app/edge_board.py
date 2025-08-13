import streamlit as st
import pandas as pd

st.title("Football Edge Board (MVP)")

st.markdown("""Upload predictions and odds to see edges and suggested stakes.""")

preds = st.file_uploader("Predictions CSV (match_id, P_H, P_D, P_A)", type=["csv"])
odds = st.file_uploader("Odds CSV (match_id, odds_H, odds_D, odds_A)", type=["csv"])

ev_thresh = st.slider("EV threshold", 0.0, 0.1, 0.02, 0.005)
kelly_frac = st.slider("Kelly fraction", 0.0, 1.0, 0.5, 0.05)

if preds and odds:
    p = pd.read_csv(preds)
    o = pd.read_csv(odds)
    df = p.merge(o, on="match_id", how="inner")

    for col in ["odds_H","odds_D","odds_A"]:
        df[f"p_book_{col[-1]}"] = (1/df[col])
    s = df[["p_book_H","p_book_D","p_book_A"]].sum(axis=1)
    for k in ["H","D","A"]:
        df[f"p_book_{k}"] = df[f"p_book_{k}"] / s

    def ev(p, odds): return odds * p - 1.0

    for k in ["H","D","A"]:
        df[f"EV_{k}"] = ev(df[f"P_{k}"], df[f"odds_{k}"])

    long = df.melt(id_vars=["match_id"], value_vars=["EV_H","EV_D","EV_A"], var_name="side", value_name="EV")
    long["side"]=long["side"].str[-1]

    bets = long[long["EV"] > ev_thresh].copy()
    bets = bets.merge(df, left_on="match_id", right_on="match_id")
    st.subheader("Bets (EV-filtered)")
    st.dataframe(bets[["match_id","side","EV","P_H","P_D","P_A","odds_H","odds_D","odds_A"]].sort_values("EV", ascending=False))
else:
    st.info("Upload predictions and odds to see suggested bets.")
