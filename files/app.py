import pandas as pd
import streamlit as st

import tennis_model as tm

st.set_page_config(page_title="Sports Predictor", page_icon="🎾", layout="wide")

LABELS = {
    "elo": "Overall rating (Elo)",
    "surf_elo": "Surface-specific rating",
    "rank": "ATP ranking",
    "form": "Recent form",
    "fatigue": "Recent workload",
    "h2h": "Head-to-head",
    "height": "Height (serve edge)",
    "age": "Age",
}


@st.cache_resource(show_spinner="Loading ATP data and training the model (first run only)...")
def load():
    df = tm.load_matches()
    model, eng = tm.train_full(df)
    last = df["date"].max()
    recent = df[df["date"] >= last - pd.Timedelta(days=540)]
    ids = pd.unique(pd.concat([recent["winner_id"], recent["loser_id"]]))
    players = sorted(
        ((eng.info[i]["name"], i) for i in ids if eng.n[i] >= 10),
        key=lambda t: eng.last_rank.get(t[1], 9999),
    )
    return df, model, eng, players, last


@st.cache_data(show_spinner="Running backtest...")
def run_backtest():
    df = tm.load_matches()
    return tm.backtest(df, "2023-01-01")


def explain(name_a, name_b, p, contrib):
    fav, dog = (name_a, name_b) if p >= 0.5 else (name_b, name_a)
    sign = 1 if p >= 0.5 else -1
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
    reasons = []
    for k, v in top:
        side = fav if v * sign > 0 else dog
        reasons.append(f"{LABELS[k]} favours {side}")
    conf = max(p, 1 - p)
    level = "high" if conf >= 0.75 else "moderate" if conf >= 0.62 else "low"
    return (f"**{fav}** is the model's pick at **{conf:.0%}**, a {level}-confidence call. "
            f"Main drivers: {'; '.join(reasons)}.")


st.title("🎾 Sports Predictor")
tab_t, tab_c, tab_b = st.tabs(["Tennis (ATP)", "Cricket", "How accurate is it?"])

with tab_t:
    df, model, eng, players, last = load()
    st.caption(f"Data through {last.date()} · {len(df):,} matches · ratings update with every match")
    names = [n for n, _ in players]
    idmap = dict(players)
    c1, c2, c3 = st.columns([2, 2, 1])
    a_name = c1.selectbox("Player A", names, index=0)
    b_name = c2.selectbox("Player B", names, index=1)
    surface = c3.selectbox("Surface", ["Hard", "Clay", "Grass"])

    if a_name == b_name:
        st.warning("Pick two different players.")
    else:
        a, b = idmap[a_name], idmap[b_name]
        p, contrib = tm.predict(model, eng, a, b, surface)
        m1, m2 = st.columns(2)
        m1.metric(a_name, f"{p:.0%}")
        m2.metric(b_name, f"{1 - p:.0%}")
        st.progress(p)
        st.markdown(explain(a_name, b_name, p, contrib))

        left, right = st.columns(2)
        with left:
            st.subheader("What drives the pick")
            cdf = pd.DataFrame({"factor": [LABELS[k] for k in contrib], "edge for " + a_name: list(contrib.values())})
            st.bar_chart(cdf.set_index("factor"))
        with right:
            st.subheader("Player snapshot")
            rows = []
            for n, i in ((a_name, a), (b_name, b)):
                rows.append({
                    "Player": n,
                    "Rank": int(eng.last_rank.get(i, 0)) or "n/a",
                    "Elo": round(eng.elo[i]),
                    f"{surface} Elo": round(eng.selo[(surface, i)]),
                    "Last 10": f"{sum(eng.recent[i])}-{len(eng.recent[i]) - sum(eng.recent[i])}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.write(f"Head-to-head: **{a_name} {eng.h2h[(a, b)]} – {eng.h2h[(b, a)]} {b_name}** (since 2005)")

with tab_c:
    st.info("Cricket (T20 / IPL / ODI) is the next module: ball-by-ball data from Cricsheet, "
            "pre-match win probability plus live win probability.")

with tab_b:
    r = run_backtest()
    st.subheader("Honest accuracy (tested on matches the model never saw)")
    k1, k2, k3 = st.columns(3)
    k1.metric("Overall accuracy", f"{r['accuracy']:.1%}")
    k2.metric("Rank-only baseline", f"{r['baseline_rank_only']:.1%}")
    k3.metric("Matches tested", f"{r['matches']:,}")
    st.write("Accuracy by confidence band. A pick shown at 70-80% should win about that often:")
    st.dataframe(pd.DataFrame(
        [{"Model confidence": k, "Actual win rate": f"{v[0]:.1%}", "Matches": v[1]} for k, v in r["bands"].items()]),
        hide_index=True)
    st.caption("No model can honestly promise 90% on every match. The strong picks (80%+) are the reliable ones, "
               "but they are only a minority of matches.")

st.divider()
st.caption("Data: Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0, non-commercial). "
           "For information and entertainment only, not betting advice.")
