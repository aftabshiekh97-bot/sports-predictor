import pandas as pd
import streamlit as st

import live_tennis as lt
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
df, model, eng, players, last = load()
tab_t, tab_l, tab_c, tab_b = st.tabs(["Tennis (ATP)", "Live Tennis", "Cricket", "How accurate is it?"])

with tab_t:
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

with tab_l:
    st.subheader("Live match win probability")
    st.caption("Type in the current score. The app re-calculates the win chance for every point, "
               "starting from the pre-match model and the exact scoreboard.")
    names_l = [n for n, _ in players]
    id_l = dict(players)
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    la_name = c1.selectbox("Player A", names_l, index=0, key="live_a")
    lb_name = c2.selectbox("Player B", names_l, index=1, key="live_b")
    l_surface = c3.selectbox("Surface", ["Hard", "Clay", "Grass"], key="live_surf")
    l_bo = c4.selectbox("Best of", [3, 5], key="live_bo")

    if la_name == lb_name:
        st.warning("Pick two different players.")
    else:
        p_model, _ = tm.predict(model, eng, id_l[la_name], id_l[lb_name], l_surface)
        override = st.checkbox("Set the pre-match chance myself", value=False)
        p_pre = st.slider(f"Pre-match chance for {la_name}", 0.05, 0.95, float(round(p_model, 2)), 0.01) \
            if override else p_model
        st.write(f"Pre-match chance for **{la_name}**: **{p_pre:.0%}**")

        st.markdown("##### Current score")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Sets won**")
            sa = st.number_input(la_name, 0, 2 if l_bo == 3 else 3, 0, key="sa")
            sb = st.number_input(lb_name, 0, 2 if l_bo == 3 else 3, 0, key="sb")
        with s2:
            st.markdown("**Games in this set**")
            ga = st.number_input(la_name, 0, 7, 0, key="ga")
            gb = st.number_input(lb_name, 0, 7, 0, key="gb")
        with s3:
            st.markdown("**Server**")
            server = st.radio("Who is serving now?", [la_name, lb_name], key="server")
        a_serves = server == la_name

        in_tb = False
        if ga == 6 and gb == 6:
            in_tb = st.checkbox("Tiebreak in progress", value=True)
        p1, p2 = st.columns(2)
        if in_tb:
            pa = p1.number_input(f"Tiebreak points - {la_name}", 0, 30, 0, key="tba")
            pb = p2.number_input(f"Tiebreak points - {lb_name}", 0, 30, 0, key="tbb")
        else:
            opts = ["0", "15", "30", "40", "AD"]
            pla = p1.selectbox(f"Game points - {la_name}", opts, key="pla")
            plb = p2.selectbox(f"Game points - {lb_name}", opts, key="plb")
            pa = pb = 0
            bad_pts = (pla == "AD" and plb != "40") or (plb == "AD" and pla != "40")
            if bad_pts:
                st.warning("Advantage means the other player is on 40.")
            else:
                pa, pb = lt.points_from_labels(pla, plb)

        need = 2 if l_bo == 3 else 3
        if sa >= need or sb >= need:
            st.success(f"Match finished: {la_name if sa >= need else lb_name} won.")
        else:
            engine = lt.calibrate(p_pre, l_surface, l_bo)
            pl = engine.live(sa, sb, ga, gb, a_serves, pa, pb, in_tb)
            m1, m2 = st.columns(2)
            m1.metric(la_name, f"{pl:.1%}", f"{(pl - p_pre) * 100:+.1f} pts vs pre-match")
            m2.metric(lb_name, f"{1 - pl:.1%}")
            st.progress(float(pl))
            lead = la_name if pl >= 0.5 else lb_name
            st.markdown(f"**{lead}** is the favourite right now at **{">99%" if max(pl, 1 - pl) > 0.995 else format(max(pl, 1 - pl), ".0%")}**.")

            if "live_log" not in st.session_state:
                st.session_state.live_log = []
            b1, b2 = st.columns(2)
            if b1.button("Log this score to the timeline"):
                label = f"{sa}-{sb} sets, {ga}-{gb} games" + (f", TB {pa}-{pb}" if in_tb else f", {pla}-{plb}")
                st.session_state.live_log.append({"Score": label, la_name: round(pl * 100, 1)})
            if b2.button("Clear timeline"):
                st.session_state.live_log = []
            if st.session_state.live_log:
                log = pd.DataFrame(st.session_state.live_log)
                st.line_chart(log[la_name].reset_index(drop=True))
                st.dataframe(log, hide_index=True, width="stretch")
            st.caption("Model note: final sets are treated like normal sets with a 7-point tiebreak at 6-6; "
                       "serve strength is estimated from the pre-match probability and the surface. "
                       "Enter the score every few games for the best view of momentum.")

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
