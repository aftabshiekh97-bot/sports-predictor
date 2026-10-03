import pandas as pd
import streamlit as st

import live_tennis as lt
import scoreboard as sb
import tennis_model as tm

st.set_page_config(page_title="PredictScore", page_icon="🎾", layout="wide")
st.markdown(sb.CSS, unsafe_allow_html=True)

LABELS = {"elo": "Overall rating (Elo)", "surf_elo": "Surface-specific rating", "rank": "Ranking",
          "form": "Recent form", "fatigue": "Recent workload", "h2h": "Head-to-head",
          "height": "Height (serve edge)", "age": "Age"}


@st.cache_resource(show_spinner="Loading ATP + WTA data and training models (first run only, ~1-2 min)...")
def load_all():
    out = {}
    for tour in ("atp", "wta"):
        df = tm.load_matches(tour=tour)
        model, eng = tm.train_full(df)
        last = df["date"].max()
        rec = df[df["date"] >= last - pd.Timedelta(days=540)]
        ids = pd.unique(pd.concat([rec["winner_id"], rec["loser_id"]]))
        players = sorted(((eng.info[i]["name"], i) for i in ids if eng.n[i] >= 10),
                         key=lambda t: eng.last_rank.get(t[1], 9999))
        out[tour] = dict(model=model, eng=eng, players=players, last=last, n=len(df), matcher=sb.Matcher(players))
    return out


@st.cache_data(ttl=120, show_spinner=False)
def api_matches(status, key):
    return sb.fetch(status, key)


@st.cache_data(show_spinner="Running backtests...")
def backtests():
    return {t: tm.backtest(tm.load_matches(tour=t), "2023-01-01") for t in ("atp", "wta")}


def explain(na, nb, p, contrib):
    fav, dog = (na, nb) if p >= 0.5 else (nb, na)
    sign = 1 if p >= 0.5 else -1
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
    why = "; ".join(f"{LABELS[k]} favours {fav if v * sign > 0 else dog}" for k, v in top)
    conf = max(p, 1 - p)
    lvl = "high" if conf >= 0.75 else "moderate" if conf >= 0.62 else "low"
    return f"**{fav}** is the pick at **{conf:.0%}** ({lvl} confidence). Main drivers: {why}."


ALL = load_all()
st.markdown('<div class="sb-title">🎾 PredictScore <span class="sb-muted">live scores + win predictions</span></div>',
            unsafe_allow_html=True)
tab_s, tab_p, tab_l, tab_c, tab_b = st.tabs(["Live Scores", "Match Predictor", "Manual Live", "Cricket", "Accuracy"])

# ====================================================================== LIVE SCORES
with tab_s:
    try:
        api_key = st.secrets.get("LIVETENNIS_API_KEY", "")
    except Exception:
        api_key = ""
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    status = f1.radio("Status", ["Live", "Upcoming"], horizontal=True, label_visibility="collapsed")
    tour_f = f2.radio("Tour", ["All", "ATP", "WTA"], horizontal=True, label_visibility="collapsed")
    tz = f3.selectbox("Time zone", ["Asia/Kolkata", "UTC", "Europe/London", "America/New_York", "Asia/Dubai"],
                      label_visibility="collapsed")
    if f4.button("Refresh"):
        api_matches.clear()

    st_key = status.lower()
    err = None
    if api_key:
        try:
            matches = api_matches(st_key, api_key)
        except Exception as e:
            matches, err = [], str(e)
    else:
        matches = [m for m in sb.demo_matches() if m["status"] == st_key]
        st.info("Demo data shown. Add your free API key to see real matches (steps in the README).")
    if err:
        st.error(err)
    if tour_f != "All":
        matches = [m for m in matches if (m.get("tour") or "").lower() == tour_f.lower()]
    matches = sorted(matches, key=lambda m: ((m.get("tour") or "z"), str(m.get("tournament")), str(m.get("scheduled_time"))))

    probs = {}
    for m in matches:
        ctx = ALL.get((m.get("tour") or "atp").lower(), ALL["atp"])
        probs[m["id"]] = sb.win_probs(m, ctx, ctx["matcher"])

    left, right = st.columns([1.15, 1])
    with left:
        if not matches:
            st.write("No matches right now." if st_key == "live" else "No upcoming matches found.")
        else:
            st.caption(f"{len(matches)} matches · probabilities are for the top player of each pair")
            st.markdown(sb.list_html(matches, probs, tz), unsafe_allow_html=True)

    with right:
        if matches:
            labels = {m["id"]: " vs ".join(sb.players_of(m)) for m in matches}
            mid = st.selectbox("Match details", list(labels), format_func=labels.get)
            m = next(x for x in matches if x["id"] == mid)
            a, b = sb.players_of(m)
            sc = sb.parse_score(m)
            p_pre, p_live, known = probs[mid]
            p = p_live if p_live is not None else p_pre
            st.markdown(
                f'<div class="sb-card"><div class="sb-muted">{m.get("tournament", "")} · {str(m.get("round") or "").split(" - ")[-1]}'
                f' · {(m.get("surface") or "").capitalize()}</div>'
                f'<div style="display:flex;justify-content:space-between;margin-top:10px"><div><b>{a}</b><br><b>{b}</b></div>'
                f'<div class="sb-big">{sc["sets"][0]}<br>{sc["sets"][1]}</div></div></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric(a, f"{p:.0%}", f"{(p - p_pre) * 100:+.1f} vs pre-match" if p_live is not None else None)
            c2.metric(b, f"{1 - p:.0%}")
            st.progress(float(p))
            st.caption("Live win probability from the exact score" if p_live is not None else "Pre-match prediction")
            if not known:
                st.warning("One or both players are not in the rating database, so the start point is 50/50. "
                           "Live probability still follows the scoreboard.")
            if p_live is not None:                                   # build a momentum timeline across refreshes
                hist = st.session_state.setdefault("hist", {}).setdefault(mid, [])
                stamp = (tuple(sc["sets"]), tuple(sc["per_set"]), tuple(sc["points"]))
                if not hist or hist[-1][0] != stamp:
                    hist.append((stamp, round(p_live * 100, 1)))
                if len(hist) > 1:
                    st.caption("Momentum (win % for " + a + ") - builds up each time you refresh")
                    st.line_chart(pd.Series([h[1] for h in hist], name=a))
            with st.expander("Raw data from API (for debugging)"):
                st.json(m)

# ====================================================================== PREDICTOR
with tab_p:
    tk = st.radio("Tour", ["ATP", "WTA"], horizontal=True, key="pt").lower()
    ctx = ALL[tk]
    eng, model, players = ctx["eng"], ctx["model"], ctx["players"]
    st.caption(f"Data through {ctx['last'].date()} · {ctx['n']:,} matches")
    names, idmap = [n for n, _ in players], dict(players)
    c1, c2, c3 = st.columns([2, 2, 1])
    na = c1.selectbox("Player A", names, index=0, key="pa")
    nb = c2.selectbox("Player B", names, index=1, key="pb")
    surf = c3.selectbox("Surface", ["Hard", "Clay", "Grass"], key="ps")
    if na == nb:
        st.warning("Pick two different players.")
    else:
        a, b = idmap[na], idmap[nb]
        p, contrib = tm.predict(model, eng, a, b, surf)
        m1, m2 = st.columns(2)
        m1.metric(na, f"{p:.0%}")
        m2.metric(nb, f"{1 - p:.0%}")
        st.progress(p)
        st.markdown(explain(na, nb, p, contrib))
        l, r = st.columns(2)
        with l:
            st.subheader("What drives the pick")
            st.bar_chart(pd.DataFrame({"factor": [LABELS[k] for k in contrib], "edge for " + na: list(contrib.values())}).set_index("factor"))
        with r:
            st.subheader("Player snapshot")
            rows = [{"Player": n, "Rank": int(eng.last_rank.get(i, 0)) or "n/a", "Elo": round(eng.elo[i]),
                     f"{surf} Elo": round(eng.selo[(surf, i)]),
                     "Last 10": f"{sum(eng.recent[i])}-{len(eng.recent[i]) - sum(eng.recent[i])}"} for n, i in ((na, a), (nb, b))]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.write(f"Head-to-head: **{na} {eng.h2h[(a, b)]} - {eng.h2h[(b, a)]} {nb}** (since 2005)")

# ====================================================================== MANUAL LIVE
with tab_l:
    st.subheader("Manual live calculator")
    st.caption("For any match not on the scoreboard: type the score and get the win probability.")
    tk2 = st.radio("Tour", ["ATP", "WTA"], horizontal=True, key="lt").lower()
    ctx = ALL[tk2]
    names_l, id_l = [n for n, _ in ctx["players"]], dict(ctx["players"])
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    la = c1.selectbox("Player A", names_l, index=0, key="live_a")
    lb = c2.selectbox("Player B", names_l, index=1, key="live_b")
    lsurf = c3.selectbox("Surface", ["Hard", "Clay", "Grass"], key="live_surf")
    lbo = c4.selectbox("Best of", [3, 5], key="live_bo")
    if la == lb:
        st.warning("Pick two different players.")
    else:
        p_model, _ = tm.predict(ctx["model"], ctx["eng"], id_l[la], id_l[lb], lsurf)
        ov = st.checkbox("Set the pre-match chance myself")
        p_pre = st.slider(f"Pre-match chance for {la}", 0.05, 0.95, float(round(p_model, 2)), 0.01) if ov else p_model
        st.write(f"Pre-match chance for **{la}**: **{p_pre:.0%}**")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Sets won**")
            sa = st.number_input(la, 0, 2 if lbo == 3 else 3, 0, key="sa")
            sbb = st.number_input(lb, 0, 2 if lbo == 3 else 3, 0, key="sb")
        with s2:
            st.markdown("**Games in this set**")
            ga = st.number_input(la, 0, 7, 0, key="ga")
            gb = st.number_input(lb, 0, 7, 0, key="gb")
        with s3:
            st.markdown("**Server**")
            server = st.radio("Who is serving now?", [la, lb], key="server")
        a_serves = server == la
        in_tb = st.checkbox("Tiebreak in progress", value=True) if (ga == 6 and gb == 6) else False
        q1, q2 = st.columns(2)
        pa = pb = 0
        if in_tb:
            pa = q1.number_input(f"Tiebreak points - {la}", 0, 30, 0, key="tba")
            pb = q2.number_input(f"Tiebreak points - {lb}", 0, 30, 0, key="tbb")
        else:
            opts = ["0", "15", "30", "40", "AD"]
            pla = q1.selectbox(f"Game points - {la}", opts, key="pla")
            plb = q2.selectbox(f"Game points - {lb}", opts, key="plb")
            if (pla == "AD" and plb != "40") or (plb == "AD" and pla != "40"):
                st.warning("Advantage means the other player is on 40.")
            else:
                pa, pb = lt.points_from_labels(pla, plb)
        need = 2 if lbo == 3 else 3
        if sa >= need or sbb >= need:
            st.success(f"Match finished: {la if sa >= need else lb} won.")
        else:
            e = lt.calibrate(p_pre, lsurf, lbo, tk2)
            pl = e.live(sa, sbb, ga, gb, a_serves, pa, pb, in_tb)
            m1, m2 = st.columns(2)
            m1.metric(la, f"{pl:.1%}", f"{(pl - p_pre) * 100:+.1f} pts vs pre-match")
            m2.metric(lb, f"{1 - pl:.1%}")
            st.progress(float(pl))

with tab_c:
    st.info("Cricket (T20 / IPL / ODI) is the next module: ball-by-ball data from Cricsheet, "
            "pre-match win probability plus live win probability.")

with tab_b:
    bt = backtests()
    st.subheader("Honest accuracy (tested on matches the model never saw, 2023 onward)")
    for t, r in bt.items():
        st.markdown(f"##### {t.upper()}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Accuracy", f"{r['accuracy']:.1%}")
        k2.metric("Rank-only baseline", f"{r['baseline_rank_only']:.1%}")
        k3.metric("Matches tested", f"{r['matches']:,}")
        st.dataframe(pd.DataFrame([{"Model confidence": k, "Actual win rate": f"{v[0]:.1%}", "Matches": v[1]}
                                   for k, v in r["bands"].items()]), hide_index=True)
    st.caption("No model can honestly promise 90% on every match. The strong picks (80%+) are the reliable ones, "
               "but they are a minority of matches.")

st.divider()
st.caption("Data: Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0, non-commercial); live scores via Live Tennis API. "
           "Independent project, not affiliated with Sofascore. For information and entertainment only, not betting advice.")
