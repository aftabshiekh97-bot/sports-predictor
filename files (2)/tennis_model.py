"""Tennis match predictor: overall + surface Elo, form, fatigue, rank, H2H -> logistic regression.
Data: Jeff Sackmann's ATP match files (CC BY-NC-SA 4.0), via public archive mirror.
"""
import glob
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MIRROR = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/{t}/{t}_matches_{y}.csv"
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
FEATURES = ["elo", "surf_elo", "rank", "form", "fatigue", "h2h", "height", "age"]


def load_matches(years=range(2005, 2027), local_dir=None, tour="atp"):
    frames = []
    for y in years:
        src = f"{local_dir}/{tour}_{y}.csv" if local_dir else MIRROR.format(t=tour, y=y)
        try:
            frames.append(pd.read_csv(src))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df = df[df["surface"].isin(SURFACES) & df["score"].notna()]
    df = df[~df["score"].astype(str).str.contains("W/O|DEF|WO", regex=True)]
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "tourney_id", "match_num"]).reset_index(drop=True)
    return df


class Engine:
    """Holds rolling ratings. update() after each match; features() before it."""

    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.selo = defaultdict(lambda: 1500.0)
        self.n = defaultdict(int)
        self.recent = defaultdict(lambda: deque(maxlen=10))      # 1/0 results
        self.minutes = defaultdict(list)                         # (date, minutes)
        self.h2h = defaultdict(int)                              # (a,b) -> a wins
        self.last_rank = {}
        self.info = {}                                           # id -> dict(name, ht, hand, ioc)

    @staticmethod
    def _k(n):  # FiveThirtyEight-style decaying K
        return 250.0 / ((n + 5) ** 0.4)

    def _fatigue(self, p, date):
        return sum(m for d, m in self.minutes[p] if 0 <= (date - d).days <= 14)

    def features(self, a, b, surface, date, ra, rb, ha, hb, aa, ab):
        sa, sb = (surface, a), (surface, b)
        form = lambda p: np.mean(self.recent[p]) if self.recent[p] else 0.5
        lr = lambda r: math.log(r) if r and r > 0 else math.log(200)
        h = lambda x: 185.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else x
        ag = lambda x: 26.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else x
        return [
            (self.elo[a] - self.elo[b]) / 100,
            (self.selo[sa] - self.selo[sb]) / 100,
            lr(rb) - lr(ra),                                     # better rank => positive
            form(a) - form(b),
            (self._fatigue(b, date) - self._fatigue(a, date)) / 300,
            (self.h2h[(a, b)] - self.h2h[(b, a)]) / 3,
            (h(ha) - h(hb)) / 10,
            (ag(ab) - ag(aa)) / 5,                               # younger slightly favoured
        ]

    def update(self, w, l, surface, date, minutes):
        for rating, key_w, key_l in ((self.elo, w, l), (self.selo, (surface, w), (surface, l))):
            ew = 1 / (1 + 10 ** ((rating[key_l] - rating[key_w]) / 400))
            kw, kl = self._k(self.n[w]), self._k(self.n[l])
            rating[key_w] += kw * (1 - ew)
            rating[key_l] -= kl * (1 - ew)
        self.n[w] += 1
        self.n[l] += 1
        self.recent[w].append(1)
        self.recent[l].append(0)
        if minutes == minutes and minutes:
            self.minutes[w].append((date, minutes))
            self.minutes[l].append((date, minutes))
        self.h2h[(w, l)] += 1


def build(df, train_until=None):
    """Walk matches in order, emit features BEFORE updating. Returns X, y, dates, engine."""
    eng = Engine()
    rows, ys, dates = [], [], []
    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        eng.info[w] = dict(name=r.winner_name, ht=r.winner_ht, hand=r.winner_hand, ioc=r.winner_ioc, age=r.winner_age)
        eng.info[l] = dict(name=r.loser_name, ht=r.loser_ht, hand=r.loser_hand, ioc=r.loser_ioc, age=r.loser_age)
        if not np.isnan(r.winner_rank):
            eng.last_rank[w] = r.winner_rank
        if not np.isnan(r.loser_rank):
            eng.last_rank[l] = r.loser_rank
        f = eng.features(w, l, r.surface, r.date, r.winner_rank, r.loser_rank, r.winner_ht, r.loser_ht, r.winner_age, r.loser_age)
        if eng.n[w] >= 5 and eng.n[l] >= 5:        # skip cold-start matches in training
            rows.append(f); ys.append(1); dates.append(r.date)
            rows.append([-x for x in f]); ys.append(0); dates.append(r.date)
        eng.update(w, l, r.surface, r.date, r.minutes)
    return np.array(rows), np.array(ys), pd.to_datetime(pd.Series(dates)).values, eng


def fit(X, y):
    return LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000).fit(X, y)


def backtest(df, test_from="2023-01-01"):
    X, y, d, _ = build(df)
    tr, te = d < np.datetime64(test_from), d >= np.datetime64(test_from)
    m = fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    yt = y[te]
    acc = ((p > 0.5) == yt).mean()
    # accuracy by confidence band (look only at the pick's side)
    conf = np.maximum(p, 1 - p)
    bands = {}
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        k = (conf >= lo) & (conf < hi)
        if k.sum():
            bands[f"{int(lo*100)}-{int(min(hi,1)*100)}%"] = (round(((p[k] > 0.5) == yt[k]).mean(), 3), int(k.sum() // 2))
    # elo-only and rank-only baselines
    base_rank = ((X[te][:, 2] > 0) == yt).mean()
    brier = np.mean((p - yt) ** 2)
    return dict(accuracy=acc, baseline_rank_only=base_rank, brier=brier, matches=int(te.sum() // 2),
                bands=bands, coef=dict(zip(FEATURES, m.coef_[0].round(3))))


def train_full(df):
    X, y, _, eng = build(df)
    return fit(X, y), eng


def predict(model, eng, a, b, surface, date=None, best_of=3):
    date = date or pd.Timestamp.today()
    ia, ib = eng.info[a], eng.info[b]
    f = eng.features(a, b, surface, date, eng.last_rank.get(a), eng.last_rank.get(b),
                     ia["ht"], ib["ht"], ia["age"], ib["age"])
    p = float(model.predict_proba([f])[0, 1])
    contrib = dict(zip(FEATURES, (np.array(f) * model.coef_[0]).round(3)))
    return p, contrib
