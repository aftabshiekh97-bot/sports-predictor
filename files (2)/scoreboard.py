"""Sofascore-style live scoreboard helpers (own branding; data from Live Tennis API free tier)."""
import html
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

import live_tennis as lt
import tennis_model as tm

BASE = "https://api.livetennisapi.com/api/public/v1"


# ---------------------------------------------------------------- data
def fetch(status, key, limit=100):
    r = requests.get(f"{BASE}/matches", params={"status": status, "limit": limit, "draw": "singles"},
                     headers={"X-API-Key": key}, timeout=15)
    if r.status_code == 429:
        raise RuntimeError("API limit reached (free plan: 100 requests/day). Try again later.")
    if r.status_code in (401, 403):
        raise RuntimeError("API key rejected. Check the key in Streamlit Secrets.")
    r.raise_for_status()
    return r.json().get("data", [])


def demo_matches():
    """Sample matches so the layout works before an API key is added."""
    def m(i, status, tour, tn, surf, a, b, sets, games, pts, server, tb=False, t="2026-10-03T08:30:00Z", rd="Quarter-finals"):
        return {"id": i, "status": status, "tour": tour, "tournament": tn, "surface": surf, "format": "BO3",
                "round": rd, "scheduled_time": t, "draw": "singles",
                "players": {"p1": {"name": a}, "p2": {"name": b}},
                "score": {"sets": sets, "games": games, "points": pts, "server": server, "is_tiebreak": tb}}
    return [
        m(1, "live", "atp", "Demo Open", "hard", "Jannik Sinner", "Alexander Zverev", [1, 0], [[6, 2], [4, 2]], ["30", "15"], 1),
        m(2, "live", "atp", "Demo Open", "hard", "Carlos Alcaraz", "Daniil Medvedev", [0, 1], [[4, 3], [6, 5]], ["15", "40"], 2),
        m(3, "live", "wta", "Demo WTA 1000", "hard", "Aryna Sabalenka", "Iga Swiatek", [1, 1], [[6, 3, 2], [4, 6, 2]], ["40", "30"], 1),
        m(4, "upcoming", "wta", "Demo WTA 1000", "hard", "Coco Gauff", "Elena Rybakina", [0, 0], [[]], [], None, t="2026-10-03T14:00:00Z"),
        m(5, "upcoming", "atp", "Demo Open", "hard", "Novak Djokovic", "Taylor Fritz", [0, 0], [[]], [], None, t="2026-10-03T16:30:00Z"),
    ]


# ---------------------------------------------------------------- parsing (defensive)
def _name(p):
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for k in ("name", "full_name", "short_name", "display_name"):
            if p.get(k):
                return str(p[k])
        fn, ln = p.get("first_name"), p.get("last_name")
        if fn or ln:
            return f"{fn or ''} {ln or ''}".strip()
    return "?"


def players_of(m):
    pl = m.get("players")
    if isinstance(pl, list) and len(pl) >= 2:
        return _name(pl[0]), _name(pl[1])
    if isinstance(pl, dict):
        for a, b in (("p1", "p2"), ("player1", "player2"), ("1", "2"), ("home", "away")):
            if a in pl and b in pl:
                return _name(pl[a]), _name(pl[b])
    return "?", "?"


def parse_score(m):
    """-> dict(sets=[a,b], games=[[...],[...]], points=(la,lb), server=1|2|None, tb=bool, per_set=[(ga,gb),...])"""
    s = m.get("score") or {}
    sets = s.get("sets") or [0, 0]
    games = s.get("games") or [[], []]
    try:
        per_set = list(zip(games[0], games[1]))
    except Exception:
        per_set = []
    pts = s.get("points") or []
    return dict(sets=list(sets) + [0] * (2 - len(sets)), per_set=per_set, points=pts,
                server=s.get("server"), tb=bool(s.get("is_tiebreak")))


def local_time(iso, tz):
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(ZoneInfo(tz)).strftime("%H:%M")
    except Exception:
        return ""


# ---------------------------------------------------------------- name matching to our rating database
def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s)


class Matcher:
    def __init__(self, players):          # players: list of (name, id)
        self.idx = {}
        for name, pid in players:
            toks = _norm(name).split()
            if len(toks) < 2:
                continue
            first = toks[0]
            for k in (1, 2, 3):
                if len(toks) - 1 >= k:
                    self.idx.setdefault("".join(toks[-k:]), []).append((pid, first))

    def find(self, api_name):
        toks = _norm(api_name).split()
        cands = []
        for i in range(len(toks)):
            for j in range(i + 1, min(i + 3, len(toks)) + 1):
                key = "".join(toks[i:j])
                rest = toks[:i] + toks[j:]
                for pid, first in self.idx.get(key, []):
                    ok = any(first.startswith(t[0]) for t in rest if t) if rest else True
                    cands.append((ok, pid))
        good = [pid for ok, pid in cands if ok]
        if good:
            return good[0]
        uniq = {pid for _, pid in cands}
        return next(iter(uniq)) if len(uniq) == 1 and not toks[1:] else None


# ---------------------------------------------------------------- probabilities
def win_probs(m, ctx, matcher):
    """-> (p_pre, p_live or None, known_both). p_* are for player 1."""
    a, b = players_of(m)
    ia, ib = matcher.find(a), matcher.find(b)
    surf = (m.get("surface") or "hard").capitalize()
    surf = surf if surf in ("Hard", "Clay", "Grass") else "Hard"
    known = ia is not None and ib is not None and ia != ib
    p_pre = tm.predict(ctx["model"], ctx["eng"], ia, ib, surf)[0] if known else 0.5
    if m.get("status") != "live":
        return p_pre, None, known
    sc = parse_score(m)
    bo = 5 if m.get("format") == "BO5" else 3
    try:
        eng = lt.calibrate(p_pre, surf, bo, m.get("tour") or "atp")
        sa, sb = sc["sets"][:2]
        ga, gb = sc["per_set"][-1] if sc["per_set"] else (0, 0)
        server = sc["server"] != 2
        if sc["tb"] and (ga, gb) != (6, 6):                       # deciding 10-point match tiebreak
            return p_pre, lt.race(eng.p_tb, ga, gb, 10), known
        if sc["tb"]:
            pa, pb = (int(x) for x in sc["points"][:2])
            return p_pre, eng.live(sa, sb, ga, gb, server, pa, pb, True), known
        lab = [str(x).upper().replace("A", "AD") if str(x).upper() in ("A", "AD") else str(x) for x in sc["points"][:2]]
        pa, pb = lt.points_from_labels(*lab) if len(lab) == 2 else (0, 0)
        return p_pre, eng.live(sa, sb, ga, gb, server, pa, pb, False), known
    except Exception:
        return p_pre, None, known


# ---------------------------------------------------------------- HTML
CSS = """
<style>
.sb-title{font-size:26px;font-weight:800;letter-spacing:.3px}
.sb-grp{background:#161b22;border-radius:10px 10px 0 0;padding:10px 14px;margin-top:14px;font-weight:700;border-bottom:1px solid #222a35}
.sb-grp span{color:#8b98a9;font-weight:500;font-size:12px;margin-left:8px}
.sb-row{display:flex;align-items:center;gap:12px;background:#0f141a;padding:9px 14px;border-bottom:1px solid #1b222c}
.sb-time{width:54px;text-align:center;font-size:12px;color:#8b98a9;line-height:1.2}
.sb-live{color:#ff4d4f;font-weight:700}
.sb-pl{flex:1;font-size:14px;line-height:1.7}
.sb-pl .w{font-weight:700;color:#fff}
.sb-sc{display:flex;gap:12px;font-size:14px;line-height:1.7;text-align:center;min-width:90px;justify-content:flex-end}
.sb-sc .col{min-width:14px}.sb-sc .tot{font-weight:800;color:#ffd24d}
.sb-pr{width:62px;text-align:right;font-size:13px;line-height:1.7;font-weight:700}
.sb-pr .hi{color:#3ddc97}.sb-pr .lo{color:#8b98a9}
.sb-bar{height:6px;border-radius:3px;background:#2a3340;overflow:hidden;margin-top:6px}
.sb-bar div{height:100%;background:linear-gradient(90deg,#3d7bff,#3ddc97)}
.sb-card{background:#161b22;border-radius:12px;padding:16px;margin-bottom:12px}
.sb-big{font-size:34px;font-weight:800}
.sb-muted{color:#8b98a9;font-size:12px}
.sb-srv{color:#ffd24d}
</style>
"""


def row_html(m, p_pre, p_live, tz, known=True):
    a, b = players_of(m)
    sc = parse_score(m)
    status = m.get("status")
    p = p_live if p_live is not None else p_pre
    srv = sc["server"]
    if status == "live":
        left = '<div class="sb-time sb-live">LIVE</div>'
    else:
        left = f'<div class="sb-time">{local_time(m.get("scheduled_time"), tz)}<br>{"Soon" if status == "upcoming" else "FT"}</div>'
    dot = lambda n: ' <span class="sb-srv">&#9679;</span>' if (status == "live" and srv == n) else ""
    cells = ""
    if status != "upcoming":
        for x, y in sc["per_set"]:
            cells += f'<div class="col"><div>{x}</div><div>{y}</div></div>'
        cells += f'<div class="col tot"><div>{sc["sets"][0]}</div><div>{sc["sets"][1]}</div></div>'
    if not known and p_live is None:
        pr = '<div class="lo">-</div><div class="lo">-</div>'
    else:
      pr = (f'<div class="{"hi" if p >= .5 else "lo"}">{p:.0%}</div>'
            f'<div class="{"hi" if p < .5 else "lo"}">{1 - p:.0%}</div>')
    return (f'<div class="sb-row">{left}'
            f'<div class="sb-pl"><div>{html.escape(a)}{dot(1)}</div><div>{html.escape(b)}{dot(2)}</div></div>'
            f'<div class="sb-sc">{cells}</div><div class="sb-pr">{pr}</div></div>')


def list_html(matches, probs, tz):
    """Group by tournament. probs: {match_id: (p_pre, p_live, known)}"""
    out, last = [], None
    for m in matches:
        key = (m.get("tour"), m.get("tournament"))
        if key != last:
            surf = (m.get("surface") or "").capitalize()
            out.append(f'<div class="sb-grp">{html.escape(str(m.get("tournament") or "Tournament"))}'
                       f'<span>{(m.get("tour") or "").upper()} · {surf} · {html.escape(str(m.get("round") or "").split(" - ")[-1])}</span></div>')
            last = key
        p_pre, p_live, known = probs[m["id"]]
        out.append(row_html(m, p_pre, p_live, tz, known))
    return "".join(out)
