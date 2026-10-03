"""Live tennis win probability.

Idea: turn the pre-match win probability into 'serve strength' for each player
(chance of winning a point on their own serve), then recompute the exact match
win probability from the current score using a point-by-point Markov model.
"""

SERVE_BASE = {  # tour-average serve-point win rate
    "atp": {"Hard": 0.650, "Clay": 0.625, "Grass": 0.670},
    "wta": {"Hard": 0.570, "Clay": 0.550, "Grass": 0.590},
}
PT_MAP = {"0": 0, "15": 1, "30": 2, "40": 3}


def race(p, a, b, target):
    """P(first player wins) from points a-b when winning a point has prob p.
    Game: target=4. Tiebreak: target=7. Win by 2."""
    q = 1 - p
    deuce = p * p / (p * p + q * q)
    memo = {}

    def go(a, b):
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        if a >= target - 1 and b >= target - 1:
            if a == b:
                return deuce
            return p + q * deuce if a > b else p * deuce
        if (a, b) not in memo:
            memo[(a, b)] = p * go(a + 1, b) + q * go(a, b + 1)
        return memo[(a, b)]

    return go(a, b)


class LiveTennis:
    def __init__(self, ps_a, ps_b, best_of=3):
        self.ps_a, self.ps_b = ps_a, ps_b
        self.need = 2 if best_of == 3 else 3
        self.p_tb = 0.5 * (ps_a + (1 - ps_b))      # tiebreak point approx (serve alternates)
        self._memo = {}

    def game_a(self, a_serves, pa=0, pb=0):
        """P(A wins the current game) from points pa-pb."""
        if a_serves:
            return race(self.ps_a, pa, pb, 4)
        return 1 - race(self.ps_b, pb, pa, 4)

    def W(self, sa, sb, ga, gb, a_serves):
        """P(A wins match) at the start of a game. a_serves = A serves the next game."""
        if sa == self.need:
            return 1.0
        if sb == self.need:
            return 0.0
        key = (sa, sb, ga, gb, a_serves)
        if key in self._memo:
            return self._memo[key]
        if ga >= 6 and ga - gb >= 2:
            v = self.W(sa + 1, sb, 0, 0, a_serves)
        elif gb >= 6 and gb - ga >= 2:
            v = self.W(sa, sb + 1, 0, 0, a_serves)
        elif ga == 6 and gb == 6:
            q = race(self.p_tb, 0, 0, 7)
            v = q * self.W(sa + 1, sb, 0, 0, not a_serves) + (1 - q) * self.W(sa, sb + 1, 0, 0, not a_serves)
        else:
            g = self.game_a(a_serves)
            v = g * self.W(sa, sb, ga + 1, gb, not a_serves) + (1 - g) * self.W(sa, sb, ga, gb + 1, not a_serves)
        self._memo[key] = v
        return v

    def live(self, sa, sb, ga, gb, a_serves, pa=0, pb=0, tiebreak=False):
        """Win probability for A from the exact live score (points inside the current game/tiebreak)."""
        if tiebreak:
            q = race(self.p_tb, pa, pb, 7)
            return q * self.W(sa + 1, sb, 0, 0, not a_serves) + (1 - q) * self.W(sa, sb + 1, 0, 0, not a_serves)
        g = self.game_a(a_serves, pa, pb)
        return g * self.W(sa, sb, ga + 1, gb, not a_serves) + (1 - g) * self.W(sa, sb, ga, gb + 1, not a_serves)

    def pre_match(self):
        return 0.5 * (self.W(0, 0, 0, 0, True) + self.W(0, 0, 0, 0, False))


def calibrate(p_pre, surface="Hard", best_of=3, tour="atp"):
    """Find serve strengths so the model's pre-match probability equals p_pre."""
    base = SERVE_BASE.get(tour, SERVE_BASE["atp"]).get(surface, 0.62)
    lo, hi = -0.14, 0.14
    for _ in range(40):
        d = (lo + hi) / 2
        if LiveTennis(base + d, base - d, best_of).pre_match() < p_pre:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return LiveTennis(base + d, base - d, best_of)


def points_from_labels(la, lb):
    """Convert '0/15/30/40/AD' labels to point counts."""
    if la == "AD":
        return 4, 3
    if lb == "AD":
        return 3, 4
    return PT_MAP[la], PT_MAP[lb]
