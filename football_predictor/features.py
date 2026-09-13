"""Situational features that layer on top of the Elo rating: rest/travel,
starting-QB continuity, weather, and game context (divisional, playoff).

QB continuity is tracked chronologically per team (``QBTracker``) so a
change from a team's last known starter — the single biggest situational
swing factor in football, ahead of weather or rest — is detected even for
future games, since the shipped schedule carries projected starters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Game

FEATURE_NAMES = [
    "elo_diff",
    "rest_diff",
    "div_game",
    "qb_change_home",
    "qb_change_away",
    "cold_game",
    "windy_game",
    "playoff",
]

# Rest is deliberately encoded as a single continuous rest_diff rather than
# separate short-week/bye binary flags: those overlap almost perfectly with
# rest_diff itself, and feeding both into the regression let collinearity
# flip a coefficient's sign in testing (a "short week" flag would sometimes
# come out *helping* the disadvantaged team). One continuous feature avoids
# the instability while still capturing the same information.
COLD_TEMP_F = 32.0
WINDY_MPH = 15.0


@dataclass
class SituationalContext:
    rest_diff: float = 0.0
    div_game: bool = False
    qb_change_home: bool = False
    qb_change_away: bool = False
    cold_game: bool = False
    windy_game: bool = False
    playoff: bool = False

    def to_vector(self, elo_diff: float) -> list[float]:
        return [
            elo_diff,
            self.rest_diff,
            float(self.div_game),
            float(self.qb_change_home),
            float(self.qb_change_away),
            float(self.cold_game),
            float(self.windy_game),
            float(self.playoff),
        ]


class QBTracker:
    """Remembers each team's most recently known starting QB so a change
    can be flagged even before that game's own result is known."""

    def __init__(self) -> None:
        self._last_qb: dict[str, str] = {}

    def changed(self, team: str, qb_id: str) -> bool:
        if not qb_id:
            return False
        previous = self._last_qb.get(team)
        return previous is not None and previous != qb_id

    def observe(self, team: str, qb_id: str) -> None:
        if qb_id:
            self._last_qb[team] = qb_id


def build_context(game: Game, qb_tracker: QBTracker) -> SituationalContext:
    outdoor = not game.is_dome
    temp_known = outdoor and game.temp is not None
    wind_known = outdoor and game.wind is not None

    return SituationalContext(
        rest_diff=float((game.home_rest or 7) - (game.away_rest or 7)),
        div_game=game.div_game,
        qb_change_home=qb_tracker.changed(game.home_team, game.home_qb_id),
        qb_change_away=qb_tracker.changed(game.away_team, game.away_qb_id),
        cold_game=bool(temp_known and game.temp <= COLD_TEMP_F),
        windy_game=bool(wind_known and game.wind >= WINDY_MPH),
        playoff=game.is_playoff,
    )
