"""Loading and normalizing historical + upcoming NFL game data.

The shipped CSV (data/nfl_games.csv) is exported from the public
nflverse/nfldata project and includes every game since 1999: final scores,
rest days, weather, starting QBs, coaches, and the closing Vegas market
line. Future/unplayed games are kept (with blank scores) so the schedule
can be used to generate real upcoming-week predictions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Franchises that relocated keep a single Elo history under their current code.
LEGACY_TEAM_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def normalize_team(code: str) -> str:
    code = code.strip().upper()
    return LEGACY_TEAM_MAP.get(code, code)


def team_display_name(code: str) -> str:
    code = normalize_team(code)
    return TEAM_NAMES.get(code, code)


@dataclass(frozen=True)
class Game:
    season: int
    week: str
    game_type: str
    date: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    home_rest: int | None = None
    away_rest: int | None = None
    div_game: bool = False
    roof: str = ""
    surface: str = ""
    temp: float | None = None
    wind: float | None = None
    home_qb_id: str = ""
    away_qb_id: str = ""
    home_qb_name: str = ""
    away_qb_name: str = ""
    home_coach: str = ""
    away_coach: str = ""
    spread_line: float | None = None  # market-implied home margin (positive = home favored)
    home_moneyline: float | None = None
    away_moneyline: float | None = None
    total_line: float | None = None

    @property
    def played(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def is_playoff(self) -> bool:
        return self.game_type != "REG"

    @property
    def is_dome(self) -> bool:
        return self.roof in ("dome", "closed")

    @property
    def margin(self) -> int:
        assert self.home_score is not None and self.away_score is not None
        return self.home_score - self.away_score


def _to_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _row_to_game(row: dict) -> Game:
    return Game(
        season=int(row["season"]),
        week=row.get("week", ""),
        game_type=row.get("game_type", "REG") or "REG",
        date=row.get("gameday", ""),
        home_team=normalize_team(row["home_team"]),
        away_team=normalize_team(row["away_team"]),
        home_score=_to_int(row.get("home_score")),
        away_score=_to_int(row.get("away_score")),
        home_rest=_to_int(row.get("home_rest")),
        away_rest=_to_int(row.get("away_rest")),
        div_game=row.get("div_game") in ("1", "True", "true"),
        roof=row.get("roof", "") or "",
        surface=row.get("surface", "") or "",
        temp=_to_float(row.get("temp")),
        wind=_to_float(row.get("wind")),
        home_qb_id=row.get("home_qb_id", "") or "",
        away_qb_id=row.get("away_qb_id", "") or "",
        home_qb_name=row.get("home_qb_name", "") or "",
        away_qb_name=row.get("away_qb_name", "") or "",
        home_coach=row.get("home_coach", "") or "",
        away_coach=row.get("away_coach", "") or "",
        spread_line=_to_float(row.get("spread_line")),
        home_moneyline=_to_float(row.get("home_moneyline")),
        away_moneyline=_to_float(row.get("away_moneyline")),
        total_line=_to_float(row.get("total_line")),
    )


def load_all_games(path: str | Path) -> list[Game]:
    """Load every row (completed and upcoming), sorted chronologically."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        games = [_row_to_game(row) for row in reader]
    games.sort(key=lambda g: (g.season, g.date, g.week))
    return games


def load_games(path: str | Path) -> list[Game]:
    """Load only completed games, for training/backtesting."""
    return [g for g in load_all_games(path) if g.played]


def load_upcoming_games(path: str | Path) -> list[Game]:
    """Load only unplayed (future) games, i.e. the remaining schedule."""
    return [g for g in load_all_games(path) if not g.played]
