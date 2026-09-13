"""Loading and normalizing historical NFL game data.

Expected CSV columns: season, week, game_type, gameday, home_team,
away_team, home_score, away_score, home_rest, away_rest, div_game.
Only the team codes and scores are required; the rest are optional and
default to sensible values when missing.
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
    home_score: int
    away_score: int
    home_rest: int | None = None
    away_rest: int | None = None
    div_game: bool = False

    @property
    def is_playoff(self) -> bool:
        return self.game_type != "REG"

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score


def _to_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def load_games(path: str | Path) -> list[Game]:
    path = Path(path)
    games: list[Game] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("home_score") or not row.get("away_score"):
                continue
            games.append(
                Game(
                    season=int(row["season"]),
                    week=row.get("week", ""),
                    game_type=row.get("game_type", "REG") or "REG",
                    date=row.get("gameday", ""),
                    home_team=normalize_team(row["home_team"]),
                    away_team=normalize_team(row["away_team"]),
                    home_score=int(float(row["home_score"])),
                    away_score=int(float(row["away_score"])),
                    home_rest=_to_int(row.get("home_rest")),
                    away_rest=_to_int(row.get("away_rest")),
                    div_game=row.get("div_game") in ("1", "True", "true"),
                )
            )

    games.sort(key=lambda g: (g.season, g.date, g.week))
    return games
