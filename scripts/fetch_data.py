#!/usr/bin/env python3
"""Refresh data/nfl_games.csv from the public nflverse/nfldata project.

Run this periodically (or via the update-predictions GitHub Action) to
pick up newly completed games, updated rest-day/QB info for the coming
week, and refreshed closing market lines.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

COLUMNS = [
    "season", "week", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score",
    "home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind",
    "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name", "home_coach", "away_coach",
    "spread_line", "home_moneyline", "away_moneyline", "total_line",
]


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "data" / "nfl_games.csv"
    print(f"fetching {SOURCE_URL} ...")
    raw = fetch(SOURCE_URL)
    reader = csv.DictReader(io.StringIO(raw))

    rows = []
    for row in reader:
        d = {c: row.get(c, "") for c in COLUMNS}
        d["home_team"] = TEAM_MAP.get(d["home_team"], d["home_team"])
        d["away_team"] = TEAM_MAP.get(d["away_team"], d["away_team"])
        rows.append(d)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
