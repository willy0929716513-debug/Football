import csv

from football_predictor.data import load_games, normalize_team, team_display_name

CSV_HEADER = ["season", "week", "game_type", "gameday", "home_team", "away_team",
              "home_score", "away_score", "home_rest", "away_rest", "div_game"]


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_legacy_team_codes_are_normalized():
    assert normalize_team("oak") == "LV"
    assert normalize_team("SD") == "LAC"
    assert normalize_team("stl") == "LA"
    assert normalize_team("KC") == "KC"


def test_team_display_name():
    assert team_display_name("KC") == "Kansas City Chiefs"
    assert team_display_name("OAK") == "Las Vegas Raiders"


def test_load_games_sorts_and_skips_unplayed(tmp_path):
    path = tmp_path / "games.csv"
    _write_csv(path, [
        {"season": "2023", "week": "2", "game_type": "REG", "gameday": "2023-09-17",
         "home_team": "KC", "away_team": "BUF", "home_score": "27", "away_score": "24",
         "home_rest": "7", "away_rest": "7", "div_game": "0"},
        {"season": "2023", "week": "1", "game_type": "REG", "gameday": "2023-09-10",
         "home_team": "OAK", "away_team": "SD", "home_score": "20", "away_score": "17",
         "home_rest": "7", "away_rest": "7", "div_game": "1"},
        {"season": "2024", "week": "1", "game_type": "REG", "gameday": "2024-09-08",
         "home_team": "KC", "away_team": "BAL", "home_score": "", "away_score": "",
         "home_rest": "7", "away_rest": "7", "div_game": "0"},
    ])

    games = load_games(path)

    assert len(games) == 2  # the unplayed 2024 game is skipped
    assert games[0].season == 2023 and games[0].date == "2023-09-10"
    assert games[0].home_team == "LV" and games[0].away_team == "LAC"
    assert games[1].date == "2023-09-17"
    assert games[0].margin == 3
    assert games[0].is_playoff is False
