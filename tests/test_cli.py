import json

import pytest

from football_predictor.cli import main


def _run(capsys, argv):
    exit_code = main(argv)
    captured = capsys.readouterr()
    return exit_code, captured.out


@pytest.fixture()
def trained_ratings(tmp_path, capsys):
    games_csv = tmp_path / "games.csv"
    games_csv.write_text(
        "season,week,game_type,gameday,home_team,away_team,home_score,away_score,home_rest,away_rest,div_game\n"
        "2022,1,REG,2022-09-11,AAA,BBB,31,10,7,7,0\n"
        "2022,2,REG,2022-09-18,BBB,AAA,17,24,7,7,0\n"
        "2023,1,REG,2023-09-10,AAA,BBB,28,14,7,7,0\n"
    )
    ratings_path = tmp_path / "ratings.json"
    exit_code, _ = _run(capsys, ["train", "--games", str(games_csv), "--out", str(ratings_path)])
    assert exit_code == 0
    return ratings_path


def test_train_command_writes_ratings_file(trained_ratings):
    assert trained_ratings.exists()


def test_predict_command_json_output(trained_ratings, capsys):
    exit_code, out = _run(capsys, [
        "predict", "--home", "AAA", "--away", "BBB",
        "--ratings", str(trained_ratings), "--json",
    ])
    assert exit_code == 0
    payload = json.loads(out)
    assert payload["home_team"] == "AAA"
    assert payload["favorite"] == "AAA"
    assert 0.0 <= payload["home_win_prob"] <= 1.0


def test_ratings_command_lists_teams(trained_ratings, capsys):
    exit_code, out = _run(capsys, ["ratings", "--ratings", str(trained_ratings), "--json"])
    assert exit_code == 0
    payload = json.loads(out)
    teams = {row["team"] for row in payload}
    assert teams == {"AAA", "BBB"}


def test_backtest_command_runs(tmp_path, capsys):
    games_csv = tmp_path / "games.csv"
    rows = ["season,week,game_type,gameday,home_team,away_team,home_score,away_score,home_rest,away_rest,div_game"]
    for season in range(2020, 2023):
        for week in range(1, 6):
            rows.append(f"{season},{week},REG,{season}-09-{week:02d},AAA,BBB,28,14,7,7,0")
    games_csv.write_text("\n".join(rows) + "\n")

    exit_code, out = _run(capsys, [
        "backtest", "--games", str(games_csv), "--start-season", "2021", "--model", "elo", "--json",
    ])
    assert exit_code == 0
    payload = json.loads(out)
    assert payload["elo"]["games"] > 0


def test_backtest_command_all_models_reports_per_model_results(tmp_path, capsys):
    games_csv = tmp_path / "games.csv"
    rows = ["season,week,game_type,gameday,home_team,away_team,home_score,away_score,home_rest,away_rest,div_game"]
    # 15 seasons (240 games) before start_season so the advanced model's
    # walk-forward refit has its required >=200-game warm-up.
    for season in range(2005, 2023):
        for week in range(1, 17):
            rows.append(f"{season},{week},REG,{season}-09-{week:02d},AAA,BBB,28,14,7,7,0")
    games_csv.write_text("\n".join(rows) + "\n")

    exit_code, out = _run(capsys, [
        "backtest", "--games", str(games_csv), "--start-season", "2020", "--model", "all", "--json",
    ])
    assert exit_code == 0
    payload = json.loads(out)
    assert payload["elo"]["games"] > 0
    assert payload["advanced"]["games"] > 0
    # this synthetic dataset carries no market odds columns, so market/blend are expected to error out
    assert "error" in payload["market"]
    assert "error" in payload["blend"]


def test_backtest_command_blend_model_with_market_data(tmp_path, capsys):
    games_csv = tmp_path / "games.csv"
    header = (
        "season,week,game_type,gameday,home_team,away_team,home_score,away_score,"
        "home_rest,away_rest,div_game,spread_line,home_moneyline,away_moneyline"
    )
    rows = [header]
    for season in range(2005, 2023):
        for week in range(1, 17):
            rows.append(
                f"{season},{week},REG,{season}-09-{week:02d},AAA,BBB,28,14,7,7,0,7.0,-300,250"
            )
    games_csv.write_text("\n".join(rows) + "\n")

    exit_code, out = _run(capsys, [
        "backtest", "--games", str(games_csv), "--start-season", "2020", "--model", "blend", "--json",
    ])
    assert exit_code == 0
    payload = json.loads(out)
    assert payload["blend"]["games"] > 0
    assert 0.0 <= payload["blend"]["accuracy"] <= 1.0


def test_export_site_writes_data_and_history_with_blend_fields(tmp_path, capsys):
    games_csv = tmp_path / "games.csv"
    header = (
        "season,week,game_type,gameday,home_team,away_team,home_score,away_score,"
        "home_rest,away_rest,div_game,spread_line,home_moneyline,away_moneyline"
    )
    rows = [header]
    for season in range(2005, 2023):
        for week in range(1, 17):
            rows.append(
                f"{season},{week},REG,{season}-09-{week:02d},AAA,BBB,28,14,7,7,0,7.0,-300,250"
            )
    # one upcoming (unplayed) game with market data, so export-site's
    # upcoming-predictions loop has something to attach blend fields to
    rows.append("2023,1,REG,2023-09-10,AAA,BBB,,,7,7,0,7.0,-300,250")
    games_csv.write_text("\n".join(rows) + "\n")

    out_dir = tmp_path / "site"
    exit_code, out = _run(capsys, [
        "export-site", "--games", str(games_csv), "--out-dir", str(out_dir), "--start-season", "2020",
    ])
    assert exit_code == 0

    data = json.loads((out_dir / "data.json").read_text())
    assert data["model_performance"]["blend"]["games"] > 0
    upcoming = data["upcoming"]
    assert len(upcoming) == 1
    assert upcoming[0]["blend_home_win_prob"] is not None
    assert upcoming[0]["market_home_win_prob"] is not None

    history = json.loads((out_dir / "backtest_history.json").read_text())
    assert history["games"]
    assert any(g["blend_correct"] is not None for g in history["games"])
