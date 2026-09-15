from football_predictor.advanced_model import (
    AdvancedPredictor, advanced_backtest, backtest_history, market_backtest,
)
from football_predictor.data import Game
from football_predictor.features import SituationalContext


def _synthetic_games(n_seasons=20, games_per_season=16):
    games = []
    for i, season in enumerate(range(2000, 2000 + n_seasons)):
        for week in range(1, games_per_season + 1):
            # AAA usually wins comfortably, but loses occasionally so the
            # outcome isn't perfectly separable (which would saturate the
            # logistic regression's probabilities at the float precision
            # limit and mask any single-feature effect).
            upset = (week == games_per_season and season % 4 == 0)
            home_score, away_score = (10, 24) if upset else (27, 13)
            games.append(Game(
                season=season, week=str(week), game_type="REG",
                date=f"{season}-09-{week:02d}",
                home_team="AAA", away_team="BBB",
                home_score=home_score, away_score=away_score,
                home_rest=7, away_rest=7,
                home_qb_id="QB-A", away_qb_id="QB-B",
            ))
    return games


def test_train_and_predict_round_trip(tmp_path):
    predictor = AdvancedPredictor()
    predictor.train(_synthetic_games())

    prediction = predictor.predict("AAA", "BBB", context=SituationalContext(), neutral=True)
    assert prediction.home_win_prob > 0.5
    assert prediction.predicted_margin > 0

    path = tmp_path / "advanced.json"
    predictor.save(path)
    restored = AdvancedPredictor.load(path)
    reloaded = restored.predict("AAA", "BBB", context=SituationalContext(), neutral=True)
    assert abs(prediction.home_win_prob - reloaded.home_win_prob) < 1e-9


def test_context_for_matchup_detects_future_qb_change():
    predictor = AdvancedPredictor()
    predictor.train(_synthetic_games())

    future_game = Game(
        season=2020, week="1", game_type="REG", date="2020-09-07",
        home_team="AAA", away_team="BBB", home_score=None, away_score=None,
        home_qb_id="QB-NEW", away_qb_id="QB-B",
    )
    ctx = predictor.context_for_matchup(future_game)
    assert ctx.qb_change_home is True
    assert ctx.qb_change_away is False


def test_advanced_backtest_runs_and_beats_coinflip():
    result = advanced_backtest(_synthetic_games(), start_season=2010)
    assert result.games > 0
    assert result.accuracy > 0.9


def test_backtest_history_only_covers_the_scored_window():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    history = backtest_history(games, start_season=2015)
    assert len(history) == sum(1 for g in games if g.season >= 2015)
    assert all(record["season"] >= 2015 for record in history)


def test_backtest_history_records_have_expected_shape():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    history = backtest_history(games, start_season=2015)
    record = history[0]

    assert record["home_team"] == "AAA" and record["away_team"] == "BBB"
    assert record["home_name_zh"] and record["away_name_zh"]
    assert isinstance(record["home_score"], int) and isinstance(record["away_score"], int)
    assert 0.0 <= record["elo_home_win_prob"] <= 1.0
    assert isinstance(record["elo_correct"], bool)
    # 200+ prior games have already been processed by 2015 in this dataset,
    # so the advanced model should already be warmed up and predicting.
    assert record["adv_home_win_prob"] is not None
    assert isinstance(record["adv_correct"], bool)
    # no market columns were supplied for this synthetic dataset
    assert record["market_home_win_prob"] is None
    assert record["market_correct"] is None


def test_backtest_history_matches_aggregate_backtest_accuracy():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    history = backtest_history(games, start_season=2015)
    adv_result = advanced_backtest(games, start_season=2015)

    history_adv_accuracy = sum(r["adv_correct"] for r in history) / len(history)
    assert abs(history_adv_accuracy - adv_result.accuracy) < 1e-9


def test_backtest_history_includes_market_fields_when_available():
    games = [
        Game(
            season=2021, week=str(w), game_type="REG", date=f"2021-09-{w:02d}",
            home_team="AAA", away_team="BBB", home_score=27, away_score=13,
            spread_line=7.0, home_moneyline=-300.0, away_moneyline=250.0,
        )
        for w in range(1, 4)
    ]
    history = backtest_history(games, start_season=2021)
    assert len(history) == 3
    for record in history:
        assert record["market_home_win_prob"] is not None
        assert record["market_spread"] == 7.0
        assert record["market_correct"] is True


def test_market_backtest_uses_moneylines_and_spread():
    games = [
        Game(
            season=2021, week="1", game_type="REG", date="2021-09-12",
            home_team="AAA", away_team="BBB", home_score=27, away_score=13,
            spread_line=7.0, home_moneyline=-300.0, away_moneyline=250.0,
        ),
        Game(
            season=2021, week="2", game_type="REG", date="2021-09-19",
            home_team="AAA", away_team="BBB", home_score=10, away_score=24,
            spread_line=3.0, home_moneyline=-150.0, away_moneyline=130.0,
        ),
    ]
    result = market_backtest(games, start_season=2021)
    assert result.games == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert result.spread_mae >= 0.0
