from football_predictor.advanced_model import (
    AdvancedPredictor, advanced_backtest, backtest_history, blend_prediction,
    blend_residual_std, blend_summary_from_history, market_backtest, pick_recommendation,
    recommendation_summary_from_history,
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


def test_blend_prediction_weight_zero_returns_pure_market():
    prob, margin = blend_prediction(
        model_home_win_prob=0.9, market_home_win_prob=0.55,
        model_margin=14.0, market_spread=3.0, weight=0.0,
    )
    assert abs(prob - 0.55) < 1e-9
    assert abs(margin - 3.0) < 1e-9


def test_blend_prediction_weight_one_returns_pure_model():
    prob, margin = blend_prediction(
        model_home_win_prob=0.9, market_home_win_prob=0.55,
        model_margin=14.0, market_spread=3.0, weight=1.0,
    )
    assert abs(prob - 0.9) < 1e-9
    assert abs(margin - 14.0) < 1e-9


def test_blend_prediction_moves_toward_the_more_confident_side():
    # model and market both favor the home team, but the market is far more
    # confident (55% vs 90%) — a small model weight should pull the blend
    # only slightly away from the market's number, not all the way to 90%.
    prob, _ = blend_prediction(
        model_home_win_prob=0.9, market_home_win_prob=0.55,
        model_margin=14.0, market_spread=3.0, weight=0.10,
    )
    assert 0.55 < prob < 0.65


def test_backtest_history_includes_blend_fields_once_warmed_up():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    # attach market data to the final season only, so we exercise blend
    # fields on games where the advanced model is definitely warmed up.
    tail_start = len(games) - 16
    patched = []
    for i, g in enumerate(games):
        if i >= tail_start:
            g = Game(
                **{**vars(g), "spread_line": 10.0, "home_moneyline": -300.0, "away_moneyline": 250.0},
            )
        patched.append(g)

    history = backtest_history(patched, start_season=2019)
    with_market = [r for r in history if r["market_home_win_prob"] is not None]
    assert len(with_market) == 16
    for record in with_market:
        assert record["blend_home_win_prob"] is not None
        assert record["blend_predicted_margin"] is not None
        assert isinstance(record["blend_correct"], bool)
        # the blend is a weighted average in log-odds space, so it must
        # fall between the two inputs (inclusive) on the probability scale
        lo = min(record["adv_home_win_prob"], record["market_home_win_prob"])
        hi = max(record["adv_home_win_prob"], record["market_home_win_prob"])
        assert lo - 1e-9 <= record["blend_home_win_prob"] <= hi + 1e-9


def test_blend_summary_from_history_matches_manual_aggregate():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    tail_start = len(games) - 16
    patched = []
    for i, g in enumerate(games):
        if i >= tail_start:
            g = Game(**{**vars(g), "spread_line": 10.0, "home_moneyline": -300.0, "away_moneyline": 250.0})
        patched.append(g)

    history = backtest_history(patched, start_season=2019)
    summary = blend_summary_from_history(history)

    with_blend = [r for r in history if r["blend_correct"] is not None]
    assert summary.games == len(with_blend)
    assert summary.accuracy == sum(r["blend_correct"] for r in with_blend) / len(with_blend)


def test_blend_summary_from_history_raises_without_any_market_data():
    games = _synthetic_games(n_seasons=5, games_per_season=16)
    history = backtest_history(games, start_season=2001)
    try:
        blend_summary_from_history(history)
        assert False, "expected ValueError when no games have both adv and market predictions"
    except ValueError:
        pass


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


def test_pick_recommendation_falls_back_to_moneyline_without_residual_std():
    rec = pick_recommendation(
        blend_home_win_prob=0.55, blend_predicted_margin=10.0, market_spread=1.0,
        residual_std=None, home_team="AAA", away_team="BBB",
    )
    assert rec["ats_confidence"] is None
    assert rec["recommendation_type"] == "moneyline"
    assert rec["recommendation_team"] == "AAA"
    assert abs(rec["recommendation_confidence"] - 0.55) < 1e-9


def test_pick_recommendation_chooses_ats_when_its_confidence_is_higher():
    # moneyline is a coin flip (55%), but the model's margin (10) clears the
    # market's spread (1) by 9 points against a small residual std (3), so
    # the ATS cover probability is far more confident than the moneyline pick.
    rec = pick_recommendation(
        blend_home_win_prob=0.55, blend_predicted_margin=10.0, market_spread=1.0,
        residual_std=3.0, home_team="AAA", away_team="BBB",
    )
    assert rec["recommendation_type"] == "ats"
    assert rec["recommendation_team"] == "AAA"
    assert rec["recommendation_confidence"] > rec["moneyline_confidence"]
    assert rec["recommendation_confidence"] == rec["ats_confidence"]


def test_pick_recommendation_chooses_moneyline_when_its_confidence_is_higher():
    # moneyline is very confident (95%), while the model's margin edge over
    # the spread is tiny relative to a large residual std, so ATS confidence
    # stays close to a coin flip.
    rec = pick_recommendation(
        blend_home_win_prob=0.95, blend_predicted_margin=10.5, market_spread=10.0,
        residual_std=10.0, home_team="AAA", away_team="BBB",
    )
    assert rec["recommendation_type"] == "moneyline"
    assert rec["recommendation_team"] == "AAA"
    assert rec["recommendation_confidence"] > rec["ats_confidence"]


def test_backtest_history_includes_recommendation_fields_once_residual_std_warmed_up():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    # attach market data to the final two seasons (32 games) so the running
    # residual-std sample has enough games to clear MIN_RESIDUAL_SAMPLES (20)
    # partway through, letting us check both the "not warmed up yet" and
    # "warmed up" states within one backtest run.
    tail_start = len(games) - 32
    patched = []
    for i, g in enumerate(games):
        if i >= tail_start:
            g = Game(**{**vars(g), "spread_line": 3.0, "home_moneyline": -300.0, "away_moneyline": 250.0})
        patched.append(g)

    history = backtest_history(patched, start_season=2018)
    with_market = [r for r in history if r["market_home_win_prob"] is not None]
    assert len(with_market) == 32

    for r in with_market[:20]:
        assert r["ats_confidence"] is None
        assert r["recommendation_type"] == "moneyline"

    for r in with_market[20:]:
        assert r["ats_confidence"] is not None
        assert r["recommendation_type"] in ("ats", "moneyline")
        assert r["recommendation_team"] in (r["home_team"], r["away_team"])
        assert r["recommendation_confidence"] >= 0.5
        assert r["recommendation_correct"] is not None


def test_recommendation_summary_from_history_matches_manual_aggregate():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    tail_start = len(games) - 32
    patched = []
    for i, g in enumerate(games):
        if i >= tail_start:
            g = Game(**{**vars(g), "spread_line": 3.0, "home_moneyline": -300.0, "away_moneyline": 250.0})
        patched.append(g)

    history = backtest_history(patched, start_season=2018)
    summary = recommendation_summary_from_history(history)

    records = [r for r in history if r["recommendation_type"] is not None]
    scored = [r for r in records if r["recommendation_correct"] is not None]
    assert summary.games == len(records)
    assert summary.ats_recommended + summary.moneyline_recommended == len(records)
    assert abs(summary.accuracy - sum(r["recommendation_correct"] for r in scored) / len(scored)) < 1e-9


def test_recommendation_summary_from_history_raises_without_any_recommendation_data():
    games = _synthetic_games(n_seasons=5, games_per_season=16)
    history = backtest_history(games, start_season=2001)
    try:
        recommendation_summary_from_history(history)
        assert False, "expected ValueError when no games have a recommendation decision"
    except ValueError:
        pass


def test_blend_residual_std_none_below_minimum_sample_size():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    tail_start = len(games) - 5
    patched = []
    for i, g in enumerate(games):
        if i >= tail_start:
            g = Game(**{**vars(g), "spread_line": 3.0, "home_moneyline": -300.0, "away_moneyline": 250.0})
        patched.append(g)

    history = backtest_history(patched, start_season=2019)
    assert blend_residual_std(history) is None


def test_blend_residual_std_positive_once_warmed_up():
    games = _synthetic_games(n_seasons=20, games_per_season=16)
    tail_start = len(games) - 32
    patched = []
    for i, g in enumerate(games):
        if i >= tail_start:
            g = Game(**{**vars(g), "spread_line": 3.0, "home_moneyline": -300.0, "away_moneyline": 250.0})
        patched.append(g)

    history = backtest_history(patched, start_season=2018)
    std = blend_residual_std(history)
    assert std is not None
    assert std >= 0.0
