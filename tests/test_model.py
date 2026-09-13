from football_predictor.data import Game
from football_predictor.elo import EloConfig, EloRatingSystem
from football_predictor.model import FootballPredictor, backtest


def _synthetic_games(n_seasons=3, games_per_season=8):
    games = []
    for season in range(2020, 2020 + n_seasons):
        for week in range(1, games_per_season + 1):
            games.append(Game(
                season=season, week=str(week), game_type="REG",
                date=f"{season}-09-{week:02d}",
                home_team="AAA", away_team="BBB",
                home_score=28, away_score=14,
            ))
    return games


def test_predict_probabilities_sum_to_one():
    predictor = FootballPredictor()
    predictor.train(_synthetic_games())
    prediction = predictor.predict("AAA", "BBB")
    assert abs((prediction.home_win_prob + prediction.away_win_prob) - 1.0) < 1e-9


def test_dominant_team_earns_higher_rating_and_positive_margin():
    predictor = FootballPredictor()
    predictor.train(_synthetic_games())
    prediction = predictor.predict("AAA", "BBB", neutral=True)
    assert prediction.home_rating > prediction.away_rating
    assert prediction.predicted_margin > 0
    assert prediction.home_win_prob > 0.5
    assert prediction.favorite == "AAA"


def test_predicted_scores_sum_to_avg_total():
    predictor = FootballPredictor()
    predictor.train(_synthetic_games())
    prediction = predictor.predict("AAA", "BBB")
    total = prediction.predicted_home_score + prediction.predicted_away_score
    assert abs(total - predictor.avg_total_points) < 1e-6


def test_save_and_load_round_trip(tmp_path):
    predictor = FootballPredictor()
    predictor.train(_synthetic_games())
    path = tmp_path / "ratings.json"
    predictor.save(path)

    restored = FootballPredictor.load(path)
    original = predictor.predict("AAA", "BBB")
    reloaded = restored.predict("AAA", "BBB")
    assert abs(original.home_win_prob - reloaded.home_win_prob) < 1e-9
    assert abs(original.predicted_margin - reloaded.predicted_margin) < 1e-6


def test_backtest_reports_reasonable_metrics():
    games = _synthetic_games(n_seasons=4, games_per_season=10)
    result = backtest(games, start_season=2022, config=EloConfig())
    assert result.games > 0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.brier_score <= 1.0
    assert result.spread_mae >= 0.0
    # the home team always wins big in this synthetic set, so the model
    # should learn to favor it strongly once ratings have warmed up
    assert result.accuracy > 0.9
