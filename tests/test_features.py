from football_predictor.data import Game
from football_predictor.features import QBTracker, build_context


def _game(**overrides):
    base = dict(
        season=2024, week="1", game_type="REG", date="2024-09-08",
        home_team="AAA", away_team="BBB", home_score=24, away_score=17,
        home_rest=7, away_rest=7, div_game=False, roof="outdoors", surface="grass",
        temp=70.0, wind=5.0, home_qb_id="QB1", away_qb_id="QB2",
        home_qb_name="", away_qb_name="", home_coach="", away_coach="",
    )
    base.update(overrides)
    return Game(**base)


def test_rest_diff_reflects_bye_week_advantage():
    tracker = QBTracker()
    ctx = build_context(_game(home_rest=13, away_rest=6), tracker)
    assert ctx.rest_diff == 7.0


def test_qb_change_not_flagged_on_a_teams_first_known_game():
    tracker = QBTracker()
    ctx = build_context(_game(home_qb_id="QB1"), tracker)
    assert ctx.qb_change_home is False


def test_qb_change_flagged_when_starter_differs_from_last_known():
    tracker = QBTracker()
    tracker.observe("AAA", "QB1")
    ctx = build_context(_game(home_qb_id="QB2"), tracker)
    assert ctx.qb_change_home is True


def test_qb_change_not_flagged_when_starter_is_unchanged():
    tracker = QBTracker()
    tracker.observe("AAA", "QB1")
    ctx = build_context(_game(home_qb_id="QB1"), tracker)
    assert ctx.qb_change_home is False


def test_dome_games_never_flag_cold_or_wind():
    tracker = QBTracker()
    ctx = build_context(_game(roof="dome", temp=20.0, wind=25.0), tracker)
    assert ctx.cold_game is False
    assert ctx.windy_game is False


def test_outdoor_extreme_weather_flags_cold_and_windy():
    tracker = QBTracker()
    ctx = build_context(_game(roof="outdoors", temp=20.0, wind=25.0), tracker)
    assert ctx.cold_game is True
    assert ctx.windy_game is True


def test_playoff_and_div_flags_pass_through():
    tracker = QBTracker()
    ctx = build_context(_game(game_type="WC", div_game=True), tracker)
    assert ctx.playoff is True
    assert ctx.div_game is True
