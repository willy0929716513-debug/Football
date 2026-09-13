from football_predictor.elo import DEFAULT_RATING, EloConfig, EloRatingSystem


def test_equal_ratings_favor_home_field():
    elo = EloRatingSystem()
    prob = elo.expected_home_win_prob("AAA", "BBB")
    assert 0.5 < prob < 1.0


def test_neutral_site_removes_home_edge():
    elo = EloRatingSystem()
    prob = elo.expected_home_win_prob("AAA", "BBB", neutral=True)
    assert prob == 0.5


def test_win_raises_winner_rating_and_lowers_losers_by_same_amount():
    elo = EloRatingSystem()
    elo.process_game(season=2024, home_team="AAA", away_team="BBB", home_score=30, away_score=10)
    assert elo.ratings["AAA"] > DEFAULT_RATING
    assert elo.ratings["BBB"] < DEFAULT_RATING
    # zero-sum: what the winner gained, the loser lost
    gained = elo.ratings["AAA"] - DEFAULT_RATING
    lost = DEFAULT_RATING - elo.ratings["BBB"]
    assert abs(gained - lost) < 1e-9


def test_larger_margin_moves_rating_further():
    close = EloRatingSystem()
    close.process_game(season=2024, home_team="AAA", away_team="BBB", home_score=24, away_score=21)

    blowout = EloRatingSystem()
    blowout.process_game(season=2024, home_team="AAA", away_team="BBB", home_score=45, away_score=3)

    close_gain = close.ratings["AAA"] - DEFAULT_RATING
    blowout_gain = blowout.ratings["AAA"] - DEFAULT_RATING
    assert blowout_gain > close_gain


def test_playoff_multiplier_increases_rating_movement():
    reg = EloRatingSystem(config=EloConfig(playoff_multiplier=1.0))
    reg.process_game(season=2024, home_team="AAA", away_team="BBB", home_score=24, away_score=21, is_playoff=False)

    playoff = EloRatingSystem(config=EloConfig(playoff_multiplier=1.2))
    playoff.process_game(season=2024, home_team="AAA", away_team="BBB", home_score=24, away_score=21, is_playoff=True)

    assert (playoff.ratings["AAA"] - DEFAULT_RATING) > (reg.ratings["AAA"] - DEFAULT_RATING)


def test_season_regression_pulls_ratings_toward_mean():
    elo = EloRatingSystem(config=EloConfig(revert_to_mean=0.5))
    elo.process_game(season=2023, home_team="AAA", away_team="BBB", home_score=40, away_score=0)
    rating_before = elo.ratings["AAA"]
    elo.start_season(2024)
    rating_after = elo.ratings["AAA"]
    assert rating_after < rating_before
    assert rating_after > DEFAULT_RATING


def test_serialization_round_trip():
    elo = EloRatingSystem()
    elo.process_game(season=2024, home_team="AAA", away_team="BBB", home_score=27, away_score=13)
    restored = EloRatingSystem.from_dict(elo.to_dict())
    assert restored.ratings == elo.ratings
    assert restored.config == elo.config
