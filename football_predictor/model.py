"""Prediction model combining Elo ratings with a calibrated spread/score
projection. No third-party ML dependencies: the rating-to-spread scale is
fit with a closed-form least-squares regression over the training games.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .data import Game, team_display_name
from .elo import DEFAULT_RATING, EloConfig, EloRatingSystem

DEFAULT_SPREAD_SCALE = 25.0  # ~25 Elo points per point of spread (538 convention)


@dataclass
class TrainingSummary:
    games_processed: int
    seasons: tuple[int, int]
    spread_scale: float
    avg_total_points: float
    log_loss: float
    accuracy: float


@dataclass
class Prediction:
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_margin: float
    predicted_home_score: float
    predicted_away_score: float
    home_rating: float
    away_rating: float
    neutral_site: bool
    context: object = None  # optional SituationalContext, set by AdvancedPredictor

    @property
    def favorite(self) -> str:
        return self.home_team if self.home_win_prob >= 0.5 else self.away_team

    def to_dict(self) -> dict:
        payload = {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_prob": round(self.home_win_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "predicted_margin": round(self.predicted_margin, 1),
            "predicted_home_score": round(self.predicted_home_score, 1),
            "predicted_away_score": round(self.predicted_away_score, 1),
            "home_rating": round(self.home_rating, 1),
            "away_rating": round(self.away_rating, 1),
            "neutral_site": self.neutral_site,
            "favorite": self.favorite,
        }
        if self.context is not None:
            payload["context"] = vars(self.context)
        return payload


class FootballPredictor:
    def __init__(self, elo: EloRatingSystem | None = None, spread_scale: float = DEFAULT_SPREAD_SCALE,
                 avg_total_points: float = 44.0):
        self.elo = elo or EloRatingSystem()
        self.spread_scale = spread_scale
        self.avg_total_points = avg_total_points

    def train(self, games: list[Game]) -> TrainingSummary:
        if not games:
            raise ValueError("no games supplied for training")

        diffs: list[float] = []
        margins: list[float] = []
        totals: list[int] = []
        correct = 0
        log_loss_sum = 0.0

        for game in games:
            pregame_home_rating = self.elo.get_rating(game.home_team)
            pregame_away_rating = self.elo.get_rating(game.away_team)
            diff_before_update = pregame_home_rating - pregame_away_rating

            home_win_prob = self.elo.process_game(
                season=game.season,
                home_team=game.home_team,
                away_team=game.away_team,
                home_score=game.home_score,
                away_score=game.away_score,
                is_playoff=game.is_playoff,
            )

            diffs.append(diff_before_update + self.elo.config.home_advantage)
            margins.append(float(game.margin))
            totals.append(game.home_score + game.away_score)

            predicted_winner_home = home_win_prob >= 0.5
            actual_winner_home = game.home_score >= game.away_score
            if predicted_winner_home == actual_winner_home:
                correct += 1

            p = min(max(home_win_prob, 1e-6), 1 - 1e-6)
            actual = 1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5)
            log_loss_sum += -(actual * math.log(p) + (1 - actual) * math.log(1 - p))

        self.spread_scale = _fit_scale(diffs, margins) or DEFAULT_SPREAD_SCALE
        self.avg_total_points = sum(totals) / len(totals)

        return TrainingSummary(
            games_processed=len(games),
            seasons=(games[0].season, games[-1].season),
            spread_scale=self.spread_scale,
            avg_total_points=self.avg_total_points,
            log_loss=log_loss_sum / len(games),
            accuracy=correct / len(games),
        )

    def predict(self, home_team: str, away_team: str, neutral: bool = False) -> Prediction:
        home_rating = self.elo.get_rating(home_team)
        away_rating = self.elo.get_rating(away_team)
        home_win_prob = self.elo.expected_home_win_prob(home_team, away_team, neutral=neutral)
        advantage = 0.0 if neutral else self.elo.config.home_advantage
        rating_diff = (home_rating + advantage) - away_rating
        predicted_margin = rating_diff / self.spread_scale

        predicted_home_score = (self.avg_total_points + predicted_margin) / 2
        predicted_away_score = (self.avg_total_points - predicted_margin) / 2

        return Prediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob,
            away_win_prob=1 - home_win_prob,
            predicted_margin=predicted_margin,
            predicted_home_score=max(predicted_home_score, 0.0),
            predicted_away_score=max(predicted_away_score, 0.0),
            home_rating=home_rating,
            away_rating=away_rating,
            neutral_site=neutral,
        )

    def power_rankings(self) -> list[tuple[str, str, float]]:
        return [
            (team, team_display_name(team), rating)
            for team, rating in self.elo.power_rankings()
        ]

    def save(self, path: str | Path) -> None:
        payload = {
            "elo": self.elo.to_dict(),
            "spread_scale": self.spread_scale,
            "avg_total_points": self.avg_total_points,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FootballPredictor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        elo = EloRatingSystem.from_dict(data["elo"])
        return cls(
            elo=elo,
            spread_scale=data.get("spread_scale", DEFAULT_SPREAD_SCALE),
            avg_total_points=data.get("avg_total_points", 44.0),
        )


def _fit_scale(diffs: list[float], margins: list[float]) -> float | None:
    """Least-squares slope of margin ~ diff, forced through the origin
    (a 0 Elo edge implies a 0-point expected margin)."""
    numerator = sum(d * m for d, m in zip(diffs, margins))
    denominator = sum(d * d for d in diffs)
    if denominator == 0:
        return None
    slope = numerator / denominator  # margin = slope * diff
    if slope == 0:
        return None
    return 1.0 / slope  # diffs per point of margin, i.e. our spread_scale


@dataclass
class BacktestResult:
    games: int
    accuracy: float
    brier_score: float
    spread_mae: float


def backtest(games: list[Game], start_season: int, config: EloConfig | None = None) -> BacktestResult:
    """Walk chronologically through *games*, generating a prediction before
    each game is used to update ratings, and score predictions for games in
    seasons >= start_season."""
    predictor = FootballPredictor(EloRatingSystem(config=config or EloConfig()))

    correct = 0
    brier_sum = 0.0
    spread_error_sum = 0.0
    evaluated = 0

    diffs: list[float] = []
    margins: list[float] = []
    totals: list[int] = []

    for game in games:
        home_rating = predictor.elo.get_rating(game.home_team)
        away_rating = predictor.elo.get_rating(game.away_team)
        home_win_prob = predictor.elo.expected_home_win_prob(game.home_team, game.away_team)

        if game.season >= start_season and diffs:
            scale = _fit_scale(diffs, margins) or DEFAULT_SPREAD_SCALE
            avg_total = sum(totals) / len(totals)
            rating_diff = (home_rating + predictor.elo.config.home_advantage) - away_rating
            predicted_margin = rating_diff / scale

            actual_winner_home = game.home_score >= game.away_score
            predicted_winner_home = home_win_prob >= 0.5
            if predicted_winner_home == actual_winner_home:
                correct += 1

            actual = 1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5)
            brier_sum += (home_win_prob - actual) ** 2
            spread_error_sum += abs(predicted_margin - game.margin)
            evaluated += 1

        predictor.elo.process_game(
            season=game.season,
            home_team=game.home_team,
            away_team=game.away_team,
            home_score=game.home_score,
            away_score=game.away_score,
            is_playoff=game.is_playoff,
        )
        diffs.append((home_rating + predictor.elo.config.home_advantage) - away_rating)
        margins.append(float(game.margin))
        totals.append(game.home_score + game.away_score)

    if evaluated == 0:
        raise ValueError("no games fell within the backtest window")

    return BacktestResult(
        games=evaluated,
        accuracy=correct / evaluated,
        brier_score=brier_sum / evaluated,
        spread_mae=spread_error_sum / evaluated,
    )
