"""The 'advanced' ensemble predictor: Elo rating difference blended with
situational features (rest/travel, starting-QB continuity, weather,
divisional and playoff context) through logistic regression (win
probability) and ridge regression (point margin). This is the model used
by the `--model advanced` CLI flag and the exported website.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Game, team_display_name, team_display_name_zh
from .elo import EloConfig, EloRatingSystem
from .features import FEATURE_NAMES, QBTracker, SituationalContext, build_context
from .model import DEFAULT_SPREAD_SCALE, Prediction, _fit_scale
from .regression import LinearModel, LogisticModel, fit_logistic, fit_ridge_linear


@dataclass
class AdvancedTrainingSummary:
    games_processed: int
    seasons: tuple[int, int]
    avg_total_points: float
    accuracy: float
    log_loss: float
    spread_mae: float
    feature_weights: dict[str, float] = field(default_factory=dict)


class AdvancedPredictor:
    def __init__(self, elo: EloRatingSystem | None = None):
        self.elo = elo or EloRatingSystem()
        self.qb_tracker = QBTracker()
        self.win_model: LogisticModel | None = None
        self.margin_model: LinearModel | None = None
        self.avg_total_points = 44.0

    def train(self, games: list[Game], l2_logit: float = 2.0, l2_linear: float = 2.0) -> AdvancedTrainingSummary:
        if not games:
            raise ValueError("no games supplied for training")

        rows: list[list[float]] = []
        home_wins: list[float] = []
        margins: list[float] = []
        totals: list[int] = []

        for game in games:
            home_rating = self.elo.get_rating(game.home_team)
            away_rating = self.elo.get_rating(game.away_team)
            elo_diff = (home_rating + self.elo.config.home_advantage) - away_rating

            ctx = build_context(game, self.qb_tracker)
            rows.append(ctx.to_vector(elo_diff))

            self.qb_tracker.observe(game.home_team, game.home_qb_id)
            self.qb_tracker.observe(game.away_team, game.away_qb_id)

            self.elo.process_game(
                season=game.season,
                home_team=game.home_team,
                away_team=game.away_team,
                home_score=game.home_score,
                away_score=game.away_score,
                is_playoff=game.is_playoff,
            )

            actual = 1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5)
            home_wins.append(actual)
            margins.append(float(game.margin))
            totals.append(game.home_score + game.away_score)

        X = np.array(rows)
        y_win = np.array(home_wins)
        y_margin = np.array(margins)

        self.win_model = fit_logistic(X, y_win, l2=l2_logit)
        self.margin_model = fit_ridge_linear(X, y_margin, l2=l2_linear)
        self.avg_total_points = sum(totals) / len(totals)

        probs = self.win_model.predict_proba(X)
        pred_margins = self.margin_model.predict(X)
        correct = int(np.sum((probs >= 0.5) == (y_win >= 0.5)))
        p_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        log_loss = float(-np.mean(y_win * np.log(p_clipped) + (1 - y_win) * np.log(1 - p_clipped)))
        spread_mae = float(np.mean(np.abs(pred_margins - y_margin)))

        weights = dict(zip(FEATURE_NAMES, self.win_model.coef))

        return AdvancedTrainingSummary(
            games_processed=len(games),
            seasons=(games[0].season, games[-1].season),
            avg_total_points=self.avg_total_points,
            accuracy=correct / len(games),
            log_loss=log_loss,
            spread_mae=spread_mae,
            feature_weights=weights,
        )

    def context_for_matchup(self, game: Game) -> SituationalContext:
        """Peek the current QB-continuity state for a scheduled (possibly
        future) game without mutating it."""
        return build_context(game, self.qb_tracker)

    def predict(self, home_team: str, away_team: str, context: SituationalContext | None = None,
                neutral: bool = False) -> Prediction:
        if self.win_model is None or self.margin_model is None:
            raise RuntimeError("model has not been trained or loaded")

        context = context or SituationalContext()
        home_rating = self.elo.get_rating(home_team)
        away_rating = self.elo.get_rating(away_team)
        advantage = 0.0 if neutral else self.elo.config.home_advantage
        elo_diff = (home_rating + advantage) - away_rating

        vector = np.array([context.to_vector(elo_diff)])
        home_win_prob = float(self.win_model.predict_proba(vector)[0])
        predicted_margin = float(self.margin_model.predict(vector)[0])

        predicted_home_score = max((self.avg_total_points + predicted_margin) / 2, 0.0)
        predicted_away_score = max((self.avg_total_points - predicted_margin) / 2, 0.0)

        return Prediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob,
            away_win_prob=1 - home_win_prob,
            predicted_margin=predicted_margin,
            predicted_home_score=predicted_home_score,
            predicted_away_score=predicted_away_score,
            home_rating=home_rating,
            away_rating=away_rating,
            neutral_site=neutral,
            context=context,
        )

    def power_rankings(self) -> list[tuple[str, str, float]]:
        return [(team, team_display_name(team), rating) for team, rating in self.elo.power_rankings()]

    def save(self, path: str | Path) -> None:
        payload = {
            "elo": self.elo.to_dict(),
            "qb_tracker": self.qb_tracker._last_qb,
            "win_model": vars(self.win_model) if self.win_model else None,
            "margin_model": vars(self.margin_model) if self.margin_model else None,
            "avg_total_points": self.avg_total_points,
            "feature_names": FEATURE_NAMES,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AdvancedPredictor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        predictor = cls(elo=EloRatingSystem.from_dict(data["elo"]))
        predictor.qb_tracker._last_qb = data.get("qb_tracker", {})
        predictor.avg_total_points = data.get("avg_total_points", 44.0)
        if data.get("win_model"):
            predictor.win_model = LogisticModel(**data["win_model"])
        if data.get("margin_model"):
            predictor.margin_model = LinearModel(**data["margin_model"])
        return predictor


@dataclass
class AdvancedBacktestResult:
    games: int
    accuracy: float
    brier_score: float
    spread_mae: float
    seasons_refit: int


def advanced_backtest(games: list[Game], start_season: int, config: EloConfig | None = None,
                       l2_logit: float = 2.0, l2_linear: float = 2.0) -> AdvancedBacktestResult:
    """Walk-forward validation: refit the regression models once at the
    start of every season using only prior seasons' data (never the season
    being scored), so no future information leaks into a prediction."""
    predictor = AdvancedPredictor(EloRatingSystem(config=config or EloConfig()))

    rows: list[list[float]] = []
    home_wins: list[float] = []
    margins: list[float] = []
    totals: list[int] = []

    correct = 0
    brier_sum = 0.0
    spread_error_sum = 0.0
    evaluated = 0
    seasons_refit = 0
    current_season: int | None = None

    for game in games:
        if game.season != current_season:
            current_season = game.season
            if game.season >= start_season and len(rows) >= 200:
                X = np.array(rows)
                predictor.win_model = fit_logistic(X, np.array(home_wins), l2=l2_logit)
                predictor.margin_model = fit_ridge_linear(X, np.array(margins), l2=l2_linear)
                predictor.avg_total_points = sum(totals) / len(totals)
                seasons_refit += 1

        home_rating = predictor.elo.get_rating(game.home_team)
        away_rating = predictor.elo.get_rating(game.away_team)
        elo_diff = (home_rating + predictor.elo.config.home_advantage) - away_rating
        ctx = build_context(game, predictor.qb_tracker)
        vector = ctx.to_vector(elo_diff)

        if game.season >= start_season and predictor.win_model is not None:
            X_row = np.array([vector])
            home_win_prob = float(predictor.win_model.predict_proba(X_row)[0])
            predicted_margin = float(predictor.margin_model.predict(X_row)[0])

            actual_winner_home = game.home_score >= game.away_score
            if (home_win_prob >= 0.5) == actual_winner_home:
                correct += 1
            actual = 1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5)
            brier_sum += (home_win_prob - actual) ** 2
            spread_error_sum += abs(predicted_margin - game.margin)
            evaluated += 1

        predictor.qb_tracker.observe(game.home_team, game.home_qb_id)
        predictor.qb_tracker.observe(game.away_team, game.away_qb_id)
        predictor.elo.process_game(
            season=game.season,
            home_team=game.home_team,
            away_team=game.away_team,
            home_score=game.home_score,
            away_score=game.away_score,
            is_playoff=game.is_playoff,
        )

        rows.append(vector)
        home_wins.append(1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5))
        margins.append(float(game.margin))
        totals.append(game.home_score + game.away_score)

    if evaluated == 0:
        raise ValueError("no games fell within the backtest window (need >=200 prior games to warm up)")

    return AdvancedBacktestResult(
        games=evaluated,
        accuracy=correct / evaluated,
        brier_score=brier_sum / evaluated,
        spread_mae=spread_error_sum / evaluated,
        seasons_refit=seasons_refit,
    )


@dataclass
class MarketBacktestResult:
    games: int
    accuracy: float
    brier_score: float
    spread_mae: float


def _american_odds_to_prob(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def market_backtest(games: list[Game], start_season: int) -> MarketBacktestResult:
    """Benchmark the closing Vegas line itself, for games where it's
    available, so the model's accuracy can be read against a real
    professional baseline rather than only a coin flip."""
    correct = 0
    brier_sum = 0.0
    spread_error_sum = 0.0
    evaluated = 0

    for game in games:
        if game.season < start_season:
            continue
        if game.spread_line is None or game.home_moneyline is None or game.away_moneyline is None:
            continue

        home_implied = _american_odds_to_prob(game.home_moneyline)
        away_implied = _american_odds_to_prob(game.away_moneyline)
        overround = home_implied + away_implied
        home_win_prob = home_implied / overround if overround > 0 else 0.5

        actual_winner_home = game.home_score >= game.away_score
        if (home_win_prob >= 0.5) == actual_winner_home:
            correct += 1
        actual = 1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5)
        brier_sum += (home_win_prob - actual) ** 2
        spread_error_sum += abs(game.spread_line - game.margin)
        evaluated += 1

    if evaluated == 0:
        raise ValueError("no games with market data fell within the backtest window")

    return MarketBacktestResult(
        games=evaluated,
        accuracy=correct / evaluated,
        brier_score=brier_sum / evaluated,
        spread_mae=spread_error_sum / evaluated,
    )


def backtest_history(games: list[Game], start_season: int, config: EloConfig | None = None,
                      l2_logit: float = 2.0, l2_linear: float = 2.0) -> list[dict]:
    """Walk-forward backtest that records one transparent, inspectable
    record per evaluated game: what the elo model, the advanced model, and
    the real Vegas market line each said beforehand, and what actually
    happened. This is the same walk-forward methodology as `backtest`,
    `advanced_backtest`, and `market_backtest` (no look-ahead: the elo
    spread-scale, the advanced model's regression, and every prediction are
    all computed only from games strictly before the one being scored), just
    combined into one pass so the site can show a real, checkable track
    record instead of only aggregate accuracy numbers.
    """
    predictor = AdvancedPredictor(EloRatingSystem(config=config or EloConfig()))

    rows: list[list[float]] = []
    home_wins: list[float] = []
    margins: list[float] = []
    totals: list[int] = []
    elo_diffs: list[float] = []  # for the elo model's own spread-scale fit

    current_season: int | None = None
    history: list[dict] = []

    for game in games:
        if game.season != current_season:
            current_season = game.season
            if game.season >= start_season and len(rows) >= 200:
                X = np.array(rows)
                predictor.win_model = fit_logistic(X, np.array(home_wins), l2=l2_logit)
                predictor.margin_model = fit_ridge_linear(X, np.array(margins), l2=l2_linear)
                predictor.avg_total_points = sum(totals) / len(totals)

        home_rating = predictor.elo.get_rating(game.home_team)
        away_rating = predictor.elo.get_rating(game.away_team)
        elo_diff = (home_rating + predictor.elo.config.home_advantage) - away_rating
        elo_home_win_prob = predictor.elo.expected_home_win_prob(game.home_team, game.away_team)
        ctx = build_context(game, predictor.qb_tracker)
        vector = ctx.to_vector(elo_diff)

        if game.season >= start_season:
            record: dict = {
                "season": game.season,
                "week": game.week,
                "game_type": game.game_type,
                "date": game.date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_name_zh": team_display_name_zh(game.home_team),
                "away_name_zh": team_display_name_zh(game.away_team),
                "home_score": game.home_score,
                "away_score": game.away_score,
                "elo_home_win_prob": round(elo_home_win_prob, 4),
                "elo_predicted_margin": round(elo_diff / (_fit_scale(elo_diffs, margins) or DEFAULT_SPREAD_SCALE), 1)
                if elo_diffs else None,
            }

            actual_winner_home = game.home_score >= game.away_score
            record["elo_correct"] = (elo_home_win_prob >= 0.5) == actual_winner_home

            if predictor.win_model is not None:
                X_row = np.array([vector])
                adv_prob = float(predictor.win_model.predict_proba(X_row)[0])
                adv_margin = float(predictor.margin_model.predict(X_row)[0])
                record["adv_home_win_prob"] = round(adv_prob, 4)
                record["adv_predicted_margin"] = round(adv_margin, 1)
                record["adv_correct"] = (adv_prob >= 0.5) == actual_winner_home
            else:
                record["adv_home_win_prob"] = None
                record["adv_predicted_margin"] = None
                record["adv_correct"] = None

            if game.spread_line is not None and game.home_moneyline is not None and game.away_moneyline is not None:
                home_implied = _american_odds_to_prob(game.home_moneyline)
                away_implied = _american_odds_to_prob(game.away_moneyline)
                overround = home_implied + away_implied
                market_prob = home_implied / overround if overround > 0 else 0.5
                record["market_home_win_prob"] = round(market_prob, 4)
                record["market_spread"] = game.spread_line
                record["market_correct"] = (market_prob >= 0.5) == actual_winner_home
            else:
                record["market_home_win_prob"] = None
                record["market_spread"] = None
                record["market_correct"] = None

            history.append(record)

        predictor.qb_tracker.observe(game.home_team, game.home_qb_id)
        predictor.qb_tracker.observe(game.away_team, game.away_qb_id)
        predictor.elo.process_game(
            season=game.season,
            home_team=game.home_team,
            away_team=game.away_team,
            home_score=game.home_score,
            away_score=game.away_score,
            is_playoff=game.is_playoff,
        )

        rows.append(vector)
        home_wins.append(1.0 if game.home_score > game.away_score else (0.0 if game.home_score < game.away_score else 0.5))
        margins.append(float(game.margin))
        totals.append(game.home_score + game.away_score)
        elo_diffs.append(elo_diff)

    return history
