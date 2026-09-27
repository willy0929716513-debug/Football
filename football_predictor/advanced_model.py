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


def _market_implied_prob(home_moneyline: float, away_moneyline: float) -> float:
    """Vig-free home win probability implied by the two moneylines."""
    home_implied = _american_odds_to_prob(home_moneyline)
    away_implied = _american_odds_to_prob(away_moneyline)
    overround = home_implied + away_implied
    return home_implied / overround if overround > 0 else 0.5


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# How much weight the blended prediction gives to our own model vs. the
# market, in log-odds space. Chosen by a walk-forward grid search (weights
# 0.00-1.00 in steps of 0.05) tuned on 2010-2020 and confirmed on a held-out
# 2021+ window: the market alone was consistently hard to beat, and any
# weight above ~0.10-0.15 on our own model made the blend *worse* on both
# windows. 0.10 keeps a small, genuine contribution from our own model
# (rest/QB/weather signals the market may weight differently) while staying
# close to the empirically-best pure-market end of the curve. See README's
# "參數是怎麼選出來的？" section for the full honest write-up.
BLEND_MODEL_WEIGHT = 0.10


def blend_prediction(model_home_win_prob: float, market_home_win_prob: float,
                      model_margin: float, market_spread: float,
                      weight: float = BLEND_MODEL_WEIGHT) -> tuple[float, float]:
    """Combine our own model's prediction with the real market line.

    Win probability is pooled in log-odds space (logistic/log-linear
    pooling), the standard way to combine two probabilistic forecasts —
    a plain average of probabilities is not well-calibrated. The margin
    is a simple weighted average since both are already point-scales.
    """
    blend_logit = weight * _logit(model_home_win_prob) + (1 - weight) * _logit(market_home_win_prob)
    blend_prob = _sigmoid(blend_logit)
    blend_margin = weight * model_margin + (1 - weight) * market_spread
    return blend_prob, blend_margin


# Minimum number of prior blend-eligible games needed before we trust an
# empirical residual standard deviation enough to compute an against-the-
# spread ("讓分") cover probability. Below this, ats_confidence is left as
# None and the recommendation always falls back to the moneyline pick.
MIN_RESIDUAL_SAMPLES = 20


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pick_recommendation(blend_home_win_prob: float, blend_predicted_margin: float,
                         market_spread: float, residual_std: float | None,
                         home_team: str, away_team: str) -> dict:
    """Compare two different bets for the same game — the straight-up
    ("不讓分") moneyline pick vs. the against-the-spread ("讓分") pick — and
    recommend whichever one we're more confident about.

    Moneyline confidence is just how far the blended win probability sits
    from a coin flip. ATS confidence needs a probability that the blend's
    predicted margin beats the market's spread line by enough to "cover":
    treating the model's historical (actual margin - blend margin) errors
    as roughly normal with standard deviation `residual_std`, the chance
    the home side covers is the normal CDF of the edge (blend margin minus
    spread line) scaled by that standard deviation. Whichever bet has the
    higher confidence is the one surfaced as "the" recommendation; both
    numbers are still reported so nothing is hidden.
    """
    moneyline_confidence = max(blend_home_win_prob, 1.0 - blend_home_win_prob)
    moneyline_pick = home_team if blend_home_win_prob >= 0.5 else away_team

    ats_edge = blend_predicted_margin - market_spread
    if residual_std is not None and residual_std > 0:
        ats_cover_prob_home = _normal_cdf(ats_edge / residual_std)
        ats_confidence = max(ats_cover_prob_home, 1.0 - ats_cover_prob_home)
        ats_pick = home_team if ats_edge >= 0 else away_team
    else:
        ats_cover_prob_home = None
        ats_confidence = None
        ats_pick = None

    if ats_confidence is not None and ats_confidence > moneyline_confidence:
        recommendation_type = "ats"
        recommendation_team = ats_pick
        recommendation_confidence = ats_confidence
    else:
        recommendation_type = "moneyline"
        recommendation_team = moneyline_pick
        recommendation_confidence = moneyline_confidence

    return {
        "ats_edge": round(ats_edge, 1),
        "ats_cover_prob_home": round(ats_cover_prob_home, 4) if ats_cover_prob_home is not None else None,
        "ats_confidence": round(ats_confidence, 4) if ats_confidence is not None else None,
        "moneyline_confidence": round(moneyline_confidence, 4),
        "recommendation_type": recommendation_type,
        "recommendation_team": recommendation_team,
        "recommendation_confidence": round(recommendation_confidence, 4),
    }


def blend_residual_std(history: list[dict]) -> float | None:
    """Standard deviation of (actual margin - blend predicted margin) over
    every game in a `backtest_history()` result that had a blend
    prediction, used to score against-the-spread confidence for *future*
    (upcoming, not-yet-played) games. Using the whole history is fine here
    — unlike inside `backtest_history` itself, an upcoming game is strictly
    after every game in the history, so this carries no look-ahead bias."""
    errors = [
        (r["home_score"] - r["away_score"]) - r["blend_predicted_margin"]
        for r in history if r["blend_predicted_margin"] is not None
    ]
    if len(errors) < MIN_RESIDUAL_SAMPLES:
        return None
    mean = sum(errors) / len(errors)
    variance = sum((e - mean) ** 2 for e in errors) / len(errors)
    return math.sqrt(max(variance, 1e-6))


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

    # Running (walk-forward: only games strictly before the one being
    # scored) mean/variance of blend margin errors, used to score
    # against-the-spread confidence for each historical game without any
    # look-ahead — see `pick_recommendation`.
    blend_error_n = 0
    blend_error_sum = 0.0
    blend_error_sumsq = 0.0

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
                market_prob = _market_implied_prob(game.home_moneyline, game.away_moneyline)
                record["market_home_win_prob"] = round(market_prob, 4)
                record["market_spread"] = game.spread_line
                record["market_correct"] = (market_prob >= 0.5) == actual_winner_home
            else:
                record["market_home_win_prob"] = None
                record["market_spread"] = None
                record["market_correct"] = None

            if record["adv_home_win_prob"] is not None and record["market_home_win_prob"] is not None:
                blend_prob, blend_margin = blend_prediction(
                    record["adv_home_win_prob"], record["market_home_win_prob"],
                    record["adv_predicted_margin"], record["market_spread"],
                )
                record["blend_home_win_prob"] = round(blend_prob, 4)
                record["blend_predicted_margin"] = round(blend_margin, 1)
                record["blend_correct"] = (blend_prob >= 0.5) == actual_winner_home

                residual_std = None
                if blend_error_n >= MIN_RESIDUAL_SAMPLES:
                    mean = blend_error_sum / blend_error_n
                    variance = max(blend_error_sumsq / blend_error_n - mean * mean, 1e-6)
                    residual_std = math.sqrt(variance)

                rec_fields = pick_recommendation(
                    blend_prob, blend_margin, record["market_spread"], residual_std,
                    game.home_team, game.away_team,
                )
                record.update(rec_fields)

                actual_margin = game.margin
                if record["recommendation_type"] == "ats":
                    if actual_margin == record["market_spread"]:
                        record["recommendation_correct"] = None  # push
                    else:
                        home_covered = actual_margin > record["market_spread"]
                        record["recommendation_correct"] = (
                            (record["recommendation_team"] == game.home_team) == home_covered
                        )
                else:
                    winner = game.home_team if actual_winner_home else game.away_team
                    record["recommendation_correct"] = record["recommendation_team"] == winner

                error = actual_margin - blend_margin
                blend_error_n += 1
                blend_error_sum += error
                blend_error_sumsq += error * error
            else:
                record["blend_home_win_prob"] = None
                record["blend_predicted_margin"] = None
                record["blend_correct"] = None
                record["ats_edge"] = None
                record["ats_cover_prob_home"] = None
                record["ats_confidence"] = None
                record["moneyline_confidence"] = None
                record["recommendation_type"] = None
                record["recommendation_team"] = None
                record["recommendation_confidence"] = None
                record["recommendation_correct"] = None

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


@dataclass
class BlendBacktestResult:
    games: int
    accuracy: float
    brier_score: float
    spread_mae: float


def blend_summary_from_history(history: list[dict]) -> BlendBacktestResult:
    """Aggregate the per-game `blend_*` fields a `backtest_history()` call
    already computed, for the same walk-forward games where both the
    advanced model and the market line were available. Reuses those
    records rather than re-running a separate backtest pass."""
    records = [r for r in history if r["blend_correct"] is not None]
    if not records:
        raise ValueError("no games with both an advanced-model and a market prediction to blend")

    correct = sum(1 for r in records if r["blend_correct"])
    brier_sum = 0.0
    spread_error_sum = 0.0
    for r in records:
        actual = 1.0 if r["home_score"] > r["away_score"] else (0.0 if r["home_score"] < r["away_score"] else 0.5)
        brier_sum += (r["blend_home_win_prob"] - actual) ** 2
        spread_error_sum += abs(r["blend_predicted_margin"] - (r["home_score"] - r["away_score"]))

    n = len(records)
    return BlendBacktestResult(
        games=n,
        accuracy=correct / n,
        brier_score=brier_sum / n,
        spread_mae=spread_error_sum / n,
    )


@dataclass
class RecommendationSummaryResult:
    games: int
    ats_recommended: int
    moneyline_recommended: int
    accuracy: float
    ats_accuracy: float | None
    moneyline_accuracy: float | None


def recommendation_summary_from_history(history: list[dict]) -> RecommendationSummaryResult:
    """Honest backtest of the "pick whichever bet type we're more confident
    about" strategy itself: how often it chose the against-the-spread pick
    vs. the plain moneyline pick, and how each performed."""
    records = [r for r in history if r["recommendation_type"] is not None]
    if not records:
        raise ValueError("no games with a recommendation decision (need blend data to be available)")

    scored = [r for r in records if r["recommendation_correct"] is not None]
    ats_scored = [r for r in scored if r["recommendation_type"] == "ats"]
    ml_scored = [r for r in scored if r["recommendation_type"] == "moneyline"]

    return RecommendationSummaryResult(
        games=len(records),
        ats_recommended=sum(1 for r in records if r["recommendation_type"] == "ats"),
        moneyline_recommended=sum(1 for r in records if r["recommendation_type"] == "moneyline"),
        accuracy=(sum(1 for r in scored if r["recommendation_correct"]) / len(scored)) if scored else 0.0,
        ats_accuracy=(sum(1 for r in ats_scored if r["recommendation_correct"]) / len(ats_scored)) if ats_scored else None,
        moneyline_accuracy=(sum(1 for r in ml_scored if r["recommendation_correct"]) / len(ml_scored)) if ml_scored else None,
    )
