"""Command-line interface for the NFL prediction toolkit.

    nflpredict train    --games data/nfl_games.csv --out ratings.json
    nflpredict predict  --home KC --away BUF
    nflpredict ratings  --top 10
    nflpredict backtest --start-season 2020
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_games, normalize_team, team_display_name
from .elo import EloConfig, EloRatingSystem
from .model import FootballPredictor, backtest as run_backtest

DEFAULT_GAMES_PATH = Path(__file__).resolve().parent.parent / "data" / "nfl_games.csv"
DEFAULT_RATINGS_PATH = Path(__file__).resolve().parent.parent / "ratings.json"


def _add_elo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=float, default=20.0, help="Elo K-factor (default: 20)")
    parser.add_argument("--home-advantage", type=float, default=48.0, help="home-field Elo bonus (default: 48)")
    parser.add_argument("--playoff-boost", type=float, default=1.2, help="K multiplier for playoff games (default: 1.2)")
    parser.add_argument("--revert", type=float, default=1.0 / 3.0, help="between-season regression to the mean (default: 1/3)")


def cmd_train(args: argparse.Namespace) -> int:
    games = load_games(args.games)
    if not games:
        print(f"error: no completed games found in {args.games}", file=sys.stderr)
        return 1

    config = EloConfig(
        k_factor=args.k,
        home_advantage=args.home_advantage,
        playoff_multiplier=args.playoff_boost,
        revert_to_mean=args.revert,
    )
    predictor = FootballPredictor(EloRatingSystem(config=config))
    summary = predictor.train(games)
    predictor.save(args.out)

    print(f"trained on {summary.games_processed} games ({summary.seasons[0]}-{summary.seasons[1]})")
    print(f"  training accuracy : {summary.accuracy:.1%}")
    print(f"  training log loss : {summary.log_loss:.4f}")
    print(f"  spread scale      : {summary.spread_scale:.2f} elo points / point")
    print(f"  avg total points  : {summary.avg_total_points:.1f}")
    print(f"ratings saved to {args.out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    predictor = FootballPredictor.load(args.ratings)
    home = normalize_team(args.home)
    away = normalize_team(args.away)
    prediction = predictor.predict(home, away, neutral=args.neutral)

    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2))
        return 0

    home_name = team_display_name(home)
    away_name = team_display_name(away)
    site = "neutral site" if args.neutral else "home field"

    print(f"{away_name} ({away}) @ {home_name} ({home})  [{site}]")
    print("-" * 60)
    print(f"  Elo ratings        : {home}: {prediction.home_rating:.1f}   {away}: {prediction.away_rating:.1f}")
    print(f"  Win probability    : {home}: {prediction.home_win_prob:.1%}   {away}: {prediction.away_win_prob:.1%}")
    print(f"  Predicted score    : {home} {prediction.predicted_home_score:.1f} - "
          f"{prediction.predicted_away_score:.1f} {away}")
    sign = "+" if prediction.predicted_margin >= 0 else ""
    print(f"  Predicted margin   : {home} {sign}{prediction.predicted_margin:.1f}")
    print(f"  Favorite           : {team_display_name(prediction.favorite)}")
    return 0


def cmd_ratings(args: argparse.Namespace) -> int:
    predictor = FootballPredictor.load(args.ratings)
    rankings = predictor.power_rankings()
    if args.team:
        team = normalize_team(args.team)
        rankings = [r for r in rankings if r[0] == team]
    if args.top:
        rankings = rankings[: args.top]

    if args.json:
        print(json.dumps(
            [{"team": t, "name": n, "rating": round(r, 1)} for t, n, r in rankings],
            indent=2,
        ))
        return 0

    print(f"{'#':<4}{'Team':<6}{'Name':<26}{'Elo':>8}")
    for i, (team, name, rating) in enumerate(rankings, start=1):
        print(f"{i:<4}{team:<6}{name:<26}{rating:>8.1f}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    games = load_games(args.games)
    config = EloConfig(
        k_factor=args.k,
        home_advantage=args.home_advantage,
        playoff_multiplier=args.playoff_boost,
        revert_to_mean=args.revert,
    )
    result = run_backtest(games, start_season=args.start_season, config=config)

    if args.json:
        print(json.dumps(vars(result), indent=2))
        return 0

    print(f"backtest: seasons >= {args.start_season}, {result.games} games")
    print(f"  winner accuracy : {result.accuracy:.1%}")
    print(f"  Brier score     : {result.brier_score:.4f}  (lower is better, 0.25 = coin flip)")
    print(f"  spread MAE      : {result.spread_mae:.2f} points")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nflpredict", description="Professional NFL game prediction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="build Elo ratings from historical games")
    p_train.add_argument("--games", default=str(DEFAULT_GAMES_PATH), help="path to games CSV")
    p_train.add_argument("--out", default=str(DEFAULT_RATINGS_PATH), help="path to write ratings.json")
    _add_elo_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_predict = subparsers.add_parser("predict", help="predict the outcome of a matchup")
    p_predict.add_argument("--home", required=True, help="home team code, e.g. KC")
    p_predict.add_argument("--away", required=True, help="away team code, e.g. BUF")
    p_predict.add_argument("--neutral", action="store_true", help="neutral-site game (no home-field bonus)")
    p_predict.add_argument("--ratings", default=str(DEFAULT_RATINGS_PATH), help="path to ratings.json")
    p_predict.add_argument("--json", action="store_true", help="output machine-readable JSON")
    p_predict.set_defaults(func=cmd_predict)

    p_ratings = subparsers.add_parser("ratings", help="show current power rankings")
    p_ratings.add_argument("--ratings", default=str(DEFAULT_RATINGS_PATH), help="path to ratings.json")
    p_ratings.add_argument("--top", type=int, default=0, help="show only the top N teams")
    p_ratings.add_argument("--team", default=None, help="show only this team code")
    p_ratings.add_argument("--json", action="store_true", help="output machine-readable JSON")
    p_ratings.set_defaults(func=cmd_ratings)

    p_backtest = subparsers.add_parser("backtest", help="validate accuracy on historical seasons")
    p_backtest.add_argument("--games", default=str(DEFAULT_GAMES_PATH), help="path to games CSV")
    p_backtest.add_argument("--start-season", type=int, required=True, help="first season to score (earlier seasons only warm up ratings)")
    p_backtest.add_argument("--json", action="store_true", help="output machine-readable JSON")
    _add_elo_args(p_backtest)
    p_backtest.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
