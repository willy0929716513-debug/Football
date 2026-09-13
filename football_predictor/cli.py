"""Command-line interface for the NFL prediction toolkit.

    nflpredict train       --model advanced --games data/nfl_games.csv
    nflpredict predict     --home KC --away BUF --model advanced
    nflpredict ratings     --top 10
    nflpredict backtest    --start-season 2015 --model all
    nflpredict export-site --out-dir docs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .advanced_model import AdvancedPredictor, advanced_backtest, market_backtest
from .data import (
    load_games, load_upcoming_games, normalize_team, team_display_name, team_display_name_zh,
    TEAM_NAMES, TEAM_NAMES_ZH,
)
from .elo import EloConfig, EloRatingSystem
from .features import FEATURE_NAMES, SituationalContext
from .model import FootballPredictor, backtest as elo_backtest

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GAMES_PATH = ROOT / "data" / "nfl_games.csv"
DEFAULT_ELO_RATINGS_PATH = ROOT / "ratings.json"
DEFAULT_ADVANCED_RATINGS_PATH = ROOT / "ratings_advanced.json"
DEFAULT_SITE_DIR = ROOT / "docs"
DEFAULT_BACKTEST_START_SEASON = 2015


def _ratings_path(args: argparse.Namespace) -> Path:
    if args.ratings:
        return Path(args.ratings)
    return DEFAULT_ADVANCED_RATINGS_PATH if args.model == "advanced" else DEFAULT_ELO_RATINGS_PATH


def _add_elo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=float, default=20.0, help="Elo K-factor (default: 20)")
    parser.add_argument("--home-advantage", type=float, default=48.0, help="home-field Elo bonus (default: 48)")
    parser.add_argument("--playoff-boost", type=float, default=1.2, help="K multiplier for playoff games (default: 1.2)")
    parser.add_argument("--revert", type=float, default=1.0 / 3.0, help="between-season regression to the mean (default: 1/3)")


def _elo_config(args: argparse.Namespace) -> EloConfig:
    return EloConfig(
        k_factor=args.k,
        home_advantage=args.home_advantage,
        playoff_multiplier=args.playoff_boost,
        revert_to_mean=args.revert,
    )


def _context_from_args(args: argparse.Namespace) -> SituationalContext:
    home_rest = args.home_rest if args.home_rest is not None else 7
    away_rest = args.away_rest if args.away_rest is not None else 7
    return SituationalContext(
        rest_diff=float(home_rest - away_rest),
        div_game=args.div,
        qb_change_home=args.home_qb_change,
        qb_change_away=args.away_qb_change,
        cold_game=args.cold,
        windy_game=args.windy,
        playoff=args.playoff,
    )


# ---------------------------------------------------------------- train ---

def cmd_train(args: argparse.Namespace) -> int:
    games = load_games(args.games)
    if not games:
        print(f"error: no completed games found in {args.games}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else (
        DEFAULT_ADVANCED_RATINGS_PATH if args.model == "advanced" else DEFAULT_ELO_RATINGS_PATH
    )

    if args.model == "elo":
        predictor = FootballPredictor(elo=EloRatingSystem(config=_elo_config(args)))
        summary = predictor.train(games)
        predictor.save(out_path)
        print(f"[elo] trained on {summary.games_processed} games ({summary.seasons[0]}-{summary.seasons[1]})")
        print(f"  training accuracy : {summary.accuracy:.1%}")
        print(f"  training log loss : {summary.log_loss:.4f}")
        print(f"  spread scale      : {summary.spread_scale:.2f} elo points / point")
        print(f"  avg total points  : {summary.avg_total_points:.1f}")
    else:
        predictor = AdvancedPredictor(elo=EloRatingSystem(config=_elo_config(args)))
        summary = predictor.train(games)
        predictor.save(out_path)
        print(f"[advanced] trained on {summary.games_processed} games ({summary.seasons[0]}-{summary.seasons[1]})")
        print(f"  training accuracy : {summary.accuracy:.1%}")
        print(f"  training log loss : {summary.log_loss:.4f}")
        print(f"  spread MAE        : {summary.spread_mae:.2f} points")
        print(f"  avg total points  : {summary.avg_total_points:.1f}")
        print("  feature weights (standardized logistic regression coefficients):")
        for name, weight in summary.feature_weights.items():
            print(f"    {name:<16}{weight:+.3f}")

    print(f"ratings saved to {out_path}")
    return 0


# -------------------------------------------------------------- predict ---

def cmd_predict(args: argparse.Namespace) -> int:
    ratings_path = _ratings_path(args)
    home = normalize_team(args.home)
    away = normalize_team(args.away)

    if args.model == "elo":
        predictor = FootballPredictor.load(ratings_path)
        prediction = predictor.predict(home, away, neutral=args.neutral)
    else:
        predictor = AdvancedPredictor.load(ratings_path)
        context = _context_from_args(args)
        prediction = predictor.predict(home, away, context=context, neutral=args.neutral)

    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2))
        return 0

    home_name = team_display_name(home)
    away_name = team_display_name(away)
    site = "neutral site" if args.neutral else "home field"

    print(f"[{args.model}] {away_name} ({away}) @ {home_name} ({home})  [{site}]")
    print("-" * 60)
    print(f"  Elo ratings        : {home}: {prediction.home_rating:.1f}   {away}: {prediction.away_rating:.1f}")
    print(f"  Win probability    : {home}: {prediction.home_win_prob:.1%}   {away}: {prediction.away_win_prob:.1%}")
    print(f"  Predicted score    : {home} {prediction.predicted_home_score:.1f} - "
          f"{prediction.predicted_away_score:.1f} {away}")
    sign = "+" if prediction.predicted_margin >= 0 else ""
    print(f"  Predicted margin   : {home} {sign}{prediction.predicted_margin:.1f}")
    print(f"  Favorite           : {team_display_name(prediction.favorite)}")
    if prediction.context is not None:
        flags = [k for k, v in vars(prediction.context).items() if v is True]
        print(f"  Context factors    : {', '.join(flags) if flags else '(none — default/neutral context)'}")
        if prediction.context.rest_diff:
            print(f"  Rest-day edge      : {home} {prediction.context.rest_diff:+.0f} days vs {away}")
    return 0


# -------------------------------------------------------------- ratings ---

def cmd_ratings(args: argparse.Namespace) -> int:
    ratings_path = _ratings_path(args)
    if args.model == "elo":
        predictor = FootballPredictor.load(ratings_path)
    else:
        predictor = AdvancedPredictor.load(ratings_path)

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


# ------------------------------------------------------------- backtest ---

def cmd_backtest(args: argparse.Namespace) -> int:
    games = load_games(args.games)
    config = _elo_config(args)
    results = {}

    if args.model in ("elo", "all"):
        try:
            results["elo"] = vars(elo_backtest(games, start_season=args.start_season, config=config))
        except ValueError as exc:
            results["elo"] = {"error": str(exc)}
    if args.model in ("advanced", "all"):
        try:
            results["advanced"] = vars(advanced_backtest(games, start_season=args.start_season, config=config))
        except ValueError as exc:
            results["advanced"] = {"error": str(exc)}
    if args.model in ("market", "all"):
        try:
            results["market"] = vars(market_backtest(games, start_season=args.start_season))
        except ValueError as exc:
            results["market"] = {"error": str(exc)}

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"backtest: seasons >= {args.start_season}")
    for name, result in results.items():
        if "error" in result:
            print(f"  [{name}] {result['error']}")
            continue
        print(f"  [{name}] {result['games']} games")
        print(f"      winner accuracy : {result['accuracy']:.1%}")
        print(f"      Brier score     : {result['brier_score']:.4f}  (lower is better, 0.25 = coin flip)")
        print(f"      spread MAE      : {result['spread_mae']:.2f} points")
    return 0


# ----------------------------------------------------------- export-site --

def cmd_export_site(args: argparse.Namespace) -> int:
    games = load_games(args.games)
    upcoming = load_upcoming_games(args.games)
    config = _elo_config(args)

    elo_predictor = FootballPredictor(elo=EloRatingSystem(config=config))
    elo_predictor.train(games)

    advanced_predictor = AdvancedPredictor(elo=EloRatingSystem(config=config))
    advanced_predictor.train(games)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    elo_predictor.save(DEFAULT_ELO_RATINGS_PATH)
    advanced_predictor.save(DEFAULT_ADVANCED_RATINGS_PATH)

    print(f"trained elo + advanced models on {len(games)} completed games")

    performance = {"start_season": args.start_season}
    try:
        performance["elo"] = vars(elo_backtest(games, start_season=args.start_season, config=config))
    except ValueError:
        performance["elo"] = None
    try:
        performance["advanced"] = vars(advanced_backtest(games, start_season=args.start_season, config=config))
    except ValueError:
        performance["advanced"] = None
    try:
        performance["market"] = vars(market_backtest(games, start_season=args.start_season))
    except ValueError:
        performance["market"] = None

    rankings = advanced_predictor.power_rankings()
    power_rankings = [
        {"rank": i, "team": team, "name": name, "name_zh": team_display_name_zh(team), "elo": round(rating, 1)}
        for i, (team, name, rating) in enumerate(rankings, start=1)
    ]

    upcoming_predictions = []
    for game in upcoming:
        context = advanced_predictor.context_for_matchup(game)
        prediction = advanced_predictor.predict(game.home_team, game.away_team, context=context)
        entry = prediction.to_dict()
        entry.update({
            "season": game.season,
            "week": game.week,
            "game_type": game.game_type,
            "date": game.date,
            "home_name": team_display_name(game.home_team),
            "away_name": team_display_name(game.away_team),
            "home_name_zh": team_display_name_zh(game.home_team),
            "away_name_zh": team_display_name_zh(game.away_team),
        })
        if game.spread_line is not None:
            entry["market_spread"] = game.spread_line
        upcoming_predictions.append(entry)

    win_model = advanced_predictor.win_model
    margin_model = advanced_predictor.margin_model

    site_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "teams": [
            {"code": code, "name": name, "name_zh": TEAM_NAMES_ZH.get(code, code)}
            for code, name in sorted(TEAM_NAMES.items())
        ],
        "power_rankings": power_rankings,
        "upcoming": upcoming_predictions,
        "model_performance": performance,
        "client_model": {
            "home_advantage": config.home_advantage,
            "ratings": {team: round(rating, 2) for team, rating in advanced_predictor.elo.ratings.items()},
            "spread_scale": elo_predictor.spread_scale,
            "avg_total_points": advanced_predictor.avg_total_points,
            "feature_names": FEATURE_NAMES,
            "win_model": vars(win_model) if win_model else None,
            "margin_model": vars(margin_model) if margin_model else None,
        },
    }

    data_path = out_dir / "data.json"
    data_path.write_text(json.dumps(site_data, indent=2, sort_keys=False), encoding="utf-8")
    print(f"site data written to {data_path} ({len(upcoming_predictions)} upcoming games, "
          f"{len(power_rankings)} teams ranked)")
    return 0


# ------------------------------------------------------------------ main --

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nflpredict", description="Professional NFL game prediction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="build ratings/model from historical games")
    p_train.add_argument("--games", default=str(DEFAULT_GAMES_PATH), help="path to games CSV")
    p_train.add_argument("--out", default=None, help="path to write the model file")
    p_train.add_argument("--model", choices=["elo", "advanced"], default="advanced", help="which model to train (default: advanced)")
    _add_elo_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_predict = subparsers.add_parser("predict", help="predict the outcome of a matchup")
    p_predict.add_argument("--home", required=True, help="home team code, e.g. KC")
    p_predict.add_argument("--away", required=True, help="away team code, e.g. BUF")
    p_predict.add_argument("--neutral", action="store_true", help="neutral-site game (no home-field bonus)")
    p_predict.add_argument("--model", choices=["elo", "advanced"], default="advanced", help="which model to use (default: advanced)")
    p_predict.add_argument("--ratings", default=None, help="path to the trained model file")
    p_predict.add_argument("--json", action="store_true", help="output machine-readable JSON")
    p_predict.add_argument("--div", action="store_true", help="divisional game")
    p_predict.add_argument("--playoff", action="store_true", help="playoff game")
    p_predict.add_argument("--home-rest", type=int, default=None, help="home team days of rest (default: 7)")
    p_predict.add_argument("--away-rest", type=int, default=None, help="away team days of rest (default: 7)")
    p_predict.add_argument("--home-qb-change", action="store_true", help="home team is starting a new/backup QB")
    p_predict.add_argument("--away-qb-change", action="store_true", help="away team is starting a new/backup QB")
    p_predict.add_argument("--cold", action="store_true", help="cold-weather outdoor game (<=32F)")
    p_predict.add_argument("--windy", action="store_true", help="windy outdoor game (>=15mph)")
    p_predict.set_defaults(func=cmd_predict)

    p_ratings = subparsers.add_parser("ratings", help="show current power rankings")
    p_ratings.add_argument("--model", choices=["elo", "advanced"], default="advanced")
    p_ratings.add_argument("--ratings", default=None, help="path to the trained model file")
    p_ratings.add_argument("--top", type=int, default=0, help="show only the top N teams")
    p_ratings.add_argument("--team", default=None, help="show only this team code")
    p_ratings.add_argument("--json", action="store_true", help="output machine-readable JSON")
    p_ratings.set_defaults(func=cmd_ratings)

    p_backtest = subparsers.add_parser("backtest", help="validate accuracy on historical seasons")
    p_backtest.add_argument("--games", default=str(DEFAULT_GAMES_PATH), help="path to games CSV")
    p_backtest.add_argument("--start-season", type=int, default=DEFAULT_BACKTEST_START_SEASON,
                             help=f"first season to score (default: {DEFAULT_BACKTEST_START_SEASON})")
    p_backtest.add_argument("--model", choices=["elo", "advanced", "market", "all"], default="all")
    p_backtest.add_argument("--json", action="store_true", help="output machine-readable JSON")
    _add_elo_args(p_backtest)
    p_backtest.set_defaults(func=cmd_backtest)

    p_export = subparsers.add_parser("export-site", help="train models and export docs/data.json for the GitHub Pages site")
    p_export.add_argument("--games", default=str(DEFAULT_GAMES_PATH), help="path to games CSV")
    p_export.add_argument("--out-dir", default=str(DEFAULT_SITE_DIR), help="directory to write data.json into (default: docs/)")
    p_export.add_argument("--start-season", type=int, default=DEFAULT_BACKTEST_START_SEASON,
                           help=f"first season to score for the transparency report (default: {DEFAULT_BACKTEST_START_SEASON})")
    _add_elo_args(p_export)
    p_export.set_defaults(func=cmd_export_site)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
