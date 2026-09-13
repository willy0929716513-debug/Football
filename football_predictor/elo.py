"""Elo rating system tuned for the NFL, following the methodology popularized
by FiveThirtyEight: margin-of-victory scaling, a home-field bonus, a playoff
weight boost, and between-season regression to the mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_RATING = 1500.0


@dataclass
class EloConfig:
    k_factor: float = 20.0
    home_advantage: float = 48.0
    playoff_multiplier: float = 1.2
    revert_to_mean: float = 1.0 / 3.0
    mov_divisor_base: float = 2.2
    mov_divisor_scale: float = 0.001


@dataclass
class EloRatingSystem:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    _current_season: int | None = field(default=None, repr=False)

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, DEFAULT_RATING)

    def expected_home_win_prob(self, home_team: str, away_team: str, neutral: bool = False) -> float:
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        advantage = 0.0 if neutral else self.config.home_advantage
        diff = (home_rating + advantage) - away_rating
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def start_season(self, season: int) -> None:
        if self._current_season is None:
            self._current_season = season
            return
        if season == self._current_season:
            return
        revert = self.config.revert_to_mean
        for team in self.ratings:
            self.ratings[team] = self.ratings[team] * (1 - revert) + DEFAULT_RATING * revert
        self._current_season = season

    def process_game(
        self,
        season: int,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        is_playoff: bool = False,
        neutral: bool = False,
    ) -> float:
        """Update ratings for one final game result.

        Returns the pre-game home win probability, useful for calibration
        and backtesting.
        """
        self.start_season(season)

        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        home_win_prob = self.expected_home_win_prob(home_team, away_team, neutral=neutral)

        if home_score > away_score:
            actual = 1.0
        elif home_score < away_score:
            actual = 0.0
        else:
            actual = 0.5

        margin = abs(home_score - away_score)
        winner_diff = (home_rating - away_rating) if home_score >= away_score else (away_rating - home_rating)
        mov_multiplier = math.log(margin + 1) * (
            self.config.mov_divisor_base
            / (winner_diff * self.config.mov_divisor_scale + self.config.mov_divisor_base)
        )

        k = self.config.k_factor * (self.config.playoff_multiplier if is_playoff else 1.0)
        delta = k * mov_multiplier * (actual - home_win_prob)

        self.ratings[home_team] = home_rating + delta
        self.ratings[away_team] = away_rating - delta

        return home_win_prob

    def power_rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def to_dict(self) -> dict:
        return {
            "config": vars(self.config),
            "ratings": dict(self.ratings),
            "current_season": self._current_season,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EloRatingSystem":
        config = EloConfig(**data.get("config", {}))
        system = cls(config=config, ratings=dict(data.get("ratings", {})))
        system._current_season = data.get("current_season")
        return system
