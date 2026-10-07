"""A squad's gameweek score: lineup, autosubs, captaincy and chips (PLAN §5 Simulator).

FPL's rules, as decided for the backtester (Phase 3 plan, Decisions):

- "Played" = minutes > 0 summed over the GW's fixtures; a blank (no row) is did-not-play.
- Autosubs: the bench GK (bench[0]) can only replace the starting GK. The outfield bench
  players, in bench order, each replace the first non-playing outfield starter (in FPL pick
  order: by position, then lineup order) whose replacement keeps the formation within the
  position minimums and maximums. Bench players who didn't play are skipped.
- Bench Boost: all 15 count, no autosubs.
- Captain ×2 (Triple Captain ×3) if he played, else the vice if he played, else nobody.

Points are generic: realized int points or xG-scored floats are summed as given, so the
same function scores both metrics (with the same minutes, hence the same autosubs).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from fplopt.backtest.rules import CHIP_NAMES, Rules

Points = int | float


class InvalidLineup(ValueError):
    """A lineup breaks the formation or selection rules."""


@dataclass(frozen=True)
class Lineup:
    """The 11 starters, the 4 bench players in autosub order (bench[0] = the GK), and the
    captain and vice (both starters). Player keys are `player_key`s."""

    starters: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    vice: int


@dataclass(frozen=True)
class GwScore:
    """`points` = Σ counted players' points × their multiplier. `counted` are the players
    whose points count (the XI after autosubs, or all 15 under Bench Boost); `multipliers`
    maps each of them to 1, 2 or 3. `captain_used` is the player who got the armband's
    multiplier (captain, else vice, else None). `bench_points` = points of the squad
    players who did not count (0 under Bench Boost)."""

    points: Points
    counted: tuple[int, ...]
    multipliers: Mapping[int, int]
    captain_used: int | None
    autosubs: tuple[tuple[int, int], ...]
    bench_points: Points


def _formation_ok(counts: Mapping[int, int], rules: Rules) -> bool:
    return all(
        rules.play_min[et] <= counts.get(et, 0) <= rules.play_max[et] for et in rules.play_min
    )


def validate_lineup(lineup: Lineup, positions: Mapping[int, int], rules: Rules) -> None:
    """Raise InvalidLineup unless: 11 starters and 4 bench players, all distinct and with a
    known position; the squad has `squad_select` players per position; the starters' formation
    is within `play_min`/`play_max`; bench[0] is the GK; captain ≠ vice, both starters."""
    starters, bench = tuple(lineup.starters), tuple(lineup.bench)
    if len(starters) != rules.squad_play:
        raise InvalidLineup(f"{len(starters)} starters, need {rules.squad_play}")
    if len(bench) != rules.squad_size - rules.squad_play:
        raise InvalidLineup(
            f"{len(bench)} bench players, need {rules.squad_size - rules.squad_play}"
        )
    squad = starters + bench
    if len(set(squad)) != len(squad):
        raise InvalidLineup(f"duplicate players in lineup: {sorted(squad)}")
    unknown = [key for key in squad if key not in positions]
    if unknown:
        raise InvalidLineup(f"no position for players {unknown}")
    counts = Counter(positions[key] for key in squad)
    if dict(counts) != dict(rules.squad_select):
        raise InvalidLineup(f"squad positions {dict(counts)} != {dict(rules.squad_select)}")
    formation = Counter(positions[key] for key in starters)
    if not _formation_ok(formation, rules):
        raise InvalidLineup(f"invalid formation {dict(sorted(formation.items()))}")
    if positions[bench[0]] != 1:
        raise InvalidLineup(f"bench[0] ({bench[0]}) must be the goalkeeper")
    if lineup.captain == lineup.vice:
        raise InvalidLineup("captain and vice must differ")
    for role, key in (("captain", lineup.captain), ("vice", lineup.vice)):
        if key not in starters:
            raise InvalidLineup(f"{role} {key} is not a starter")


def score_gameweek(
    lineup: Lineup,
    outcomes: Mapping[int, tuple[Points, int]],
    positions: Mapping[int, int],
    rules: Rules,
    chip: str | None = None,
) -> GwScore:
    """Score a lineup with `outcomes` (player_key -> (points, minutes), summed over the
    GW's fixtures, e.g. from `gw_outcomes`). Players missing from `outcomes` scored 0 in 0
    minutes. `chip` is None or a chip name; only `bboost` and `3xc` change the score."""
    if chip is not None and chip not in CHIP_NAMES:
        raise ValueError(f"unknown chip {chip!r}")
    validate_lineup(lineup, positions, rules)

    def points(key: int) -> Points:
        return outcomes.get(key, (0, 0))[0]

    def played(key: int) -> bool:
        return outcomes.get(key, (0, 0))[1] > 0

    xi = list(lineup.starters)
    autosubs: list[tuple[int, int]] = []
    if chip == "bboost":
        counted = xi + list(lineup.bench)
    else:
        formation = Counter(positions[key] for key in xi)
        # FPL pick order: by position, then the order given.
        order = sorted(range(len(xi)), key=lambda i: (positions[xi[i]], i))
        for sub in lineup.bench:
            if not played(sub):
                continue
            sub_et = positions[sub]
            for i in order:
                out = xi[i]
                out_et = positions[out]
                if played(out) or (out_et == 1) != (sub_et == 1):
                    continue
                trial = formation.copy()
                trial[out_et] -= 1
                trial[sub_et] += 1
                if _formation_ok(trial, rules):
                    xi[i], formation = sub, trial
                    autosubs.append((out, sub))
                    break
        counted = xi

    multiplier = 3 if chip == "3xc" else 2
    captain_used = next((key for key in (lineup.captain, lineup.vice) if played(key)), None)
    multipliers = {key: 1 for key in counted}
    if captain_used is not None:
        multipliers[captain_used] = multiplier
    total = sum(points(key) * multipliers[key] for key in counted)
    bench = [key for key in (*lineup.starters, *lineup.bench) if key not in multipliers]
    return GwScore(
        points=total,
        counted=tuple(counted),
        multipliers=MappingProxyType(multipliers),
        captain_used=captain_used,
        autosubs=tuple(autosubs),
        bench_points=sum(points(key) for key in bench),
    )


def gw_outcomes(
    scored: pd.DataFrame, points: str = "points", minutes: str = "minutes"
) -> dict[int, tuple[Points, int]]:
    """player_key -> (points, minutes) summed over the GW's rows (a double GW has two rows
    per player), from a frame with `player_key`, `points` and `minutes` columns. Int points
    stay int; float (xG-scored) points stay float."""
    sums = scored.groupby("player_key", sort=True)[[points, minutes]].sum()
    # itertuples yields Python scalars (int stays int, float stays float).
    return {int(key): (pts, int(mins)) for key, pts, mins in sums.itertuples(index=True, name=None)}
