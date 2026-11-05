//! A gameweek's outcomes per player (Python `fplopt.backtest.simulator.gw_matches` /
//! `read_outcomes`, `backtest.xg_points`): the GW's `player_match` rows with the player's
//! position, re-scored under the backtest rules (realized points) and xG-scored (expected
//! goals, assists, clean sheets and goals conceded in place of the realized ones), summed per
//! player over the GW's fixtures.
//!
//! xG points: player xG/xA come as a pair from one source, Understat when both are non-null,
//! else FPL; the opponent's team xG is the first non-null of Understat, FPL and
//! football-data. Clean sheets (minutes ≥ `long_play_minutes`) score CS points × P(N = 0) and
//! goals conceded (any minutes) GC points × E[floor(N / per point)], N ~ Poisson(opponent xG
//! × minutes / 90). Null when a needed input is missing; a player-GW is null if any of its
//! rows is.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};
use fplopt_build::scoring::{Rules, StatLine, score};
use fplopt_core::seasons::season_label;
use fplopt_store::{AsOfView, DataStore, Micros, Table};

/// Results are available at the lockdown and views are strict.
pub const OUTCOME_DELAY: Micros = 1;
/// N = 0..POISSON_CAP in the Poisson sums (the tail is below 1e-15 for any match xG).
pub const POISSON_CAP: usize = 40;

pub const MATCH_COLUMNS: [&str; 26] = [
    "player_key",
    "season",
    "gw",
    "fixture_key",
    "opponent_team_key",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "us_xg",
    "us_xa",
    "fpl_xg",
    "fpl_xa",
    "starts",
];
const TEAM_COLUMNS: [&str; 6] = ["fixture_key", "team_key", "season", "us_xg", "fpl_xg", "fd_xg"];

/// One GW's outcomes per player_key: realized (points, minutes) and xG-scored (points,
/// minutes; NaN points when not computable). Players without a row are absent.
#[derive(Clone, Debug, Default)]
pub struct GwOutcomes {
    pub realized: HashMap<i64, (i64, i64)>,
    pub xg: HashMap<i64, (f64, i64)>,
}

impl GwOutcomes {
    pub fn is_empty(&self) -> bool {
        self.realized.is_empty()
    }
}

/// One scored row of `gw_matches`.
#[derive(Clone, Debug)]
pub struct MatchRow {
    pub player_key: i64,
    pub fixture_key: i64,
    pub opponent_team_key: Option<i64>,
    pub element_type: i64,
    pub line: StatLine,
    pub us: (Option<f64>, Option<f64>),
    pub fpl: (Option<f64>, Option<f64>),
    /// Null where the source has no starts (before 2022/23 GW16).
    pub starts: Option<i64>,
}

/// The `player_match` rows of (season, gw) visible in `view` with the player's
/// `player_season.element_type` (rows without one are dropped with a warning), in the
/// table's order.
pub fn gw_matches(view: &AsOfView, season: i64, gw: i64) -> Result<Vec<MatchRow>> {
    let m = view.table("player_match", Some(&MATCH_COLUMNS))?;
    let (seasons, gws) = (m.i64("season")?, m.i64("gw")?);
    let rows: Vec<usize> =
        (0..m.height()).filter(|&i| seasons[i] == season && gws[i] == gw).collect();
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let ps = view.table("player_season", Some(&["player_key", "season", "element_type"]))?;
    let (pk, ps_season, ps_et) =
        (ps.i64("player_key")?, ps.i64("season")?, ps.opt_i64("element_type")?);
    let mut position: HashMap<i64, Option<i64>> = HashMap::new();
    for i in 0..ps.height() {
        if ps_season[i] == season {
            position.insert(pk[i], ps_et[i]); // keep="last"
        }
    }
    let ints = |name: &str| m.opt_i64(name);
    let floats = |name: &str| m.opt_f64(name);
    let keys = m.i64("player_key")?;
    let fixture = m.i64("fixture_key")?;
    let opponent = ints("opponent_team_key")?;
    let stat = |name: &str| -> Result<Vec<Option<i64>>> { ints(name) };
    let (minutes, goals, assists, cs, gc) = (
        stat("minutes")?,
        stat("goals_scored")?,
        stat("assists")?,
        stat("clean_sheets")?,
        stat("goals_conceded")?,
    );
    let (og, ps_saved, pm, yc, rc, saves, bonus) = (
        stat("own_goals")?,
        stat("penalties_saved")?,
        stat("penalties_missed")?,
        stat("yellow_cards")?,
        stat("red_cards")?,
        stat("saves")?,
        stat("bonus")?,
    );
    let (dc, cbi, tackles, recoveries) = (
        stat("defensive_contribution")?,
        stat("clearances_blocks_interceptions")?,
        stat("tackles")?,
        stat("recoveries")?,
    );
    let starts = stat("starts")?;
    let (us_xg, us_xa, fpl_xg, fpl_xa) =
        (floats("us_xg")?, floats("us_xa")?, floats("fpl_xg")?, floats("fpl_xa")?);
    let mut out = Vec::with_capacity(rows.len());
    let mut unknown = 0;
    for i in rows {
        let Some(Some(et)) = position.get(&keys[i]).copied() else {
            unknown += 1;
            continue;
        };
        let need = |v: &[Option<i64>], name: &str| -> Result<i64> {
            match v[i] {
                Some(x) => Ok(x),
                None => bail!("score_matches: null values in ['{name}']"),
            }
        };
        let line = StatLine {
            element_type: et,
            minutes: need(&minutes, "minutes")?,
            goals_scored: need(&goals, "goals_scored")?,
            assists: need(&assists, "assists")?,
            clean_sheets: need(&cs, "clean_sheets")?,
            goals_conceded: need(&gc, "goals_conceded")?,
            own_goals: need(&og, "own_goals")?,
            penalties_saved: need(&ps_saved, "penalties_saved")?,
            penalties_missed: need(&pm, "penalties_missed")?,
            yellow_cards: need(&yc, "yellow_cards")?,
            red_cards: need(&rc, "red_cards")?,
            saves: need(&saves, "saves")?,
            bonus: need(&bonus, "bonus")?,
            defensive_contribution: dc[i],
            clearances_blocks_interceptions: cbi[i],
            tackles: tackles[i],
            recoveries: recoveries[i],
        };
        out.push(MatchRow {
            player_key: keys[i],
            fixture_key: fixture[i],
            opponent_team_key: opponent[i],
            element_type: et,
            line,
            us: (us_xg[i], us_xa[i]),
            fpl: (fpl_xg[i], fpl_xa[i]),
            starts: starts[i],
        });
    }
    if unknown > 0 {
        log::warn!(
            "{} GW{gw}: {unknown} player_match row(s) without a player_season position are not \
             scored",
            season_label(season as i32)
        );
    }
    Ok(out)
}

/// (fixture_key, team_key) -> team xG (first non-null of Understat, summed FPL,
/// football-data), from `team_match` rows (the last row of a key wins).
pub fn team_xg_table(teams: &Table) -> Result<HashMap<(i64, i64), Option<f64>>> {
    let (fixture, team) = (teams.i64("fixture_key")?, teams.i64("team_key")?);
    let sources = [teams.opt_f64("us_xg")?, teams.opt_f64("fpl_xg")?, teams.opt_f64("fd_xg")?];
    let mut out = HashMap::new();
    for i in 0..teams.height() {
        let xg = sources.iter().find_map(|s| s[i].filter(|x| !x.is_nan()));
        out.insert((fixture[i], team[i]), xg);
    }
    Ok(out)
}

/// E[floor(N / k)] for N ~ Poisson(rate), N = 0..POISSON_CAP (p(n) = p(n−1) × rate / n).
pub fn poisson_expected_floor(rate: f64, k: i64) -> f64 {
    let base = (-rate).exp();
    let mut product = 1.0;
    let mut total = 0.0;
    for n in 0..=POISSON_CAP {
        if n > 0 {
            product *= rate / n as f64;
        }
        total += base * product * (n as i64).div_euclid(k) as f64;
    }
    total
}

/// The xG-scored points of one row (None when an input it needs is missing).
pub fn xg_points(row: &MatchRow, opp_xg: Option<f64>, rules: &Rules) -> Result<Option<f64>> {
    let m = &row.line;
    let et = row.element_type;
    let realized = score(m, rules)?;
    let attack_def = m.goals_scored * rules.goals_scored[&et]
        + m.assists * rules.assists
        + m.clean_sheets * rules.clean_sheets[&et]
        + m.goals_conceded.div_euclid(rules.goals_conceded_per_point) * rules.goals_conceded[&et];
    let rest = (realized - attack_def) as f64;
    let played = m.minutes > 0;
    let long_play = m.minutes >= rules.long_play_minutes;
    let pair = |(xg, xa): (Option<f64>, Option<f64>)| match (xg, xa) {
        (Some(g), Some(a)) if !g.is_nan() && !a.is_nan() => Some((g, a)),
        _ => None,
    };
    let player = pair(row.us).or_else(|| pair(row.fpl));
    let cs_points = rules.clean_sheets[&et];
    let gc_points = rules.goals_conceded[&et];
    let needs_cs = long_play && cs_points != 0;
    let needs_gc = played && gc_points != 0;
    if (played && player.is_none()) || ((needs_cs || needs_gc) && opp_xg.is_none()) {
        return Ok(None);
    }
    let rate = opp_xg.unwrap_or(0.0) * m.minutes as f64 / 90.0;
    let cs_term = if needs_cs { cs_points as f64 * (-rate).exp() } else { 0.0 };
    let gc_term = if needs_gc {
        gc_points as f64 * poisson_expected_floor(rate, rules.goals_conceded_per_point)
    } else {
        0.0
    };
    let attack = match player {
        Some((xg, xa)) if played => xg * rules.goals_scored[&et] as f64 + xa * rules.assists as f64,
        _ => 0.0,
    };
    Ok(Some(rest + attack + cs_term + gc_term))
}

/// pandas' grouped float sum (Kahan, in row order).
#[derive(Clone, Copy, Default)]
struct Kahan {
    sum: f64,
    compensation: f64,
}

impl Kahan {
    fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = t - self.sum - y;
        self.sum = t;
    }
}

/// Outcomes of (season, gw) from `store.as_of(lockdown + 1 µs)`.
pub fn read_outcomes(
    store: &DataStore,
    rules: &Rules,
    season: i64,
    gw: i64,
    lockdown: Micros,
) -> Result<GwOutcomes> {
    let view = store.as_of(lockdown + OUTCOME_DELAY);
    let rows = gw_matches(&view, season, gw)?;
    if rows.is_empty() {
        return Ok(GwOutcomes::default());
    }
    let teams = view.table("team_match", Some(&TEAM_COLUMNS))?;
    let fixtures: std::collections::HashSet<i64> = rows.iter().map(|r| r.fixture_key).collect();
    let keep: Vec<usize> = {
        let f = teams.i64("fixture_key")?;
        (0..teams.height()).filter(|&i| fixtures.contains(&f[i])).collect()
    };
    let team_xg = team_xg_table(&teams.take(&keep))?;
    let mut realized: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    let mut xg: BTreeMap<i64, (Kahan, i64, bool)> = BTreeMap::new();
    for row in &rows {
        let points = score(&row.line, rules)?;
        let opp = row
            .opponent_team_key
            .and_then(|o| team_xg.get(&(row.fixture_key, o)).copied().flatten());
        let x = xg_points(row, opp, rules)?;
        let r = realized.entry(row.player_key).or_default();
        r.0 += points;
        r.1 += row.line.minutes;
        let e = xg.entry(row.player_key).or_insert((Kahan::default(), 0, false));
        match x {
            Some(v) => e.0.add(v),
            None => e.2 = true,
        }
        e.1 += row.line.minutes;
    }
    Ok(GwOutcomes {
        realized: realized.into_iter().collect(),
        xg: xg
            .into_iter()
            .map(|(k, (sum, mins, null))| (k, (if null { f64::NAN } else { sum.sum }, mins)))
            .collect(),
    })
}
