//! Value-of-information oracles (spec `docs/superpowers/specs/2026-11-09-value-of-information-design.md`,
//! plan `docs/superpowers/plans/2026-11-09-value-of-information.md`): xP models
//! `v1+oracle-<ARM>` that are v1 with one input replaced by the truth, to measure what
//! knowing that input perfectly is worth to the planner in season points.
//!
//! **This reads the future by design.** `Truth::load` reads a whole season from a view that
//! sees everything. Only the backtester's decision paths build these models
//! (`Caches::xp_for`); the model names are not in `fplopt_models::MODELS`, so `optimize
//! plan`, `models eval`, the bot and `live` cannot ask for them, and the leakage check lists
//! them as excluded (`oracle::LEAKAGE_EXCLUDED`).
//!
//! No arm uses a realized goal, assist, clean sheet or point: only who played and started,
//! the closing odds, and season-long scoring rates.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, bail};
use fplopt_core::seasons::HOLDOUT_SEASONS;
use fplopt_features::history::infer_starts;
use fplopt_models::assemble::{Fixture, V1Params, set_play};
use fplopt_models::team::closing_lambdas;
use fplopt_models::util::opt_ts;
use fplopt_store::{AsOfView, DataStore};

/// The model-name prefix: `v1+oracle-M0` etc.
pub const PREFIX: &str = "v1+oracle-";
/// The v1 model the oracles are built on.
pub const BASE_MODEL: &str = "v1";
/// Arm R replaces a player's rates only with at least this many season minutes with xG.
pub const R_MIN_MINUTES: f64 = 450.0;
/// Arm R's first season (Understat player xG covers 91%+ of minutes from 2019/20).
pub const R_FIRST_SEASON: i64 = 2019;
/// The flag arms' first season (status flags are archived from 2021/22).
pub const FLAG_FIRST_SEASON: i64 = 2021;
/// M2 knows horizons 0..=this (the coming GW and the two after).
pub const M2_MAX_HORIZON: i64 = 2;

/// What an oracle model knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arm {
    /// Who plays and starts in the coming GW.
    M0,
    /// M0 for a deterministic half of the players.
    M0h,
    /// Who plays and starts in every horizon GW.
    MA,
    /// MA for the players flagged at the deadline only (`flagged_at`).
    MAf,
    /// MA for the players not flagged at the deadline only.
    MAu,
    /// MA for the coming GW and the two after only.
    M2,
    /// Who plays in the coming GW (not who starts).
    A0,
    /// The closing odds' team λ in the coming GW.
    T0,
    /// Each player's season npxG and xA per 90, in every horizon GW: a team's expected
    /// non-penalty goals and assists are re-split among its players by true rate × expected
    /// minutes (the team's totals stay v1's).
    R,
}

pub const ARMS: [Arm; 9] =
    [Arm::M0, Arm::M0h, Arm::MA, Arm::MAf, Arm::MAu, Arm::M2, Arm::A0, Arm::T0, Arm::R];

impl Arm {
    pub fn code(self) -> &'static str {
        match self {
            Arm::M0 => "M0",
            Arm::M0h => "M0h",
            Arm::MA => "MA",
            Arm::MAf => "MAf",
            Arm::MAu => "MAu",
            Arm::M2 => "M2",
            Arm::A0 => "A0",
            Arm::T0 => "T0",
            Arm::R => "R",
        }
    }

    /// The arm of an oracle model name (`v1+oracle-M0`), None for any other name.
    pub fn parse(model: &str) -> Option<Arm> {
        let code = model.strip_prefix(PREFIX)?;
        ARMS.into_iter().find(|a| a.code() == code)
    }

    pub fn model(self) -> String {
        format!("{PREFIX}{}", self.code())
    }

    /// Whether the arm needs the deadline's flag set (`flagged_at`).
    pub fn needs_flags(self) -> bool {
        matches!(self, Arm::MAf | Arm::MAu)
    }
}

/// The players flagged at the view's deadline: status not `a` in the newest snapshot of the
/// view's season (read through the as-of view, like v1's own flags). Empty before the
/// season's first snapshot.
pub fn flagged_at(view: &AsOfView) -> Result<HashSet<i64>> {
    let (season, _) = view.gameweek_for_deadline()?;
    let columns = ["snapshot_at", "season", "status"];
    let snaps = view.latest("player_snapshot", &["player_key"], Some(&columns))?;
    let keys = snaps.i64("player_key")?;
    let seasons = snaps.i64("season")?;
    let at = opt_ts(&snaps, "snapshot_at")?;
    let status = snaps.opt_str("status")?;
    let in_season: Vec<usize> = (0..snaps.height()).filter(|&i| seasons[i] == season).collect();
    let Some(newest) = in_season.iter().filter_map(|&i| at[i]).max() else {
        return Ok(HashSet::new());
    };
    Ok(in_season
        .into_iter()
        .filter(|&i| at[i] == Some(newest) && status[i].as_deref().is_some_and(|s| s != "a"))
        .map(|i| keys[i])
        .collect())
}

/// The oracle model names, for error messages.
pub fn model_names() -> Vec<String> {
    ARMS.into_iter().map(Arm::model).collect()
}

/// A player's season scoring rates (Understat, per 90).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    pub npxg90: f64,
    pub xa90: f64,
}

/// One season's truth. **Reads the future.**
#[derive(Clone, Debug, Default)]
pub struct Truth {
    pub season: i64,
    /// (player_key, fixture_key) → (played, started).
    play: HashMap<(i64, i64), (bool, bool)>,
    /// The (fixture_key, team_key) sides with at least one player row (the match was played).
    sides: HashSet<(i64, i64)>,
    /// fixture_key → (home team, λ_home, λ_away, P(home CS), P(away CS)) from closing odds.
    closing: HashMap<i64, (i64, f64, f64, f64, f64)>,
    /// player_key → season rates (players with at least `R_MIN_MINUTES` with xG).
    rates: HashMap<i64, Rates>,
}

/// What an edit did, for the run's log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditCount {
    pub rows: usize,
    pub changed: usize,
    /// Rows left as v1's because the truth has nothing for them.
    pub unknown: usize,
    /// Rows that now play but had no expected minutes before (expectations stay 0).
    pub no_rate: usize,
}

fn refuse_holdout(season: i64) -> Result<()> {
    if HOLDOUT_SEASONS.iter().any(|&s| i64::from(s) == season) {
        bail!("v1+oracle: season {season} is the holdout season");
    }
    Ok(())
}

/// splitmix64's finalizer: a fixed hash for M0h's half.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Whether M0h knows this player in this GW (a fixed half of the player-GWs).
pub fn in_half(player_key: i64, gw_index: i64) -> bool {
    mix((player_key as u64) ^ mix(gw_index as u64)) & 1 == 0
}

impl Truth {
    /// `season`'s truth from a view that sees everything: **the future, by design**.
    pub fn load(store: &DataStore, season: i64) -> Result<Truth> {
        refuse_holdout(season)?;
        let view = store.as_of(i64::MAX);
        let mut truth = Truth { season, ..Truth::default() };

        // who played and started, and the scoring rates
        let pm = view.table(
            "player_match",
            Some(&[
                "season",
                "player_key",
                "fixture_key",
                "team_key",
                "minutes",
                "starts",
                "us_npxg",
                "us_xa",
            ]),
        )?;
        let seasons = pm.i64("season")?;
        let rows: Vec<usize> = (0..pm.height()).filter(|&i| seasons[i] == season).collect();
        if rows.is_empty() {
            bail!("v1+oracle: no player_match rows for season {season}");
        }
        let take_i = |name: &str| -> Result<Vec<i64>> {
            let col = pm.opt_i64(name)?;
            Ok(rows.iter().map(|&i| col[i].unwrap_or(0)).collect())
        };
        let (players, fixtures, teams) =
            (take_i("player_key")?, take_i("fixture_key")?, take_i("team_key")?);
        let minutes = take_i("minutes")?;
        let starts_col = pm.opt_i64("starts")?;
        let starts: Vec<Option<i64>> = rows.iter().map(|&i| starts_col[i]).collect();
        let inferred = infer_starts(&fixtures, &teams, &minutes, &players);
        let (npxg_col, xa_col) = (pm.opt_f64("us_npxg")?, pm.opt_f64("us_xa")?);
        let mut sums: BTreeMap<i64, (f64, f64, f64)> = BTreeMap::new();
        for (n, &i) in rows.iter().enumerate() {
            let started = starts[n].map_or(inferred[n], |s| s == 1);
            truth.play.insert((players[n], fixtures[n]), (minutes[n] > 0, started));
            truth.sides.insert((fixtures[n], teams[n]));
            if let (Some(g), Some(a)) = (npxg_col[i], xa_col[i]) {
                let e = sums.entry(players[n]).or_default();
                e.0 += g;
                e.1 += a;
                e.2 += minutes[n] as f64;
            }
        }
        truth.rates = sums
            .into_iter()
            .filter(|(_, (_, _, m))| *m >= R_MIN_MINUTES)
            .map(|(k, (g, a, m))| (k, Rates { npxg90: g / m * 90.0, xa90: a / m * 90.0 }))
            .collect();

        // closing odds
        let fx = view.table("fixture", Some(&["season", "fixture_key", "home_team_key"]))?;
        let (fs, fk, fh) =
            (fx.i64("season")?, fx.i64("fixture_key")?, fx.opt_i64("home_team_key")?);
        let home: HashMap<i64, i64> = (0..fx.height())
            .filter(|&i| fs[i] == season)
            .filter_map(|i| fh[i].map(|h| (fk[i], h)))
            .collect();
        let keys: HashSet<i64> = home.keys().copied().collect();
        let params = V1Params::for_model(BASE_MODEL)?.team;
        for (key, (lh, la, cs_h, cs_a)) in closing_lambdas(&view, &keys, &params)? {
            truth.closing.insert(key, (home[&key], lh, la, cs_h, cs_a));
        }
        Ok(truth)
    }

    /// Refuses an arm this season cannot support.
    pub fn check(&self, arm: Arm) -> Result<()> {
        match arm {
            Arm::R if self.season < R_FIRST_SEASON => bail!(
                "{}: season {} is before {R_FIRST_SEASON} (player xG coverage)",
                arm.model(),
                self.season
            ),
            Arm::R if self.rates.is_empty() => {
                bail!("{}: season {} has no player xG", arm.model(), self.season)
            }
            Arm::MAf | Arm::MAu if self.season < FLAG_FIRST_SEASON => bail!(
                "{}: season {} is before {FLAG_FIRST_SEASON} (status flags)",
                arm.model(),
                self.season
            ),
            Arm::T0 if self.closing.is_empty() => {
                bail!("{}: season {} has no closing odds", arm.model(), self.season)
            }
            _ => Ok(()),
        }
    }

    /// (played, started) of a player-fixture: no row for him in a played match means he did
    /// not play; None when his club has no row at all for the fixture (not played as listed:
    /// the final schedule's known gap), so v1's values stand.
    fn played(&self, f: &Fixture) -> Option<(bool, bool)> {
        match self.play.get(&(f.key.player_key, f.key.fixture_key)) {
            Some(&p) => Some(p),
            None if self.sides.contains(&(f.key.fixture_key, f.key.team_key)) => {
                Some((false, false))
            }
            None => None,
        }
    }

    /// Applies `arm` to v1's calibrated per-fixture inputs. `flagged`: the players flagged
    /// at the deadline (`flagged_at`), read by the flag arms only.
    pub fn apply(&self, arm: Arm, flagged: &HashSet<i64>, frame: &mut [Fixture]) -> EditCount {
        let mut count = EditCount::default();
        for f in frame.iter_mut() {
            if f.key.season != self.season {
                continue;
            }
            let h0 = f.key.horizon == 0;
            let in_scope = match arm {
                Arm::M0 | Arm::A0 | Arm::T0 => h0,
                Arm::M0h => h0 && in_half(f.key.player_key, f.key.gw_index),
                Arm::MA | Arm::R => true,
                Arm::MAf => flagged.contains(&f.key.player_key),
                Arm::MAu => !flagged.contains(&f.key.player_key),
                Arm::M2 => f.key.horizon <= M2_MAX_HORIZON,
            };
            if !in_scope {
                continue;
            }
            count.rows += 1;
            match arm {
                Arm::M0 | Arm::M0h | Arm::MA | Arm::MAf | Arm::MAu | Arm::M2 | Arm::A0 => {
                    let Some((played, started)) = self.played(f) else {
                        count.unknown += 1;
                        continue;
                    };
                    let (p_start, p_play) = match (arm, played) {
                        (_, false) => (0.0, 0.0),
                        (Arm::A0, true) if f.p_play > 0.0 => (f.p_start / f.p_play, 1.0),
                        (Arm::A0, true) => (0.5, 1.0),
                        (_, true) => (if started { 1.0 } else { 0.0 }, 1.0),
                    };
                    if !set_play(f, p_start, p_play) {
                        count.no_rate += 1;
                    }
                    count.changed += 1;
                }
                Arm::T0 => {
                    let Some(&(home, lh, la, cs_h, cs_a)) = self.closing.get(&f.key.fixture_key)
                    else {
                        count.unknown += 1;
                        continue;
                    };
                    let (lambda_for, lambda_against, p_cs) =
                        if f.key.team_key == home { (lh, la, cs_h) } else { (la, lh, cs_a) };
                    let ratio = if f.lambda_for > 0.0 { lambda_for / f.lambda_for } else { 1.0 };
                    f.lambda_for = lambda_for;
                    f.lambda_against = lambda_against;
                    f.p_cs = p_cs;
                    f.e_np_goals *= ratio;
                    f.e_pen_goals *= ratio;
                    f.e_assists *= ratio;
                    count.changed += 1;
                }
                Arm::R => {
                    if self.rates.contains_key(&f.key.player_key) {
                        count.changed += 1;
                    } else {
                        count.unknown += 1;
                    }
                }
            }
        }
        if arm == Arm::R {
            self.resplit(frame);
        }
        count
    }

    /// Arm R: per team and fixture, the expected non-penalty goals (and assists) v1 gives the
    /// players with known season rates are re-split among them in proportion to true rate ×
    /// expected minutes. The team's totals, the other players and penalties are unchanged; a
    /// group whose weights sum to 0 is left alone.
    fn resplit(&self, frame: &mut [Fixture]) {
        // (fixture, team) → (v1 goals, v1 assists, goal weights, assist weights) of the known
        let mut groups: BTreeMap<(i64, i64), [f64; 4]> = BTreeMap::new();
        let weights = |f: &Fixture, r: &Rates| (r.npxg90 * f.e_minutes, r.xa90 * f.e_minutes);
        for f in frame.iter().filter(|f| f.key.season == self.season) {
            if let Some(r) = self.rates.get(&f.key.player_key) {
                let (wg, wa) = weights(f, r);
                let g = groups.entry((f.key.fixture_key, f.key.team_key)).or_default();
                g[0] += f.e_np_goals;
                g[1] += f.e_assists;
                g[2] += wg;
                g[3] += wa;
            }
        }
        for f in frame.iter_mut().filter(|f| f.key.season == self.season) {
            let Some(r) = self.rates.get(&f.key.player_key) else { continue };
            let g = groups[&(f.key.fixture_key, f.key.team_key)];
            let (wg, wa) = weights(f, r);
            if g[2] > 0.0 {
                f.e_np_goals = g[0] * wg / g[2];
            }
            if g[3] > 0.0 {
                f.e_assists = g[1] * wa / g[3];
            }
        }
    }

    #[cfg(test)]
    pub fn for_test(season: i64) -> Truth {
        Truth { season, ..Truth::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_models::minutes::Key;

    const SEASON: i64 = 2022;

    fn fixture(player: i64, fixture_key: i64, team: i64, horizon: i64) -> Fixture {
        Fixture {
            key: Key {
                player_key: player,
                fixture_key,
                team_key: team,
                season: SEASON,
                gw: 10 + horizon,
                gw_index: 10 + horizon,
                horizon,
            },
            p_start: 0.6,
            p_sub: 0.2,
            p_play: 0.8,
            p_60: 0.54,
            e_minutes: 52.0,
            c60: 0.9,
            csub: 0.5,
            m_start: 80.0,
            m_sub: 20.0,
            lambda_for: 1.5,
            lambda_against: 1.0,
            p_cs: 0.35,
            e_np_goals: 0.26,
            e_pen_goals: 0.0,
            e_assists: 0.13,
            ..Fixture::default()
        }
    }

    /// Player 1 started fixture 100 and came on in 101; player 2 has no row in 100 (his club,
    /// team 7, played it); fixture 102 has no rows at all.
    fn truth() -> Truth {
        let mut t = Truth::for_test(SEASON);
        t.play.insert((1, 100), (true, true));
        t.play.insert((1, 101), (true, false));
        t.sides.insert((100, 7));
        t.sides.insert((101, 7));
        t
    }

    fn none() -> HashSet<i64> {
        HashSet::new()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn names_round_trip_and_other_names_are_not_arms() {
        for arm in ARMS {
            assert_eq!(Arm::parse(&arm.model()), Some(arm));
        }
        assert_eq!(Arm::M0.model(), "v1+oracle-M0");
        for name in ["v1", "v1+oracle-", "v1+oracle-ZZ", "v1+oracle-m0", "rolling"] {
            assert_eq!(Arm::parse(name), None, "{name}");
        }
    }

    #[test]
    fn m0_sets_the_coming_gw_only_and_ma_every_gw() {
        let frame = || vec![fixture(1, 100, 7, 0), fixture(1, 101, 7, 1), fixture(2, 100, 7, 0)];
        let mut m0 = frame();
        let count = truth().apply(Arm::M0, &none(), &mut m0);
        assert_eq!(count, EditCount { rows: 2, changed: 2, unknown: 0, no_rate: 0 });
        // player 1 started: 80 minutes, expectations × 80 / 52
        assert!(close(m0[0].p_start, 1.0) && close(m0[0].p_play, 1.0));
        assert!(close(m0[0].e_minutes, 80.0) && close(m0[0].e_np_goals, 0.26 * 80.0 / 52.0));
        // horizon 1 is v1's
        assert!(close(m0[1].p_start, 0.6) && close(m0[1].e_np_goals, 0.26));
        // player 2 has no row in a played match: he did not play
        assert!(m0[2].p_play == 0.0 && m0[2].e_minutes == 0.0 && m0[2].e_assists == 0.0);

        let mut ma = frame();
        let count = truth().apply(Arm::MA, &none(), &mut ma);
        assert_eq!(count.rows, 3);
        // horizon 1: he came on
        assert!(close(ma[1].p_start, 0.0) && close(ma[1].p_play, 1.0));
        assert!(close(ma[1].e_minutes, 20.0));
    }

    #[test]
    fn the_flag_arms_split_ma_by_player_and_m2_by_horizon() {
        // player 1 (flagged) at horizons 0, 1 and 3; player 2 (not flagged) at horizon 0
        let mut t = truth();
        t.play.insert((1, 104), (true, true));
        t.sides.insert((104, 7));
        let frame = || {
            vec![
                fixture(1, 100, 7, 0),
                fixture(1, 101, 7, 1),
                fixture(1, 104, 7, 3),
                fixture(2, 100, 7, 0),
            ]
        };
        let flagged: HashSet<i64> = [1].into_iter().collect();
        let mut ma = frame();
        t.apply(Arm::MA, &flagged, &mut ma);

        let mut maf = frame();
        let count = t.apply(Arm::MAf, &flagged, &mut maf);
        assert_eq!(count, EditCount { rows: 3, changed: 3, unknown: 0, no_rate: 0 });
        let mut mau = frame();
        let count = t.apply(Arm::MAu, &flagged, &mut mau);
        assert_eq!(count, EditCount { rows: 1, changed: 1, unknown: 0, no_rate: 0 });
        let v1 = frame();
        for i in 0..4 {
            let (in_f, in_u) = if i < 3 { (&ma[i], &v1[i]) } else { (&v1[i], &ma[i]) };
            assert!(close(maf[i].p_play, in_f.p_play) && close(maf[i].e_minutes, in_f.e_minutes));
            assert!(close(mau[i].p_play, in_u.p_play) && close(mau[i].e_minutes, in_u.e_minutes));
        }
        // with nobody flagged MAf changes nothing and MAu is MA
        assert_eq!(t.apply(Arm::MAf, &none(), &mut frame()), EditCount::default());
        assert_eq!(t.apply(Arm::MAu, &none(), &mut frame()).rows, 4);

        let mut m2 = frame();
        let count = t.apply(Arm::M2, &none(), &mut m2);
        assert_eq!(count.rows, 3);
        assert!(close(m2[1].e_minutes, ma[1].e_minutes) && close(m2[2].e_minutes, 52.0));
        assert!(close(ma[2].e_minutes, 80.0));

        let early = Truth::for_test(2020);
        assert!(early.check(Arm::MAf).is_err() && early.check(Arm::MAu).is_err());
        assert!(early.check(Arm::M2).is_ok() && truth().check(Arm::MAf).is_ok());
    }

    #[test]
    fn a0_knows_who_plays_but_keeps_v1s_share_of_starts() {
        let mut frame = vec![fixture(1, 100, 7, 0), fixture(2, 100, 7, 0)];
        truth().apply(Arm::A0, &none(), &mut frame);
        assert!(close(frame[0].p_play, 1.0) && close(frame[0].p_start, 0.75));
        assert!(close(frame[0].e_minutes, 0.75 * 80.0 + 0.25 * 20.0));
        assert_eq!(frame[1].p_play, 0.0);
    }

    #[test]
    fn a_fixture_with_no_rows_for_the_club_is_left_alone() {
        let mut frame = vec![fixture(1, 102, 7, 0)];
        let count = truth().apply(Arm::M0, &none(), &mut frame);
        assert_eq!(count, EditCount { rows: 1, changed: 0, unknown: 1, no_rate: 0 });
        assert!(close(frame[0].p_start, 0.6) && close(frame[0].e_minutes, 52.0));
    }

    #[test]
    fn another_seasons_fixtures_are_untouched() {
        let mut frame = vec![fixture(1, 100, 7, 0)];
        frame[0].key.season = SEASON + 1;
        assert_eq!(truth().apply(Arm::M0, &none(), &mut frame), EditCount::default());
        assert!(close(frame[0].p_start, 0.6));
    }

    #[test]
    fn m0h_knows_a_fixed_half() {
        let known = (0..1000).filter(|&p| in_half(p, 10)).count();
        assert!((450..=550).contains(&known), "{known}");
        assert_eq!(known, (0..1000).filter(|&p| in_half(p, 10)).count());
        // the half changes with the GW
        assert!((0..1000).any(|p| in_half(p, 10) != in_half(p, 11)));
        let mut t = truth();
        for p in 0..1000 {
            t.play.insert((p, 100), (true, true));
        }
        let mut frame: Vec<Fixture> = (0..1000).map(|p| fixture(p, 100, 7, 0)).collect();
        let count = t.apply(Arm::M0h, &none(), &mut frame);
        assert_eq!(count.changed, known);
        for f in &frame {
            let want = if in_half(f.key.player_key, 10) { 1.0 } else { 0.6 };
            assert!(close(f.p_start, want));
        }
    }

    #[test]
    fn t0_takes_the_closing_lambdas_by_side_at_the_coming_gw() {
        let mut t = truth();
        t.closing.insert(100, (7, 2.0, 0.8, 0.45, 0.14));
        let mut frame = vec![fixture(1, 100, 7, 0), fixture(3, 100, 8, 0), fixture(1, 101, 7, 1)];
        frame.push(fixture(1, 103, 7, 0)); // no closing odds
        let count = t.apply(Arm::T0, &none(), &mut frame);
        assert_eq!(count, EditCount { rows: 3, changed: 2, unknown: 1, no_rate: 0 });
        // home side
        assert!(close(frame[0].lambda_for, 2.0) && close(frame[0].lambda_against, 0.8));
        assert!(close(frame[0].p_cs, 0.45) && close(frame[0].e_np_goals, 0.26 * 2.0 / 1.5));
        assert!(close(frame[0].e_assists, 0.13 * 2.0 / 1.5));
        // away side
        assert!(close(frame[1].lambda_for, 0.8) && close(frame[1].p_cs, 0.14));
        // later GW and no-odds fixture unchanged
        assert!(close(frame[2].lambda_for, 1.5) && close(frame[3].lambda_for, 1.5));
    }

    #[test]
    fn r_resplits_a_teams_goals_by_true_rate_and_minutes() {
        let mut t = truth();
        t.rates.insert(1, Rates { npxg90: 0.6, xa90: 0.1 });
        t.rates.insert(3, Rates { npxg90: 0.2, xa90: 0.3 });
        // players 1 and 3 of team 7 in fixture 100, player 2 without rates, and player 1
        // again in a later fixture on his own
        let mut frame = vec![
            fixture(1, 100, 7, 0),
            fixture(3, 100, 7, 0),
            fixture(2, 100, 7, 0),
            fixture(1, 101, 7, 3),
        ];
        frame[1].e_minutes = 26.0; // half of player 1's 52
        frame[0].e_pen_goals = 0.05;
        let count = t.apply(Arm::R, &none(), &mut frame);
        assert_eq!(count, EditCount { rows: 4, changed: 3, unknown: 1, no_rate: 0 });
        // goals: 0.52 in all, weights 0.6·52 : 0.2·26 = 6 : 1
        assert!(close(frame[0].e_np_goals, 0.52 * 6.0 / 7.0));
        assert!(close(frame[1].e_np_goals, 0.52 / 7.0));
        // assists: 0.26 in all, weights 0.1·52 : 0.3·26 = 2 : 3
        assert!(close(frame[0].e_assists, 0.26 * 0.4) && close(frame[1].e_assists, 0.26 * 0.6));
        // penalties and minutes untouched; the player without rates is v1's
        assert!(close(frame[0].e_pen_goals, 0.05) && close(frame[0].e_minutes, 52.0));
        assert!(close(frame[2].e_np_goals, 0.26) && close(frame[2].e_assists, 0.13));
        // alone in his group, he keeps v1's total
        assert!(close(frame[3].e_np_goals, 0.26) && close(frame[3].e_assists, 0.13));
    }

    #[test]
    fn check_refuses_what_a_season_cannot_support() {
        let mut t = Truth::for_test(2018);
        t.rates.insert(1, Rates { npxg90: 0.5, xa90: 0.2 });
        assert!(t.check(Arm::R).is_err());
        assert!(t.check(Arm::T0).is_err());
        assert!(t.check(Arm::M0).is_ok());
        t.season = 2019;
        assert!(t.check(Arm::R).is_ok());
    }

    /// On real `data/` (the worktree's `../../../data`; skipped if absent), 2022/23 GW10: M0
    /// changes nothing after the coming GW and, in it, only rows whose minutes it set; MA
    /// reaches later GWs; R keeps the league's expected goals within 10% of v1's; T0 moves
    /// only the coming GW's fixtures with closing odds.
    #[test]
    #[ignore = "needs the real data/ directory"]
    fn oracle_arms_real() {
        use fplopt_models::assemble::{fit_v1, predict_fixtures, predict_fixtures_with};
        use fplopt_models::fitted::{REFIT_EVERY, refit_deadline};

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../../data");
        if !dir.join("gameweek.parquet").exists() {
            eprintln!("skipped: no data/ at {}", dir.display());
            return;
        }
        // v1's points need `config/scoring`, found relative to the working directory (the
        // workspace root; this test runs on its own, `--ignored`).
        std::env::set_current_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .unwrap();
        let store = DataStore::open(&dir).unwrap();
        let deadline = crate::simulator::season_schedule(&store, SEASON, 10).unwrap()[0].deadline;
        let view = store.as_of(deadline);
        let fit_view = view.earlier(refit_deadline(&view, REFIT_EVERY).unwrap()).unwrap();
        let fit = fit_v1(&fit_view, &V1Params::for_model(BASE_MODEL).unwrap()).unwrap();
        let base = predict_fixtures(&view, &fit).unwrap();
        let truth = Truth::load(&store, SEASON).unwrap();
        let flagged = flagged_at(&view).unwrap();
        eprintln!("{} players flagged", flagged.len());
        assert!((10..400).contains(&flagged.len()), "{}", flagged.len());
        let run = |arm: Arm| {
            truth.check(arm).unwrap();
            let count = std::cell::Cell::new(EditCount::default());
            let out = predict_fixtures_with(&view, &fit, &|frame| {
                count.set(truth.apply(arm, &flagged, frame))
            })
            .unwrap();
            assert_eq!(out.len(), base.len());
            (out, count.get())
        };
        let same = |a: &Fixture, b: &Fixture| a.xp.to_bits() == b.xp.to_bits();
        let h0 = base.iter().filter(|f| f.key.horizon == 0).count();

        let (m0, count) = run(Arm::M0);
        eprintln!("M0: {count:?} of {h0} horizon-0 rows");
        assert_eq!(count.rows, h0);
        assert!(count.unknown * 50 <= h0 && count.no_rate * 50 <= h0, "{count:?}");
        for (a, b) in base.iter().zip(&m0) {
            if a.key.horizon > 0 {
                assert!(same(a, b), "M0 changed horizon {} of {}", a.key.horizon, a.key.player_key);
            } else if !same(a, b) {
                assert!(a.p_play != b.p_play || a.p_start != b.p_start);
            }
        }
        // it knows who plays: every horizon-0 row is now 0 or 1
        assert!(
            m0.iter()
                .filter(|f| f.key.horizon == 0)
                .filter(|f| f.p_play > 0.0 && f.p_play < 1.0)
                .count()
                <= count.unknown
        );

        let (ma, _) = run(Arm::MA);
        assert!(base.iter().zip(&ma).any(|(a, b)| a.key.horizon > 0 && !same(a, b)));

        // the flag arms partition MA's rows; M2 stops after horizon 2
        let (maf, cf) = run(Arm::MAf);
        let (mau, cu) = run(Arm::MAu);
        let (m2, c2) = run(Arm::M2);
        eprintln!("MAf: {cf:?}; MAu: {cu:?}; M2: {c2:?}");
        assert_eq!(cf.rows + cu.rows, base.len());
        assert!(cf.rows > 0 && cu.rows > cf.rows);
        for (i, a) in base.iter().enumerate() {
            let f = flagged.contains(&a.key.player_key);
            assert!(same(if f { &mau[i] } else { &maf[i] }, a));
            assert!(same(if f { &maf[i] } else { &mau[i] }, &ma[i]));
            assert!(same(&m2[i], if a.key.horizon <= M2_MAX_HORIZON { &ma[i] } else { a }));
        }

        let (m0h, half) = run(Arm::M0h);
        assert!(half.rows * 10 >= h0 * 4 && half.rows * 10 <= h0 * 6, "{half:?} of {h0}");
        assert!(base.iter().zip(&m0h).all(|(a, b)| a.key.horizon == 0 || same(a, b)));

        let (t0, count) = run(Arm::T0);
        eprintln!("T0: {count:?}");
        assert!(count.changed * 2 >= count.rows, "{count:?}");
        assert!(base.iter().zip(&t0).all(|(a, b)| a.key.horizon == 0 || same(a, b)));

        let (r, count) = run(Arm::R);
        eprintln!("R: {count:?}");
        let goals = |v: &[Fixture]| {
            v.iter().filter(|f| f.key.horizon == 0).map(|f| f.e_np_goals).sum::<f64>()
        };
        let moved = base.iter().zip(&r).filter(|(a, b)| a.e_np_goals != b.e_np_goals).count();
        eprintln!(
            "R: horizon-0 expected non-penalty goals {:.3} vs v1 {:.3}; {moved} rows moved",
            goals(&r),
            goals(&base)
        );
        assert!((goals(&r) / goals(&base) - 1.0).abs() < 1e-9);
        assert!(moved * 4 >= count.changed);
    }

    #[test]
    fn the_holdout_is_refused() {
        assert!(refuse_holdout(2025).is_err());
        assert!(refuse_holdout(2022).is_ok());
    }
}
