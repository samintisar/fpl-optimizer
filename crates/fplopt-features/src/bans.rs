//! Suspensions, served where they really are (Python `fplopt.features.bans`; spec
//! docs/superpowers/specs/2026-10-27-espn-cup-data-design.md §2). Pure functions over rows.
//!
//! - A red card in a domestic match (PL, FA Cup, EFL Cup) bans the club's next *domestic*
//!   matches, so a cup tie in between uses a match of the ban up. A straight red is 1 certain
//!   match plus `STRAIGHT_RED_EXTRA` likely ones, a second yellow (or a red without ESPN
//!   data) 1.
//! - PL yellow-card bans (`BAN_RULES`) are served in PL fixtures only.
//!
//! `ban_kind` codes: `BAN_NONE` 0, `BAN_LIKELY` 1, `BAN_CERTAIN` 2 (overlapping bans combine
//! with `max`).

use std::collections::{BTreeMap, HashMap};

use fplopt_store::Micros;

pub const BAN_NONE: i64 = 0;
pub const BAN_LIKELY: i64 = 1;
pub const BAN_CERTAIN: i64 = 2;
/// Likely matches after the certain one for a straight red.
pub const STRAIGHT_RED_EXTRA: i64 = 2;
/// (yellow cards this season, last gw_index it counts by, matches banned): PLAN §6.3.
pub const BAN_RULES: &[(f64, i64, i64)] = &[(5.0, 19, 1), (10.0, 32, 2), (15.0, 99, 3)];
/// ESPN league codes of the domestic cups.
pub const DOMESTIC_CUPS: &[&str] = &["eng.fa", "eng.league_cup"];

/// A league match row with a card (history's match rows).
#[derive(Clone, Debug)]
pub struct LeagueRow {
    pub player_key: i64,
    pub season: i64,
    pub gw_index: i64,
    pub fixture_key: i64,
    pub team_key: i64,
    pub kickoff: Micros,
    pub yellow_cards: f64,
    pub red_cards: f64,
}

/// Whether ESPN's PL row of a red card says it was a second yellow.
#[derive(Clone, Debug)]
pub struct RedKind {
    pub player_key: i64,
    pub fixture_key: i64,
    pub second_yellow: bool,
}

/// A red card in a domestic cup match.
#[derive(Clone, Debug)]
pub struct CupRed {
    pub player_key: i64,
    pub team_key: i64,
    pub kickoff: Micros,
    pub second_yellow: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Served in the club's domestic matches (league and cups).
    Domestic,
    /// Served in league fixtures only.
    League,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BanEvent {
    pub player_key: i64,
    pub team_key: i64,
    pub time: Micros,
    pub scope: Scope,
    pub n_certain: i64,
    pub n_likely: i64,
}

/// One of a club's domestic matches: a league fixture or (fixture None) a cup match.
#[derive(Clone, Debug)]
pub struct TimelineEntry {
    pub team_key: i64,
    pub kickoff: Micros,
    pub fixture_key: Option<i64>,
}

/// (n_certain, n_likely) of a red card (`second_yellow` None: no ESPN data).
fn red_lengths(second_yellow: Option<bool>) -> (i64, i64) {
    let straight = second_yellow == Some(false);
    (1, if straight { STRAIGHT_RED_EXTRA } else { 0 })
}

/// One event per ban: PL yellow-card bans, then PL reds, then domestic cup reds.
pub fn ban_events(rows: &[LeagueRow], red_kinds: &[RedKind], cup_reds: &[CupRed]) -> Vec<BanEvent> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| (rows[i].player_key, rows[i].season, rows[i].gw_index));
    let mut out = Vec::new();
    let mut totals: HashMap<(i64, i64), f64> = HashMap::new();
    for &i in &order {
        let row = &rows[i];
        let total = totals.entry((row.player_key, row.season)).or_insert(0.0);
        let before = *total;
        *total += row.yellow_cards;
        let mut length = 0;
        for &(cards, last_gw, matches) in BAN_RULES {
            if before < cards && *total >= cards && row.gw_index <= last_gw {
                length += matches;
            }
        }
        if length > 0 {
            out.push(BanEvent {
                player_key: row.player_key,
                team_key: row.team_key,
                time: row.kickoff,
                scope: Scope::League,
                n_certain: length,
                n_likely: 0,
            });
        }
    }
    let mut kinds: HashMap<(i64, i64), bool> = HashMap::new();
    for kind in red_kinds {
        kinds.entry((kind.player_key, kind.fixture_key)).or_insert(kind.second_yellow);
    }
    for &i in &order {
        let row = &rows[i];
        if row.red_cards > 0.0 {
            let (n_certain, n_likely) =
                red_lengths(kinds.get(&(row.player_key, row.fixture_key)).copied());
            out.push(BanEvent {
                player_key: row.player_key,
                team_key: row.team_key,
                time: row.kickoff,
                scope: Scope::Domestic,
                n_certain,
                n_likely,
            });
        }
    }
    for red in cup_reds {
        let (n_certain, n_likely) = red_lengths(Some(red.second_yellow));
        out.push(BanEvent {
            player_key: red.player_key,
            team_key: red.team_key,
            time: red.kickoff,
            scope: Scope::Domestic,
            n_certain,
            n_likely,
        });
    }
    out
}

/// Each club's domestic matches in time order (league fixtures before cup matches at the
/// same kickoff): `league` (team_key, fixture_key, kickoff) and `cups` (team_key, kickoff),
/// one row per PL side of every domestic cup match.
pub fn club_timelines(league: &[(i64, i64, Micros)], cups: &[(i64, Micros)]) -> Vec<TimelineEntry> {
    let mut out: Vec<TimelineEntry> = league
        .iter()
        .map(|&(team_key, fixture_key, kickoff)| TimelineEntry {
            team_key,
            kickoff,
            fixture_key: Some(fixture_key),
        })
        .chain(cups.iter().map(|&(team_key, kickoff)| TimelineEntry {
            team_key,
            kickoff,
            fixture_key: None,
        }))
        .collect();
    out.sort_by_key(|e| (e.team_key, e.kickoff));
    out
}

/// (player_key, fixture_key) -> ban_kind for the league fixtures each ban is served in: the
/// club's matches after the ban's time, in order (scope League: league fixtures only); the
/// first `n_certain` are certain, the next `n_likely` likely; a cup match uses up its
/// position without a row. Overlapping bans keep the maximum.
pub fn serve_bans(events: &[BanEvent], timelines: &[TimelineEntry]) -> BTreeMap<(i64, i64), i64> {
    let mut timelines = timelines.to_vec();
    timelines.sort_by_key(|e| (e.team_key, e.kickoff));
    let mut out: BTreeMap<(i64, i64), i64> = BTreeMap::new();
    for event in events {
        let lo = timelines.partition_point(|e| e.team_key < event.team_key);
        let hi = timelines.partition_point(|e| e.team_key <= event.team_key);
        let club = &timelines[lo..hi];
        let after = club.partition_point(|e| e.kickoff <= event.time);
        let upcoming = club[after..]
            .iter()
            .filter(|e| event.scope == Scope::Domestic || e.fixture_key.is_some())
            .take((event.n_certain + event.n_likely).max(0) as usize);
        for (position, entry) in upcoming.enumerate() {
            if let Some(fixture) = entry.fixture_key {
                let kind =
                    if (position as i64) < event.n_certain { BAN_CERTAIN } else { BAN_LIKELY };
                let slot = out.entry((event.player_key, fixture)).or_insert(kind);
                *slot = (*slot).max(kind);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: Micros = 86_400_000_000;

    fn row(player: i64, gw: i64, yellow: f64, red: f64) -> LeagueRow {
        LeagueRow {
            player_key: player,
            season: 2024,
            gw_index: gw,
            fixture_key: 100 + gw,
            team_key: 1,
            kickoff: gw * 7 * D,
            yellow_cards: yellow,
            red_cards: red,
        }
    }

    fn league(gws: &[i64]) -> Vec<(i64, i64, Micros)> {
        gws.iter().map(|&gw| (1, 100 + gw, gw * 7 * D)).collect()
    }

    #[test]
    fn fifth_yellow_bans_the_next_league_fixture_only() {
        let rows: Vec<_> = (1..=5).map(|gw| row(7, gw, 1.0, 0.0)).collect();
        let events = ban_events(&rows, &[], &[]);
        assert_eq!(events.len(), 1);
        assert_eq!(
            (events[0].scope, events[0].n_certain, events[0].time),
            (Scope::League, 1, 35 * D)
        );
        // A cup match between GW5 and GW6 does not serve a league-only ban.
        let timelines = club_timelines(&league(&[5, 6, 7]), &[(1, 38 * D)]);
        let served = serve_bans(&events, &timelines);
        assert_eq!(served, BTreeMap::from([((7, 106), BAN_CERTAIN)]));
    }

    #[test]
    fn straight_red_is_one_certain_and_two_likely_and_cups_use_them_up() {
        let rows = vec![row(7, 1, 0.0, 1.0)];
        let kinds = vec![RedKind { player_key: 7, fixture_key: 101, second_yellow: false }];
        let events = ban_events(&rows, &kinds, &[]);
        assert_eq!((events[0].n_certain, events[0].n_likely), (1, 2));
        let timelines = club_timelines(&league(&[1, 2, 3, 4]), &[(1, 10 * D)]);
        let served = serve_bans(&events, &timelines);
        assert_eq!(served, BTreeMap::from([((7, 102), BAN_LIKELY), ((7, 103), BAN_LIKELY)]));
        // A second yellow, or no ESPN row: one match.
        let kinds = vec![RedKind { player_key: 7, fixture_key: 101, second_yellow: true }];
        assert_eq!(ban_events(&rows, &kinds, &[])[0].n_likely, 0);
        assert_eq!(ban_events(&rows, &[], &[])[0].n_likely, 0);
    }

    #[test]
    fn cup_reds_and_overlaps_keep_the_maximum() {
        let cup = vec![CupRed { player_key: 7, team_key: 1, kickoff: 8 * D, second_yellow: false }];
        let rows = vec![row(7, 2, 0.0, 1.0)];
        let events = ban_events(&rows, &[], &cup);
        assert_eq!(events.len(), 2);
        let served = serve_bans(&events, &club_timelines(&league(&[1, 2, 3, 4, 5]), &[]));
        assert_eq!(
            served,
            BTreeMap::from([
                ((7, 102), BAN_CERTAIN),
                ((7, 103), BAN_CERTAIN),
                ((7, 104), BAN_LIKELY)
            ])
        );
    }
}
