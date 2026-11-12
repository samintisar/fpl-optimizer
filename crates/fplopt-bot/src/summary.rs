//! The facts of a plan for the bot's messages (spec
//! docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md): plain data, no text.
//! `messages` words them.

use fplopt_optimize::{Plan, XpFrame};

use crate::context::Fixture;

/// A player's expected points summed over `gws` (every row of a player-GW counts, so a
/// double adds both games; NaN or missing counts 0); None if the frame has no row for him in
/// those GWs.
pub fn horizon_points(xp: &XpFrame, player_key: i64, gws: &[i64]) -> Option<f64> {
    let mut total = None;
    for i in 0..xp.len() {
        if xp.player_key[i] == player_key && gws.contains(&xp.gw[i]) {
            let x = xp.xp[i].filter(|v| !v.is_nan()).unwrap_or(0.0);
            total = Some(total.unwrap_or(0.0) + x);
        }
    }
    total
}

/// A club's games per GW in `gws` (in that order): (gw, [(opponent, home)]), empty for a
/// blank.
pub fn gw_fixtures(
    fixtures: &[Fixture],
    team_key: i64,
    gws: &[i64],
) -> Vec<(i64, Vec<(String, bool)>)> {
    gws.iter()
        .map(|&gw| {
            let games = fixtures
                .iter()
                .filter(|f| f.gw == gw && f.team_key == team_key)
                .map(|f| (f.opponent.clone(), f.home))
                .collect();
            (gw, games)
        })
        .collect()
}

/// Plans whose objectives differ by less than the solver's gap: equally good as far as the
/// solver can tell (PLAN §8 live finding).
pub fn tied(a: &Plan, b: &Plan) -> bool {
    let gap = a.mip_gap.max(b.mip_gap) * a.total_objective().abs();
    (a.total_objective() - b.total_objective()).abs() <= gap
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> XpFrame {
        XpFrame {
            player_key: vec![1, 1, 1, 1, 2],
            gw: vec![6, 7, 7, 8, 6],
            gw_index: vec![6, 7, 7, 8, 6],
            horizon: vec![0, 1, 1, 2, 0],
            xp: vec![Some(2.0), Some(3.0), Some(1.5), Some(f64::NAN), None],
            p_play: None,
            p_play_gw: None,
            e_minutes: None,
        }
    }

    #[test]
    fn horizon_points_add_a_players_rows_in_the_gws() {
        // GW7 is a double (two rows); NaN and missing count 0.
        assert_eq!(horizon_points(&frame(), 1, &[6, 7]), Some(6.5));
        assert_eq!(horizon_points(&frame(), 1, &[6, 7, 8]), Some(6.5));
        assert_eq!(horizon_points(&frame(), 2, &[6]), Some(0.0));
        assert_eq!(horizon_points(&frame(), 3, &[6]), None);
    }

    #[test]
    fn gw_fixtures_group_games_per_gw_and_mark_blanks() {
        let f = |gw, team_key, opponent: &str, home| Fixture {
            gw,
            team_key,
            opponent: opponent.into(),
            home,
        };
        let fixtures = vec![
            f(6, 1, "BRE", true),
            f(6, 1, "LEE", false),
            f(8, 1, "WOL", true),
            f(6, 2, "X", true),
        ];
        assert_eq!(
            gw_fixtures(&fixtures, 1, &[6, 7, 8]),
            vec![
                (6, vec![("BRE".to_string(), true), ("LEE".to_string(), false)]),
                (7, vec![]),
                (8, vec![("WOL".to_string(), true)]),
            ]
        );
    }
}
