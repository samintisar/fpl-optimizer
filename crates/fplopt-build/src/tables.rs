//! `TABLES`: metadata for every table the build writes to `data/` (Python
//! `fplopt.build.tables`): its kind (how `available_at` is set and how as-of reads treat it),
//! its unique key and, for static tables, what `AsOfView::lookup` may return.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Reference data (`available_at` = epoch), read only through `AsOfView::lookup`.
    Static,
    /// Known from publication; historically only the final version exists.
    Schedule,
    /// One row per thing that happened, each with its own `available_at`.
    Event,
    /// The state at each snapshot time (`available_at` = the snapshot time).
    Snapshot,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Static => "static",
            Kind::Schedule => "schedule",
            Kind::Event => "event",
            Kind::Snapshot => "snapshot",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TableSpec {
    pub name: &'static str,
    pub kind: Kind,
    pub key: &'static [&'static str],
    /// Static tables: the columns `lookup` returns (the key first) ...
    pub public_columns: &'static [&'static str],
    /// ... for the keys visible in this as-of (table, column).
    pub visible_via: Option<(&'static str, &'static str)>,
}

const fn spec(name: &'static str, kind: Kind, key: &'static [&'static str]) -> TableSpec {
    TableSpec { name, kind, key, public_columns: &[], visible_via: None }
}

const fn static_spec(
    name: &'static str,
    key: &'static [&'static str],
    public_columns: &'static [&'static str],
    visible_via: (&'static str, &'static str),
) -> TableSpec {
    TableSpec { name, kind: Kind::Static, key, public_columns, visible_via: Some(visible_via) }
}

pub const TABLES: &[TableSpec] = &[
    static_spec(
        "team_dim",
        &["team_key"],
        &["team_key", "short_name", "fpl_names"],
        ("team_rating", "team_key"),
    ),
    static_spec(
        "player_dim",
        &["player_key"],
        &["player_key", "first_name", "second_name", "web_name"],
        ("player_season", "player_key"),
    ),
    static_spec(
        "understat_map",
        &["understat_id"],
        &["understat_id", "player_key"],
        ("understat_player_match", "understat_id"),
    ),
    static_spec(
        "espn_map",
        &["espn_athlete_id"],
        &["espn_athlete_id", "player_key"],
        ("espn_player_match", "espn_athlete_id"),
    ),
    spec("schedule", Kind::Schedule, &["fixture_key"]),
    spec("gameweek", Kind::Schedule, &["season", "gw"]),
    spec("fixture", Kind::Event, &["fixture_key"]),
    spec("gameweek_result", Kind::Event, &["season", "gw"]),
    spec("player_season", Kind::Event, &["player_key", "season"]),
    spec("player_match", Kind::Event, &["player_key", "fixture_key"]),
    spec("understat_player_match", Kind::Event, &["understat_id", "fixture_key"]),
    spec("team_match", Kind::Event, &["fixture_key", "team_key"]),
    spec("player_gw", Kind::Event, &["player_key", "season", "gw"]),
    spec("player_gw_ownership", Kind::Event, &["player_key", "season", "gw"]),
    spec("team_rating", Kind::Event, &["team_key", "event_time"]),
    spec("espn_match", Kind::Event, &["espn_event_id"]),
    spec("espn_player_match", Kind::Event, &["espn_athlete_id", "espn_event_id"]),
    spec("club_match", Kind::Event, &["espn_event_id"]),
    spec("player_snapshot", Kind::Snapshot, &["snapshot_at", "source", "player_key"]),
    spec("fixture_snapshot", Kind::Snapshot, &["snapshot_at", "fixture_key"]),
    spec("espn_schedule_snapshot", Kind::Snapshot, &["snapshot_at", "espn_event_id"]),
    spec(
        "odds_snapshot",
        Kind::Snapshot,
        &[
            "fixture_key",
            "source",
            "bookmaker",
            "market",
            "outcome",
            "line",
            "is_closing",
            "snapshot_at",
        ],
    ),
];

/// The spec of `name`, if it is a registered table.
pub fn table_spec(name: &str) -> Option<&'static TableSpec> {
    TABLES.iter().find(|spec| spec.name == name)
}

/// Every registered table name, sorted (for error messages).
pub fn table_names() -> Vec<&'static str> {
    let mut names: Vec<_> = TABLES.iter().map(|spec| spec.name).collect();
    names.sort_unstable();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builder_output_is_registered() {
        // Builders that write several tables write these instead of their own name.
        let written = |name: &'static str| -> Vec<&'static str> {
            match name {
                "understat" => vec!["understat_player_match", "understat_map"],
                "espn" => vec![
                    "espn_match",
                    "espn_player_match",
                    "espn_schedule_snapshot",
                    "espn_map",
                    "club_match",
                ],
                other => vec![other],
            }
        };
        let built: std::collections::BTreeSet<_> =
            crate::ORDER.iter().flat_map(|name| written(name)).collect();
        let registered: std::collections::BTreeSet<_> = TABLES.iter().map(|s| s.name).collect();
        assert_eq!(built, registered);
    }
}
