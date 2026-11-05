use super::*;

const H: Micros = 3_600_000_000;

fn store(tables: Vec<(&str, Table)>) -> DataStore {
    DataStore::from_tables(tables.into_iter().map(|(n, t)| (n.to_string(), t)).collect()).unwrap()
}

fn ts(v: &[i64]) -> Column {
    Column::Ts(v.iter().map(|h| h * H).collect())
}

fn ints(v: &[i64]) -> Column {
    Column::I64(v.to_vec())
}

fn strs(v: &[&str]) -> Column {
    Column::Str(v.iter().map(|s| s.to_string()).collect())
}

#[test]
fn table_keeps_rows_strictly_before_the_deadline_in_available_order() {
    let rows = Table::new()
        .with("player_key", ints(&[1, 2, 3]))
        .with("fixture_key", ints(&[10, 10, 10]))
        .with("available_at", ts(&[5, 1, 3]));
    let s = store(vec![("player_match", rows)]);
    let view = s.as_of(5 * H);
    let out = view.table("player_match", Some(&["player_key"])).unwrap();
    assert_eq!(out.i64("player_key").unwrap(), &[2, 3]);
    assert_eq!(s.as_of(H).table("player_match", None).unwrap().height(), 0);
    assert!(view.earlier(6 * H).is_err());
    assert_eq!(view.earlier(4 * H).unwrap().table("player_match", None).unwrap().height(), 2);
}

#[test]
fn latest_takes_the_newest_visible_row_per_group_and_prefers_fpl_at_ties() {
    let rows = Table::new()
        .with("snapshot_at", ts(&[1, 1, 2, 2, 4]))
        .with("source", strs(&["fpl", "fplcache", "fplcache", "fpl", "fpl"]))
        .with("player_key", Column::OptI64(vec![Some(7), Some(7), Some(7), Some(7), None]))
        .with("now_cost", ints(&[50, 51, 52, 53, 54]))
        .with("available_at", ts(&[1, 1, 2, 2, 4]));
    let s = store(vec![("player_snapshot", rows)]);
    let out =
        s.as_of(3 * H).latest("player_snapshot", &["player_key"], Some(&["now_cost"])).unwrap();
    assert_eq!(out.names(), ["player_key", "now_cost"]);
    assert_eq!(out.i64("now_cost").unwrap(), &[53]);
    let out =
        s.as_of(5 * H).latest("player_snapshot", &["player_key"], Some(&["now_cost"])).unwrap();
    // The null group sorts last.
    assert_eq!(out.opt_i64("player_key").unwrap(), [Some(7), None]);
    assert_eq!(out.i64("now_cost").unwrap(), &[53, 54]);
    assert!(s.as_of(H).latest("player_snapshot", &[], None).unwrap().height() == 0);
}

#[test]
fn static_tables_are_read_only_through_lookup() {
    let dim = Table::new()
        .with("player_key", ints(&[3, 1, 2]))
        .with("first_name", strs(&["c", "a", "b"]))
        .with("second_name", strs(&["C", "A", "B"]))
        .with("web_name", strs(&["Cc", "Aa", "Bb"]))
        .with("birth", ints(&[0, 0, 0]))
        .with("available_at", ts(&[0, 0, 0]));
    let seasons = Table::new()
        .with("player_key", ints(&[1, 3, 2]))
        .with("season", ints(&[2020, 2020, 2021]))
        .with("available_at", ts(&[1, 1, 9]));
    let s = store(vec![("player_dim", dim), ("player_season", seasons)]);
    let view = s.as_of(5 * H);
    assert!(view.table("player_dim", None).is_err());
    let out = view.lookup("player_dim", &[2, 3, 1, 99]).unwrap();
    assert_eq!(out.names(), ["player_key", "first_name", "second_name", "web_name"]);
    assert_eq!(out.i64("player_key").unwrap(), &[1, 3]);
    assert!(view.lookup("player_season", &[1]).is_err());
}

#[test]
fn schedule_prefers_the_newest_fixture_snapshot_of_the_season() {
    let schedule = Table::new()
        .with("fixture_key", ints(&[2, 1, 9]))
        .with("season", ints(&[2020, 2020, 2019]))
        .with("gw", ints(&[1, 1, 1]))
        .with("gw_index", ints(&[1, 1, 1]))
        .with("kickoff_time", ts(&[20, 20, 20]))
        .with("home_team_key", ints(&[1, 3, 5]))
        .with("away_team_key", ints(&[2, 4, 6]))
        .with("schedule_source", strs(&["final", "final", "final"]))
        .with("available_at", ts(&[0, 0, 0]));
    let snapshots = Table::new()
        .with("snapshot_at", ts(&[2, 2, 4]))
        .with("season", ints(&[2020, 2020, 2020]))
        .with("fixture_key", ints(&[2, 1, 1]))
        .with("gw", Column::OptI64(vec![Some(2), None, Some(1)]))
        .with("kickoff_time", ts(&[30, 31, 32]))
        .with("home_team_key", ints(&[1, 3, 3]))
        .with("away_team_key", ints(&[2, 4, 4]))
        .with("available_at", ts(&[2, 2, 4]));
    let gameweek = Table::new()
        .with("season", ints(&[2020, 2020]))
        .with("gw", ints(&[1, 2]))
        .with("gw_index", ints(&[1, 2]))
        .with("deadline_time", ts(&[19, 25]))
        .with("available_at", ts(&[0, 0]));
    let s = store(vec![
        ("schedule", schedule),
        ("fixture_snapshot", snapshots),
        ("gameweek", gameweek),
    ]);
    let out = s.as_of(H).schedule(2020).unwrap();
    assert_eq!(out.names(), SCHEDULE_COLUMNS);
    assert_eq!(out.i64("fixture_key").unwrap(), &[1, 2]);
    assert_eq!(out.str("schedule_source").unwrap()[0], "final");
    let out = s.as_of(3 * H).schedule(2020).unwrap();
    assert_eq!(out.i64("fixture_key").unwrap(), &[1, 2]);
    assert_eq!(out.opt_i64("gw_index").unwrap(), [None, Some(2)]);
    assert_eq!(out.str("schedule_source").unwrap()[0], "snapshot");
    assert_eq!(s.as_of(5 * H).schedule(2020).unwrap().i64("fixture_key").unwrap(), &[1]);
    assert_eq!(s.as_of(25 * H).gameweek_for_deadline().unwrap(), (2020, 2));
    assert!(s.as_of(24 * H).gameweek_for_deadline().is_err());
}

#[test]
fn files_blocked_stops_opening_data_but_not_memory_stores() {
    let dir = tempfile::tempdir().unwrap();
    files_blocked("in a test", || {
        let err = DataStore::open(dir.path()).err().unwrap();
        assert!(err.to_string().contains("blocked in a test"));
        let rows = Table::new().with("player_key", ints(&[1])).with("available_at", ts(&[0]));
        let s = store(vec![("player_season", rows)]);
        assert_eq!(s.as_of(H).table("player_season", None).unwrap().height(), 1);
    });
    assert!(DataStore::open(dir.path()).is_ok());
}

#[test]
fn unknown_tables_and_columns_fail() {
    assert!(DataStore::from_tables([("nope".to_string(), Table::new())].into()).is_err());
    let rows = Table::new().with("player_key", ints(&[1])).with("available_at", ts(&[0]));
    let s = store(vec![("player_season", rows)]);
    assert!(s.as_of(H).table("player_season", Some(&["season"])).is_err());
    assert!(s.as_of(H).table("player_match", None).is_err());
}
