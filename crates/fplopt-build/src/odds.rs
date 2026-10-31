//! `odds_snapshot`: bookmaker prices per fixture in long format. Port of `fplopt.build.odds`
//! (its module docstring has the football-data allowlist, the timing rules and the checks).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Context, Result, bail};
use chrono::Datelike;
use fplopt_archive::raw_store::RawStore;
use fplopt_core::seasons::football_data_code;
use serde_json::Value;

use crate::common::{
    BuildContext, HOUR, date_from_days, from_micros, json_f64, micros, parse_iso, parse_opt_float,
    season_of, uk_date, uk_wall_clock,
};
use crate::fixtures::{FIRST_SEASON, FdRow, Fixture, football_data_frame, football_data_frames};
use crate::rules::py_float_repr;
use crate::table::{Column, Micros, Table};
use crate::teams::TeamResolver;

const ERA_B_FIRST_SEASON: i64 = 2019;
const TOTALS_LINE: f64 = 2.5;
const OVERROUND_MIN: f64 = 1.0;
const OVERROUND_MAX: f64 = 1.2;
const MAX_OUTLIER_SHARE: f64 = 0.01;
const COLLECTION_HOUR: u32 = 15;
const LAST_COLLECTION_BEFORE_KICKOFF: Micros = HOUR;

pub const SORT_BY: &[&str] = &[
    "fixture_key",
    "source",
    "bookmaker",
    "market",
    "outcome",
    "line",
    "is_closing",
    "snapshot_at",
];

fn outcomes(market: &str) -> &'static [&'static str] {
    match market {
        "h2h" => &["home", "draw", "away"],
        "totals" => &["over", "under"],
        _ => &["home", "away"],
    }
}

#[derive(Clone, Debug)]
struct OddsColumn {
    column: &'static str,
    bookmaker: &'static str,
    market: &'static str,
    outcome: &'static str,
    is_closing: bool,
    line_column: Option<&'static str>,
}

fn market(
    out: &mut Vec<OddsColumn>,
    bookmaker: &'static str,
    market: &'static str,
    columns: &[&'static str],
    closing: bool,
    line_column: Option<&'static str>,
) {
    for (column, outcome) in columns.iter().zip(outcomes(market)) {
        out.push(OddsColumn {
            column,
            bookmaker,
            market,
            outcome,
            is_closing: closing,
            line_column,
        });
    }
}

fn era_a() -> Vec<OddsColumn> {
    let mut v = Vec::new();
    market(&mut v, "avg", "h2h", &["BbAvH", "BbAvD", "BbAvA"], false, None);
    market(&mut v, "avg", "totals", &["BbAv>2.5", "BbAv<2.5"], false, None);
    market(&mut v, "avg", "ah", &["BbAvAHH", "BbAvAHA"], false, Some("BbAHh"));
    market(&mut v, "pinnacle", "h2h", &["PSH", "PSD", "PSA"], false, None);
    market(&mut v, "pinnacle", "h2h", &["PSCH", "PSCD", "PSCA"], true, None);
    v
}

fn era_b() -> Vec<OddsColumn> {
    let mut v = Vec::new();
    market(&mut v, "avg", "h2h", &["AvgH", "AvgD", "AvgA"], false, None);
    market(&mut v, "avg", "totals", &["Avg>2.5", "Avg<2.5"], false, None);
    market(&mut v, "avg", "ah", &["AvgAHH", "AvgAHA"], false, Some("AHh"));
    market(&mut v, "avg", "h2h", &["AvgCH", "AvgCD", "AvgCA"], true, None);
    market(&mut v, "avg", "totals", &["AvgC>2.5", "AvgC<2.5"], true, None);
    market(&mut v, "avg", "ah", &["AvgCAHH", "AvgCAHA"], true, Some("AHCh"));
    market(&mut v, "pinnacle", "h2h", &["PSH", "PSD", "PSA"], false, None);
    market(&mut v, "pinnacle", "totals", &["P>2.5", "P<2.5"], false, None);
    market(&mut v, "pinnacle", "ah", &["PAHH", "PAHA"], false, Some("AHh"));
    market(&mut v, "pinnacle", "h2h", &["PSCH", "PSCD", "PSCA"], true, None);
    market(&mut v, "pinnacle", "totals", &["PC>2.5", "PC<2.5"], true, None);
    market(&mut v, "pinnacle", "ah", &["PCAHH", "PCAHA"], true, Some("AHCh"));
    market(&mut v, "betfair_ex", "h2h", &["BFEH", "BFED", "BFEA"], false, None);
    market(&mut v, "betfair_ex", "totals", &["BFE>2.5", "BFE<2.5"], false, None);
    market(&mut v, "betfair_ex", "ah", &["BFEAHH", "BFEAHA"], false, Some("AHh"));
    market(&mut v, "betfair_ex", "h2h", &["BFECH", "BFECD", "BFECA"], true, None);
    market(&mut v, "betfair_ex", "totals", &["BFEC>2.5", "BFEC<2.5"], true, None);
    market(&mut v, "betfair_ex", "ah", &["BFECAHH", "BFECAHA"], true, Some("AHCh"));
    v
}

fn allowlist_columns() -> Vec<&'static str> {
    let mut set = BTreeSet::new();
    for c in era_a().into_iter().chain(era_b()) {
        set.insert(c.column);
        if let Some(line) = c.line_column {
            set.insert(line);
        }
    }
    set.into_iter().collect()
}

/// A long football-data odds row (before placing it on a fixture).
#[derive(Clone, Debug)]
struct FdOdds {
    season: i64,
    fd_date: i32,
    home: i64,
    away: i64,
    bookmaker: &'static str,
    market: &'static str,
    outcome: &'static str,
    line: Option<f64>,
    price: f64,
    is_closing: bool,
}

fn cell_float(row: &FdRow, column: &str) -> Result<Option<f64>> {
    parse_opt_float(row.cells.get(column).and_then(|c| c.as_deref()), column)
        .map(|v| v.filter(|f| !f.is_nan()))
}

fn football_data_odds(fd: &[FdRow]) -> Result<Vec<FdOdds>> {
    let mut by_season: BTreeMap<i64, Vec<&FdRow>> = BTreeMap::new();
    for r in fd {
        by_season.entry(r.season).or_default().push(r);
    }
    let mut out = Vec::new();
    for (season, rows) in by_season {
        let specs = if season >= ERA_B_FIRST_SEASON { era_b() } else { era_a() };
        let columns: HashSet<&String> = rows.iter().flat_map(|r| r.cells.keys()).collect();
        for spec in specs {
            if !columns.contains(&spec.column.to_string()) {
                continue;
            }
            if let Some(line) = spec.line_column
                && !columns.contains(&line.to_string())
            {
                continue;
            }
            for r in &rows {
                let Some(price) = cell_float(r, spec.column)? else { continue };
                let line = match spec.line_column {
                    Some(column) => cell_float(r, column)?,
                    None if spec.market == "totals" => Some(TOTALS_LINE),
                    None => None,
                };
                if line.is_none() && spec.market != "h2h" {
                    continue;
                }
                out.push(FdOdds {
                    season,
                    fd_date: r.fd_date,
                    home: r.home_team_key,
                    away: r.away_team_key,
                    bookmaker: spec.bookmaker,
                    market: spec.market,
                    outcome: spec.outcome,
                    line,
                    price,
                    is_closing: spec.is_closing,
                });
            }
        }
    }
    Ok(out)
}

/// When football-data's pre-match odds were collected: Tue/Wed/Thu kickoffs that week's
/// Tuesday 15:00 UK, else the Friday on or before, 15:00 UK; capped at kickoff - 1h.
pub fn prematch_available_at(kickoff: Micros) -> Micros {
    let day = uk_date(kickoff);
    let weekday = i64::from(day.weekday().num_days_from_monday());
    let days_back =
        if (1..=3).contains(&weekday) { weekday - 1 } else { (weekday - 4).rem_euclid(7) };
    let collected = uk_wall_clock(day - chrono::Duration::days(days_back), COLLECTION_HOUR);
    collected.min(kickoff - LAST_COLLECTION_BEFORE_KICKOFF)
}

type MarketKey = (i64, i64, i64, &'static str, &'static str, bool);

fn market_key(o: &FdOdds) -> MarketKey {
    (o.season, o.home, o.away, o.bookmaker, o.market, o.is_closing)
}

/// Each market's full set of (outcome, line, price) as one comparable string.
fn market_prices(odds: &[FdOdds]) -> HashMap<MarketKey, String> {
    let mut items: HashMap<MarketKey, Vec<String>> = HashMap::new();
    for o in odds {
        let line = o.line.map(py_float_repr).unwrap_or_default();
        items.entry(market_key(o)).or_default().push(format!(
            "{}@{line}={}",
            o.outcome,
            py_float_repr(o.price)
        ));
    }
    items
        .into_iter()
        .map(|(k, mut v)| {
            v.sort();
            (k, v.join("|"))
        })
        .collect()
}

/// (market, prices) -> the first archived file holding exactly that price set, over every
/// raw file of each season archived live.
fn football_data_live_first_seen(
    ctx: &BuildContext,
    resolver: &TeamResolver,
    seasons: &[i64],
    fixtures: &[Fixture],
) -> Result<HashMap<(MarketKey, String), Micros>> {
    let columns = allowlist_columns();
    let mut out: HashMap<(MarketKey, String), Micros> = HashMap::new();
    for &season in seasons {
        let code = football_data_code(season as i32);
        let entries = ctx.store.entries("football-data", &format!("E0/{code}"), ".csv.gz")?;
        let last =
            fixtures.iter().filter(|f| f.season == season).filter_map(|f| f.kickoff_time).max();
        let Some((first, _)) = entries.first() else { continue };
        if !last.is_none_or(|last| micros(first) < last) {
            continue;
        }
        for (at, path) in &entries {
            let rows = football_data_frame(path, season, resolver, &columns)?;
            for (key, prices) in market_prices(&football_data_odds(&rows)?) {
                let e = out.entry((key, prices)).or_insert(micros(at));
                *e = (*e).min(micros(at));
            }
        }
        log::info!("football-data {code} archived live: first seen over {} file(s)", entries.len());
    }
    Ok(out)
}

/// An odds_snapshot row.
#[derive(Clone, Debug)]
struct OddsRow {
    fixture_key: i64,
    season: i64,
    source: &'static str,
    bookmaker: String,
    market: String,
    outcome: String,
    line: Option<f64>,
    price: f64,
    is_closing: bool,
    snapshot_at: Micros,
    event_time: Micros,
}

fn football_data_snapshots(
    odds: &[FdOdds],
    fixtures: &[Fixture],
    first_seen: &HashMap<(MarketKey, String), Micros>,
) -> Result<Vec<OddsRow>> {
    let by_sides: HashMap<(i64, i64, i64), &Fixture> =
        fixtures.iter().map(|f| ((f.season, f.home_team_key, f.away_team_key), f)).collect();
    let (mut unmatched, mut wrong) = (BTreeSet::new(), BTreeSet::new());
    for o in odds {
        match by_sides.get(&(o.season, o.home, o.away)) {
            None => {
                unmatched.insert((o.season, o.home, o.away, o.fd_date));
            }
            Some(f) if f.fd_date != Some(o.fd_date) => {
                wrong.insert((o.season, o.home, o.away, o.fd_date));
            }
            _ => {}
        }
    }
    let mut problems = Vec::new();
    if !unmatched.is_empty() {
        problems.push(format!(
            "{} football-data match(es) without a fixture: {:?}",
            unmatched.len(),
            unmatched.iter().take(10).collect::<Vec<_>>()
        ));
    }
    if !wrong.is_empty() {
        problems.push(format!(
            "{} football-data date mismatch(es) with fixture.fd_date (rebuild `fixture` after a football-data refresh): {:?}",
            wrong.len(),
            wrong.iter().take(10).map(|w| (w.0, w.1, w.2, date_from_days(w.3))).collect::<Vec<_>>()
        ));
    }
    if !problems.is_empty() {
        bail!("football-data odds: {}", problems.join("\n"));
    }
    let live_seasons: HashSet<i64> = first_seen.keys().map(|(k, _)| k.0).collect();
    let prices = if live_seasons.is_empty() { HashMap::new() } else { market_prices(odds) };
    let (mut missing, mut later, mut after_kickoff) = (Vec::new(), 0, 0);
    let mut out = Vec::new();
    for o in odds {
        let f = by_sides[&(o.season, o.home, o.away)];
        let kickoff = f.kickoff_time.context("odds for a fixture without a kickoff")?;
        let mut available = if o.is_closing { kickoff } else { prematch_available_at(kickoff) };
        if live_seasons.contains(&o.season) {
            let key = market_key(o);
            match first_seen.get(&(key, prices[&key].clone())) {
                None => missing.push(format!("{key:?} {} {:?} {}", o.outcome, o.line, o.price)),
                Some(&seen) => {
                    if seen > available {
                        later += 1;
                        if seen >= kickoff {
                            after_kickoff += 1;
                        }
                        available = seen;
                    }
                }
            }
        }
        out.push(OddsRow {
            fixture_key: f.fixture_key,
            season: o.season,
            source: "football-data",
            bookmaker: o.bookmaker.to_string(),
            market: o.market.to_string(),
            outcome: o.outcome.to_string(),
            line: o.line,
            price: o.price,
            is_closing: o.is_closing,
            snapshot_at: available,
            event_time: f.event_time,
        });
    }
    if !missing.is_empty() {
        bail!(
            "{} live-season football-data price(s) in no archived file:\n{}",
            missing.len(),
            missing[..missing.len().min(10)].join("\n")
        );
    }
    if later > 0 {
        log::info!(
            "football-data: {later} live-season row(s) archived after the assumed time ({after_kickoff} after kickoff)"
        );
    }
    Ok(out)
}

/// Long odds rows from one Odds API snapshot, plus skipped-event counts by reason.
fn odds_api_rows(
    events: &[Value],
    taken_at: Micros,
    by_sides: &HashMap<(i64, i64, i64), &Fixture>,
    resolver: &TeamResolver,
    skipped: &mut BTreeMap<&'static str, usize>,
) -> Result<Vec<OddsRow>> {
    let mut rows = Vec::new();
    for event in events {
        let commence =
            parse_iso(event["commence_time"].as_str().context("event without commence_time")?)?;
        if commence <= taken_at {
            *skipped.entry("started").or_default() += 1;
            continue;
        }
        let home = event["home_team"].as_str().unwrap_or_default();
        let away = event["away_team"].as_str().unwrap_or_default();
        let season = season_of(commence);
        let Some(fx) = by_sides.get(&(season, resolver.odds_api(home)?, resolver.odds_api(away)?))
        else {
            *skipped.entry("no fixture").or_default() += 1;
            continue;
        };
        if fx.kickoff_time.is_some_and(|k| k <= taken_at) {
            *skipped.entry("started").or_default() += 1;
            continue;
        }
        if let Some(k) = fx.kickoff_time
            && k != commence
        {
            log::info!(
                "Odds API {home} v {away}: commence {}, FPL kickoff {}",
                from_micros(commence),
                from_micros(k)
            );
        }
        for book in event["bookmakers"].as_array().into_iter().flatten() {
            let book_key = book["key"].as_str().unwrap_or_default();
            for m in book["markets"].as_array().into_iter().flatten() {
                let market_key = m["key"].as_str().unwrap_or_default();
                for outcome in m["outcomes"].as_array().into_iter().flatten() {
                    let name = outcome["name"].as_str().unwrap_or_default();
                    let (label, line) = if market_key == "h2h" {
                        let label = if name == home {
                            Some("home")
                        } else if name == away {
                            Some("away")
                        } else if name == "Draw" {
                            Some("draw")
                        } else {
                            None
                        };
                        (label, None)
                    } else if market_key == "totals"
                        && outcome.get("point").and_then(Value::as_f64) == Some(TOTALS_LINE)
                    {
                        let label = match name {
                            "Over" => Some("over"),
                            "Under" => Some("under"),
                            _ => None,
                        };
                        (label, Some(TOTALS_LINE))
                    } else {
                        continue;
                    };
                    let Some(label) = label else {
                        bail!(
                            "Odds API {home} v {away} {book_key} {market_key}: unknown outcome '{name}'"
                        );
                    };
                    rows.push(OddsRow {
                        fixture_key: fx.fixture_key,
                        season,
                        source: "odds-api",
                        bookmaker: book_key.to_string(),
                        market: market_key.to_string(),
                        outcome: label.to_string(),
                        line,
                        price: json_f64(&outcome["price"], "price")?,
                        is_closing: false,
                        snapshot_at: taken_at,
                        event_time: fx.event_time,
                    });
                }
            }
        }
    }
    Ok(rows)
}

fn check_overround(rows: &[OddsRow]) -> Result<()> {
    let mut groups: BTreeMap<(i64, &str, &str, bool, Micros), (f64, BTreeSet<&str>)> =
        BTreeMap::new();
    for r in rows.iter().filter(|r| r.market == "h2h") {
        let e = groups
            .entry((r.fixture_key, r.source, &r.bookmaker, r.is_closing, r.snapshot_at))
            .or_default();
        e.0 += 1.0 / r.price;
        e.1.insert(&r.outcome);
    }
    let bad: Vec<String> = groups
        .iter()
        .filter(|(_, (sum, outcomes))| {
            outcomes.len() != 3 || *sum < OVERROUND_MIN || *sum > OVERROUND_MAX
        })
        .map(|(k, (sum, outcomes))| format!("{k:?} overround {sum:.6} outcomes {}", outcomes.len()))
        .collect();
    if !bad.is_empty() {
        let share = bad.len() as f64 / groups.len() as f64;
        let message = format!(
            "h2h overround outside [{OVERROUND_MIN}, {OVERROUND_MAX}] or incomplete in {} of {} group(s) ({:.2}%):\n{}",
            bad.len(),
            groups.len(),
            share * 100.0,
            bad[..bad.len().min(10)].join("\n")
        );
        if share > MAX_OUTLIER_SHARE {
            bail!(message);
        }
        log::warn!("{message}");
    }
    Ok(())
}

fn consistent(r: &OddsRow) -> bool {
    let valid = outcomes(&r.market).contains(&r.outcome.as_str())
        && ["h2h", "totals", "ah"].contains(&r.market.as_str());
    let line_ok = match r.market.as_str() {
        "h2h" => r.line.is_none(),
        "totals" => r.line == Some(TOTALS_LINE),
        _ => r.line.is_some_and(|l| (l * 4.0).round() == l * 4.0),
    };
    let fd = r.source == "football-data";
    let timing = if r.is_closing {
        r.snapshot_at >= r.event_time
    } else if fd {
        r.snapshot_at >= prematch_available_at(r.event_time)
    } else {
        r.snapshot_at < r.event_time
    };
    valid
        && line_ok
        && timing
        && r.fixture_key.div_euclid(1000) == r.season
        && (!r.is_closing || fd)
        && r.price > 1.0
        && r.season >= FIRST_SEASON
}

pub fn build_odds_snapshot(ctx: &mut BuildContext) -> Result<Table> {
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let fd = football_data_frames(ctx, &resolver, &allowlist_columns(), Some(FIRST_SEASON))?;
    let seasons: Vec<i64> =
        fd.iter().map(|r| r.season).collect::<BTreeSet<_>>().into_iter().collect();
    let first_seen = football_data_live_first_seen(ctx, &resolver, &seasons, &fixtures)?;
    let mut rows = football_data_snapshots(&football_data_odds(&fd)?, &fixtures, &first_seen)?;
    let by_sides: HashMap<(i64, i64, i64), &Fixture> =
        fixtures.iter().map(|f| ((f.season, f.home_team_key, f.away_team_key), f)).collect();
    let mut skipped = BTreeMap::new();
    for (taken_at, path) in ctx.store.entries("odds", "soccer_epl", ".json.gz")? {
        let payload = RawStore::read_json(&path)?;
        let events = payload.as_array().context("odds snapshot is not a list")?;
        rows.extend(odds_api_rows(events, micros(&taken_at), &by_sides, &resolver, &mut skipped)?);
    }
    if !skipped.is_empty() {
        log::info!("Odds API events skipped: {skipped:?}");
    }
    check_overround(&rows)?;
    let mut keys = HashSet::new();
    for r in &rows {
        let key = (
            r.fixture_key,
            r.source,
            &r.bookmaker,
            &r.market,
            &r.outcome,
            r.line.map(f64::to_bits),
            r.is_closing,
            r.snapshot_at,
        );
        if !consistent(r) || !keys.insert(key) {
            bail!("table 'odds_snapshot' failed validation at {r:?}");
        }
    }
    Ok(Table::new()
        .with("fixture_key", Column::I64(rows.iter().map(|r| r.fixture_key).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("source", Column::Str(rows.iter().map(|r| r.source.to_string()).collect()))
        .with("bookmaker", Column::Str(rows.iter().map(|r| r.bookmaker.clone()).collect()))
        .with("market", Column::Str(rows.iter().map(|r| r.market.clone()).collect()))
        .with("outcome", Column::Str(rows.iter().map(|r| r.outcome.clone()).collect()))
        .with("line", Column::OptF64(rows.iter().map(|r| r.line).collect()))
        .with("price", Column::F64(rows.iter().map(|r| r.price).collect()))
        .with("is_closing", Column::Bool(rows.iter().map(|r| r.is_closing).collect()))
        .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.snapshot_at).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.snapshot_at).collect())))
}
