//! A real manager's squad for `optimize plan` (Python `fplopt.entry`): imported from FPL's
//! public API by team ID, from a squad file (TOML, prices in £m), or both, the file's values
//! overriding the import's; then turned into the backtester's `SquadState`.
//!
//! **Import** (`import_squad`): the squad and bank of the last GW before the planned one (the
//! GW before it when that GW was a Free Hit); purchase prices from the latest transfer in
//! (Free Hit GWs excepted), else the price at the manager's first GW's deadline; free
//! transfers replayed with `next_state`; the chips played. No HTTP here: the caller passes an
//! `EntryClient`.
//!
//! Errors meant for the user are `EntryError` (inside the `anyhow::Error`; the CLI reports
//! them as usage errors, exit 2). Messages match Python's.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;

use anyhow::Result;
use fplopt_build::rules::py_float_repr;
use fplopt_build::scoring::Rules;
use serde_json::Value;

use crate::state::{Holding, PoolPlayer, SquadState, TransferRecord, chip_window, next_state};

/// The squad can't be imported or read (a message for the user).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryError(pub String);

impl fmt::Display for EntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EntryError {}

fn entry_error<T>(message: impl Into<String>) -> Result<T> {
    Err(EntryError(message.into()).into())
}

/// The FPL client methods the import uses (raw JSON bodies).
pub trait EntryClient {
    fn entry_history(&self, entry_id: i64) -> Result<Vec<u8>>;
    fn entry_picks(&self, entry_id: i64, gw: i64) -> Result<Vec<u8>>;
    fn entry_transfers(&self, entry_id: i64) -> Result<Vec<u8>>;
}

/// A held player: FPL `element_id` (this season's), purchase price in tenths of £m; `name`
/// is informational.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SquadPlayer {
    pub element_id: i64,
    pub purchase_price: i64,
    pub name: String,
}

impl SquadPlayer {
    pub fn new(element_id: i64, purchase_price: i64) -> SquadPlayer {
        SquadPlayer { element_id, purchase_price, name: String::new() }
    }
}

/// A manager's squad before GW `gw` of `season`: bank in tenths of £m, free transfers for GW
/// `gw`, chips played as (name, FPL GW); `notes`: how values were inferred.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SquadSpec {
    pub season: i64,
    pub gw: i64,
    pub players: Vec<SquadPlayer>,
    pub bank: i64,
    pub free_transfers: i64,
    pub chips_used: Vec<(String, i64)>,
    pub notes: Vec<String>,
}

impl SquadSpec {
    pub fn new(season: i64, gw: i64, players: Vec<SquadPlayer>, bank: i64, ft: i64) -> SquadSpec {
        SquadSpec {
            season,
            gw,
            players,
            bank,
            free_transfers: ft,
            chips_used: Vec::new(),
            notes: Vec::new(),
        }
    }
}

/// FPL GW number -> gw_index for one season.
pub type GwIndex = BTreeMap<i64, i64>;

// --- import ---------------------------------------------------------------------------------

/// The squad of FPL team `entry_id` before GW `gw` (module docs). `initial_prices(gw)` gives
/// element_id -> price at that GW's deadline.
pub fn import_squad(
    fpl: &dyn EntryClient,
    entry_id: i64,
    season: i64,
    gw: i64,
    rules: &Rules,
    gw_index: &GwIndex,
    initial_prices: &mut dyn FnMut(i64) -> Result<HashMap<i64, i64>>,
) -> Result<SquadSpec> {
    let history = json(&fpl.entry_history(entry_id)?, &format!("team {entry_id}'s history"))?;
    let transfers = json(&fpl.entry_transfers(entry_id)?, &format!("team {entry_id}'s transfers"))?;
    let transfers = match transfers {
        Value::Array(items) => items,
        _ => anyhow::bail!("team {entry_id}'s transfers: not a list"),
    };
    let mut picks =
        |event: i64| json(&fpl.entry_picks(entry_id, event)?, &format!("GW{event} picks"));
    squad_from_api(season, gw, &history, &mut picks, &transfers, rules, gw_index, initial_prices)
}

fn json(body: &[u8], what: &str) -> Result<Value> {
    let body = body.strip_prefix(b"\xef\xbb\xbf").unwrap_or(body);
    match serde_json::from_slice(body) {
        Ok(v) => Ok(v),
        Err(e) => entry_error(format!("{what}: not JSON ({e})")),
    }
}

/// Python's `int(x)` of a JSON value (an integer, a whole float or a numeric string).
fn int_of(value: Option<&Value>, what: &str) -> Result<i64> {
    let parsed = match value {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Some(Value::String(s)) => s.trim().parse().ok(),
        Some(Value::Bool(b)) => Some(i64::from(*b)),
        _ => None,
    };
    parsed.ok_or_else(|| anyhow::anyhow!("{what}: expected an integer, got {value:?}"))
}

fn items<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// Python's `str(x)` of a JSON value.
fn str_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

/// `import_squad` on parsed responses (`picks(gw)` fetches a GW's picks).
#[allow(clippy::too_many_arguments)]
pub fn squad_from_api(
    season: i64,
    gw: i64,
    history: &Value,
    picks: &mut dyn FnMut(i64) -> Result<Value>,
    transfers: &[Value],
    rules: &Rules,
    gw_index: &GwIndex,
    initial_prices: &mut dyn FnMut(i64) -> Result<HashMap<i64, i64>>,
) -> Result<SquadSpec> {
    let mut rows: Vec<(i64, &Value)> = Vec::new();
    for r in items(history, "current") {
        let event = int_of(r.get("event"), "history event")?;
        if event < gw {
            rows.push((event, r));
        }
    }
    rows.sort_by_key(|(e, _)| *e);
    let Some(&(last, _)) = rows.last() else {
        return entry_error(format!(
            "the team has no finished GW before GW{gw}: there is no squad to import yet"
        ));
    };
    if last != gw - 1 {
        return entry_error(format!(
            "GW{}'s deadline hasn't passed (the history ends at GW{last}): plan GW{}, or give \
             the squad in a squad file",
            gw - 1,
            last + 1
        ));
    }
    let mut chips: BTreeMap<i64, String> = BTreeMap::new();
    for c in items(history, "chips") {
        chips.insert(
            int_of(c.get("event"), "chip event")?,
            c.get("name").map(str_of).unwrap_or_default(),
        );
    }
    let known: HashSet<&str> = rules.chips.iter().map(|c| c.name.as_str()).collect();
    let unknown: Vec<String> = chips
        .values()
        .filter(|n| !known.contains(n.as_str()))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if !unknown.is_empty() {
        return entry_error(format!("unknown chip(s) in the history: {}", py_repr_strs(&unknown)));
    }
    let mut notes = Vec::new();
    let mut squad_row = *rows.last().unwrap();
    if chips.get(&last).map(String::as_str) == Some("freehit") {
        if rows.len() < 2 {
            return entry_error("the team's only GW was a Free Hit: no squad to revert to");
        }
        squad_row = rows[rows.len() - 2];
        notes.push(format!("GW{last} was a Free Hit: squad and bank of GW{}", squad_row.0));
    }
    let squad_event = squad_row.0;
    let picked = picks(squad_event)?;
    let elements: Vec<i64> = picked
        .get("picks")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("GW{squad_event} picks: no 'picks'"))?
        .iter()
        .map(|p| int_of(p.get("element"), "pick element"))
        .collect::<Result<_>>()?;
    if elements.len() as i64 != rules.squad_size {
        return entry_error(format!("GW{squad_event} picks hold {} players", elements.len()));
    }

    let row_values: Vec<(i64, i64)> = rows
        .iter()
        .map(|(e, r)| {
            let n = match r.get("event_transfers") {
                None => 0,
                v => int_of(v, "event_transfers")?,
            };
            Ok((*e, n))
        })
        .collect::<Result<_>>()?;
    let free_transfers = replay_free_transfers(season, gw, &row_values, &chips, rules, gw_index)?;
    let freehits: HashSet<i64> =
        chips.iter().filter(|(_, n)| n.as_str() == "freehit").map(|(e, _)| *e).collect();
    let mut sorted: Vec<(i64, String, &Value)> = transfers
        .iter()
        .map(|t| {
            let time = t.get("time").map(str_of).unwrap_or_default();
            Ok((int_of(t.get("event"), "transfer event")?, time, t))
        })
        .collect::<Result<_>>()?;
    sorted.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    let mut bought: HashMap<i64, i64> = HashMap::new();
    for (event, _, t) in sorted {
        if freehits.contains(&event) || event > squad_event {
            continue;
        }
        bought.insert(
            int_of(t.get("element_in"), "element_in")?,
            int_of(t.get("element_in_cost"), "element_in_cost")?,
        );
    }
    let first = rows[0].0;
    let mut start_prices: Option<HashMap<i64, i64>> = None;
    let mut players = Vec::new();
    for &element in &elements {
        if let Some(&cost) = bought.get(&element) {
            players.push(SquadPlayer::new(element, cost));
            continue;
        }
        if start_prices.as_ref().is_none_or(HashMap::is_empty) {
            start_prices = Some(initial_prices(first)?);
        }
        let Some(&price) = start_prices.as_ref().and_then(|p| p.get(&element)) else {
            return entry_error(format!(
                "no GW{first} price for element {element} (held since GW{first}): give its \
                 purchase_price in the squad file"
            ));
        };
        players.push(SquadPlayer::new(element, price));
    }
    // FPL prices don't change before the GW1 deadline, so a squad held since GW1 has its
    // exact prices; a later first GW's squad may have been picked days before that deadline.
    if gw_index.get(&first) != Some(&1)
        && players.iter().any(|p| !bought.contains_key(&p.element_id))
    {
        notes.push(format!(
            "purchase price of players held since GW{first}: their price at the GW{first} \
             deadline (newest snapshot before it)"
        ));
    }
    notes.push(format!(
        "free transfers {free_transfers}: replayed from the transfers and chips of \
         GW{first}-GW{last}"
    ));
    notes.push(format!(
        "transfers or a chip made for GW{gw} before its deadline aren't public: if you made \
         any, edit the squad file"
    ));
    let mut chips_used: Vec<(String, i64)> = chips.iter().map(|(e, n)| (n.clone(), *e)).collect();
    chips_used.sort_by_key(|c| c.1);
    Ok(SquadSpec {
        season,
        gw,
        players,
        bank: int_of(squad_row.1.get("bank"), "bank")?,
        free_transfers,
        chips_used,
        notes,
    })
}

/// Free transfers for GW `gw` after the history `rows` ((FPL GW, transfers made) before
/// `gw`, ascending): `next_state` GW by GW. The manager's first GW is a fresh squad (0 FTs
/// carried in, so 1 after it, as after GW1).
pub fn replay_free_transfers(
    season: i64,
    gw: i64,
    rows: &[(i64, i64)],
    chips: &BTreeMap<i64, String>,
    rules: &Rules,
    gw_index: &GwIndex,
) -> Result<i64> {
    let Some(&(first, _)) = rows.first() else { anyhow::bail!("no history rows") };
    let mut state = SquadState::new(season, index(gw_index, first)?, Vec::new(), 0, 0)?;
    for (k, &(event, n)) in rows.iter().enumerate() {
        let next = rows.get(k + 1).map_or(gw, |r| r.0);
        let mut current = state.clone();
        current.gw_index = index(gw_index, event)?;
        let chip = chips.get(&event);
        let mut chip_id = None;
        if let Some(chip) = chip {
            match chip_window(&current, chip, rules) {
                Ok(id) => chip_id = Some(id),
                Err(e) => return entry_error(format!("GW{event} {chip}: {e}")),
            }
        }
        if chip.map(String::as_str) == Some("freehit") {
            current.freehit_backup = Some(Vec::new());
            current.freehit_bank = Some(0);
        }
        let record = TransferRecord { n_transfers: n, chip: chip.cloned(), chip_id };
        state = next_state(&current, &record, rules, index(gw_index, next)?)?;
    }
    Ok(state.free_transfers)
}

fn index(gw_index: &GwIndex, gw: i64) -> Result<i64> {
    match gw_index.get(&gw) {
        Some(&i) => Ok(i),
        None => entry_error(format!("GW{gw} is not in the season's schedule")),
    }
}

// --- squad file -----------------------------------------------------------------------------

/// A squad file player: by element_id or by (web) name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlayerRef {
    ElementId(i64),
    Name(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilePlayer {
    pub player: PlayerRef,
    pub purchase_price: i64,
}

/// A squad file's keys (prices in tenths of £m); None: not in the file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SquadOverrides {
    pub season: Option<i64>,
    pub gw: Option<i64>,
    pub bank: Option<i64>,
    pub free_transfers: Option<i64>,
    pub chips_used: Option<Vec<(String, i64)>>,
    pub players: Option<Vec<FilePlayer>>,
}

const KNOWN_KEYS: [&str; 6] = ["bank", "chips_used", "free_transfers", "gw", "players", "season"];

/// The file's keys (module docs), checked; prices in tenths of £m.
pub fn read_squad_file(path: &Path) -> Result<SquadOverrides> {
    let shown = path.display().to_string();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return entry_error(format!("squad file {shown}: {e}")),
    };
    let raw: toml::Table = match toml::from_str(&text) {
        Ok(t) => t,
        Err(e) => return entry_error(format!("squad file {shown}: {}", e.to_string().trim_end())),
    };
    parse_squad_table(&raw, &shown)
}

fn parse_squad_table(raw: &toml::Table, path: &str) -> Result<SquadOverrides> {
    let extra: Vec<String> =
        raw.keys().filter(|k| !KNOWN_KEYS.contains(&k.as_str())).cloned().collect();
    if !extra.is_empty() {
        return entry_error(format!(
            "squad file {path}: unknown key(s) {} (known: {})",
            py_repr_strs(&extra),
            py_repr_strs(&KNOWN_KEYS)
        ));
    }
    let mut out = SquadOverrides::default();
    if let Some(v) = raw.get("season") {
        let label = toml_str(v);
        match fplopt_core_season(&label) {
            Some(s) => out.season = Some(s),
            None => {
                return entry_error(format!(
                    "squad file {path}: season: not a season label like 2026-27: {}",
                    py_repr_str(&label)
                ));
            }
        }
    }
    if let Some(v) = raw.get("gw") {
        out.gw = Some(int_value(v, "gw", path, 1)?);
    }
    if let Some(v) = raw.get("bank") {
        out.bank = Some(tenths(v, "bank", path)?);
    }
    if let Some(v) = raw.get("free_transfers") {
        out.free_transfers = Some(int_value(v, "free_transfers", path, 0)?);
    }
    if let Some(v) = raw.get("chips_used") {
        let mut chips = Vec::new();
        for item in toml_items(v) {
            match item.as_array() {
                Some(pair) if pair.len() == 2 => {
                    chips
                        .push((toml_str(&pair[0]), int_value(&pair[1], "chips_used gw", path, 1)?));
                }
                _ => {
                    return entry_error(format!(
                        "squad file {path}: chips_used entries are [name, gw]"
                    ));
                }
            }
        }
        out.chips_used = Some(chips);
    }
    if let Some(v) = raw.get("players") {
        let mut players = Vec::new();
        for (i, item) in toml_items(v).iter().enumerate() {
            let i = i + 1;
            let Some(table) = item.as_table().filter(|t| t.contains_key("purchase_price")) else {
                return entry_error(format!("squad file {path}: player {i} needs purchase_price"));
            };
            if table.contains_key("element_id") == table.contains_key("name") {
                return entry_error(format!(
                    "squad file {path}: player {i}: give element_id or name"
                ));
            }
            let purchase_price = tenths(&table["purchase_price"], "purchase_price", path)?;
            let player = match table.get("element_id") {
                Some(e) => PlayerRef::ElementId(int_value(e, "element_id", path, 0)?),
                None => PlayerRef::Name(toml_str(&table["name"])),
            };
            players.push(FilePlayer { player, purchase_price });
        }
        out.players = Some(players);
    }
    Ok(out)
}

fn fplopt_core_season(label: &str) -> Option<i64> {
    // Python's parse_season_label: YYYY-YY with consecutive years.
    let (a, b) = label.split_once('-')?;
    if a.len() != 4 || b.len() != 2 || !a.bytes().chain(b.bytes()).all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (start, end): (i64, i64) = (a.parse().ok()?, b.parse().ok()?);
    ((start + 1) % 100 == end).then_some(start)
}

/// What Python iterates over for a TOML value (an array's items; a string's characters).
fn toml_items(value: &toml::Value) -> Vec<toml::Value> {
    match value {
        toml::Value::Array(items) => items.clone(),
        toml::Value::String(s) => s.chars().map(|c| toml::Value::String(c.to_string())).collect(),
        toml::Value::Table(t) => t.keys().map(|k| toml::Value::String(k.clone())).collect(),
        other => vec![other.clone()],
    }
}

/// Python's `str(x)` of a TOML value.
fn toml_str(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => py_float_repr(*f).replace("Infinity", "inf").replace("NaN", "nan"),
        toml::Value::Boolean(true) => "True".to_owned(),
        toml::Value::Boolean(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

/// Python's `repr(x)` of a TOML value.
fn toml_repr(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => py_repr_str(s),
        other => toml_str(other),
    }
}

/// £m (e.g. 7.5) -> tenths (75); a whole number of tenths, >= 0.
fn tenths(value: &toml::Value, key: &str, path: &str) -> Result<i64> {
    let x = match value {
        toml::Value::Integer(i) => *i as f64,
        toml::Value::Float(f) => *f,
        other => {
            return entry_error(format!(
                "squad file {path}: {key} must be a number of £m, got {}",
                toml_repr(other)
            ));
        }
    };
    let tenths = (x * 10.0).round_ties_even();
    if !x.is_finite() || (x * 10.0 - tenths).abs() > 1e-6 || tenths < 0.0 {
        return entry_error(format!(
            "squad file {path}: {key} must be >= 0 in steps of 0.1, got {}",
            toml_str(value)
        ));
    }
    Ok(tenths as i64)
}

fn int_value(value: &toml::Value, key: &str, path: &str, minimum: i64) -> Result<i64> {
    match value {
        toml::Value::Integer(i) if *i >= minimum => Ok(*i),
        _ => entry_error(format!("squad file {path}: {key} must be an integer >= {minimum}")),
    }
}

/// The spec with the file's values in place of the import's; without an import (`spec`
/// None) the file must give bank, free_transfers and players. The file's season and gw, when
/// given, must be the planned ones. `resolve(name)` -> element_id.
pub fn apply_overrides(
    spec: Option<SquadSpec>,
    overrides: &SquadOverrides,
    season: i64,
    gw: i64,
    resolve: &mut dyn FnMut(&str) -> Result<i64>,
) -> Result<SquadSpec> {
    if let Some(s) = overrides.season
        && s != season
    {
        return entry_error(format!(
            "the squad file is for season {}, the plan is for {season}",
            season_label(s)
        ));
    }
    if let Some(g) = overrides.gw
        && g != gw
    {
        return entry_error(format!("the squad file is for gw {g}, the plan is for {gw}"));
    }
    let mut spec = match spec {
        Some(spec) => spec,
        None => {
            let missing: Vec<&str> = [
                ("bank", overrides.bank.is_none()),
                ("free_transfers", overrides.free_transfers.is_none()),
                ("players", overrides.players.is_none()),
            ]
            .into_iter()
            .filter(|(_, m)| *m)
            .map(|(k, _)| k)
            .collect();
            if !missing.is_empty() {
                return entry_error(format!(
                    "without --entry the squad file needs {}",
                    missing.join(", ")
                ));
            }
            SquadSpec::new(season, gw, Vec::new(), 0, 0)
        }
    };
    // Sorted key names, as Python's `sorted(changed)`.
    let mut changed = Vec::new();
    if let Some(bank) = overrides.bank {
        spec.bank = bank;
        changed.push("bank");
    }
    if let Some(chips) = &overrides.chips_used {
        spec.chips_used = chips.clone();
        changed.push("chips_used");
    }
    if let Some(ft) = overrides.free_transfers {
        spec.free_transfers = ft;
        changed.push("free_transfers");
    }
    if let Some(items) = &overrides.players {
        let mut players = Vec::new();
        for item in items {
            let (element_id, name) = match &item.player {
                PlayerRef::ElementId(e) => (*e, String::new()),
                PlayerRef::Name(n) => (resolve(n)?, n.clone()),
            };
            players.push(SquadPlayer { element_id, purchase_price: item.purchase_price, name });
        }
        spec.players = players;
        changed.push("players");
    }
    if !changed.is_empty() {
        spec.notes.push(format!("from the squad file: {}", changed.join(", ")));
    }
    Ok(spec)
}

/// The spec as a squad file to edit (byte-identical to Python's `write_squad_file`);
/// `names`: element_id -> name, written as comments.
pub fn squad_file_text(spec: &SquadSpec, names: &HashMap<i64, String>) -> String {
    let mut lines =
        vec!["# Squad for `fplopt optimize plan --squad-file`. Prices in £m.".to_owned()];
    lines.extend(spec.notes.iter().map(|n| format!("# {n}")));
    lines.push(format!("season = \"{}\"", season_label(spec.season)));
    lines.push(format!("gw = {}", spec.gw));
    lines.push(format!("bank = {}", money(spec.bank)));
    lines.push(format!("free_transfers = {}", spec.free_transfers));
    let chips: Vec<String> =
        spec.chips_used.iter().map(|(name, g)| format!("[\"{name}\", {g}]")).collect();
    lines.push(format!("chips_used = [{}]", chips.join(", ")));
    for p in &spec.players {
        let name = names.get(&p.element_id).unwrap_or(&p.name);
        lines.push(String::new());
        lines.push("[[players]]".to_owned());
        let comment = if name.is_empty() { String::new() } else { format!("  # {name}") };
        lines.push(format!("element_id = {}{comment}", p.element_id));
        lines.push(format!("purchase_price = {}", money(p.purchase_price)));
    }
    lines.join("\n") + "\n"
}

pub fn write_squad_file(spec: &SquadSpec, path: &Path, names: &HashMap<i64, String>) -> Result<()> {
    std::fs::write(path, squad_file_text(spec, names))
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// Tenths of £m as Python's `f"{x / 10:.1f}"`.
pub fn money(tenths: i64) -> String {
    format!("{:.1}", tenths as f64 / 10.0)
}

/// 2026 -> "2026-27".
pub fn season_label(season: i64) -> String {
    format!("{season}-{:02}", (season + 1).rem_euclid(100))
}

/// Python's `repr` of a string.
fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_repr_strs<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| py_repr_str(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

// --- the backtester's state -----------------------------------------------------------------

/// The spec as a `SquadState` at GW `spec.gw`: `pool` is the planned GW's player pool,
/// `element_keys` this season's element_id -> player_key. Checks the squad's size,
/// duplicates and position counts; chips are mapped to their windows.
pub fn squad_state(
    spec: &SquadSpec,
    pool: &[PoolPlayer],
    element_keys: &HashMap<i64, i64>,
    rules: &Rules,
    gw_index: &GwIndex,
) -> Result<SquadState> {
    let by_key: HashMap<i64, &PoolPlayer> = pool.iter().map(|p| (p.player_key, p)).collect();
    let mut holdings = Vec::new();
    for p in &spec.players {
        let Some(&key) = element_keys.get(&p.element_id) else {
            return entry_error(format!(
                "element {} is not a {} player",
                p.element_id,
                season_label(spec.season)
            ));
        };
        let Some(row) = by_key.get(&key) else {
            let who = if p.name.is_empty() { key.to_string() } else { p.name.clone() };
            return entry_error(format!(
                "element {} ({who}) is not in the GW{} player pool (left the league?): sell them \
                 in the squad file",
                p.element_id, spec.gw
            ));
        };
        holdings.push(Holding {
            player_key: key,
            element_type: row.element_type,
            team_key: row.team_key,
            purchase_price: p.purchase_price,
            price: row.price,
        });
    }
    check_squad(&holdings, rules)?;
    let gw = index(gw_index, spec.gw)?;
    let mut state = SquadState::new(spec.season, gw, holdings, spec.bank, spec.free_transfers)?;
    if spec.free_transfers > rules.max_free_transfers {
        return entry_error(format!(
            "free_transfers {} > the rules' cap {}",
            spec.free_transfers, rules.max_free_transfers
        ));
    }
    state.chips_used = chip_ids(spec, rules, gw_index)?;
    Ok(state)
}

/// The spec's chips as `SquadState.chips_used` ((chip_id, gw_index)), in GW order; each takes
/// the window `chip_window` gives at its GW.
pub fn chip_ids(spec: &SquadSpec, rules: &Rules, gw_index: &GwIndex) -> Result<Vec<(i64, i64)>> {
    let mut state = SquadState::new(spec.season, 1, Vec::new(), 0, 0)?;
    let mut chips = spec.chips_used.clone();
    chips.sort_by_key(|c| c.1);
    for (name, gw) in chips {
        if gw >= spec.gw {
            return entry_error(format!("chips_used: {name} in GW{gw}, not before GW{}", spec.gw));
        }
        let mut at = state.clone();
        at.gw_index = index(gw_index, gw)?;
        match chip_window(&at, &name, rules) {
            Ok(chip_id) => state.chips_used.push((chip_id, at.gw_index)),
            Err(e) => return entry_error(format!("chips_used: {name} in GW{gw}: {e}")),
        }
    }
    Ok(state.chips_used)
}

fn check_squad(holdings: &[Holding], rules: &Rules) -> Result<()> {
    let keys: HashSet<i64> = holdings.iter().map(|h| h.player_key).collect();
    if keys.len() != holdings.len() {
        return entry_error("the squad holds a player twice");
    }
    if holdings.len() as i64 != rules.squad_size {
        return entry_error(format!(
            "the squad has {} players, not {}",
            holdings.len(),
            rules.squad_size
        ));
    }
    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for h in holdings {
        *counts.entry(h.element_type).or_default() += 1;
    }
    if counts != rules.squad_select {
        return entry_error(format!(
            "position counts {} != {}",
            py_dict(&counts),
            py_dict(&rules.squad_select)
        ));
    }
    Ok(())
}

/// Python's `repr` of an int -> int dict (in key order).
pub fn py_dict(map: &BTreeMap<i64, i64>) -> String {
    let parts: Vec<String> = map.iter().map(|(k, v)| format!("{k}: {v}")).collect();
    format!("{{{}}}", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::scoring::backtest_rules;
    use serde_json::json;

    fn rules() -> Rules {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/scoring");
        backtest_rules(2026, &dir).unwrap() // max 5 FTs, WC/FH retain FTs, chips per half
    }

    fn gw_index() -> GwIndex {
        (1..39).map(|g| (g, g)).collect()
    }

    const ELEMENTS: [i64; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    fn position(e: i64) -> i64 {
        if e <= 2 {
            1
        } else if e <= 7 {
            2
        } else if e <= 12 {
            3
        } else {
            4
        }
    }

    fn start_price(e: i64) -> i64 {
        40 + e
    }

    fn start_prices() -> HashMap<i64, i64> {
        (1..30).map(|e| (e, start_price(e))).collect()
    }

    fn history(rows: &[(i64, i64, i64)], chips: &[(&str, i64)]) -> Value {
        json!({
            "current": rows.iter().map(|(e, bank, n)| json!({"event": e, "bank": bank, "event_transfers": n})).collect::<Vec<_>>(),
            "chips": chips.iter().map(|(name, e)| json!({"name": name, "event": e})).collect::<Vec<_>>(),
        })
    }

    fn transfer(event: i64, out: i64, into: i64, cost: i64, time: &str) -> Value {
        json!({"event": event, "element_out": out, "element_in": into, "element_in_cost": cost, "time": time})
    }

    const T: &str = "2026-09-01T10:00:00Z";

    fn imported_with(
        rows: &[(i64, i64, i64)],
        squads: &[(i64, Vec<i64>)],
        transfers: &[Value],
        chips: &[(&str, i64)],
        gw: Option<i64>,
        gw_index: &GwIndex,
    ) -> Result<SquadSpec> {
        let gw = gw.unwrap_or_else(|| rows.iter().map(|r| r.0).max().unwrap() + 1);
        let squads: HashMap<i64, Vec<i64>> = squads.iter().cloned().collect();
        let mut picks = |event: i64| -> Result<Value> {
            Ok(
                json!({"picks": squads[&event].iter().map(|e| json!({"element": e})).collect::<Vec<_>>()}),
            )
        };
        let mut prices = |_gw: i64| -> Result<HashMap<i64, i64>> { Ok(start_prices()) };
        squad_from_api(
            2026,
            gw,
            &history(rows, chips),
            &mut picks,
            transfers,
            &rules(),
            gw_index,
            &mut prices,
        )
    }

    fn imported(
        rows: &[(i64, i64, i64)],
        squads: &[(i64, Vec<i64>)],
        transfers: &[Value],
        chips: &[(&str, i64)],
        gw: Option<i64>,
    ) -> Result<SquadSpec> {
        imported_with(rows, squads, transfers, chips, gw, &gw_index())
    }

    fn message(result: Result<impl fmt::Debug>) -> String {
        let err = result.unwrap_err();
        assert!(err.downcast_ref::<EntryError>().is_some(), "not an EntryError: {err:#}");
        err.to_string()
    }

    fn prices(spec: &SquadSpec) -> HashMap<i64, i64> {
        spec.players.iter().map(|p| (p.element_id, p.purchase_price)).collect()
    }

    #[test]
    fn import_squad_bank_prices_and_free_transfers() {
        let mut squad = ELEMENTS[..14].to_vec();
        squad.push(20); // element 15 sold for 20 in GW3
        let spec = imported(
            &[(1, 5, 0), (2, 5, 0), (3, 3, 1), (4, 3, 0)],
            &[(4, squad)],
            &[transfer(3, 15, 20, 62, T)],
            &[],
            None,
        )
        .unwrap();
        assert_eq!((spec.gw, spec.bank), (5, 3));
        assert_eq!(prices(&spec)[&20], 62);
        assert_eq!(prices(&spec)[&1], start_price(1));
        // GW1 -> 1, GW2 (0 made) -> 2, GW3 (1 made) -> 2, GW4 (0 made) -> 3.
        assert_eq!(spec.free_transfers, 3);
        assert!(spec.chips_used.is_empty());
        assert!(spec.notes.iter().any(|n| n.contains("GW5 before its deadline aren't public")));
    }

    #[test]
    fn the_held_since_note_only_for_a_squad_picked_after_gw1() {
        let from_gw1 =
            imported(&[(1, 0, 0), (2, 0, 0)], &[(2, ELEMENTS.to_vec())], &[], &[], None).unwrap();
        assert!(!from_gw1.notes.iter().any(|n| n.contains("held since")));
        let from_gw3 =
            imported(&[(3, 0, 0), (4, 0, 0)], &[(4, ELEMENTS.to_vec())], &[], &[], None).unwrap();
        assert!(from_gw3.notes.iter().any(|n| n.contains("held since GW3")));
    }

    #[test]
    fn the_latest_transfer_in_sets_the_price() {
        let spec = imported(
            &[(1, 0, 0), (2, 0, 1), (3, 0, 1)],
            &[(3, ELEMENTS.to_vec())],
            &[transfer(2, 5, 21, 50, T), transfer(3, 21, 5, 47, "2026-09-10T10:00:00Z")],
            &[],
            None,
        )
        .unwrap();
        assert_eq!(prices(&spec)[&5], 47);
    }

    #[test]
    fn wildcard_keeps_the_free_transfers_and_its_transfers_set_prices() {
        let spec = imported(
            &[(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 9), (5, 0, 0)],
            &[(5, ELEMENTS.to_vec())],
            &[transfer(4, 21, 3, 55, T)],
            &[("wildcard", 4)],
            None,
        )
        .unwrap();
        assert_eq!(spec.free_transfers, 4);
        assert_eq!(spec.chips_used, vec![("wildcard".to_owned(), 4)]);
        assert_eq!(prices(&spec)[&3], 55);
    }

    #[test]
    fn free_hit_reverts_squad_and_bank_and_its_buys_set_no_price() {
        let mut free_hit = ELEMENTS[..14].to_vec();
        free_hit.push(25);
        let spec = imported(
            &[(1, 4, 0), (2, 4, 0), (3, 9, 6)],
            &[(2, ELEMENTS.to_vec()), (3, free_hit)],
            &[transfer(3, 15, 25, 70, T)],
            &[("freehit", 3)],
            None,
        )
        .unwrap();
        assert_eq!(spec.players.iter().map(|p| p.element_id).collect::<Vec<_>>(), ELEMENTS);
        assert_eq!(spec.bank, 4);
        assert!(spec.players.iter().all(|p| p.purchase_price == start_price(p.element_id)));
        assert_eq!(spec.free_transfers, 2);
        assert!(spec.notes.iter().any(|n| n.contains("Free Hit")));
    }

    #[test]
    fn a_late_starter_gets_one_free_transfer_after_their_first_gw() {
        let ft = replay_free_transfers(
            2026,
            5,
            &[(3, 0), (4, 0)],
            &BTreeMap::new(),
            &rules(),
            &gw_index(),
        );
        assert_eq!(ft.unwrap(), 2);
        let spec = imported(&[(3, 0, 0)], &[(3, ELEMENTS.to_vec())], &[], &[], None).unwrap();
        assert_eq!(spec.free_transfers, 1);
    }

    #[test]
    fn free_transfers_cap_at_five() {
        let rows: Vec<_> = (1..10).map(|e| (e, 0, 0)).collect();
        let spec = imported(&rows, &[(9, ELEMENTS.to_vec())], &[], &[], None).unwrap();
        assert_eq!(spec.free_transfers, 5);
    }

    #[test]
    fn import_errors() {
        let m = message(imported(&[(1, 0, 0)], &[(1, ELEMENTS.to_vec())], &[], &[], Some(1)));
        assert!(m.contains("no finished GW before GW1"), "{m}");
        let rows = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
        let m = message(imported(&rows, &[(4, ELEMENTS.to_vec())], &[], &[], Some(6)));
        assert!(m.contains("GW5's deadline hasn't passed") && m.contains("plan GW5"), "{m}");
        let mut short: GwIndex = (1..9).map(|g| (g, g)).collect();
        short.insert(10, 10);
        let m = message(imported_with(
            &[(9, 0, 0)],
            &[(9, ELEMENTS.to_vec())],
            &[],
            &[],
            Some(10),
            &short,
        ));
        assert!(m.contains("GW9 is not in the season's schedule"), "{m}");
        let m = message(imported(&[(1, 0, 0)], &[(1, ELEMENTS[..14].to_vec())], &[], &[], None));
        assert!(m.contains("hold 14 players"), "{m}");
        let m = message(imported(
            &[(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            &[(3, ELEMENTS.to_vec())],
            &[],
            &[("wildcard", 2), ("wildcard", 3)],
            None,
        ));
        assert!(m.contains("already used"), "{m}");
    }

    // --- squad file ---------------------------------------------------------------------

    fn write(dir: &Path, text: &str) -> std::path::PathBuf {
        let path = dir.join("squad.toml");
        std::fs::write(&path, text).unwrap();
        path
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fplopt-entry-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn spec_of(players: &[i64], bank: i64, ft: i64, chips: &[(&str, i64)]) -> SquadSpec {
        let mut spec = SquadSpec::new(
            2026,
            6,
            players.iter().map(|&e| SquadPlayer::new(e, start_price(e))).collect(),
            bank,
            ft,
        );
        spec.chips_used = chips.iter().map(|(n, g)| (n.to_string(), *g)).collect();
        spec
    }

    fn no_resolve(_: &str) -> Result<i64> {
        Ok(0)
    }

    #[test]
    fn a_partial_file_overrides_only_its_keys() {
        let dir = tmp("partial");
        let path = write(&dir, "free_transfers = 2\n");
        let spec = apply_overrides(
            Some(spec_of(&ELEMENTS, 0, 4, &[])),
            &read_squad_file(&path).unwrap(),
            2026,
            6,
            &mut no_resolve,
        )
        .unwrap();
        assert_eq!(spec.free_transfers, 2);
        assert_eq!(spec.players, spec_of(&ELEMENTS, 0, 1, &[]).players);
        assert_eq!(spec.notes.last().unwrap(), "from the squad file: free_transfers");
    }

    #[test]
    fn a_full_file_without_an_import() {
        let dir = tmp("full");
        let players: String = ELEMENTS[1..]
            .iter()
            .map(|&e| {
                format!(
                    "[[players]]\nelement_id = {e}\npurchase_price = {}\n\n",
                    py_float_repr(start_price(e) as f64 / 10.0)
                )
            })
            .collect();
        let path = write(
            &dir,
            &format!(
                "season = \"2026-27\"\ngw = 6\nbank = 1.5\nfree_transfers = 2\nchips_used = \
                 [[\"wildcard\", 4]]\n\n[[players]]\nname = \"Keeper\"\npurchase_price = \
                 4.1\n\n{players}"
            ),
        );
        let mut resolve = |n: &str| -> Result<i64> {
            assert_eq!(n, "Keeper");
            Ok(1)
        };
        let spec =
            apply_overrides(None, &read_squad_file(&path).unwrap(), 2026, 6, &mut resolve).unwrap();
        assert_eq!((spec.bank, spec.free_transfers), (15, 2));
        assert_eq!(spec.chips_used, vec![("wildcard".to_owned(), 4)]);
        assert_eq!(spec.players.iter().map(|p| p.element_id).collect::<Vec<_>>(), ELEMENTS);
        assert_eq!(spec.players[0].purchase_price, 41);
        assert_eq!(spec.players[0].name, "Keeper");
    }

    #[test]
    fn file_errors() {
        let dir = tmp("errors");
        let read = |text: &str| read_squad_file(&write(&dir, text));
        let m = message(apply_overrides(
            None,
            &read("free_transfers = 1\n").unwrap(),
            2026,
            6,
            &mut no_resolve,
        ));
        assert!(m.contains("needs bank, players"), "{m}");
        let m = message(apply_overrides(
            Some(spec_of(&ELEMENTS, 0, 1, &[])),
            &read("gw = 7\n").unwrap(),
            2026,
            6,
            &mut no_resolve,
        ));
        assert!(m.contains("for gw 7"), "{m}");
        let m = message(apply_overrides(
            Some(spec_of(&ELEMENTS, 0, 1, &[])),
            &read("season = \"2025-26\"\n").unwrap(),
            2026,
            6,
            &mut no_resolve,
        ));
        assert_eq!(m, "the squad file is for season 2025-26, the plan is for 2026");
        let bad = dir.join("latin1.toml");
        std::fs::write(&bad, b"bank = 1.0  # Gro\xdf\n").unwrap();
        assert!(message(read_squad_file(&bad)).contains("latin1.toml"));
        let m = message(read("free_transfer = 1\n"));
        assert!(m.contains("unknown key(s) ['free_transfer'] (known: ['bank', 'chips_used', 'free_transfers', 'gw', 'players', 'season'])"), "{m}");
        let m = message(read("bank = 0.55\n"));
        assert!(m.ends_with("bank must be >= 0 in steps of 0.1, got 0.55"), "{m}");
        let m = message(read("bank = \"x\"\n"));
        assert!(m.ends_with("bank must be a number of £m, got 'x'"), "{m}");
        assert!(
            message(read("[[players]]\npurchase_price = 5.0\n")).contains("element_id or name")
        );
        assert!(
            message(read("[[players]]\nelement_id = 3\n"))
                .contains("player 1 needs purchase_price")
        );
        assert!(message(read("gw = 0\n")).ends_with("gw must be an integer >= 1"));
        assert!(
            message(read("chips_used = [[\"wildcard\"]]\n"))
                .contains("chips_used entries are [name, gw]")
        );
        let m = message(read("season = \"2026-28\"\n"));
        assert!(m.ends_with("season: not a season label like 2026-27: '2026-28'"), "{m}");
        assert_eq!(read("bank = 2\n").unwrap().bank, Some(20));
    }

    #[test]
    fn written_file_reads_back_and_matches_python_text() {
        let dir = tmp("written");
        let mut spec = spec_of(&ELEMENTS, 7, 3, &[("wildcard", 4)]);
        spec.notes = vec!["a note".to_owned()];
        let path = dir.join("out.toml");
        let names: HashMap<i64, String> = [(1, "Raya".to_owned())].into();
        write_squad_file(&spec, &path, &names).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("element_id = 1  # Raya"));
        let head = "# Squad for `fplopt optimize plan --squad-file`. Prices in £m.\n# a note\n\
                    season = \"2026-27\"\ngw = 6\nbank = 0.7\nfree_transfers = 3\nchips_used = \
                    [[\"wildcard\", 4]]\n\n[[players]]\nelement_id = 1  # Raya\npurchase_price = \
                    4.1\n\n[[players]]\nelement_id = 2\npurchase_price = 4.2\n";
        assert!(text.starts_with(head), "{text}");
        assert!(text.ends_with("element_id = 15\npurchase_price = 5.5\n"));
        let back =
            apply_overrides(None, &read_squad_file(&path).unwrap(), 2026, 6, &mut no_resolve)
                .unwrap();
        assert_eq!(
            (back.players, back.bank, back.free_transfers, back.chips_used),
            (spec.players, spec.bank, spec.free_transfers, spec.chips_used)
        );
        assert_eq!(money(25), "2.5");
        assert_eq!(money(1005), "100.5");
        assert_eq!(money(-5), "-0.5");
    }

    // --- the backtester's state ---------------------------------------------------------

    fn pool(elements: impl IntoIterator<Item = i64>) -> Vec<PoolPlayer> {
        elements
            .into_iter()
            .map(|e| PoolPlayer {
                player_key: 1000 + e,
                element_type: position(e),
                team_key: e % 10,
                price: start_price(e) + 3,
            })
            .collect()
    }

    fn keys() -> HashMap<i64, i64> {
        (1..30).map(|e| (e, 1000 + e)).collect()
    }

    #[test]
    fn squad_state_from_spec() {
        let spec = spec_of(&ELEMENTS, 5, 2, &[("wildcard", 4)]);
        let state = squad_state(&spec, &pool(1..30), &keys(), &rules(), &gw_index()).unwrap();
        assert_eq!((state.gw_index, state.bank, state.free_transfers), (6, 5, 2));
        assert_eq!(state.chips_used, vec![(1, 4)]); // the first-half Wildcard window
        let held = state.holdings.iter().find(|h| h.player_key == 1001).unwrap();
        assert_eq!((held.purchase_price, held.price), (start_price(1), start_price(1) + 3));
    }

    #[test]
    fn squad_state_errors() {
        let (r, g, k) = (rules(), gw_index(), keys());
        let mut players = ELEMENTS[1..].to_vec();
        players.push(16);
        let m = message(squad_state(&spec_of(&players, 0, 1, &[]), &pool(1..30), &k, &r, &g));
        assert_eq!(m, "position counts {1: 1, 2: 5, 3: 5, 4: 4} != {1: 2, 2: 5, 3: 5, 4: 3}");
        let m = message(squad_state(&spec_of(&ELEMENTS, 0, 1, &[]), &pool(2..30), &k, &r, &g));
        assert!(m.contains("not in the GW6 player pool"), "{m}");
        let m = message(squad_state(&spec_of(&ELEMENTS, 0, 6, &[]), &pool(1..30), &k, &r, &g));
        assert!(m.contains("cap 5"), "{m}");
        let m = message(squad_state(
            &spec_of(&ELEMENTS, 0, 1, &[("bboost", 6)]),
            &pool(1..30),
            &k,
            &r,
            &g,
        ));
        assert!(m.contains("not before GW6"), "{m}");
        let mut twice = ELEMENTS[..14].to_vec();
        twice.push(1);
        let m = message(squad_state(&spec_of(&twice, 0, 1, &[]), &pool(1..30), &k, &r, &g));
        assert_eq!(m, "the squad holds a player twice");
    }
}
