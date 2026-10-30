//! `optimize plan` and `optimize squad` (Python `cli._optimize_plan`, `_optimize_squad`,
//! `_real_squad`, `plan_text`, `lineup_text`; the checks of `main`). By design the xP frame
//! comes from `--xp-file` (the models are ported in batch 6) and `--xp` only labels it.
//! Usage errors (bad flags, a bad
//! team ID, squad file or squad: `EntryError`) print `error: ...` and exit 2; other failures
//! exit 1. The printed text matches Python's.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use fplopt_archive::adapters::{base_url, fpl};
use fplopt_archive::fetch::{FetchError, Http};
use fplopt_build::rules::py_float_repr;
use fplopt_build::scoring::{Rules, backtest_rules, default_config_dir};
use fplopt_build::table::read_parquet;
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_core::settings::Settings;
use fplopt_features::baseline::player_pool;
use fplopt_optimize::entry::{
    EntryClient, EntryError, GwIndex, SquadSpec, apply_overrides, import_squad, money,
    read_squad_file, squad_file_text, squad_state,
};
use fplopt_optimize::plans::{GwPlan, PlanSet};
use fplopt_optimize::start::{
    MAX_BANK, plan_gw_index, pool_players, random_state, season_gw_index, start_deadline,
    template_state,
};
use fplopt_optimize::state::SquadState;
use fplopt_optimize::{OptimizerParams, PlanInput, Search, XpFrame, optimize};
use fplopt_store::{AsOfView, DataStore};

use crate::optimize::{INTERACTIVE_TIME_LIMIT, MAX_PLAN_HORIZON, PlanArgs, SquadArgs, SquadSource};

/// Seasons with built tables and backtest rules (Python `BACKTEST_SEASONS`).
const BACKTEST_SEASONS: std::ops::RangeInclusive<i64> = 2016..=2026;
/// FPL's ep_next is in the snapshots from 2020/21 GW32 on.
const EP_NEXT_FIRST_SEASON: i64 = 2021;
const EP_NEXT_MODELS: [&str; 2] = ["ep_next", "ep_next_fade"];

/// A usage error (Python's `parser.error`): exit 2.
#[derive(Debug)]
pub(crate) struct Usage(pub String);

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Usage {}

pub(crate) fn usage<T>(message: impl Into<String>) -> Result<T> {
    Err(Usage(message.into()).into())
}

/// Usage and squad errors -> exit 2 (no alert), anything else -> exit 1.
pub(crate) fn exit_code(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            if let Some(u) = err.downcast_ref::<Usage>() {
                eprintln!("error: {u}");
                ExitCode::from(2)
            } else if let Some(e) = err.downcast_ref::<EntryError>() {
                eprintln!("error: {e}");
                ExitCode::from(2)
            } else {
                eprintln!("error: {err:#}");
                ExitCode::from(1)
            }
        }
    }
}

pub fn plan(args: &PlanArgs) -> Result<ExitCode> {
    Ok(exit_code(run_plan(args)))
}

pub fn squad(args: &SquadArgs) -> Result<ExitCode> {
    Ok(exit_code(run_squad(args)))
}

// --- arguments ------------------------------------------------------------------------------

/// Python's `parse_seasons` (YYYY, YYYY-YYYY, YYYY-YY or a comma list): sorted start years;
/// usage errors name `arg`.
pub fn parse_seasons(text: &str, arg: &str) -> Result<Vec<i64>> {
    let mut seasons = BTreeSet::new();
    for item in text.split(',').map(str::trim) {
        let bad = || {
            usage(format!(
                "argument {arg}: bad season '{item}': expected YYYY, YYYY-YYYY (a range) or \
                 YYYY-YY (a label)"
            ))
        };
        let (head, tail) = match item.split_once('-') {
            Some((h, t)) => (h, Some(t)),
            None => (item, None),
        };
        let digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());
        if !digits(head, 4) {
            return bad();
        }
        let first: i64 = head.parse()?;
        match tail {
            None => {
                seasons.insert(first);
            }
            Some(t) if digits(t, 2) => {
                if (first + 1) % 100 != t.parse::<i64>()? {
                    return usage(format!(
                        "argument {arg}: not a season label like 2026-27: '{item}'"
                    ));
                }
                seasons.insert(first);
            }
            Some(t) if digits(t, 4) => {
                let last: i64 = t.parse()?;
                if last < first {
                    return usage(format!(
                        "argument {arg}: bad season range '{item}': end before start"
                    ));
                }
                seasons.extend(first..=last);
            }
            Some(_) => return bad(),
        }
    }
    let label = |s: &i64| season_label(*s as i32);
    let outside: Vec<String> =
        seasons.iter().filter(|s| !BACKTEST_SEASONS.contains(s)).map(label).collect();
    if !outside.is_empty() {
        return usage(format!(
            "argument {arg}: no backtest data for {} (seasons {} to {})",
            outside.join(", "),
            label(BACKTEST_SEASONS.start()),
            label(BACKTEST_SEASONS.end())
        ));
    }
    let holdout: Vec<String> = seasons
        .iter()
        .filter(|s| HOLDOUT_SEASONS.iter().any(|h| i64::from(*h) == **s))
        .map(label)
        .collect();
    if !holdout.is_empty() {
        return usage(format!(
            "argument {arg}: {} is the holdout season: backtests refuse it until Phase 6 \
             (e.g. use 2016-2024,2026)",
            holdout.join(", ")
        ));
    }
    Ok(seasons.into_iter().collect())
}

/// `parse_seasons`, then one season only.
fn parse_season(text: &str) -> Result<i64> {
    let seasons = parse_seasons(text, "--season")?;
    if seasons.len() != 1 {
        return usage("--season: give one season");
    }
    Ok(seasons[0])
}

/// `max_hits`: an int >= 0, or `none`/`inf` for unlimited (Python `parse_max_hits`).
fn parse_max_hits(text: &str) -> Result<Option<i64>, String> {
    let t = text.trim().to_lowercase();
    if t == "none" || t == "inf" {
        return Ok(None);
    }
    let value: i64 = text
        .trim()
        .replace('_', "")
        .parse()
        .map_err(|_| format!("invalid literal for int() with base 10: '{text}'"))?;
    if value < 0 {
        return Err(format!("max_hits must be >= 0 or none, got '{text}'"));
    }
    Ok(Some(value))
}

/// `optimize plan`'s planner settings over `OptimizerParams::default()`, with the
/// interactive wall-clock cap (Python `plan_params`).
fn plan_params(args: &PlanArgs) -> Result<OptimizerParams> {
    if !(1..=MAX_PLAN_HORIZON).contains(&args.horizon) {
        return usage(format!("--horizon must be in 1..{MAX_PLAN_HORIZON} (the xP frames' GWs)"));
    }
    let mut params = OptimizerParams {
        horizon: args.horizon,
        time_limit: Some(INTERACTIVE_TIME_LIMIT),
        ..OptimizerParams::default()
    };
    if let Some(text) = &args.max_hits {
        params.max_hits = parse_max_hits(text).map_err(|e| Usage(format!("--max-hits: {e}")))?;
    }
    if let Some(m) = args.hit_margin {
        params.hit_margin = m;
    }
    if let Some(d) = args.decay {
        params.decay = d;
    }
    if let Err(e) = params.validate() {
        return usage(e.to_string());
    }
    Ok(params)
}

/// The checks Python's `main` makes for both commands (after argparse's).
fn check_source(source: &SquadSource) -> Result<()> {
    if let Some(path) = &source.squad_file
        && !path.is_file()
    {
        return usage(format!("--squad-file: no file {}", path.display()));
    }
    if let Some(entry) = source.entry
        && entry < 1
    {
        return usage("--entry: an FPL team ID is a positive number");
    }
    Ok(())
}

/// The data store and the GW's gw_index (`--gw: ... has no GW..` otherwise).
fn open_store(season: i64, gw: i64) -> Result<(DataStore, i64)> {
    let settings = Settings::from_env();
    let store = DataStore::open(&settings.data_dir)?;
    let gw_index = match plan_gw_index(&store, season, gw) {
        Ok(g) => g,
        Err(e) => return usage(format!("{e:#}")),
    };
    match gw_index {
        Some(g) => Ok((store, g)),
        None => usage(format!("--gw: {} has no GW{gw}", season_label(season as i32))),
    }
}

// --- optimize plan --------------------------------------------------------------------------

fn run_plan(args: &PlanArgs) -> Result<()> {
    let season = parse_season(&args.season)?;
    let mut seed: Option<u64> = None;
    if let Some(start) = &args.start {
        let random = start
            .strip_prefix("random:")
            .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
        if let Some(digits) = random {
            match digits.parse() {
                Ok(s) => seed = Some(s),
                Err(_) => return usage(format!("--start {start}: seed out of range")),
            }
        } else if start != "template" {
            return usage(format!(
                "argument --start: --start: expected template or random:SEED, got '{start}'"
            ));
        }
    }
    check_source(&args.squad)?;
    let real = args.squad.entry.is_some() || args.squad.squad_file.is_some();
    if args.start.is_some() && real {
        return usage("--start: not with --entry/--squad-file (the squad is yours)");
    }
    if EP_NEXT_MODELS.contains(&args.xp.as_str()) && season < EP_NEXT_FIRST_SEASON {
        return usage(format!(
            "--xp {} uses ep_next, which exists only from {} (FPL snapshots); drop {}",
            args.xp,
            season_label(EP_NEXT_FIRST_SEASON as i32),
            season_label(season as i32)
        ));
    }
    if args.top_k < 1 {
        return usage("--top-k must be >= 1");
    }
    let params = plan_params(args)?;
    if !args.xp_file.is_file() {
        return usage(format!("--xp-file: no file {}", args.xp_file.display()));
    }
    let chips = !args.no_chips;
    let (store, gw_index) = open_store(season, args.gw)?;

    let view = store.as_of(start_deadline(&store, season, gw_index)?);
    let rules = backtest_rules(season, &default_config_dir())?;
    let (state, start, notes) = if real {
        let real = real_squad(&args.squad, &store, &view, season, args.gw, &rules)?;
        let start = match args.squad.entry {
            Some(entry) => format!("team {entry}"),
            None => "squad-file".to_owned(),
        };
        (real.state, start, real.spec.notes)
    } else if let Some(seed) = seed {
        let state = random_state(&view, &rules, seed, Some(MAX_BANK))?;
        (state, args.start.clone().unwrap_or_default(), Vec::new())
    } else {
        (template_state(&view, &rules)?, "template".to_owned(), Vec::new())
    };
    let began = Instant::now();
    let pool = pool_players(&player_pool(&view)?)?;
    let xp = XpFrame::from_table(&read_parquet(&args.xp_file, None)?)?;
    let problem = PlanInput::from_context(&state, &pool, &xp, &rules, &params)?;
    let plans = optimize(&problem, &params, args.top_k, chips, true, Search::Bound)?;
    let runtime = began.elapsed().as_secs_f64();

    let mut keys: Vec<i64> = problem.state.player_keys();
    for plan in plans.plans.iter().chain(&plans.roll) {
        for g in &plan.gws {
            keys.extend(g.transfers.iter().flat_map(|t| [t.out_key, t.in_key]));
            keys.extend([g.captain, g.vice]);
            keys.extend(&g.starters);
            keys.extend(&g.bench);
        }
    }
    let names = player_names(&view, &keys)?;
    let squad: Vec<&str> = problem.state.player_keys().iter().map(|k| names[k].as_str()).collect();
    let used: Vec<String> = state.chips_used.iter().map(|(c, g)| format!("{c}@{g}")).collect();
    let used = if used.is_empty() { "none".to_owned() } else { used.join(", ") };
    let max_hits = params.max_hits.map_or("none".to_owned(), |m| m.to_string());
    let mut text = format!(
        "{} GW{} (gw_index {}), {start} squad, xP {}, horizon {} GW(s), decay {}, {}, max_hits \
         {max_hits}, hit_margin {}; bank {}, FT {}, chips used (id@gw_index) {used}; {} \
         candidates ({runtime:.1}s)\nSquad: {}\n",
        season_label(season as i32),
        args.gw,
        state.gw_index,
        args.xp,
        problem.gws.len(),
        py_float_repr(params.decay),
        if chips { "chips" } else { "no chips" },
        py_float_repr(params.hit_margin),
        money(state.bank),
        state.free_transfers,
        problem.players.len(),
        squad.join(", "),
    );
    for note in &notes {
        text += &format!("  note: {note}\n");
    }
    text += &format!("\n{}", plan_text(&plans, &names));
    println!("{text}");
    if let Some(first) = plans.plans.first().map(|p| &p.gws[0]) {
        println!("\n{}", lineup_text(first, &names));
    }
    if chips {
        let solved: Vec<String> =
            plans.scenario_objectives.iter().map(|(k, v)| format!("{k} {v:.2}")).collect();
        println!("\nChip scenarios solved (total objective): {}", solved.join(", "));
    }
    Ok(())
}

/// player_key -> web_name (the key itself where unknown).
fn player_names(view: &AsOfView, keys: &[i64]) -> Result<HashMap<i64, String>> {
    let unique: Vec<i64> = keys.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let mut out: HashMap<i64, String> = unique.iter().map(|k| (*k, k.to_string())).collect();
    let found = view.lookup("player_dim", &unique)?;
    for (k, n) in found.opt_i64("player_key")?.into_iter().zip(found.opt_str("web_name")?) {
        if let Some(k) = k {
            out.insert(k, n.unwrap_or_else(|| "None".to_owned()));
        }
    }
    Ok(out)
}

/// A plain aligned table: the first `left` columns left-aligned, the rest right-aligned
/// (widths in characters, as Python's `len`).
pub fn text_table(headers: &[&str], rows: &[Vec<String>], left: usize) -> String {
    let width = |s: &str| s.chars().count();
    let mut widths: Vec<usize> = headers.iter().map(|h| width(h)).collect();
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(width(cell));
        }
    }
    let line = |cells: Vec<&str>| -> String {
        let out: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let pad = " ".repeat(widths[i] - width(cell));
                if i < left { format!("{cell}{pad}") } else { format!("{pad}{cell}") }
            })
            .collect();
        out.join("  ").trim_end().to_owned()
    };
    let mut lines = vec![line(headers.to_vec())];
    lines.extend(rows.iter().map(|r| line(r.iter().map(String::as_str).collect())));
    lines.join("\n")
}

/// The plans (#1..#k, then the roll plan) per GW: transfers, captain, chip, xP of the XI
/// with the captain, hits, bank and FTs; per plan its objective and gain vs the roll plan.
pub fn plan_text(plans: &PlanSet, names: &HashMap<i64, String>) -> String {
    let mut labelled: Vec<(String, _)> =
        plans.plans.iter().enumerate().map(|(i, p)| (format!("Plan #{}", i + 1), p)).collect();
    if let Some(roll) = &plans.roll {
        labelled.push(("Roll plan".to_owned(), roll));
    }
    let mut blocks = Vec::new();
    for (label, plan) in labelled {
        let gain = plan.gain_vs_roll.map_or("-".to_owned(), |g| format!("{g:+.2}"));
        let head = format!(
            "{label}: gain vs roll {gain}, objective {:.2} (chip terminal value {:.2}), scenario \
             {}, {}, gap {:.2}%, {} nodes",
            plan.total_objective(),
            plan.terminal_value,
            plan.scenario,
            plan.status,
            plan.mip_gap * 100.0,
            plan.n_nodes
        );
        let rows: Vec<Vec<String>> = plan
            .gws
            .iter()
            .map(|g| {
                let moves: Vec<String> = g
                    .transfers
                    .iter()
                    .map(|t| format!("{} -> {}", names[&t.out_key], names[&t.in_key]))
                    .collect();
                vec![
                    g.gw.to_string(),
                    g.chip.clone().unwrap_or_else(|| "-".to_owned()),
                    if moves.is_empty() { "-".to_owned() } else { moves.join(", ") },
                    format!("{} ({})", names[&g.captain], names[&g.vice]),
                    format!("{:.2}", g.xp),
                    g.hits.to_string(),
                    money(g.bank),
                    g.ft.to_string(),
                ]
            })
            .collect();
        let headers =
            ["GW", "chip", "transfers (out -> in)", "captain (vice)", "xP", "hits", "bank", "FT"];
        blocks.push(format!("{head}\n{}", text_table(&headers, &rows, 4)));
    }
    blocks.join("\n\n")
}

/// A plan GW's lineup: the XI (captain C, vice V) and the bench in autosub order.
pub fn lineup_text(g: &GwPlan, names: &HashMap<i64, String>) -> String {
    let mark = |k: i64| {
        if k == g.vice {
            " (V)"
        } else if k == g.captain {
            " (C)"
        } else {
            ""
        }
    };
    let xi: Vec<String> = g.starters.iter().map(|k| format!("{}{}", names[k], mark(*k))).collect();
    let bench: Vec<&str> = g.bench.iter().map(|k| names[k].as_str()).collect();
    let chip = g.chip.as_ref().map_or(String::new(), |c| format!(", chip {c}"));
    format!(
        "Plan #1 lineup for GW{}{chip}:\n  XI: {}\n  Bench (in order): {}",
        g.gw,
        xi.join(", "),
        bench.join(", ")
    )
}

// --- optimize squad -------------------------------------------------------------------------

fn run_squad(args: &SquadArgs) -> Result<()> {
    let season = parse_season(&args.season)?;
    if args.squad.entry.is_none() {
        return usage("the following arguments are required: --entry");
    }
    check_source(&args.squad)?;
    let (store, gw_index) = open_store(season, args.gw)?;
    let view = store.as_of(start_deadline(&store, season, gw_index)?);
    let rules = backtest_rules(season, &default_config_dir())?;
    let real = real_squad(&args.squad, &store, &view, season, args.gw, &rules)?;
    let text = squad_file_text(&real.spec, &real.names);
    std::fs::write(&args.out, &text).map_err(|e| anyhow::anyhow!("{}: {e}", args.out.display()))?;
    println!("{text}");
    println!("Written to {}", args.out.display());
    Ok(())
}

// --- a real squad ---------------------------------------------------------------------------

/// A real squad: the spec, its `SquadState` and element_id -> web name.
struct RealSquad {
    spec: SquadSpec,
    state: SquadState,
    names: HashMap<i64, String>,
}

/// The FPL client as the entry module's `EntryClient`.
struct FplEntry<'a>(fpl::FplClient<'a>);

impl EntryClient for FplEntry<'_> {
    fn entry_history(&self, entry_id: i64) -> Result<Vec<u8>> {
        self.0.entry_history(entry_id)
    }

    fn entry_picks(&self, entry_id: i64, gw: i64) -> Result<Vec<u8>> {
        self.0.entry_picks(entry_id, gw)
    }

    fn entry_transfers(&self, entry_id: i64) -> Result<Vec<u8>> {
        self.0.entry_transfers(entry_id)
    }
}

/// --entry (import from FPL's public API) and/or --squad-file (overrides), as a state at GW
/// `gw`'s deadline (Python `_real_squad`).
fn real_squad(
    source: &SquadSource,
    store: &DataStore,
    view: &AsOfView,
    season: i64,
    gw: i64,
    rules: &Rules,
) -> Result<RealSquad> {
    let gw_index: GwIndex = season_gw_index(store, season)?;
    let table =
        view.table("player_season", Some(&["player_key", "season", "element_id", "web_name"]))?;
    let (keys, seasons) = (table.opt_i64("player_key")?, table.opt_i64("season")?);
    let (elements, web) = (table.opt_i64("element_id")?, table.opt_str("web_name")?);
    // This season's rows, deduplicated on element_id keeping the last (in row order).
    let rows: Vec<usize> = (0..table.height()).filter(|&i| seasons[i] == Some(season)).collect();
    let mut last: HashMap<Option<i64>, usize> = HashMap::new();
    for &i in &rows {
        last.insert(elements[i], i);
    }
    let kept: Vec<usize> =
        rows.into_iter().filter(|i| last.get(&elements[*i]) == Some(i)).collect();
    let mut element_keys: HashMap<i64, i64> = HashMap::new();
    let mut names: HashMap<i64, String> = HashMap::new();
    let mut key_elements: HashMap<i64, i64> = HashMap::new();
    let mut players: Vec<(i64, String)> = Vec::new();
    for &i in &kept {
        let (Some(e), Some(k)) = (elements[i], keys[i]) else { continue };
        let name = web[i].clone().unwrap_or_else(|| "None".to_owned());
        element_keys.insert(e, k);
        names.insert(e, name.clone());
        players.push((e, name));
    }
    // Python's {k: e for e, k in element_keys.items()}: the last element of a key wins.
    for &i in &kept {
        if let (Some(e), Some(k)) = (elements[i], keys[i]) {
            key_elements.insert(k, e);
        }
    }

    let mut initial_prices = |first_gw: i64| -> Result<HashMap<i64, i64>> {
        let Some(&first_index) = gw_index.get(&first_gw) else {
            anyhow::bail!("GW{first_gw} is not in the season's schedule");
        };
        let deadline = start_deadline(store, season, first_index)?;
        let pool = pool_players(&player_pool(&store.as_of(deadline))?)?;
        Ok(pool
            .iter()
            .filter_map(|p| key_elements.get(&p.player_key).map(|e| (*e, p.price)))
            .collect())
    };
    let mut resolve = |name: &str| -> Result<i64> {
        let folded = name.to_lowercase();
        let hits: Vec<i64> =
            players.iter().filter(|(_, n)| n.to_lowercase() == folded).map(|(e, _)| *e).collect();
        if hits.len() != 1 {
            let found = if hits.is_empty() {
                "no player".to_owned()
            } else {
                format!("{} players", hits.len())
            };
            return Err(EntryError(format!(
                "squad file: {found} named {} in {}",
                fplopt_archive::py_repr_str(name),
                season_label(season as i32)
            ))
            .into());
        }
        Ok(hits[0])
    };

    let mut spec = None;
    if let Some(entry) = source.entry {
        let http = Http::new()?;
        let client = FplEntry(fpl::FplClient::new(&http, &base_url("FPL", fpl::BASE_URL)));
        match import_squad(&client, entry, season, gw, rules, &gw_index, &mut initial_prices) {
            Ok(s) => spec = Some(s),
            Err(err)
                if err.downcast_ref::<FetchError>().and_then(FetchError::status) == Some(404) =>
            {
                return Err(EntryError(format!("FPL has no team with ID {entry}")).into());
            }
            Err(err) => return Err(err),
        }
    }
    if let Some(path) = &source.squad_file {
        let overrides = read_squad_file(Path::new(path))?;
        spec = Some(apply_overrides(spec, &overrides, season, gw, &mut resolve)?);
    }
    let Some(spec) = spec else {
        return Err(EntryError("give --entry and/or --squad-file".to_owned()).into());
    };
    let pool = pool_players(&player_pool(view)?)?;
    let state = squad_state(&spec, &pool, &element_keys, rules, &gw_index)?;
    Ok(RealSquad { spec, state, names })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seasons() {
        assert_eq!(parse_season("2024").unwrap(), 2024);
        assert_eq!(parse_season("2024-25").unwrap(), 2024);
        assert_eq!(parse_season("2024-2024").unwrap(), 2024);
        let err = |t: &str| parse_season(t).unwrap_err().to_string();
        assert_eq!(err("2023,2024"), "--season: give one season");
        assert!(err("2025").contains("holdout"));
        assert!(err("2015").contains("no backtest data for 2015-16"));
        assert!(err("2024-26").contains("not a season label"));
        assert!(err("24").contains("bad season '24'"));
    }

    #[test]
    fn max_hits() {
        assert_eq!(parse_max_hits("none"), Ok(None));
        assert_eq!(parse_max_hits("INF"), Ok(None));
        assert_eq!(parse_max_hits("2"), Ok(Some(2)));
        assert!(parse_max_hits("-1").unwrap_err().contains(">= 0 or none"));
        assert!(parse_max_hits("x").is_err());
    }

    #[test]
    fn table_alignment() {
        let rows = vec![vec!["1".to_owned(), "Ødegaard".to_owned(), "2.50".to_owned()]];
        let text = text_table(&["GW", "name", "xP"], &rows, 2);
        assert_eq!(text, "GW  name        xP\n1   Ødegaard  2.50");
    }
}
