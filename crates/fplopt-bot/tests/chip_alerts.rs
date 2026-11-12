//! The alerts' chip section and `chip_calls` log (spec docs/superpowers/specs/
//! 2026-11-05-chip-alerts-design.md), Rust only: the synthetic bot golden
//! (`tests/fixtures/rust_golden/bot/`) replayed with `chip_alerts: true` must give Python's
//! alert texts with the chip section in front of `alert24` (and of an `alert2` whose chip
//! call changed), and log one `chip_calls` row per alert run. Without the key the replay is
//! Python's golden byte for byte (`tests/golden.rs`).

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use fplopt_bot::context::{ContextStore, PlanContext};
use fplopt_bot::planner::{chip_options, chip_params, live_params, plan_user};
use fplopt_bot::replay::FileFpl;
use fplopt_bot::service::{AlertOutcome, BUDGET_SKIP, BotService, ChipPlanner, SendError};
use fplopt_bot::store::{ChipCallRow, UserStore};
use fplopt_bot::util::parse_isoformat;
use fplopt_optimize::io::params_with;
use serde_json::{Value, json};

const HEADER: &str = "<b>Chip this week:</b>";
const EVIDENCE: &str = "Evidence: on 2023/24 and 2024/25";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn golden() -> PathBuf {
    repo().join("tests/fixtures/rust_golden/bot")
}

fn read(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(golden().join(name)).unwrap()).unwrap()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Rules and chip values are read relative to the repo root, as the CLI runs: every test
/// runs from there (the process-wide working directory, set once; `golden.rs` is another
/// binary).
fn at_repo_root() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| std::env::set_current_dir(repo()).unwrap());
}

/// A fresh copy of the golden's state directory.
fn state(name: &str) -> PathBuf {
    at_repo_root();
    let state =
        std::env::temp_dir().join(format!("fplopt-chip-alerts-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&state);
    copy_dir(&golden().join("state"), &state);
    state
}

/// The alert recommendations' messages in log order (`kind`, message), from Python's `db.json`.
fn golden_alerts() -> Vec<(String, String)> {
    read("db.json")["recommendations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"].as_str().unwrap().starts_with("alert"))
        .map(|r| {
            (
                r["kind"].as_str().unwrap().to_string(),
                r["message"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect()
}

/// `text` as `<section>\n\n<rest>`: the section (which starts with the header and has no blank
/// line) and the rest.
fn split_section(text: &str) -> (&str, &str) {
    assert!(text.starts_with(HEADER), "no chip section: {text}");
    text.split_once("\n\n").expect("a blank line after the section")
}

/// The alert recommendations in log order, from `bot.sqlite`.
struct AlertRec {
    kind: String,
    season: i64,
    gw: i64,
    context_id: Option<String>,
    message: Option<String>,
    error: Option<String>,
}

fn alert_recs(state: &Path) -> Vec<AlertRec> {
    let db = rusqlite::Connection::open(state.join("bot.sqlite")).unwrap();
    let mut q = db
        .prepare(
            "SELECT kind, season, gw, context_id, message, error FROM recommendations \
             WHERE kind IN ('alert24', 'alert2') ORDER BY rec_id",
        )
        .unwrap();
    q.query_map([], |r| {
        Ok(AlertRec {
            kind: r.get(0)?,
            season: r.get(1)?,
            gw: r.get(2)?,
            context_id: r.get(3)?,
            message: r.get(4)?,
            error: r.get(5)?,
        })
    })
    .unwrap()
    .map(Result::unwrap)
    .collect()
}

fn shown(row: &ChipCallRow) -> (String, String, String) {
    (
        row.planner_call.clone(),
        row.rules_call.clone(),
        row.forced.clone().unwrap_or_else(|| "none".to_string()),
    )
}

#[test]
fn replay_without_the_key_is_pythons_golden() {
    let state = state("no-key");
    let results =
        fplopt_bot::replay::replay(&read("script.json"), &state, &golden().join("fpl")).unwrap();
    assert_eq!(results, read("results.json"));
    let users = UserStore::new(state.join("bot.sqlite")).unwrap();
    let uid = users.user(5).unwrap().unwrap().user_id;
    assert!(users.chip_calls(uid).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&state);
}

#[test]
fn replay_with_chip_alerts_prepends_the_section_and_logs_the_calls() {
    let state = state("replay");
    let mut script = read("script.json");
    script["chip_alerts"] = json!(true);
    let started = Instant::now();
    let results = fplopt_bot::replay::replay(&script, &state, &golden().join("fpl")).unwrap();
    eprintln!("replay with chip_alerts: {:.1} s", started.elapsed().as_secs_f64());

    // Every step but the alerts is Python's; alert outcomes are too (none is silent here).
    let expected = read("results.json");
    let steps = script["steps"].as_array().unwrap();
    let (got, want) = (results.as_array().unwrap(), expected.as_array().unwrap());
    assert_eq!(got.len(), want.len());
    for (i, ((g, w), step)) in got.iter().zip(want).zip(steps).enumerate() {
        if step.get("alert").is_some() {
            assert_eq!(g["alert"], w["alert"], "step {i}");
            if step["kind"] == "alert24" {
                let first = g["texts"][0][1].as_str().unwrap();
                assert!(first.starts_with(HEADER), "step {i}: {first}");
            }
        } else {
            assert_eq!(g, w, "step {i}");
        }
    }

    let users = UserStore::new(state.join("bot.sqlite")).unwrap();
    let uid = users.user(5).unwrap().unwrap().user_id;
    let alerts = alert_recs(&state);
    let python = golden_alerts();
    assert_eq!(alerts.len(), python.len());
    let calls = users.chip_calls(uid).unwrap();
    assert_eq!(calls.len(), alerts.len(), "one chip_calls row per alert run");
    let alert24 = &calls[0];
    for ((rec, (kind, message)), row) in alerts.iter().zip(&python).zip(&calls) {
        assert_eq!(&rec.kind, kind);
        assert_eq!(&row.kind, kind);
        assert_eq!(row.context_id, rec.context_id);
        assert_eq!((row.season, row.gw), (rec.season, rec.gw));
        assert!(row.error.is_none(), "{row:?}");
        assert!(row.v6_call.is_none(), "V6 is logged from GW20 only: {row:?}");
        assert!(row.planner_gain.is_some());
        assert!(row.gains.starts_with('{') && row.gains.contains("bench_xp"), "{}", row.gains);
        assert!(row.scenarios.starts_with("{\"none\": "), "{}", row.scenarios);
        let text = rec.message.as_deref().unwrap();
        let shows_section = kind == "alert24" || shown(row) != shown(alert24);
        if shows_section {
            let (section, rest) = split_section(text);
            assert!(section.contains(EVIDENCE), "{section}");
            assert_eq!(rest, message, "{kind}: the rest is Python's text");
        } else {
            assert_eq!(text, message, "{kind}: an unchanged chip call adds no section");
        }
    }
    assert_eq!(calls[0].kind, "alert24");
    let _ = std::fs::remove_dir_all(&state);
}

/// A service on a copy of the golden state after the script's registration steps (up to the
/// first alert), with the script's planner settings and the chip section on.
fn registered_service(name: &str) -> (PathBuf, BotService) {
    let state = state(name);
    let mut script = read("script.json");
    let steps = script["steps"].as_array().unwrap();
    let first_alert = steps.iter().position(|s| s.get("alert").is_some()).unwrap();
    let now = steps[..first_alert]
        .iter()
        .rev()
        .find_map(|s| s.get("now"))
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    script["steps"] = Value::Array(steps[..first_alert].to_vec());
    fplopt_bot::replay::replay(&script, &state, &golden().join("fpl")).unwrap();

    let fpl = FileFpl { dir: golden().join("fpl"), down: Default::default() };
    let mut service = BotService::new(
        UserStore::new(state.join("bot.sqlite")).unwrap(),
        ContextStore::new(state.join("context")),
        Box::new(fpl),
        script["admin"].as_i64(),
    );
    let now = parse_isoformat(&now).unwrap();
    service.now = Box::new(move || now);
    let params = params_with(live_params(), &script["params"]).unwrap();
    service.planner = Box::new(move |fpl, entry, context, ft, made| {
        plan_user(fpl, entry, context, ft, &params, made)
    });
    service.chip_planner = script_chip_planner();
    service.chip_alerts = true;
    (state, service)
}

/// `/chips`'s search with the script's `chip_params` (fast on the synthetic league).
fn script_chip_planner() -> ChipPlanner {
    let chips = params_with(chip_params(), &read("script.json")["chip_params"]).unwrap();
    Box::new(move |fpl, entry, context, ft, made| {
        chip_options(fpl, entry, context, ft, &chips, made)
    })
}

/// Run user 5's alert of `kind` on the newest context; the outcome and the full text sent.
fn alert(service: &BotService, kind: &str, refuse: bool) -> (AlertOutcome, String) {
    let context = service.contexts.latest().unwrap().unwrap();
    alert_on(service, kind, &context, refuse)
}

/// `alert` on `context`.
fn alert_on(
    service: &BotService,
    kind: &str,
    context: &PlanContext,
    refuse: bool,
) -> (AlertOutcome, String) {
    let user = service.users.user(5).unwrap().unwrap();
    let sent = RefCell::new(String::new());
    let mut send = |_chat: i64, text: &str| -> Result<(), SendError> {
        if refuse {
            return Err(SendError { message: "Forbidden".into(), permanent: true });
        }
        sent.borrow_mut().push_str(text);
        Ok(())
    };
    let outcome = service.alert_user(kind, context, &user, &mut send).unwrap();
    (outcome, sent.into_inner())
}

#[test]
fn alert2_shows_the_section_only_when_the_chip_call_changed() {
    let (state, service) = registered_service("alert2");
    let uid = service.users.user(5).unwrap().unwrap().user_id;

    let started = Instant::now();
    let (outcome, text) = alert(&service, "alert24", false);
    eprintln!("one alert24 with the chip search: {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, rest) = split_section(&text);
    assert!(section.contains(EVIDENCE));
    assert!(rest.starts_with("<b>GW6 plan</b>"), "{rest}");

    // Same plan, same chip call: silent, and still logged.
    let (outcome, text) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Silent);
    assert!(text.is_empty());
    let calls = service.users.chip_calls(uid).unwrap();
    assert_eq!(calls.iter().map(|c| c.kind.as_str()).collect::<Vec<_>>(), ["alert24", "alert2"]);
    assert_eq!(shown(&calls[0]), shown(&calls[1]));

    // A different alert24 call logged for this GW: the update is sent for the chip alone,
    // with the section and no transfer change.
    let logged = &calls[0];
    let other = if logged.planner_call == "wildcard" { "freehit" } else { "wildcard" };
    service
        .users
        .record_chip_call(&ChipCallRow { planner_call: other.into(), ..logged.clone() })
        .unwrap();
    let (outcome, text) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (_, rest) = split_section(&text);
    assert!(rest.starts_with("<b>GW6 plan</b>"), "{rest}");
    // the two runs, the inserted alert24 row and this run's
    let kinds: Vec<String> =
        service.users.chip_calls(uid).unwrap().into_iter().map(|c| c.kind).collect();
    assert_eq!(kinds, ["alert24", "alert2", "alert24", "alert2"]);
    let _ = std::fs::remove_dir_all(&state);
}

#[test]
fn a_failed_chip_search_gives_the_failure_line_and_never_fails_the_alert() {
    let (state, mut service) = registered_service("failure");
    let uid = service.users.user(5).unwrap().unwrap().user_id;
    service.chip_planner = Box::new(|_, _, _, _, _| anyhow::bail!("solver timed out"));

    let (outcome, text) = alert(&service, "alert24", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, rest) = split_section(&text);
    assert_eq!(section, fplopt_bot::messages::chip_failure_section());
    assert!(rest.starts_with("<b>GW6 plan</b>"), "{rest}");
    let calls = service.users.chip_calls(uid).unwrap();
    assert_eq!(calls.len(), 1);
    let row = &calls[0];
    assert_eq!((row.planner_call.as_str(), row.rules_call.as_str()), ("none", "none"));
    assert_eq!((row.gains.as_str(), row.scenarios.as_str()), ("{}", "{}"));
    assert!(row.error.as_deref().unwrap().contains("solver timed out"), "{row:?}");

    // A failed alert2 run shows no section: the plan is unchanged, so it stays silent.
    let (outcome, _) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Silent);
    assert_eq!(service.users.chip_calls(uid).unwrap().len(), 2);

    // Undeliverable: alerts off, both rows logged.
    let (outcome, _) = alert(&service, "alert24", true);
    assert_eq!(outcome, AlertOutcome::Undeliverable);
    assert!(!service.users.user(5).unwrap().unwrap().alerts);
    let recs = alert_recs(&state);
    let last = recs.last().unwrap();
    assert!(last.error.as_deref().unwrap().starts_with("undeliverable"));
    assert!(last.message.as_deref().unwrap().starts_with(HEADER));
    assert_eq!(service.users.chip_calls(uid).unwrap().len(), 3);
    let _ = std::fs::remove_dir_all(&state);
}

#[test]
fn chip_alerts_off_logs_no_chip_calls() {
    let (state, mut service) = registered_service("off");
    service.chip_alerts = false;
    service.chip_planner = Box::new(|_, _, _, _, _| panic!("no chip search with chip_alerts off"));
    let uid = service.users.user(5).unwrap().unwrap().user_id;
    let (outcome, text) = alert(&service, "alert24", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    assert!(text.starts_with("<b>GW6 plan</b>"), "{text}");
    assert!(service.users.chip_calls(uid).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&state);
}

/// The newest context as if it were GW20's (`gw_index` 20): V6 is searched on alert24.
fn gw20(service: &BotService) -> PlanContext {
    let mut context = (*service.contexts.latest().unwrap().unwrap()).clone();
    context.gw_index = 20;
    context
}

fn uid(service: &BotService) -> i64 {
    service.users.user(5).unwrap().unwrap().user_id
}

// Fix 1a: an alert2 that is the user's first alert this GW is treated like alert24.
#[test]
fn a_first_alert2_carries_the_section_like_alert24() {
    let (state, service) = registered_service("first-alert2");
    let (outcome, text) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, rest) = split_section(&text);
    assert!(section.contains(EVIDENCE), "{section}");
    assert!(rest.starts_with("<b>GW6 plan</b>"), "{rest}");
    let _ = std::fs::remove_dir_all(&state);

    let (state, mut service) = registered_service("first-alert2-failed");
    service.chip_planner = Box::new(|_, _, _, _, _| anyhow::bail!("solver timed out"));
    let (outcome, text) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, _) = split_section(&text);
    assert_eq!(section, fplopt_bot::messages::chip_failure_section());
    let _ = std::fs::remove_dir_all(&state);
}

// Fix 1b: no alert24 chip row logged (alert24 went out before chip alerts were on).
#[test]
fn alert2_without_a_logged_alert24_call_shows_the_section() {
    let (state, mut service) = registered_service("no-alert24-row");
    service.chip_alerts = false;
    assert_eq!(alert(&service, "alert24", false).0, AlertOutcome::Sent);
    service.chip_alerts = true;
    let (outcome, text) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (_, rest) = split_section(&text);
    assert!(rest.starts_with("<b>GW6 plan</b>"), "{rest}");
    let _ = std::fs::remove_dir_all(&state);
}

// Fix 1c: the alert24 calculation failed, so the user never saw a call.
#[test]
fn alert2_after_a_failed_alert24_calculation_shows_the_section() {
    let (state, mut service) = registered_service("failed-alert24");
    service.chip_planner = Box::new(|_, _, _, _, _| anyhow::bail!("solver timed out"));
    assert_eq!(alert(&service, "alert24", false).0, AlertOutcome::Sent);
    service.chip_planner = script_chip_planner();
    let (outcome, text) = alert(&service, "alert2", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, rest) = split_section(&text);
    assert_ne!(section, fplopt_bot::messages::chip_failure_section());
    assert!(section.contains(EVIDENCE), "{section}");
    assert!(rest.starts_with("<b>GW6 plan</b>"), "{rest}");
    let calls = service.users.chip_calls(uid(&service)).unwrap();
    assert!(calls[1].error.is_none(), "{:?}", calls[1]);
    let _ = std::fs::remove_dir_all(&state);
}

// Fix 4: a panicking chip search is a failed one.
#[test]
fn a_panicking_chip_search_gives_the_failure_line() {
    let (state, mut service) = registered_service("panic");
    service.chip_planner = Box::new(|_, _, _, _, _| panic!("chip search blew up"));
    let (outcome, text) = alert(&service, "alert24", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, _) = split_section(&text);
    assert_eq!(section, fplopt_bot::messages::chip_failure_section());
    let row = &service.users.chip_calls(uid(&service)).unwrap()[0];
    assert!(row.error.as_deref().unwrap().contains("panicked: chip search blew up"), "{row:?}");

    // `/plan` never runs the chip search.
    let replies = service.run("plan", 5, &[]).unwrap();
    assert!(replies[0].text.starts_with("<b>GW6 plan</b>"), "{}", replies[0].text);
    let _ = std::fs::remove_dir_all(&state);
}

// The Told path (the squad can't be imported) runs no chip search and logs no chip row.
#[test]
fn an_entry_error_is_told_without_a_chip_search() {
    let (state, mut service) = registered_service("told");
    let nowhere = state.join("no-fpl");
    service.fpl = Box::new(FileFpl { dir: nowhere, down: Default::default() });
    service.chip_planner = Box::new(|_, _, _, _, _| panic!("no chip search on the Told path"));
    let (outcome, text) = alert(&service, "alert24", false);
    assert_eq!(outcome, AlertOutcome::Told);
    assert!(text.starts_with("Couldn't plan GW6"), "{text}");
    assert!(service.users.chip_calls(uid(&service)).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&state);
}

// Fix 5: no chip search once the tick's budget has run out.
#[test]
fn an_exhausted_time_budget_skips_the_chip_search() {
    let (state, service) = registered_service("budget");
    service.chip_budget_until.set(Some(Instant::now()));
    let (outcome, text) = alert(&service, "alert24", false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, _) = split_section(&text);
    assert_eq!(section, fplopt_bot::messages::chip_failure_section());
    let row = &service.users.chip_calls(uid(&service)).unwrap()[0];
    assert_eq!(row.error.as_deref(), Some(BUDGET_SKIP));
    let _ = std::fs::remove_dir_all(&state);
}

// Fix 5: the budget runs out during P's search: H's MILPs and V6 are skipped (the failure
// line). V6 skipped with P and H shown needs the budget to run out during H, which a test
// can't time.
#[test]
fn a_budget_running_out_during_p_skips_h_and_v6() {
    let (state, mut service) = registered_service("budget-v6");
    let p = script_chip_planner();
    service.chip_planner = Box::new(move |fpl, entry, context, ft, made| {
        std::thread::sleep(Duration::from_secs(4));
        p(fpl, entry, context, ft, made)
    });
    service.v6_planner = Box::new(|_, _, _, _, _| panic!("V6 runs over budget"));
    let context = gw20(&service);
    // The plan takes well under 3 s, so P starts within the budget and ends after it.
    service.chip_budget_until.set(Some(Instant::now() + Duration::from_secs(3)));
    let (outcome, text) = alert_on(&service, "alert24", &context, false);
    assert_eq!(outcome, AlertOutcome::Sent);
    let (section, _) = split_section(&text);
    assert_eq!(section, fplopt_bot::messages::chip_failure_section());
    let row = &service.users.chip_calls(uid(&service)).unwrap()[0];
    assert_eq!(row.error.as_deref(), Some(BUDGET_SKIP));
    assert!(row.v6_call.is_none());
    let _ = std::fs::remove_dir_all(&state);
}

// V6 from GW20: logged on alert24 only; a V6 failure keeps P's section and is in the row.
#[test]
fn v6_is_logged_on_alert24_from_gw20() {
    let (state, mut service) = registered_service("v6");
    service.v6_planner = script_chip_planner();
    let context = gw20(&service);
    let (outcome, text) = alert_on(&service, "alert24", &context, false);
    assert_eq!(outcome, AlertOutcome::Sent);
    split_section(&text);
    let (outcome, _) = alert_on(&service, "alert2", &context, false);
    assert_eq!(outcome, AlertOutcome::Silent);
    let calls = service.users.chip_calls(uid(&service)).unwrap();
    assert!(calls[0].v6_call.is_some(), "{:?}", calls[0]);
    assert!(calls[0].error.is_none(), "{:?}", calls[0]);
    assert!(calls[1].v6_call.is_none(), "{:?}", calls[1]);

    let failing: Vec<(&str, ChipPlanner)> = vec![
        ("error", Box::new(|_, _, _, _, _| anyhow::bail!("no window table"))),
        ("panic", Box::new(|_, _, _, _, _| panic!("V6 blew up"))),
    ];
    for (name, planner) in failing {
        service.v6_planner = planner;
        let (outcome, text) = alert_on(&service, "alert24", &context, false);
        assert_eq!(outcome, AlertOutcome::Sent, "{name}");
        let (section, _) = split_section(&text);
        assert_ne!(section, fplopt_bot::messages::chip_failure_section(), "{name}");
        let row = service.users.chip_calls(uid(&service)).unwrap().pop().unwrap();
        let error = row.error.clone().unwrap();
        assert!(error.starts_with("v6: "), "{name}: {row:?}");
        assert!(row.v6_call.is_none() && row.gains != "{}", "{name}: {row:?}");
        // P's call was shown: an unchanged alert2 stays silent.
        assert_eq!(alert_on(&service, "alert2", &context, false).0, AlertOutcome::Silent, "{name}");
    }
    let _ = std::fs::remove_dir_all(&state);
}
