//! Walk-forward fitted xP models (Python `fplopt.models.fitted`). A fitted model splits into
//! `fit(view) -> fit` (frozen parameters) and `predict(view, fit) -> xP frame`. Called on a
//! view, it fits on `view.earlier(cutoff(view))` and predicts at the view. The cutoff is the
//! deadline of the season's latest refit GW at or before the view's (`gw_index` 1,
//! 1 + every, 1 + 2·every, …).

use anyhow::{Result, bail};
use fplopt_store::{AsOfView, Micros};

pub const REFIT_EVERY: i64 = 4;

/// Deadline of the latest refit GW (`gw_index` ≡ 1 mod `every`) of the view's season at or
/// before the view's GW.
pub fn refit_deadline(view: &AsOfView, every: i64) -> Result<Micros> {
    if every < 1 {
        bail!("every must be >= 1, got {every}");
    }
    let (season, gw) = view.gameweek_for_deadline()?;
    let gameweeks = view.table("gameweek", Some(&["season", "gw", "gw_index", "deadline_time"]))?;
    let (seasons, gws, index) =
        (gameweeks.i64("season")?, gameweeks.i64("gw")?, gameweeks.i64("gw_index")?);
    let deadlines = gameweeks.opt_i64("deadline_time")?;
    let in_season: Vec<usize> = (0..gameweeks.height()).filter(|&i| seasons[i] == season).collect();
    let Some(&target) = in_season.iter().find(|&&i| gws[i] == gw) else {
        bail!("season {season}: no gameweek {gw}");
    };
    let refit_index = 1 + (index[target] - 1).div_euclid(every) * every;
    let refit: Vec<usize> = in_season.into_iter().filter(|&i| index[i] == refit_index).collect();
    if refit.len() != 1 {
        bail!("season {season}: {} gameweeks are the refit GW of GW {gw}", refit.len());
    }
    match deadlines[refit[0]] {
        Some(deadline) => Ok(deadline),
        None => bail!("season {season}: the refit GW of GW {gw} has no deadline"),
    }
}
