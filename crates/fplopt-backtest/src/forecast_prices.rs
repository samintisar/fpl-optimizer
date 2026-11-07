//! Forecast price paths (`prices=forecast`; price planner spec Stage C, plan
//! `docs/superpowers/plans/2026-11-07-price-planner-stage-c.md` Task 1): the planner's
//! candidate prices from the Gate-B price forecast (`fplopt_models::price`).
//!
//! The forecast is fitted once per season S with the frozen settings
//! (`config/prices/forecast.json`: preset and `first_fit`) on the training rows of seasons
//! first_fit..S−1 (the holdout left out) at S's first deadline, with v1 xP from the
//! backtester's caches, exactly as `research price-eval` does. At each decision the
//! candidates' expected changes dp_h give the path: path[t] = price + dp_{h=t} for t ≥ 1
//! (h = 6 carried forward past the forecast's horizons). Unlike `prices=oracle` nothing here
//! reads the future: the fit sees only what is visible at S's first deadline, the features
//! only the decision's view.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_models::price::{
    HORIZONS, PriceFit, PriceParams, deadline_features, fit_price, predict_changes, training_rows,
};
use fplopt_optimize::HorizonGw;
use fplopt_store::{AsOfView, DataStore, Table};

use crate::price_eval::{XP_MODEL, fit_seasons};
use crate::simulator::{Caches, season_schedule};

/// The frozen forecast settings: the `PriceParams` preset and the first season fitted on.
#[derive(Clone, Debug, PartialEq)]
pub struct ForecastConfig {
    pub params: String,
    pub first_fit: i64,
}

impl ForecastConfig {
    /// `config/prices/forecast.json`, located like the scoring and chip configs.
    pub fn default_path() -> PathBuf {
        fplopt_build::scoring::default_config_dir()
            .parent()
            .map_or_else(|| PathBuf::from("config/prices"), |c| c.join("prices"))
            .join("forecast.json")
    }

    /// Reads `params` (a known preset) and `first_fit` from the file.
    pub fn load(path: &Path) -> Result<ForecastConfig> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("prices=forecast: cannot read {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("prices=forecast: {} is not JSON", path.display()))?;
        let params = json
            .get("params")
            .and_then(|v| v.as_str())
            .with_context(|| format!("prices=forecast: {} has no 'params'", path.display()))?;
        let first_fit = json
            .get("first_fit")
            .and_then(|v| v.as_i64())
            .with_context(|| format!("prices=forecast: {} has no 'first_fit'", path.display()))?;
        PriceParams::preset(params)?;
        Ok(ForecastConfig { params: params.to_owned(), first_fit })
    }

    /// Refuses the holdout and any season not after `first_fit` (S is fitted on
    /// first_fit..S−1, so it needs at least one earlier season).
    pub fn check_season(&self, season: i64) -> Result<()> {
        if HOLDOUT_SEASONS.iter().any(|&s| i64::from(s) == season) {
            bail!("prices=forecast: season {season} is the holdout season");
        }
        if season <= self.first_fit {
            bail!(
                "prices=forecast: season {} is not after the forecast's first_fit {} (each \
                 season is fitted on the seasons before it)",
                season_label(season as i32),
                season_label(self.first_fit as i32)
            );
        }
        Ok(())
    }
}

/// One season's fitted price forecast.
pub struct ForecastPrices {
    pub season: i64,
    pub fit: PriceFit,
}

impl ForecastPrices {
    /// The forecast for `season`: fitted at its first deadline on first_fit..season−1.
    pub fn fit(
        store: &DataStore,
        caches: &Caches,
        season: i64,
        config: &ForecastConfig,
    ) -> Result<ForecastPrices> {
        config.check_season(season)?;
        let schedule = season_schedule(store, season, 1)?;
        let Some(first) = schedule.first() else {
            bail!("prices=forecast: no deadlines in {}", season_label(season as i32))
        };
        let xp_at =
            |view: &AsOfView| -> Result<Table> { Ok(caches.xp(XP_MODEL, view)?.table.clone()) };
        let seasons = fit_seasons(season, config.first_fit);
        let rows = training_rows(store, first.deadline, &seasons, &xp_at)?;
        let fit = fit_price(&rows, &PriceParams::preset(&config.params)?)?;
        log::info!(
            "prices=forecast: {} fitted ({}, seasons {seasons:?}, {} rows)",
            season_label(season as i32),
            config.params,
            rows.height()
        );
        Ok(ForecastPrices { season, fit })
    }

    /// Each priced player's path over `gws` at the view (`xp`: v1's frame there); players
    /// without a feature row are absent and keep fixed prices.
    pub fn paths(
        &self,
        view: &AsOfView,
        xp: &Table,
        gws: &[HorizonGw],
    ) -> Result<HashMap<i64, Vec<f64>>> {
        let features = deadline_features(view, xp)?;
        let changes = predict_changes(&self.fit, &features)?;
        let prices: HashMap<i64, f64> = features
            .i64("player_key")?
            .iter()
            .copied()
            .zip(features.f64("price")?.iter().copied())
            .filter(|(_, p)| p.is_finite())
            .collect();
        paths_from_changes(&prices, &changes, gws.len())
    }
}

/// Price paths of `n_gws` entries from `changes` (`player_key, h, dp`, h = 1..6): path[0] is
/// a placeholder (0; the planner puts today's price there), path[t] = price + dp at h = t,
/// h = 6's change carried forward past the forecast's horizons. A non-finite dp counts as no
/// change. Players absent from `prices` or from `changes` get no path.
pub fn paths_from_changes(
    prices: &HashMap<i64, f64>,
    changes: &Table,
    n_gws: usize,
) -> Result<HashMap<i64, Vec<f64>>> {
    let (keys, hs, dps) = (changes.i64("player_key")?, changes.i64("h")?, changes.f64("dp")?);
    let mut by_key: HashMap<i64, [f64; HORIZONS.len()]> = HashMap::new();
    for i in 0..changes.height() {
        let h = hs[i];
        if !(1..=HORIZONS.len() as i64).contains(&h) {
            bail!("price changes: h = {h} outside 1..{}", HORIZONS.len());
        }
        let dp = if dps[i].is_finite() { dps[i] } else { 0.0 };
        by_key.entry(keys[i]).or_insert([0.0; HORIZONS.len()])[(h - 1) as usize] = dp;
    }
    Ok(by_key
        .into_iter()
        .filter_map(|(key, dp)| {
            let price = *prices.get(&key)?;
            let path = (0..n_gws)
                .map(|t| if t == 0 { 0.0 } else { price + dp[t.min(HORIZONS.len()) - 1] })
                .collect();
            Some((key, path))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_store::Column;

    fn changes() -> Table {
        // player 1: dp 1, 2, 2, 3, 3, 4; player 2 (unpriced): 5 throughout
        let mut keys = vec![1; 6];
        keys.extend([2; 6]);
        let hs: Vec<i64> = (1..=6).chain(1..=6).collect();
        let mut dp = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        dp.extend([5.0; 6]);
        Table::new()
            .with("player_key", Column::I64(keys))
            .with("h", Column::I64(hs))
            .with("dp", Column::F64(dp))
    }

    #[test]
    fn paths_add_the_changes_to_todays_price() {
        let prices = HashMap::from([(1, 50.0), (3, 60.0)]);
        let paths = paths_from_changes(&prices, &changes(), 4).unwrap();
        assert_eq!(paths[&1], vec![0.0, 51.0, 52.0, 52.0]);
        // player 2 has no price, player 3 no changes: neither gets a path
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn paths_carry_the_last_horizon_forward() {
        let prices = HashMap::from([(1, 50.0)]);
        let paths = paths_from_changes(&prices, &changes(), 8).unwrap();
        assert_eq!(paths[&1], vec![0.0, 51.0, 52.0, 52.0, 53.0, 53.0, 54.0, 54.0]);
        assert_eq!(paths_from_changes(&prices, &changes(), 1).unwrap()[&1], vec![0.0]);
    }

    #[test]
    fn config_loads_and_refuses_seasons() {
        let dir = std::env::temp_dir().join(format!("fplopt-forecast-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("forecast.json");
        std::fs::write(&path, r#"{"params": "p4", "first_fit": 2021, "horizons": [1]}"#).unwrap();
        let config = ForecastConfig::load(&path).unwrap();
        assert_eq!(config, ForecastConfig { params: "p4".to_owned(), first_fit: 2021 });
        assert!(config.check_season(2022).is_ok());
        let err = config.check_season(2021).unwrap_err().to_string();
        assert!(err.contains("not after"), "{err}");
        let err = config.check_season(2025).unwrap_err().to_string();
        assert!(err.contains("holdout"), "{err}");
        std::fs::write(&path, r#"{"params": "p9", "first_fit": 2021}"#).unwrap();
        assert!(ForecastConfig::load(&path).is_err());
        std::fs::write(&path, r#"{"params": "p4"}"#).unwrap();
        assert!(ForecastConfig::load(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
        // the repo's frozen settings
        assert!(ForecastConfig::default_path().ends_with("config/prices/forecast.json"));
    }
}
