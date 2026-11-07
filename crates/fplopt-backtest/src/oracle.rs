//! Oracle price paths (`prices=oracle`): the real deadline prices of a plan's horizon GWs,
//! read from `player_gw.value` (a GW's `value` is its deadline price).
//!
//! **This reads the future by design.** It is a backtest-only foresight bound (price planner
//! spec, Stage A): how much a perfect price forecast could be worth. Only the backtester's
//! decision paths (`simulator::decide_step`, `money_value::solve_at`) hand it to a policy;
//! the leakage check lists it as excluded (`LEAKAGE_EXCLUDED`), and nothing outside the
//! backtest specs can ask for it.

use std::collections::HashMap;

use anyhow::{Result, bail};
use fplopt_optimize::{HorizonGw, Player};
use fplopt_store::DataStore;

/// What `check leakage` does not check, and why: (name, reason).
pub const LEAKAGE_EXCLUDED: &[(&str, &str)] = &[("prices=oracle", "backtest-only foresight bound")];

/// One season's deadline prices: (season, gw, player_key) -> `player_gw.value` (tenths).
#[derive(Clone, Debug, Default)]
pub struct FuturePrices {
    by_key: HashMap<(i64, i64, i64), i64>,
}

impl FuturePrices {
    /// Every `player_gw` price of `season`, read from a view that sees everything (the
    /// latest representable instant): **the future, by design** (backtest only).
    pub fn load(store: &DataStore, season: i64) -> Result<FuturePrices> {
        if fplopt_core::seasons::HOLDOUT_SEASONS.iter().any(|&s| i64::from(s) == season) {
            bail!("prices=oracle: season {season} is the holdout season");
        }
        let view = store.as_of(i64::MAX);
        let rows = view.table("player_gw", Some(&["season", "gw", "player_key", "value"]))?;
        let (seasons, gws) = (rows.i64("season")?, rows.i64("gw")?);
        let (keys, values) = (rows.i64("player_key")?, rows.opt_i64("value")?);
        let mut by_key = HashMap::new();
        for i in 0..rows.height() {
            if seasons[i] != season {
                continue;
            }
            if let Some(v) = values[i] {
                by_key.insert((season, gws[i], keys[i]), v);
            }
        }
        if by_key.is_empty() {
            bail!("prices=oracle: no player_gw prices for season {season}");
        }
        Ok(FuturePrices { by_key })
    }

    #[cfg(test)]
    pub fn from_rows(rows: impl IntoIterator<Item = (i64, i64, i64, i64)>) -> FuturePrices {
        FuturePrices { by_key: rows.into_iter().map(|(s, g, k, v)| ((s, g, k), v)).collect() }
    }

    /// The player's price path over `gws`: `[0]` today's price (`player.price`; the planner
    /// overwrites it anyway), then each horizon GW's deadline price, the previous entry
    /// carried forward where he has none (a blank, or not yet in the game).
    pub fn path(&self, season: i64, player: &Player, gws: &[HorizonGw]) -> Vec<f64> {
        let mut path = Vec::with_capacity(gws.len());
        let mut last = player.price as f64;
        for (t, gw) in gws.iter().enumerate() {
            if t > 0
                && let Some(&v) = self.by_key.get(&(season, gw.gw, player.player_key))
            {
                last = v as f64;
            }
            path.push(last);
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn player(player_key: i64, price: i64) -> Player {
        Player {
            player_key,
            element_type: 3,
            team_key: 1,
            price,
            sell_price: price,
            owned: false,
            buyable: true,
            xp: vec![0.0; 4],
            purchase_price: None,
            price_path: None,
        }
    }

    #[test]
    fn path_starts_at_todays_price_and_carries_forward() {
        // GW5's own row (49) is ignored: entry 0 is today's price
        let fp = FuturePrices::from_rows([
            (2020, 5, 1, 49),
            (2020, 6, 1, 52),
            (2020, 8, 1, 55),
            (2021, 7, 1, 99),
        ]);
        let gws: Vec<HorizonGw> = (5..=8)
            .enumerate()
            .map(|(h, gw)| HorizonGw { gw, gw_index: gw, horizon: h as i64 })
            .collect();
        assert_eq!(fp.path(2020, &player(1, 50), &gws), vec![50.0, 52.0, 52.0, 55.0]);
        // nobody's rows: today's price throughout
        assert_eq!(fp.path(2020, &player(2, 45), &gws), vec![45.0; 4]);
    }

    #[test]
    fn load_reads_one_season_including_the_far_future() {
        use fplopt_build::table::{Column, Table};
        let rows = Table::new()
            .with("season", Column::I64(vec![2020, 2020, 2021]))
            .with("gw", Column::I64(vec![6, 7, 6]))
            .with("player_key", Column::I64(vec![1, 1, 1]))
            .with("value", Column::I64(vec![52, 53, 99]))
            // available long after any deadline: the oracle still sees it
            .with("available_at", Column::Ts(vec![1, i64::MAX - 1, 2]));
        let store = DataStore::from_tables([("player_gw".to_owned(), rows)].into()).unwrap();
        let fp = FuturePrices::load(&store, 2020).unwrap();
        assert_eq!(fp.by_key.len(), 2);
        assert_eq!(fp.by_key[&(2020, 7, 1)], 53);
        assert!(FuturePrices::load(&store, 2019).is_err());
        assert!(FuturePrices::load(&store, 2025).unwrap_err().to_string().contains("holdout"));
    }
}
