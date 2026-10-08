"""Walk-forward evaluation of xP models (Phase 5 plan, *Evaluation (criterion 1)*; PLAN §5
*Metrics*): realized outcomes (`outcomes`), metric functions (`metrics`) and the runner
behind `fplopt models eval` (`run.evaluate`). An orchestrator like the simulator: it opens
the `DataStore` and reads only through `as_of` views."""
