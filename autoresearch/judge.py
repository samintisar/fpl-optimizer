"""Keep or discard a model experiment (autoresearch/program.md). Frozen: the loop never
edits this file.

    judge.py summarize <calibration json> <sha> <seconds>   -> metrics.json on stdout
    judge.py compare <baseline metrics.json> <candidate metrics.json>

`summarize` takes the `chosen` rows (the live calibration parts) of `dev/calibrate_v1.py
fit`'s report: per develop season 2017/18-2022/23 and their mean.

`compare` prints KEEP or DISCARD with the reasons and exits 0 / 1. KEEP needs every rule:
1. mean xP MSE at horizon 0 lower by at least MIN_GAIN;
2. mean xP MSE at horizons 1-5 no higher than SLACK_H15 above the baseline;
3. per season (MSE horizon 0): more seasons better than worse (|change| < TIE is a tie, so a
   change that touches only the flag seasons can pass), none worse by more than MAX_SEASON_LOSS;
4. no component metric worse by more than COMPONENT_SLACK (relative): P(start) Brier h0 and
   h1-5, minutes log loss h0, P(goal) Brier, P(CS) Brier, goals log-likelihood;
5. calibration in the large: |mean xP - mean points| at horizon 0 no more than CAL_SLACK worse.
"""

from __future__ import annotations

import json
import sys

MIN_GAIN = 0.0005
SLACK_H15 = 0.0002
TIE = 1e-6
MAX_SEASON_LOSS = 0.003
COMPONENT_SLACK = 0.005
CAL_SLACK = 0.01
METRICS = (
    "mse_h0",
    "mse_h15",
    "mean_xp_h0",
    "mean_points_h0",
    "brier_start_h0",
    "brier_start_h15",
    "minutes_ll_h0",
    "brier_cs",
    "brier_goal_h0",
    "goals_ll_h0",
)
LOWER_IS_BETTER = (
    "brier_start_h0",
    "brier_start_h15",
    "minutes_ll_h0",
    "brier_cs",
    "brier_goal_h0",
)
HIGHER_IS_BETTER = ("goals_ll_h0",)


def summarize(path: str, sha: str, seconds: str) -> dict:
    report = json.loads(open(path, encoding="utf-8").read())
    rows = [r for r in report["rows"] if r["variant"] == "chosen"]
    per_season = {str(r["season"]): {m: r[m] for m in METRICS} for r in rows}
    mean = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
    return {"sha": sha, "seconds": int(seconds), "mean": mean, "per_season": per_season}


def compare(base: dict, cand: dict) -> tuple[bool, list[str]]:
    b, c = base["mean"], cand["mean"]
    reasons, ok = [], True

    def rule(passed: bool, text: str) -> None:
        nonlocal ok
        ok = ok and passed
        reasons.append(("pass  " if passed else "FAIL  ") + text)

    gain = b["mse_h0"] - c["mse_h0"]
    rule(gain >= MIN_GAIN, f"MSE h0 {b['mse_h0']:.5f} -> {c['mse_h0']:.5f} (gain {gain:+.5f})")
    d15 = c["mse_h15"] - b["mse_h15"]
    rule(d15 <= SLACK_H15, f"MSE h1-5 {b['mse_h15']:.5f} -> {c['mse_h15']:.5f} ({d15:+.5f})")
    deltas = {
        s: cand["per_season"][s]["mse_h0"] - base["per_season"][s]["mse_h0"]
        for s in base["per_season"]
    }
    better = sum(d < -TIE for d in deltas.values())
    worse = sum(d > TIE for d in deltas.values())
    worst = max(deltas.values())
    shown = ", ".join(f"{s} {d:+.4f}" for s, d in sorted(deltas.items()))
    rule(better > worse, f"seasons better {better}, worse {worse} ({shown})")
    rule(worst <= MAX_SEASON_LOSS, f"worst season {worst:+.5f} (max +{MAX_SEASON_LOSS})")
    for m in LOWER_IS_BETTER:
        rel = (c[m] - b[m]) / abs(b[m])
        rule(rel <= COMPONENT_SLACK, f"{m} {b[m]:.5f} -> {c[m]:.5f} ({rel:+.2%})")
    for m in HIGHER_IS_BETTER:
        rel = (b[m] - c[m]) / abs(b[m])
        rule(rel <= COMPONENT_SLACK, f"{m} {b[m]:.5f} -> {c[m]:.5f} ({-rel:+.2%})")
    cal_b = abs(b["mean_xp_h0"] - b["mean_points_h0"])
    cal_c = abs(c["mean_xp_h0"] - c["mean_points_h0"])
    rule(cal_c - cal_b <= CAL_SLACK, f"|mean xP - points| h0 {cal_b:.4f} -> {cal_c:.4f}")
    return ok, reasons


def main() -> None:
    if sys.argv[1] == "summarize":
        print(json.dumps(summarize(*sys.argv[2:5]), indent=1))
        return
    base = json.loads(open(sys.argv[2], encoding="utf-8").read())
    cand = json.loads(open(sys.argv[3], encoding="utf-8").read())
    ok, reasons = compare(base, cand)
    print("KEEP" if ok else "DISCARD")
    print("\n".join(reasons))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
