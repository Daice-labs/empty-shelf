#!/usr/bin/env python3
"""Verify the shipped T11 records against the expected-output manifest.
Recomputes the identification result, the zero-linkage null, the primary AUROC,
and the purity law directly from the per-task records."""
import json, sys
import numpy as np, pandas as pd

exp = json.load(open("expected_outputs.json")); fail = []
def check(name, got, want, tol=1e-4):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(("  OK   " if ok else "  FAIL ") + f"{name}: {round(got,6) if isinstance(got,float) else got} (expected {want})")
    if not ok: fail.append(name)

per = pd.read_csv("results/per_task_truth_and_confidence.csv")
sig = pd.read_csv("results/signals.csv")
m = per.merge(sig[["id", "max_sim", "z"]], on="id")

def auroc(score, y):
    a, b = np.asarray(score, float)[y == 1], np.asarray(score, float)[y == 0]
    return float(np.less.outer(b, a).mean() + 0.5 * np.equal.outer(b, a).mean())

order = np.argsort(m.max_sim.values); flag = np.zeros(len(m), bool); flag[order[:82]] = True
tfp = m.true_fail_prob.values; z = m.z.values
check("attempt_fail_rate_supported", float(tfp[z == 0].mean()), exp["q0_attempt"], 1e-3)
check("attempt_fail_rate_unsupported", float(tfp[z == 1].mean()), exp["q1_attempt"], 1e-3)

u, v = float(flag[z == 1].mean()), float(flag[z == 0].mean())
check("detector_u", u, exp["u"], 1e-3); check("detector_v", v, exp["v"], 1e-3)
w1 = 0.5 * u / (0.5 * u + 0.5 * v)
yv = m.y_R9_main_seed12.values
r1, r0 = float(yv[z == 1].mean()), float(yv[z == 0].mean())
pred = w1 * r1 + (1 - w1) * r0   # Proposition 1: operating point x stratum-conditional label rates
meas = float(m.loc[flag, "y_R9_main_seed12"].mean())
check("predicted_flag_precision", pred, exp["predicted_precision"], 2e-3)
check("measured_flag_precision", meas, exp["measured_precision"], 1e-3)
check("identity_residual_below_binomial_sd", abs(meas - pred) < 0.048, True)

check("zero_linkage_auroc_ladder", auroc(-m.max_sim.values, m.y_zero_linkage_seed13.values), exp["zero_linkage_ladder"], 1e-3)
check("auroc_max_sim_primary", auroc(-m.max_sim.values, m.y_R9_main_seed12.values), exp["auroc_max_sim"], 1e-3)

g3 = json.load(open("results/g3_reproducibility.json"))
check("a3_realized_bias", float(g3["a3_sensitivity"]["realized_identification_bias"]), exp["a3_bias"], 5e-6)
check("split_hash_prefix", str(g3["split_integrity_sha256"])[:16], exp["split_hash_prefix"])

pur = pd.read_csv("results/label_purity.csv")
for R, want in exp["purity_measured"].items():
    check(f"purity_measured_R{R}", float(pur[pur.R == int(R)].iloc[0].measured_leakage), want, 1e-3)

print("\n" + ("ALL CHECKS PASSED" if not fail else f"{len(fail)} CHECK(S) FAILED"))
sys.exit(0 if not fail else 1)
