# ZIL-H Wildlife Strike Cost Model — Analysis Code

Reproducible analysis pipeline for the paper

> **Asymmetric Cost Distribution in Wildlife Strikes: A Zero-Inflated Lognormal
> Hurdle Model for Repair Cost Prediction and Risk-Adjusted Insurance Premium
> Estimation**

Everything the manuscript reports — the descriptive statistics, the two-stage
hurdle model, the model comparison, the Gamma tail risk metrics, the route
premium schedule, and all six figures — is produced by the two scripts in this
folder from the raw FAA National Wildlife Strike Database (NWSD) export.

---

## Files

| File | What it does |
|------|--------------|
| `zilh_analysis.py` | The statistical pipeline. Loads and cleans the NWSD, computes the cost descriptives (Gini, percentiles), fits Stage 1 (logistic hurdle) and Stage 2 (lognormal severity), runs the 80/20 model comparison, calibrates the Gamma tail for VaR/TVaR, builds the premium schedule, and draws Figures 1, 4, and 6. |
| `zilh_figures_extra.py` | The three summary figures that read off aggregate numbers rather than the raw fit: Figure 2 (model horse-race + architecture), Figure 3 (cost by engine type and mass class), and Figure 5 (premium schedule + tail risk). Kept separate so the main file stays focused on the statistics. |
| `README.md` | This file. |

The split is deliberate: `zilh_analysis.py` is where the modelling lives, and
`zilh_figures_extra.py` holds the presentation figures so it can be re-run on
its own while you tune the layout, without re-fitting anything.

---

## Data

You need the **Public** NWSD export. Download it from the FAA:

> https://wildlife.faa.gov  →  *Search / Download* → export the full public
> dataset as `Public.xlsx` (or CSV).

The scripts do **not** ship any data, and no synthetic values are used
anywhere — every number in the paper comes from this file.

### Expected columns

The NWSD renames fields slightly between serial releases, so all the column
names are collected in one place at the top of `zilh_analysis.py`:

```python
COLS = {
    "cost":     "COST_REPAIRS_INFL_ADJ",   # inflation-adjusted repair cost, USD
    "mass":     "AC_MASS",                  # 1..4 mass class
    "engine":   "TYPE_ENG",                 # engine-type code (A/B/C/D...)
    "phase":    "PHASE_OF_FLIGHT",          # phase-of-flight label
    "ingested": "INGESTED",                 # engine ingestion flag
    "year":     "INCIDENT_YEAR",
}
```

If your export uses different names (e.g. `COST_REPAIRS` instead of
`COST_REPAIRS_INFL_ADJ`), edit the right-hand side here and nothing else
changes.

Mass classes map to the kg boundaries used in the paper:

| Class | Boundary |
|-------|----------|
| 1 | < 2,269 kg |
| 2 | 2,269 – 5,670 kg |
| 3 | 5,670 – 27,215 kg |
| 4 | > 27,215 kg |

---

## Requirements

```
python >= 3.10
pandas  >= 2.0
numpy   >= 1.24
scipy   >= 1.10
statsmodels >= 0.14
matplotlib  >= 3.7
openpyxl    >= 3.1     # only needed to read the .xlsx export
```

Install everything with:

```bash
pip install pandas numpy scipy statsmodels matplotlib openpyxl
```

---

## Running it

Put `Public.xlsx` next to the scripts and run:

```bash
# the full statistical pipeline + Figures 1, 4, 6
python zilh_analysis.py --data Public.xlsx --outdir figures

# the three summary figures (2, 3, 5)
python zilh_figures_extra.py --outdir figures
```

If you leave `--data` off, `zilh_analysis.py` looks for `Public.xlsx` in the
current directory. `--outdir` defaults to `./figures`.

### What you get

In the output folder:

```
figures/
├── Figure1_cost_distribution.png    # zero/positive split, histogram, percentiles
├── Figure2_model_comparison.png     # AIC horse-race + two-stage architecture
├── Figure3_cost_by_platform.png     # cost by engine type and mass class
├── Figure4_coefficients.png         # Stage 1 odds ratios, Stage 2 multipliers (95% CI)
├── Figure5_premium_tail_risk.png    # route premium schedule + VaR95/TVaR99
├── Figure6_diagnostics.png          # Stage 1 calibration + Stage 2 Q-Q plot
└── premium_schedule.csv             # the Table 2 numbers, machine-readable
```

The console also prints the headline numbers as it goes (zero share, Gini,
Stage 1/2 AIC, combined AIC/BIC, holdout RMSE and improvement, Gamma shape,
and the full premium table).

---

## How the analysis works, step by step

This mirrors the manuscript's Methods (§3) and Results (§4) so you can line the
code up against the text.

### 1. Load and clean (`load_nwsd`, `split_costs`)
The raw export is read as-is and renamed to the canonical column set. The cost
field is coerced to numeric — anything blank or non-numeric becomes a
**structural zero** (no cost reported, which under NWSD voluntary reporting is
the overwhelming majority). Records with a positive cost below **$100** are
dropped as administrative placeholders (round-number filing minimums), leaving
the economically meaningful positive-cost observations.

### 2. Descriptive pass (`describe_costs`, `gini`)
Computes the numbers behind the abstract and §4.1: the zero share, mean vs
median (and their ratio), the P75/P90/P95/P99 percentiles, the maximum, and the
**Gini coefficient** of the positive costs (sorted-cumulative formulation).
Also reports the share of total cost sitting in the top decile of events —
the concentration figure quoted in the text.

### 3. Stage 1 — logistic hurdle (`build_design`, `fit_stage1`)
A logistic regression for **P(cost > 0)** on aircraft mass class, engine type,
phase of flight, and a centred year trend. Piston / mass class 1 / Approach are
the reference levels, so the odds ratios read directly against those baselines.
The Brier score and McFadden pseudo-R² are pulled off for the diagnostics.
Newton is the default optimiser; if a sparse dummy cell makes the Hessian
singular the fit falls back to BFGS automatically.

### 4. Stage 2 — lognormal severity (`fit_stage2`)
An OLS regression of **log(cost)** over the positive records only — this is the
lognormal severity stage, `ln(cost) ~ N(Xβ, σ²)`. The residual variance σ² is
kept because the combined expectation needs it.

### 5. Combined expected cost (`expected_cost`)
The two stages multiply back together as
`E[Y] = p · exp(μ + σ²/2)` — the hurdle probability times the lognormal mean on
the natural scale. This per-event expectation is what the premium schedule uses.

### 6. Model comparison (`model_comparison`)
An 80/20 stratified split: both stages are fit on the training 80% and scored on
the held-out 20%. Returns the ZIL-H combined AIC/BIC and the held-out RMSE on
the log scale, plus the lognormal-only RMSE the **7% improvement** is measured
against. Note the alternative single-component AICs (OLS, Tobit, Gamma) are on
their own likelihood scales — they're for indicative ranking only, which is why
the paper reports them as relative. There is deliberately **no ZIP+Gamma** here:
a Poisson count model on a continuous cost is a category error.

### 7. Gamma tail → VaR / TVaR (`calibrate_gamma_tail`, `var_tvar`, `tail_by_mass`)
The lognormal fits the bulk of the distribution well but rolls off too fast in
the extreme upper tail (visible in the Figure 6 Q-Q plot), so the tail-risk
metrics come from a **Gamma** calibrated to the same positive-cost vector
(shape φ ≈ 0.62). `var_tvar` returns Value-at-Risk (the α-quantile) and
Tail-VaR (the closed-form conditional mean above it) at any level; `tail_by_mass`
does this per mass class, falling back to the pooled shape when a class has too
few positive records to fit its own.

### 8. Premium schedule (`premium_schedule`, `delta_ci`)
For each route stratum (defined in §3.3) the pure premium is
`λ(r|k) · E[cost]`, where λ is the platform-conditional strike rate per flight
from NWSD frequency counts. This is loaded by 1.45 (30% expense + 15% profit)
and annualised over 365 flights. The E[cost] confidence intervals use a delta-
method relative standard error propagated from both stages. The result is
written to `premium_schedule.csv` (the Table 2 numbers).

### 9. Figures
- **Figure 1** — three panels: zero/positive split, positive-cost histogram,
  and key percentiles on a log axis.
- **Figure 4** — Stage 1 odds ratios and Stage 2 severity multipliers, each
  with 95% CIs read straight off the fitted covariance matrices.
- **Figure 6** — Stage 1 calibration (predicted vs. observed by decile on the
  holdout) and the Stage 2 Q-Q plot.
- **Figures 2, 3, 5** — the horse-race, cost-by-platform, and premium/tail
  figures, in the companion script.

---

## Reproducibility notes

- The 80/20 split is seeded (`seed=20240517` in `model_comparison`), so the
  reported RMSE is stable run to run. Change the seed if you want to check
  sensitivity to the split — or better, swap in k-fold, which is noted in the
  paper's limitations as the recommended next step.
- All figures render at **300 dpi** for print.
- Nothing here touches the network or writes outside `--outdir` (plus the one
  CSV). Re-running is safe and idempotent.
- The Gamma tail parameter φ is estimated by maximum likelihood on the positive
  sub-sample; the ~0.62 in the paper is what the current public export gives.
  A future NWSD release will shift it slightly — that's expected, and the
  console prints the fitted value each run.

---

## Contact

Questions about the pipeline or the column mapping can go to the corresponding
author of the manuscript.
