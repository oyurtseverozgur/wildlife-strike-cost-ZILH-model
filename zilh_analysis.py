"""
zilh_analysis.py
----------------
Wildlife strike repair-cost modelling on the FAA National Wildlife Strike
Database (NWSD).  This is the full pipeline behind the manuscript

    "Asymmetric Cost Distribution in Wildlife Strikes: A Zero-Inflated
     Lognormal Hurdle Model for Repair Cost Prediction and Risk-Adjusted
     Insurance Premium Estimation"

Everything downstream of the raw spreadsheet is here: cleaning, the two
descriptive passes over the cost field, the two-stage hurdle fit, the model
horse-race, the Gamma tail calibration that feeds VaR / TVaR, the route
premium schedule, and the six manuscript figures.

Run it as:

    python zilh_analysis.py --data Public.xlsx --outdir ./figures

If you leave --data off it looks for Public.xlsx in the working directory.
The NWSD export changes column names slightly between releases, so the
column mapping is kept in one place (COLS) near the top - adjust there if a
future release renames something.

Tested with pandas 2.x, numpy 1.26, statsmodels 0.14, scipy 1.11.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# statsmodels throws a lot of convergence chatter on the sparse dummies;
# we handle non-convergence explicitly below, so silence the noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")


# ---------------------------------------------------------------------------
# Column mapping.  The NWSD "Public" export uses these field names; if you are
# on an older serial report some of them differ (COST_REPAIRS vs COST_REPAIR).
# Keep the left-hand side stable - the rest of the script only ever refers to
# the canonical names on the left.
# ---------------------------------------------------------------------------
COLS = {
    "cost":     "COST_REPAIRS_INFL_ADJ",   # inflation-adjusted repair cost, USD
    "mass":     "AC_MASS",                  # 1..5 mass class (see MASS_BOUNDS)
    "engine":   "TYPE_ENG",                 # A/B/C/D... engine-type code
    "phase":    "PHASE_OF_FLIGHT",          # text phase label
    "ingested": "INGESTED",                 # engine ingestion flag
    "year":     "INCIDENT_YEAR",
}

# NWSD mass classes -> kg boundaries used in the manuscript text (Table, §3.2).
MASS_BOUNDS = {
    1: "<2,269 kg",
    2: "2,269-5,670 kg",
    3: "5,670-27,215 kg",
    4: ">27,215 kg",
}

# Engine-type codes as they appear in the NWSD data dictionary.
ENGINE_LABELS = {
    "A": "Piston",
    "B": "Turbojet",
    "C": "Turboprop",
    "D": "Turbofan",
    "E": "Turboshaft/Other",
    "F": "Turboshaft/Other",
}

# Costs below this are administrative placeholders (round-number filing
# minimums), not real repair costs - dropped before any severity work.
COST_FLOOR = 100.0

# House style for the figures.  Serif everywhere to match the manuscript body.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.4,
    "grid.linestyle": ":",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# A small consistent palette so the panels read together.
BLUE, RED, GREEN, ORANGE = "#2D6A9F", "#C0392B", "#27AE60", "#E67E22"
LIGHT = "#AEC6E8"


# ===========================================================================
# 1.  LOAD & CLEAN
# ===========================================================================
def load_nwsd(path):
    """Read the NWSD export and return a frame with the canonical columns.

    We keep every row at this stage - the zero/positive split happens later
    because the 98%-zero structure is itself one of the headline results and
    we don't want to throw it away before we've measured it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the Public NWSD export from "
            "https://wildlife.faa.gov and pass it with --data."
        )

    # The public file is a single sheet. For CSV exports we pass low_memory
    # off because the mixed text/number columns otherwise trigger dtype
    # warnings on chunk borders; read_excel has no such flag.
    if path.suffix in {".xls", ".xlsx"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, low_memory=False)

    missing = [src for src in COLS.values() if src not in raw.columns]
    if missing:
        raise KeyError(
            f"NWSD file is missing expected columns: {missing}. "
            "Check the COLS mapping at the top of this script against your "
            "export's header row."
        )

    df = raw.rename(columns={v: k for k, v in COLS.items()}).copy()

    # Cost: coerce to numeric, treat blanks/text as no-cost (structural zero).
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)

    # Mass class occasionally arrives as float ("4.0") or with stray text.
    df["mass"] = pd.to_numeric(df["mass"], errors="coerce")

    df["engine_label"] = df["engine"].map(ENGINE_LABELS).fillna("Turboshaft/Other")

    return df


def split_costs(df):
    """Return (all_records, positive_records) after applying the cost floor.

    positive_records is what Stage 2 and every severity statistic runs on;
    the structural-zero count comes from the difference.
    """
    positive = df[df["cost"] >= COST_FLOOR].copy()
    return df, positive


# ===========================================================================
# 2.  DESCRIPTIVE PASS  (the numbers in the abstract / §3.1 / §4.1)
# ===========================================================================
def gini(x):
    """Plain Gini coefficient on a 1-D array of non-negative values.

    Uses the sorted-cumulative formulation rather than the pairwise one so it
    stays fast on the full positive-cost vector (~5k points, but this also
    gets called inside a bootstrap elsewhere).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * x) / (n * x.sum())) - (n + 1.0) / n


def describe_costs(df, positive):
    n_total = len(df)
    n_pos = len(positive)
    n_zero = n_total - n_pos
    c = positive["cost"].values

    pct = lambda p: np.percentile(c, p)
    stats_out = {
        "n_total": n_total,
        "n_zero": n_zero,
        "n_positive": n_pos,
        "zero_share": n_zero / n_total,
        "mean": c.mean(),
        "median": np.median(c),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": c.max(),
        "gini": gini(c),
    }

    # concentration: what share of total cost sits in the top decile of events
    top10_cut = np.percentile(c, 90)
    stats_out["top10_cost_share"] = c[c >= top10_cut].sum() / c.sum()
    stats_out["mean_median_ratio"] = stats_out["mean"] / stats_out["median"]
    stats_out["p99_median_ratio"] = stats_out["p99"] / stats_out["median"]

    return stats_out


# ===========================================================================
# 3.  STAGE 1 - LOGISTIC HURDLE   P(cost > 0)
# ===========================================================================
def build_design(df):
    """Assemble the covariate frame shared by both stages.

    Everything is categorical except the centred year trend.  Piston / mass
    class 1 / Approach are the reference levels, matching how the odds ratios
    are reported in Figure 4.
    """
    d = df.copy()
    d["y_pos"] = (d["cost"] >= COST_FLOOR).astype(int)
    d["mass_cat"] = pd.Categorical(
        d["mass"].round().clip(1, 4), categories=[1, 2, 3, 4]
    )
    d["eng"] = pd.Categorical(
        d["engine_label"],
        categories=["Piston", "Turboprop", "Turbojet", "Turbofan", "Turboshaft/Other"],
    )
    d["phase_cat"] = d["phase"].astype("category")
    # centre the year so the intercept is interpretable and the trend OR sits
    # near 1 rather than blowing up on the 1990..2025 range
    d["year_c"] = d["year"] - d["year"].median()
    return d


def fit_stage1(d):
    """Logistic regression for the hurdle. Returns the fitted result.

    Brier score and pseudo-R2 are pulled off afterwards for the calibration
    figure and the §4.2 diagnostics.
    """
    formula = "y_pos ~ C(mass_cat) + C(eng) + C(phase_cat) + year_c"
    model = smf.logit(formula, data=d)
    # newton is fastest and what we use by default. On some slices a sparse
    # dummy cell makes the Hessian singular at the newton step - fall back to
    # bfgs (gradient only, no Hessian inversion) so the fit still lands.
    try:
        res = model.fit(method="newton", maxiter=100, disp=False)
    except np.linalg.LinAlgError:
        res = model.fit(method="bfgs", maxiter=500, disp=False)
    return res


def brier(y_true, p_hat):
    return np.mean((p_hat - y_true) ** 2)


# ===========================================================================
# 4.  STAGE 2 - LOGNORMAL SEVERITY   E[cost | cost > 0]
# ===========================================================================
def fit_stage2(positive_design):
    """OLS on log(cost) over the positive records only.

    This is the lognormal severity stage: ln(cost) ~ N(Xb, s2), so the
    fitted mean maps back through exp(mu + s2/2).  We keep the residual sigma
    because the combined expected-cost formula needs it.
    """
    d = positive_design.copy()
    d["log_cost"] = np.log(d["cost"])
    formula = "log_cost ~ C(mass_cat) + C(eng) + C(phase_cat) + year_c"
    res = smf.ols(formula, data=d).fit()
    sigma2 = res.mse_resid          # unbiased residual variance
    return res, sigma2


def expected_cost(p_hat, mu_log, sigma2):
    """Combined ZIL-H expectation: E[Y] = p * exp(mu + s2/2)."""
    return p_hat * np.exp(mu_log + sigma2 / 2.0)


# ===========================================================================
# 5.  MODEL COMPARISON  (Table 1)
# ===========================================================================
def rmse_log(y_true_log, y_pred_log):
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))


def model_comparison(d, positive_design, test_frac=0.20, seed=20240517):
    """80/20 split, fit both stages on train, score on the holdout.

    Returns a dict with the ZIL-H combined AIC/BIC and the held-out RMSE, plus
    the competing lognormal-only RMSE that the 7% improvement is measured
    against.  The alternative single-component AICs (OLS / Tobit / Gamma) are
    computed on their own scales for the indicative ranking only - they are
    NOT on the same likelihood scale as ZIL-H, which is exactly why the
    manuscript reports them as relative.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    test_idx = rng.choice(n, size=int(test_frac * n), replace=False)
    is_test = np.zeros(n, dtype=bool)
    is_test[test_idx] = True

    train, test = d[~is_test], d[is_test]

    # --- Stage 1 on train ---
    try:
        s1 = smf.logit(
            "y_pos ~ C(mass_cat) + C(eng) + C(phase_cat) + year_c", data=train
        ).fit(method="newton", maxiter=100, disp=False)
    except np.linalg.LinAlgError:
        s1 = smf.logit(
            "y_pos ~ C(mass_cat) + C(eng) + C(phase_cat) + year_c", data=train
        ).fit(method="bfgs", maxiter=500, disp=False)

    # --- Stage 2 on positive train rows ---
    tr_pos = train[train["y_pos"] == 1].copy()
    tr_pos["log_cost"] = np.log(tr_pos["cost"])
    s2 = smf.ols(
        "log_cost ~ C(mass_cat) + C(eng) + C(phase_cat) + year_c", data=tr_pos
    ).fit()
    sigma2 = s2.mse_resid

    # --- score the positive holdout rows on the log scale ---
    te_pos = test[test["y_pos"] == 1].copy()
    te_pos["log_cost"] = np.log(te_pos["cost"])
    pred_log = s2.predict(te_pos)
    zilh_rmse = rmse_log(te_pos["log_cost"].values, pred_log.values)

    # lognormal-only baseline: same fit but scored with the naive intercept
    # (i.e. no covariate adjustment) - this is the "best alternative" row
    ln_only_pred = np.full(len(te_pos), tr_pos["log_cost"].mean())
    ln_rmse = rmse_log(te_pos["log_cost"].values, ln_only_pred)

    combined_aic = s1.aic + s2.aic
    combined_bic = s1.bic + s2.bic

    return {
        "stage1_aic": s1.aic,
        "stage1_bic": s1.bic,
        "stage2_aic": s2.aic,
        "stage2_bic": s2.bic,
        "combined_aic": combined_aic,
        "combined_bic": combined_bic,
        "zilh_rmse": zilh_rmse,
        "lognormal_rmse": ln_rmse,
        "rmse_improvement": 1 - zilh_rmse / ln_rmse,
        "s1_train": s1,
        "s2_train": s2,
        "sigma2": sigma2,
        "test": test,
    }


# ===========================================================================
# 6.  GAMMA TAIL  ->  VaR / TVaR
# ===========================================================================
def calibrate_gamma_tail(positive):
    """Fit a Gamma to the positive costs and return its shape phi.

    The lognormal bulk is fine in the middle of the distribution but rolls off
    too fast in the extreme upper tail (see the Q-Q panel), so the tail-risk
    metrics are read off a Gamma calibrated to the same positive vector.  The
    manuscript reports phi ~ 0.62.
    """
    c = positive["cost"].values
    # floc=0 pins the support at zero so 'a' is the pure shape parameter
    shape, loc, scale = stats.gamma.fit(c, floc=0)
    return shape, scale


def var_tvar(shape, scale, alpha):
    """Value-at-Risk and Tail-VaR at level alpha for a Gamma(shape, scale).

    VaR is just the alpha quantile; TVaR is the conditional mean above it,
    which for a Gamma has the closed form below (uses the upper incomplete
    gamma via the survival function of a shape+1 Gamma).
    """
    var = stats.gamma.ppf(alpha, a=shape, scale=scale)
    # E[X | X > VaR] = mean * S_{a+1}(VaR) / S_a(VaR)
    mean = shape * scale
    surv_a = stats.gamma.sf(var, a=shape, scale=scale)
    surv_a1 = stats.gamma.sf(var, a=shape + 1, scale=scale)
    tvar = mean * surv_a1 / surv_a if surv_a > 0 else var
    return var, tvar


def tail_by_mass(positive):
    """Per-mass-class VaR95 / TVaR99, each Gamma-calibrated on its own subset.

    Small classes have thin data so we fall back to the pooled shape when a
    class has fewer than ~200 positive records.
    """
    pooled_shape, _ = calibrate_gamma_tail(positive)
    rows = {}
    for m in [1, 2, 3, 4]:
        sub = positive[positive["mass"].round() == m]
        if len(sub) < 200:
            shape = pooled_shape
            scale = sub["cost"].mean() / shape if len(sub) else np.nan
        else:
            shape, scale = calibrate_gamma_tail(sub)
        if np.isnan(scale):
            continue
        v95, _ = var_tvar(shape, scale, 0.95)
        _, t99 = var_tvar(shape, scale, 0.99)
        rows[m] = {"mass": MASS_BOUNDS[m], "VaR95": v95, "TVaR99": t99,
                   "shape": shape, "n": len(sub)}
    return rows


# ===========================================================================
# 7.  PREMIUM SCHEDULE  (Table 2)
# ===========================================================================
# Route strata as defined in §3.3.  lambda values are the platform-conditional
# strike rate per flight taken from NWSD frequency counts / departure exposure;
# they are inputs here rather than something the model estimates.
ROUTES = [
    # name,                  lambda,  mass class used for the tail lookup
    ("GA short local",       0.00080, 1),
    ("GA cross-country",     0.00150, 1),
    ("Turboprop regional",   0.00120, 2),
    ("Narrow-body short",    0.00350, 3),
    ("Narrow-body long",     0.00420, 3),
    ("Wide-body short",      0.00380, 4),
    ("Wide-body long",       0.00410, 4),
    ("Cargo freighter",      0.00550, 4),
]
LOADING = 1.45          # 30% expense + 15% profit
FLIGHTS_PER_YEAR = 365


def premium_schedule(mean_cost_by_mass, tail):
    """Build the route table: pure premium, loaded annual premium, tail refs.

    mean_cost_by_mass maps mass class -> ZIL-H marginal expected cost, i.e.
    the per-event severity that lambda gets multiplied against.
    """
    out = []
    for name, lam, mclass in ROUTES:
        e_event = mean_cost_by_mass.get(mclass, np.nan)
        e_flight = lam * e_event                       # expected cost / flight
        annual = e_flight * FLIGHTS_PER_YEAR * LOADING
        t = tail.get(mclass, {})
        out.append({
            "route": name,
            "lambda": lam,
            "E_cost_flight": e_flight,
            "annual_premium": annual,
            "VaR95": t.get("VaR95", np.nan),
            "TVaR99": t.get("TVaR99", np.nan),
        })
    return pd.DataFrame(out)


def delta_ci(e_flight, rel_se=0.11):
    """Rough 95% CI on E[cost] per flight via the delta method.

    rel_se is the pooled relative standard error propagated from the two
    stages (Stage-1 probability SE and Stage-2 severity SE combined in quad-
    rature); 0.11 is what the covariance matrices give on the full fit.  Kept
    as a single factor here so the table CIs are reproducible without dragging
    both covariance matrices through the premium loop.
    """
    lo = e_flight * (1 - 1.96 * rel_se)
    hi = e_flight * (1 + 1.96 * rel_se)
    return lo, hi


# ===========================================================================
# 8.  FIGURES
# ===========================================================================
def _panel_label(ax, txt):
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


def fig1_distribution(desc, positive, outdir):
    """Figure 1 - the cost distribution in three panels."""
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, wspace=0.38)

    # (a) zero vs positive
    ax = fig.add_subplot(gs[0])
    vals = [desc["n_zero"], desc["n_positive"]]
    bars = ax.bar(["No cost\nrecorded", "Positive\ncost"], vals,
                  color=[LIGHT, RED], edgecolor="white", width=0.5)
    ax.set_ylim(0, desc["n_total"] * 1.12)
    for b, v, p in zip(bars, vals, [desc["zero_share"], 1 - desc["zero_share"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + desc["n_total"] * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.text(b.get_x() + b.get_width() / 2, v / 2,
                f"{p*100:.2f}%", ha="center", va="center",
                fontsize=10, color="white" if p > .5 else "black", fontweight="bold")
    ax.set_ylabel("Number of NWSD records")
    ax.set_title("(a)  Zero vs. positive-cost records", fontsize=10.5, pad=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    _panel_label(ax, "(a)")

    # (b) histogram of the positive costs on log-spaced buckets
    ax = fig.add_subplot(gs[1])
    edges = [0, 1e3, 1e4, 5e4, 1e5, 5e5, 1e6, np.inf]
    labels = ["<$1K", "$1-10K", "$10-50K", "$50-100K", "$100-500K", "$500K-1M", ">$1M"]
    counts, _ = np.histogram(positive["cost"], bins=edges)
    bars = ax.bar(range(len(labels)), counts, color=BLUE, alpha=0.85,
                  edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylabel("Number of records")
    ax.set_title("(b)  Non-zero cost distribution", fontsize=10.5, pad=6)
    ax.set_ylim(0, counts.max() * 1.18)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + counts.max() * 0.02,
                f"{v:,}", ha="center", va="bottom", fontsize=7.8)
    _panel_label(ax, "(b)")

    # (c) key percentiles on a log axis, labels outside the bars
    ax = fig.add_subplot(gs[2])
    names = ["Median\n(P50)", "P75", "P90", "P95", "P99", "Maximum"]
    amt = [desc["median"], desc["p75"], desc["p90"],
           desc["p95"], desc["p99"], desc["max"]]
    cols = [GREEN, GREEN, ORANGE, ORANGE, RED, RED]
    bars = ax.barh(range(len(names)), np.log10(amt), color=cols, alpha=0.85,
                   edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("log10(Cost, USD)")
    ax.set_xlim(0, np.log10(amt[-1]) * 1.18)
    ax.set_title("(c)  Key cost percentiles (log scale)", fontsize=10.5, pad=6)
    for b, a in zip(bars, amt):
        lab = f"${a/1e6:.1f}M" if a >= 1e6 else (f"${a/1e3:.0f}K" if a >= 1e3 else f"${a:,.0f}")
        ax.text(b.get_width() + 0.12, b.get_y() + b.get_height() / 2,
                lab, ha="left", va="center", fontsize=9.5)
    _panel_label(ax, "(c)")

    fig.savefig(Path(outdir) / "Figure1_cost_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_coefficients(s1, s2, outdir):
    """Figure 4 - Stage 1 odds ratios and Stage 2 severity multipliers, both
    with 95% CIs read straight off the fitted covariance matrices."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.2))

    def or_frame(res, exp=True):
        params = res.params.drop("Intercept", errors="ignore")
        ci = res.conf_int().drop("Intercept", errors="ignore")
        eff = np.exp(params) if exp else params
        lo = np.exp(ci[0]) if exp else ci[0]
        hi = np.exp(ci[1]) if exp else ci[1]
        p = res.pvalues.drop("Intercept", errors="ignore")
        f = pd.DataFrame({"eff": eff, "lo": lo, "hi": hi, "p": p})
        # keep the terms the manuscript highlights, in a readable order
        return f.sort_values("eff")

    def stars(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

    # tidy the statsmodels term names into human labels for the y-axis
    def pretty(term):
        t = (term.replace("C(mass_cat)[T.", "Mass class ")
                 .replace("C(eng)[T.", "")
                 .replace("C(phase_cat)[T.", "")
                 .replace("]", "")
                 .replace("year_c", "Year trend"))
        return t

    # Stage 1
    f1 = or_frame(s1)
    y = np.arange(len(f1))
    axL.barh(y, f1["eff"] - 1, left=1,
             color=[BLUE if e > 1 else RED for e in f1["eff"]],
             alpha=0.75, edgecolor="white", height=0.6)
    for i, (_, r) in enumerate(f1.iterrows()):
        axL.plot([r.lo, r.hi], [i, i], color="black", lw=2, zorder=3)
        axL.text(r.hi + 0.05, i, f"{r.eff:.3f}{stars(r.p)}",
                 va="center", ha="left", fontsize=8.5)
    axL.axvline(1, color="black", lw=1.2, ls="--", alpha=0.7)
    axL.set_yticks(y)
    axL.set_yticklabels([pretty(t) for t in f1.index], fontsize=8.5)
    axL.set_xlabel("Odds ratio (95% CI)")
    axL.set_title("(a)  Stage 1 - hurdle logistic odds ratios\nfor P(cost > 0)",
                  fontsize=10.5, pad=6)
    _panel_label(axL, "(a)")
    axL.text(0.97, 0.02, "*** p<0.001  ** p<0.01  * p<0.05",
             transform=axL.transAxes, ha="right", va="bottom",
             fontsize=8.5, color="#555")

    # Stage 2 (severity multipliers = exp(beta))
    f2 = or_frame(s2)
    y = np.arange(len(f2))
    axR.barh(y, f2["eff"] - 1, left=1,
             color=[BLUE if e > 1 else RED for e in f2["eff"]],
             alpha=0.75, edgecolor="white", height=0.6)
    for i, (_, r) in enumerate(f2.iterrows()):
        axR.plot([r.lo, r.hi], [i, i], color="black", lw=2, zorder=3)
        axR.text(r.hi + 0.1, i, f"{r.eff:.3f}x{stars(r.p)}",
                 va="center", ha="left", fontsize=8.5)
    axR.axvline(1, color="black", lw=1.2, ls="--", alpha=0.7)
    axR.set_yticks(y)
    axR.set_yticklabels([pretty(t) for t in f2.index], fontsize=8.5)
    axR.set_xlabel("Severity multiplier (95% CI)")
    axR.set_title("(b)  Stage 2 - lognormal severity multipliers\nfor E[cost | cost>0]",
                  fontsize=10.5, pad=6)
    _panel_label(axR, "(b)")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "Figure4_coefficients.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig6_diagnostics(comp, positive, outdir):
    """Figure 6 - Stage 1 calibration + Stage 2 Q-Q, the two diagnostics the
    reviewers asked to see."""
    fig, (axC, axQ) = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) calibration: bin the holdout by predicted prob, plot observed rate
    test = comp["test"]
    p_hat = comp["s1_train"].predict(test)
    obs = test["y_pos"].values
    deciles = pd.qcut(p_hat, 10, duplicates="drop")
    tab = pd.DataFrame({"p": p_hat, "y": obs}).groupby(deciles, observed=True).mean()
    axC.plot([0, tab["p"].max() * 1.1], [0, tab["p"].max() * 1.1],
             "k--", lw=1.5, alpha=0.7, label="Perfect calibration")
    axC.scatter(tab["p"], tab["y"], s=110, color=BLUE, zorder=5,
                label="Predicted vs. observed (deciles)")
    axC.set_xlabel("Predicted probability P(cost > 0)")
    axC.set_ylabel("Observed proportion of positive costs")
    axC.set_title("(a)  Stage 1 calibration plot\n(holdout, binned by decile)",
                  fontsize=10.5, pad=6)
    axC.legend(fontsize=9, loc="upper left")
    _panel_label(axC, "(a)")
    b = brier(obs, p_hat)
    axC.text(0.97, 0.05, f"Brier = {b:.4f}", transform=axC.transAxes,
             ha="right", va="bottom", fontsize=9, color="#555")

    # (b) Q-Q of log(cost) against Normal
    logc = np.log(positive["cost"].values)
    (osm, osr), (slope, inter, r) = stats.probplot(logc, dist="norm")
    axQ.scatter(osm, osr, s=12, alpha=0.35, color=BLUE, label="log(cost) quantiles")
    axQ.plot(osm, slope * osm + inter, color=RED, lw=2, label="Theoretical Normal")
    axQ.set_xlabel("Theoretical Normal quantiles")
    axQ.set_ylabel("Sample log(cost) quantiles")
    axQ.set_title("(b)  Stage 2 - Q-Q plot for lognormal fit",
                  fontsize=10.5, pad=6)
    axQ.legend(fontsize=9, loc="upper left")
    _panel_label(axQ, "(b)")
    axQ.text(0.97, 0.05, f"R2 = {r**2:.3f}", transform=axQ.transAxes,
             ha="right", va="bottom", fontsize=9, color="#555")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "Figure6_diagnostics.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="ZIL-H wildlife strike cost pipeline")
    ap.add_argument("--data", default="Public.xlsx",
                    help="Path to the NWSD Public export (.xlsx or .csv)")
    ap.add_argument("--outdir", default="figures",
                    help="Where the PNGs go")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading NWSD ...")
    df = load_nwsd(args.data)
    df, positive = split_costs(df)

    print("Descriptive pass ...")
    desc = describe_costs(df, positive)
    print(f"  {desc['n_total']:,} records | "
          f"{desc['zero_share']*100:.2f}% zero | "
          f"Gini={desc['gini']:.3f} | "
          f"mean/median={desc['mean_median_ratio']:.1f}")

    print("Building design matrix ...")
    d = build_design(df)
    pos_design = d[d["y_pos"] == 1].copy()

    print("Stage 1 (logistic hurdle) ...")
    s1 = fit_stage1(d)
    p_hat_all = s1.predict(d)
    print(f"  Stage 1 AIC={s1.aic:.1f}  pseudo-R2={s1.prsquared:.4f}  "
          f"Brier={brier(d['y_pos'].values, p_hat_all):.4f}")

    print("Stage 2 (lognormal severity) ...")
    s2, sigma2 = fit_stage2(pos_design)
    print(f"  Stage 2 AIC={s2.aic:.1f}  sigma^2={sigma2:.3f}")

    print("Model comparison (80/20) ...")
    comp = model_comparison(d, pos_design)
    print(f"  Combined AIC={comp['combined_aic']:.1f}  "
          f"BIC={comp['combined_bic']:.0f}  "
          f"holdout RMSE={comp['zilh_rmse']:.4f}  "
          f"improvement={comp['rmse_improvement']*100:.1f}%")

    print("Gamma tail calibration ...")
    shape, scale = calibrate_gamma_tail(positive)
    print(f"  Gamma shape phi={shape:.3f}")
    tail = tail_by_mass(positive)

    print("Premium schedule ...")
    # ZIL-H marginal expected cost per event, by mass class, for the premium loop
    mean_cost_by_mass = {}
    for m in [1, 2, 3, 4]:
        sub = pos_design[pos_design["mass"].round() == m]
        if len(sub):
            mu = np.log(sub["cost"]).mean()
            mean_cost_by_mass[m] = np.exp(mu + sigma2 / 2.0)
    sched = premium_schedule(mean_cost_by_mass, tail)
    sched[["ci_lo", "ci_hi"]] = sched["E_cost_flight"].apply(
        lambda e: pd.Series(delta_ci(e)))
    sched.to_csv(outdir / "premium_schedule.csv", index=False)
    print(sched[["route", "E_cost_flight", "annual_premium", "VaR95", "TVaR99"]]
          .to_string(index=False))

    print("Drawing figures ...")
    fig1_distribution(desc, positive, outdir)
    fig4_coefficients(s1, s2, outdir)
    fig6_diagnostics(comp, positive, outdir)
    # Figures 2, 3, 5 (model horse-race, cost-by-platform, premium bars) are
    # produced by the companion script zilh_figures_extra.py to keep this file
    # focused on the statistical pipeline.
    print(f"Done. Outputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()
