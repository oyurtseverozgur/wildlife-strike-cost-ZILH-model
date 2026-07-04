"""
zilh_figures_extra.py
---------------------
The three "results-summary" figures that read off aggregates rather than the
raw fit: the model horse-race (Fig 2), cost-by-platform (Fig 3), and the
premium / tail-risk bars (Fig 5).

These are split out from zilh_analysis.py on purpose - that file is the
statistical pipeline and I didn't want 300 lines of matplotlib buried in it.
Everything here takes summary numbers as input (either passed in or read from
the CSV the main script writes), so the two files can be run independently.

    python zilh_figures_extra.py --outdir figures

If you want these regenerated straight from the fitted model rather than from
the hard-coded summary values, import the functions and hand them the frames
from zilh_analysis.main() instead.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4,
    "grid.linestyle": ":", "figure.facecolor": "white", "axes.facecolor": "white",
})

BLUE, RED, GREEN, ORANGE = "#2D6A9F", "#C0392B", "#27AE60", "#E67E22"
LIGHT, MED = "#AEC6E8", "#8B6914"


def _plabel(ax, t):
    ax.text(0.02, 0.97, t, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# FIGURE 2 - model comparison + architecture sketch
# ---------------------------------------------------------------------------
def figure2(outdir, aic_values=None):
    """AIC bars on the left, the two-stage flow on the right.

    Note there is no ZIP+Gamma bar here: a Poisson count model on a continuous
    cost response is a category error, so it was dropped from the comparison.
    """
    if aic_values is None:
        # summary AICs; ZIL-H is the only one on the shared likelihood scale,
        # the rest are indicative (their own samples/scales)
        aic_values = {
            "OLS\n(single-stage)": 61840,
            "Tobit\n(censored)":   60920,
            "Gamma GLM\n(no zeros)": 60180,
            "Two-part\nLognormal": 58420,
            "ZIL-H\n(this study)": 58240,
        }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    names = list(aic_values)
    vals = list(aic_values.values())
    cols = [LIGHT, LIGHT, LIGHT, BLUE, RED]
    bars = axL.bar(range(len(names)), vals, color=cols, edgecolor="white", width=0.62)
    axL.set_ylim(min(vals) - 2200, max(vals) + 1300)
    axL.set_xticks(range(len(names)))
    axL.set_xticklabels(names, fontsize=9)
    axL.set_ylabel("AIC (lower = better)")
    axL.set_title("(a)  Model comparison by AIC\n(ZIL-H directly computed; others indicative)",
                  fontsize=10.5, pad=6)
    for b, v in zip(bars, vals):
        axL.text(b.get_x() + b.get_width() / 2, v + 80,
                 f"{v:,}", ha="center", va="bottom", fontsize=9)
    axL.annotate("delta AIC = -180\nvs two-part LN", xy=(4, vals[-1]),
                 xytext=(2.55, max(vals) - 400), fontsize=8.5, color=RED,
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axL.legend(handles=[mpatches.Patch(color=RED, label="ZIL-H (preferred)"),
                        mpatches.Patch(color=BLUE, label="Two-part Lognormal"),
                        mpatches.Patch(color=LIGHT, label="Alternative specs")],
               fontsize=9, loc="upper right")
    _plabel(axL, "(a)")

    # right panel: boxes + arrows, no data, just the flow
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.axis("off")
    axR.set_title("(b)  ZIL-H two-stage architecture", fontsize=10.5, pad=6)
    _plabel(axR, "(b)")

    def box(x, y, w, h, txt, fc, ec, fs=9.5):
        axR.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                      boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=1.8))
        axR.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                 fontsize=fs, fontweight="bold")

    box(1, 8, 8, 0.9, "NWSD full record  (n = 342,075)", "#F0F3F4", "#555")
    axR.annotate("", xy=(5, 7.4), xytext=(5, 7.05),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))
    box(0.5, 5.8, 4, 1.2,
        "Stage 1 - Logistic hurdle\nP(cost > 0)", "#D5E8D4", GREEN)
    box(5.2, 5.8, 4.3, 1.2,
        "Stage 2 - Lognormal severity\nln(cost | cost>0)", "#DAE8FC", BLUE)
    axR.text(5.0, 5.85, "x", ha="center", va="bottom", fontsize=13, color="#555")
    box(2.5, 3.8, 5, 1.1,
        "E[Y] = p * exp(mu + sigma^2/2)", "#FCE4D6", RED)
    axR.annotate("", xy=(5, 3.8), xytext=(5, 5.7),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))
    box(1, 1.8, 3.8, 1.5, "Actuarial premiums\nVaR95, TVaR99", "#FFF2CC", ORANGE)
    box(5.2, 1.8, 3.8, 1.5, "Gamma tail model\n(shape phi = 0.62)", "#E1D5E7", "#7B4F9E")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "Figure2_model_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3 - repair cost by engine type and mass class
# ---------------------------------------------------------------------------
def figure3(outdir, by_engine=None, by_mass=None):
    if by_engine is None:
        by_engine = {  # label: (mean, p95, n)
            "Turboshaft/Other": (15200, 12000, 120),
            "Piston":           (20282, 31400, 1830),
            "Turboprop":        (48760, 105000, 620),
            "Turbojet":         (651150, 1820000, 38),
            "Turbofan":         (363178, 1356000, 2781),
        }
    if by_mass is None:
        by_mass = {
            "Class 1\n<2,269 kg":      (20594, 28000, 1830),
            "Class 2\n2,269-5,670 kg": (45780, 89000, 640),
            "Class 3\n5,670-27,215 kg":(185320, 420000, 1100),
            "Class 4\n>27,215 kg":     (454171, 1530000, 1807),
        }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    def draw(ax, data, title, plabel):
        names = list(data)
        means = [data[k][0] for k in names]
        p95s = [data[k][1] for k in names]
        ns = [data[k][2] for k in names]
        cols = [LIGHT, LIGHT, BLUE, ORANGE, RED][:len(names)]
        y = np.arange(len(names))
        bars = ax.barh(y, means, color=cols, alpha=0.88, edgecolor="white",
                       height=0.55, label="Mean cost")
        ax.barh(y + 0.3, p95s, color=cols, alpha=0.35, edgecolor="white",
                height=0.22, label="P95 cost")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlim(0, max(means) * 1.55)
        ax.set_xlabel("Repair cost (USD)")
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"${x/1e3:.0f}K" if x < 1e6 else f"${x/1e6:.1f}M"))
        for b, m, n in zip(bars, means, ns):
            lab = f"${m/1e3:.0f}K  (n={n:,})" if m < 1e6 else f"${m/1e6:.1f}M  (n={n:,})"
            ax.text(b.get_width() + max(means) * 0.02, b.get_y() + b.get_height() / 2,
                    lab, ha="left", va="center", fontsize=8.5)
        ax.legend(fontsize=9, loc="lower right")
        _plabel(ax, plabel)

    draw(axL, by_engine,
         "(a)  Mean repair cost by engine type\n(positive-cost records only)", "(a)")
    draw(axR, by_mass,
         "(b)  Mean repair cost by aircraft mass class\n(positive-cost records only)", "(b)")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "Figure3_cost_by_platform.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 5 - premium schedule + tail risk
# ---------------------------------------------------------------------------
def figure5(outdir, routes=None, tail=None):
    if routes is None:
        # (label, E[cost]/flight)
        routes = [
            ("Light GA\nlocal", 16.48),
            ("Light GA\nregional", 48.2),
            ("Turboprop\nregional", 143.5),
            ("Narrowbody\nmedium-haul", 402.1),
            ("Narrowbody\nlong-haul", 798.4),
            ("Widebody\nlong-haul", 1862.0),
        ]
    if tail is None:
        # mass class: (VaR95, TVaR99)
        tail = {
            "Class 1\n<2,269 kg":       (37800, 68200),
            "Class 2\n2,269-5,670 kg":  (124300, 225100),
            "Class 3\n5,700-27,215 kg": (505000, 912000),
            "Class 4\n>27,215 kg":      (1154879, 1966195),
        }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.8))

    # (a) expected cost per flight + annual premium (twin axis)
    labels = [r[0] for r in routes]
    ecost = [r[1] for r in routes]
    premium_k = [e * 1.45 * 365 / 1000 for e in ecost]
    cols = [LIGHT, LIGHT, BLUE, BLUE, ORANGE, RED]
    x = np.arange(len(labels))
    w = 0.38
    b1 = axL.bar(x - w / 2, ecost, w, color=cols, alpha=0.88, edgecolor="white")
    axLr = axL.twinx()
    b2 = axLr.bar(x + w / 2, premium_k, w, color=cols, alpha=0.42,
                  edgecolor="white", hatch="//")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=8.5)
    axL.set_ylabel("Expected cost per flight (USD)")
    axLr.set_ylabel("Annual premium (USD thousands)", color=ORANGE)
    axLr.tick_params(axis="y", labelcolor=ORANGE)
    axL.set_ylim(0, max(ecost) * 1.3)
    axLr.set_ylim(0, max(premium_k) * 1.3)
    axL.set_title("(a)  ZIL-H expected cost per flight and\nannual premium by route type",
                  fontsize=10.5, pad=6)
    for b, v in zip(b1, ecost):
        lab = f"${v:.0f}" if v < 100 else f"${v:,.0f}"
        axL.text(b.get_x() + b.get_width() / 2, b.get_height() + max(ecost) * 0.015,
                 lab, ha="center", va="bottom", fontsize=7.8, rotation=35)
    axL.legend([b1, b2], ["Expected cost/flight", "Annual premium"],
               fontsize=9, loc="upper left")
    _plabel(axL, "(a)")

    # (b) VaR95 vs TVaR99 by mass class
    mnames = list(tail)
    var95 = [tail[k][0] / 1000 for k in mnames]
    tvar99 = [tail[k][1] / 1000 for k in mnames]
    x = np.arange(len(mnames))
    w = 0.38
    ba = axR.bar(x - w / 2, var95, w, color=BLUE, alpha=0.88,
                 edgecolor="white", label="VaR95")
    bb = axR.bar(x + w / 2, tvar99, w, color=RED, alpha=0.88,
                 edgecolor="white", label="TVaR99")
    axR.set_xticks(x)
    axR.set_xticklabels(mnames, fontsize=9)
    axR.set_ylabel("Cost per event (USD thousands)")
    axR.set_ylim(0, max(tvar99) * 1.2)
    axR.set_title("(b)  VaR95 and TVaR99 by aircraft mass class\n(Gamma tail model)",
                  fontsize=10.5, pad=6)
    for bars, vals, c in [(ba, var95, "black"), (bb, tvar99, RED)]:
        for b, v in zip(bars, vals):
            lab = f"${v:.0f}K" if v < 1000 else f"${v/1000:.2f}M"
            axR.text(b.get_x() + b.get_width() / 2, b.get_height() + max(tvar99) * 0.012,
                     lab, ha="center", va="bottom", fontsize=8.8, color=c)
    axR.legend(fontsize=10, loc="upper left")
    _plabel(axR, "(b)")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "Figure5_premium_tail_risk.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    figure2(out)
    figure3(out)
    figure5(out)
    print(f"Figures 2, 3, 5 written to {out.resolve()}")


if __name__ == "__main__":
    main()
