# Findings: Probability of Collision — Key Trends

Empirical observations confirmed by plots and tests.
Regenerate figures: `PYTHONPATH=src uv run python plots/plot_pc_findings.py`

---

## 1. Pc vs. Covariance Size Is NOT Always Monotone

**Plot:** `pc_findings.png`, left panel.

The orange curve (high-Pc crossing, 204 m miss) rises steeply as covariance
grows from 0.01× to ~3× default, then falls again — a clear peak.  The blue
(head-on) and green (near-miss) curves are monotone-decreasing across the whole
range.

**Why:**

The Fowler method integrates a 2D Gaussian over the hard-body disk.  Two competing
effects determine whether growing the covariance raises or lowers Pc:

| Regime | Condition | Effect of growing covariance |
|--------|-----------|------------------------------|
| Rising | miss >> projected 2D sigma | Gaussian tail barely reaches disk — growing spreads it toward the disk → **Pc rises** |
| Falling | miss << projected 2D sigma | Gaussian nearly flat over disk — growing dilutes it → **Pc falls** |

Whether a scenario is in the rising or falling regime depends on where the miss
distance sits relative to the projected 2D sigma — not just the absolute size of
either.

**Observed for the high-Pc crossing (204 m miss, 500 m/s, HBR = 10 m):**

| Covariance level | pos σ (R/T/N) | Pc |
|------------------|---------------|----|
| tight (0.1× default) | 10/50/5 m | ~1e-160 (miss is 200+ σ away) |
| default | 100/500/50 m | ~3e-5 |
| loose (3× default) | 300/1500/150 m | ~1e-4 ← **peak** |
| very loose (10×) | 1000/5000/500 m | ~1e-5 |

**Practical implication:** receiving a tighter covariance from a new observation
can cause Pc to go *up* if you were previously in the rising regime.  This is
physically correct — better tracking reveals you were worse off than you thought.

---

## 2. Encounter Geometry Determines the Projected Covariance Width

**Plot:** `pc_findings.png`, right panel.

At the same miss distance and same v_rel (500 m/s), crossing conjunctions have
much lower Pc than head-on across the entire miss-distance range.  The gap is
roughly 3–6 orders of magnitude.

**Why:**

The encounter plane is perpendicular to `v_rel`.  For a **head-on** encounter,
`v_rel` is roughly along-track, so the encounter plane is the R-N plane — it
avoids the large along-track uncertainty.  For a **crossing** encounter, `v_rel`
is roughly cross-track, so the encounter plane is the R-T plane — and the full
500 m along-track sigma projects into it, creating a much wider 2D Gaussian.

Same miss distance, same speed, same covariance, but very different Pc — purely
because of which direction the two spacecraft approach from.

**Key takeaway for test design:** do not use a slow or wide-covariance crossing
scenario to produce "interesting" Pc values.  Use head-on or a near-miss instead.

---

## 3. Relative Speed Has Almost No Effect on Pc for Head-On Encounters

**Plot:** `pc_findings.png`, middle panel (three lines nearly on top of each other).

At a fixed 200 m miss distance, varying v_rel from 50 to 500 m/s for head-on
conjunctions changes Pc by less than 1%:

| v_rel | 2D projected sigmas | Pc (HBR = 10 m) |
|-------|--------------------|--------------------|
| 50 m/s | (127 m, 707 m) | 5.35e-4 |
| 200 m/s | (127 m, 707 m) | 5.36e-4 |
| 500 m/s | (127 m, 707 m) | 5.34e-4 |

The projected 2D covariance is nearly identical across all speeds.  For head-on
encounters the direction of `v_rel` is always roughly along-track regardless of
speed, so the same axes of the covariance project into the encounter plane.  Speed
magnitude does not directly enter the Fowler calculation — only the *direction* of
`v_rel` matters for determining the encounter plane.
