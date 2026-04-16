# ProbofCollision

A Python library for computing spacecraft probability of collision (Pc) in low Earth orbit. Implements synthetic conjunction generation, TCA finding, covariance modeling, and analytic Pc methods — built toward a survey of Pc algorithms.

## Setup

```bash
uv sync
```

Requires Python 3.12+. Dependencies: `brahe`, `numpy`, `scipy`.

## Project structure

```
src/collision/
    conjunction.py      — synthetic conjunction scenario generator
    tca.py              — Time of Closest Approach finder
    covariance.py       — RTN covariance builder, rotates to ECI
    fowler.py           — Fowler (1993) analytic Pc method

tests/
    conftest.py         — session-scoped fixtures (scenarios, TCA results, covariances)
    test_conjunction.py
    test_tca.py
    test_fowler.py

plots/
    plot_conjunctions.py        — 4-panel per-scenario: RTN trajectory, miss distance,
                                  encounter plane, RTN covariance cross-sections
    plot_3d_orbits.py           — 3D RTN relative motion for all scenarios
    plot_3d_eci.py              — full ECI orbit + zoomed TCA view per scenario
    plot_scenario_comparison.py — side-by-side scenario comparisons
    plot_pc_findings.py         — Pc sensitivity: covariance scale and miss distance

report/
    report.tex          — LaTeX report (source only; build to get PDF)
```

## Running the tests

```bash
uv run pytest tests/ -v
```

All 70 tests pass in ~35 seconds.

## Running the plots

All plot scripts are run from the repo root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src uv run python plots/plot_conjunctions.py
PYTHONPATH=src uv run python plots/plot_3d_orbits.py
PYTHONPATH=src uv run python plots/plot_3d_eci.py
PYTHONPATH=src uv run python plots/plot_scenario_comparison.py
PYTHONPATH=src uv run python plots/plot_pc_findings.py
```

Output PNGs are written to `plots/` (gitignored — regenerate as needed).

## Building the report

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # second pass resolves cross-references and TOC
```

Only `report.tex` is committed. The second `pdflatex` pass is required to resolve figure references and the table of contents. Generated files (`report.pdf`, `.aux`, `.log`, etc.) are gitignored.
