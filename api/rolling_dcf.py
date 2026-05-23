"""
Rolling DCF Valuation Model — Damodaran Methodology
======================================================
Produces year-end target prices for 2026, 2027, 2028 using:
  - Actual data  : 2021–2025
  - Consensus    : 2026–2028  (Stockanalysis / manual input)
  - Extrapolated : 2029–2038  (decay + margin convergence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

FirmType = Literal["mature", "high_growth", "unprofitable"]


@dataclass
class Financials2025:
    """Snapshot of 2025 actual / preliminary actuals used for classification."""
    ebit_margin: float        # EBIT / Revenue  (e.g. 0.31)
    revenue_growth: float     # YoY  (e.g. 0.06 for 6 %)
    revenue: float            # $B
    ebit: float               # $B
    da: float                 # Depreciation & Amortisation  $B
    capex: float              # Capital Expenditure  $B (positive)
    delta_nwc: float          # Increase in Net Working Capital  $B
    cash: float               # Cash & equivalents  $B
    debt: float               # Total debt  $B
    shares: float             # Diluted shares outstanding  B shares


@dataclass
class ConsensusYear:
    """Single-year consensus estimate."""
    year: int
    revenue: float            # $B
    ebit_margin: float        # fraction
    da: float                 # $B
    capex: float              # $B
    delta_nwc: float          # $B


@dataclass
class DCFParams:
    """Global parameters for the DCF engine."""
    tax_rate: float = 0.21
    risk_free_rate: float = 0.044          # US 10-Y Treasury
    erp: float = 0.055                     # Equity Risk Premium
    beta: float = 1.0
    terminal_growth: float = 0.025         # = risk_free_rate long-run proxy
    industry_ebit_margin: float = 0.25     # Convergence target
    da_pct_revenue: float = 0.04           # D&A as % rev in extrapolation
    capex_pct_revenue: float = 0.05        # CapEx as % rev in extrapolation
    nwc_pct_revenue: float = 0.02          # NWC as % rev (delta = growth × NWC)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FIRM CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_firm(f: Financials2025) -> tuple[FirmType, int]:
    """
    Classify the firm and return (firm_type, projection_horizon).

    Rules (Damodaran):
      - Mature      : EBIT margin > 0  AND  revenue growth <= 10 %  → 5-year
      - High Growth  : EBIT margin > 0  AND  revenue growth >  10 %  → 10-year
      - Unprofitable : EBIT margin <= 0                               → 10-year
    """
    if f.ebit_margin <= 0:
        return "unprofitable", 10
    if f.revenue_growth > 0.10:
        return "high_growth", 10
    return "mature", 5


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FCF EXTRAPOLATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _decay_growth(
    base_growth: float,
    terminal_growth: float,
    step: int,
    horizon: int,
) -> float:
    """
    Linear decay from base_growth → terminal_growth over [1 .. horizon] steps.
    step = 1 is the first extrapolated year beyond the consensus window.
    """
    if horizon <= 1:
        return terminal_growth
    alpha = (step - 1) / (horizon - 1)          # 0 → 1
    return base_growth * (1 - alpha) + terminal_growth * alpha


def _converge_margin(
    base_margin: float,
    target_margin: float,
    step: int,
    horizon: int,
) -> float:
    """Linear convergence of EBIT margin over the horizon."""
    if horizon <= 1:
        return target_margin
    alpha = (step - 1) / (horizon - 1)
    return base_margin * (1 - alpha) + target_margin * alpha


def extrapolate_fcfs(
    consensus: list[ConsensusYear],
    params: DCFParams,
    firm_type: FirmType,
    horizon: int,
    extrapolate_from_year: int,
    extrapolate_to_year: int,
) -> pd.DataFrame:
    """
    Generate FCF rows for years [extrapolate_from_year .. extrapolate_to_year].

    FCF = EBIT(1-t) + D&A - CapEx - dNWC    (Damodaran FCFF definition)

    Growth decay  : last consensus rev growth → risk_free_rate
    Margin conv.  : last consensus EBIT margin → industry_ebit_margin
    D&A / CapEx / NWC: held at params percentages of revenue
    """
    # ── seed values from last consensus year ──────────────────────────────
    last = max(consensus, key=lambda c: c.year)
    # compute implied revenue growth from last two consensus points
    second_last = sorted(consensus, key=lambda c: c.year)[-2] if len(consensus) >= 2 else None
    if second_last:
        seed_growth = (last.revenue / second_last.revenue) - 1
    else:
        seed_growth = params.risk_free_rate + 0.02

    seed_margin = last.ebit_margin
    seed_rev    = last.revenue

    # ── extrapolation horizon relative to consensus end ───────────────────
    n_steps = extrapolate_to_year - extrapolate_from_year + 1

    rows: list[dict] = []
    prev_rev = seed_rev

    for step, yr in enumerate(range(extrapolate_from_year, extrapolate_to_year + 1), start=1):
        g = _decay_growth(seed_growth, params.terminal_growth, step, n_steps)
        m = _converge_margin(seed_margin, params.industry_ebit_margin, step, n_steps)

        rev    = prev_rev * (1 + g)
        ebit   = rev * m
        nopat  = ebit * (1 - params.tax_rate)
        da     = rev * params.da_pct_revenue
        capex  = rev * params.capex_pct_revenue
        # delta NWC = change in (NWC_pct × revenue)
        d_nwc  = (rev - prev_rev) * params.nwc_pct_revenue
        fcf    = nopat + da - capex - d_nwc

        rows.append({
            "year":        yr,
            "revenue":     round(rev, 3),
            "rev_growth":  round(g * 100, 2),
            "ebit_margin": round(m * 100, 2),
            "ebit":        round(ebit, 3),
            "nopat":       round(nopat, 3),
            "da":          round(da, 3),
            "capex":       round(capex, 3),
            "delta_nwc":   round(d_nwc, 3),
            "fcf":         round(fcf, 3),
            "source":      "extrapolated",
        })
        prev_rev = rev

    return pd.DataFrame(rows).set_index("year")


def consensus_to_fcf(
    consensus: list[ConsensusYear],
    params: DCFParams,
) -> pd.DataFrame:
    """Convert consensus estimates to FCF rows (same schema as extrapolated)."""
    rows: list[dict] = []
    rev_prev: float | None = None

    for c in sorted(consensus, key=lambda x: x.year):
        nopat = c.ebit_margin * c.revenue * (1 - params.tax_rate)
        d_nwc = ((c.revenue - rev_prev) * params.nwc_pct_revenue
                 if rev_prev is not None else c.delta_nwc)
        fcf   = nopat + c.da - c.capex - d_nwc
        g     = (c.revenue / rev_prev - 1) if rev_prev else float("nan")

        rows.append({
            "year":        c.year,
            "revenue":     round(c.revenue, 3),
            "rev_growth":  round(g * 100, 2) if not math.isnan(g) else None,
            "ebit_margin": round(c.ebit_margin * 100, 2),
            "ebit":        round(c.ebit_margin * c.revenue, 3),
            "nopat":       round(nopat, 3),
            "da":          round(c.da, 3),
            "capex":       round(c.capex, 3),
            "delta_nwc":   round(d_nwc, 3),
            "fcf":         round(fcf, 3),
            "source":      "consensus",
        })
        rev_prev = c.revenue

    return pd.DataFrame(rows).set_index("year")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROLLING DCF CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _wacc(params: DCFParams) -> float:
    return params.risk_free_rate + params.beta * params.erp


def _pv_stream(
    fcf_series: pd.Series,
    wacc: float,
    base_year: int,
) -> float:
    """
    Discount a Series of FCFs (index = year) back to base_year end.
    Year base_year+1 is discounted 1 period, base_year+2 two periods, etc.
    """
    total = 0.0
    for yr, fcf in fcf_series.items():
        n = yr - base_year
        if n <= 0:
            continue
        total += fcf / (1 + wacc) ** n
    return total


def _terminal_value(
    last_fcf: float,
    wacc: float,
    terminal_growth: float,
    base_year: int,
    terminal_year: int,
) -> float:
    """
    Gordon-Growth TV at terminal_year, discounted back to base_year.
    TV = FCF_terminal × (1 + g) / (WACC - g)
    """
    if wacc <= terminal_growth:
        raise ValueError(f"WACC ({wacc:.3f}) must exceed terminal growth ({terminal_growth:.3f})")
    tv      = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    n       = terminal_year - base_year
    return tv / (1 + wacc) ** n


def calculate_rolling_targets(
    f2025: Financials2025,
    consensus: list[ConsensusYear],
    params: DCFParams | None = None,
    cash_per_year: dict[int, float] | None = None,
) -> pd.DataFrame:
    """
    Compute year-end target prices for 2026, 2027, 2028.

    Mechanics per Damodaran Rolling DCF:
      For each valuation date T ∈ {2026, 2027, 2028}:
        Base Cash  = Cash_{T-1} + FCF_T
        EV         = PV(FCF_{T+1 … T+H}) + PV(TV at T+H)   discounted to T
        Equity     = EV + Base Cash − Debt
        Target     = Equity / Shares

    Parameters
    ----------
    cash_per_year : optional mapping {2025: X, 2026: Y, 2027: Z}
                    Override cumulative cash balances if known.
    """
    if params is None:
        params = DCFParams()

    firm_type, horizon = classify_firm(f2025)
    wacc               = _wacc(params)

    # ── build full FCF table: consensus (2026-2028) + extrapolated ────────
    con_df = consensus_to_fcf(consensus, params)

    # extrapolate from 2029 to cover the longest window needed (2028+horizon)
    last_con_year  = max(c.year for c in consensus)
    ext_start      = last_con_year + 1
    ext_end        = 2028 + horizon          # worst-case terminal year

    ext_df = extrapolate_fcfs(
        consensus          = consensus,
        params             = params,
        firm_type          = firm_type,
        horizon            = horizon,
        extrapolate_from_year = ext_start,
        extrapolate_to_year   = ext_end,
    )

    all_fcf: pd.Series = pd.concat([con_df["fcf"], ext_df["fcf"]])

    # ── rolling cash balances ─────────────────────────────────────────────
    # Cash_T = Cash_{T-1} + FCF_T  (simplified; ignores dividends/buybacks)
    rolling_cash: dict[int, float] = {}
    cash_prev = cash_per_year.get(2025, f2025.cash) if cash_per_year else f2025.cash
    for yr in sorted([c.year for c in consensus]):
        fcf_t = float(all_fcf.loc[yr])
        rolling_cash[yr] = cash_prev + fcf_t
        cash_prev = rolling_cash[yr]

    # ── valuation loop ────────────────────────────────────────────────────
    results: list[dict] = []

    for base_year in (2026, 2027, 2028):
        # projection window: base_year+1 … base_year+horizon
        proj_start   = base_year + 1
        proj_end     = base_year + horizon
        terminal_yr  = proj_end

        # FCFs within the projection window
        window_fcf = all_fcf[
            (all_fcf.index >= proj_start) & (all_fcf.index <= proj_end)
        ]

        pv_fcfs = _pv_stream(window_fcf, wacc, base_year)

        last_fcf_in_window = float(all_fcf.loc[terminal_yr]) if terminal_yr in all_fcf.index else float(window_fcf.iloc[-1])
        pv_tv   = _terminal_value(last_fcf_in_window, wacc, params.terminal_growth,
                                  base_year, terminal_yr)

        ev           = pv_fcfs + pv_tv
        base_cash    = rolling_cash.get(base_year, f2025.cash)
        equity_value = ev + base_cash - f2025.debt
        target_price = equity_value / f2025.shares if f2025.shares else float("nan")

        results.append({
            "valuation_date":  base_year,
            "firm_type":       firm_type,
            "horizon_yrs":     horizon,
            "wacc_pct":        round(wacc * 100, 2),
            "proj_window":     f"{proj_start}-{proj_end}",
            "pv_fcfs_B":       round(pv_fcfs, 2),
            "pv_tv_B":         round(pv_tv, 2),
            "ev_B":            round(ev, 2),
            "base_cash_B":     round(base_cash, 2),
            "debt_B":          round(f2025.debt, 2),
            "equity_B":        round(equity_value, 2),
            "shares_B":        round(f2025.shares, 3),
            "target_price":    round(target_price, 2),
        })

    return pd.DataFrame(results).set_index("valuation_date")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DIAGNOSTICS  (full FCF schedule)
# ─────────────────────────────────────────────────────────────────────────────

def build_full_schedule(
    consensus: list[ConsensusYear],
    params: DCFParams,
    firm_type: FirmType,
    horizon: int,
) -> pd.DataFrame:
    """Return the complete FCF schedule for inspection."""
    con_df = consensus_to_fcf(consensus, params)
    ext_df = extrapolate_fcfs(
        consensus             = consensus,
        params                = params,
        firm_type             = firm_type,
        horizon               = horizon,
        extrapolate_from_year = max(c.year for c in consensus) + 1,
        extrapolate_to_year   = 2028 + horizon,
    )
    return pd.concat([con_df, ext_df])


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MOCK EXECUTION  (AAPL-proxy dummy data)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)

    # ── 2025 actuals snapshot ─────────────────────────────────────────────
    aapl_2025 = Financials2025(
        ebit_margin    = 0.318,   # 31.8 %
        revenue_growth = 0.064,   # 6.4 %  → classified as MATURE
        revenue        = 416.16,
        ebit           = 132.34,
        da             = 16.70,
        capex          = 10.90,
        delta_nwc      = 3.20,
        cash           = 53.77,   # cash + ST investments
        debt           = 97.34,
        shares         = 15.12,   # diluted, B shares
    )

    # ── 2026–2028 consensus (Stockanalysis-style) ─────────────────────────
    consensus_aapl: list[ConsensusYear] = [
        ConsensusYear(
            year=2026, revenue=486.77, ebit_margin=0.330,
            da=17.50, capex=11.50, delta_nwc=3.80,
        ),
        ConsensusYear(
            year=2027, revenue=528.50, ebit_margin=0.338,
            da=18.20, capex=12.10, delta_nwc=4.10,
        ),
        ConsensusYear(
            year=2028, revenue=573.00, ebit_margin=0.342,
            da=19.00, capex=12.80, delta_nwc=4.40,
        ),
    ]

    # ── DCF parameters ────────────────────────────────────────────────────
    params = DCFParams(
        tax_rate               = 0.155,   # AAPL effective
        risk_free_rate         = 0.044,
        erp                    = 0.055,
        beta                   = 1.24,
        terminal_growth        = 0.025,
        industry_ebit_margin   = 0.28,    # Tech hardware industry avg
        da_pct_revenue         = 0.042,
        capex_pct_revenue      = 0.026,
        nwc_pct_revenue        = 0.018,
    )

    # ── classify ──────────────────────────────────────────────────────────
    firm_type, horizon = classify_firm(aapl_2025)
    wacc = _wacc(params)

    print("=" * 65)
    print("  ROLLING DCF  -  AAPL-proxy  (Damodaran methodology)")
    print("=" * 65)
    print(f"  Firm type   : {firm_type.upper()}   |   Horizon : {horizon} yrs")
    print(f"  WACC        : {wacc*100:.2f}%  "
          f"(rf {params.risk_free_rate*100:.1f}% + b{params.beta} x ERP {params.erp*100:.1f}%)")
    print(f"  Terminal g  : {params.terminal_growth*100:.1f}%")
    print()

    # ── full FCF schedule ─────────────────────────────────────────────────
    schedule = build_full_schedule(consensus_aapl, params, firm_type, horizon)
    print("-- FCF Schedule (B$) " + "-" * 45)
    display_cols = ["revenue", "rev_growth", "ebit_margin", "nopat", "da", "capex", "fcf", "source"]
    print(schedule[display_cols].to_string())
    print()

    # ── rolling target prices ─────────────────────────────────────────────
    targets = calculate_rolling_targets(aapl_2025, consensus_aapl, params)
    print("-- Rolling Target Prices " + "-" * 41)
    print(targets.to_string())
    print()

    # ── per-year summary ──────────────────────────────────────────────────
    print("-- Target Price Summary " + "-" * 42)
    for yr, row in targets.iterrows():
        print(f"  Year-End {yr} -> ${row['target_price']:.2f}  "
              f"(EV ${row['ev_B']:.1f}B | Cash ${row['base_cash_B']:.1f}B "
              f"| Debt ${row['debt_B']:.1f}B | Equity ${row['equity_B']:.1f}B)")
