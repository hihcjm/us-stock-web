"""
Rolling DCF Valuation Model — Damodaran Methodology (v2, Safety-Guarded)
=========================================================================
Produces year-end target prices for 2026, 2027, 2028.

Safety guards (mandatory, applied to ALL extrapolated years):
  Guard 1 — Growth Decay      : rev growth decays linearly -> Risk_Free_Rate by Y10
  Guard 2 — Margin Convergence: EBIT margin converges linearly -> Target_Industry_Margin by Y10
  Guard 3 — Sales-to-Capital  : Reinvestment = DeltaRevenue / SalesToCapital (no separate CapEx/DA)
  Guard 4 — Terminal Rate Cap : TV = FCF_Y11 / (WACC - rf);  terminal_g = rf;  WACC > rf enforced
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

CONSENSUS_YEARS = (2026, 2027, 2028)
ROLLING_BASES   = (2026, 2027, 2028)


@dataclass
class Financials:
    """Latest-year actual snapshot (2025)."""
    revenue:        float   # $B
    ebit_margin:    float   # fraction  (negative allowed)
    revenue_growth: float   # YoY fraction
    cash:           float   # $B  (cash + ST investments)
    debt:           float   # $B  total debt
    shares:         float   # B diluted shares


@dataclass
class ConsensusYear:
    year:           int
    revenue:        float   # $B
    ebit_margin:    float   # fraction  (negative allowed)


@dataclass
class DCFParams:
    risk_free_rate:         float = 0.044   # US 10-Y  (= terminal growth)
    erp:                    float = 0.055   # Equity Risk Premium
    beta:                   float = 1.0
    tax_rate:               float = 0.21
    target_industry_margin: float = 0.15   # Guard 2 convergence target
    sales_to_capital:       float = 1.5    # Guard 3  ($1 reinvestment -> $X rev)
    wacc:                   float = field(init=False, repr=True, default=0.0)

    def __post_init__(self) -> None:
        raw = self.risk_free_rate + self.beta * self.erp
        # Guard 4: WACC must strictly exceed rf; minimum spread = 100 bps
        self.wacc = max(raw, self.risk_free_rate + 0.01)

    @property
    def terminal_growth(self) -> float:
        """Guard 4: terminal g === rf (immutable)."""
        return self.risk_free_rate


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FIRM CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_firm(f: Financials) -> tuple[FirmType, int]:
    """
    Damodaran classification -> projection horizon.
      Mature      : EBIT margin > 0 AND rev growth <= 10%  -> 5-year
      High Growth : EBIT margin > 0 AND rev growth > 10%   -> 10-year
      Unprofitable: EBIT margin <= 0                        -> 10-year
    """
    if f.ebit_margin <= 0:
        return "unprofitable", 10
    if f.revenue_growth > 0.10:
        return "high_growth", 10
    return "mature", 5


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONSENSUS -> FCF  (2026-2028, Guard 3 applied)
# ─────────────────────────────────────────────────────────────────────────────

def consensus_to_fcf(
    consensus: list[ConsensusYear],
    actuals:   Financials,
    params:    DCFParams,
) -> pd.DataFrame:
    """
    Convert 2026-2028 consensus into FCF rows.
    Guard 3: Reinvestment = DeltaRevenue / sales_to_capital
    FCF = NOPAT - Reinvestment   (NOPAT = EBIT * (1-t), EBIT may be negative)
    """
    rows: list[dict] = []
    prev_rev = actuals.revenue

    for c in sorted(consensus, key=lambda x: x.year):
        ebit  = c.revenue * c.ebit_margin
        nopat = ebit * (1 - params.tax_rate)
        reinv = (c.revenue - prev_rev) / params.sales_to_capital
        fcf   = nopat - reinv
        g     = (c.revenue / prev_rev - 1) if prev_rev else float("nan")

        rows.append({
            "year":         c.year,
            "revenue":      round(c.revenue, 4),
            "rev_growth":   round(g * 100, 2) if math.isfinite(g) else None,
            "ebit_margin":  round(c.ebit_margin * 100, 2),
            "ebit":         round(ebit, 4),
            "nopat":        round(nopat, 4),
            "reinvestment": round(reinv, 4),
            "fcf":          round(fcf, 4),
            "source":       "consensus",
        })
        prev_rev = c.revenue

    return pd.DataFrame(rows).set_index("year")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXTRAPOLATION ENGINE  (Guards 1-3 enforced per year)
# ─────────────────────────────────────────────────────────────────────────────

def _linspace(start: float, end: float, step: int, total_steps: int) -> float:
    """Linear interpolation: step=1 -> start, step=total_steps -> end."""
    if total_steps <= 1:
        return end
    alpha = (step - 1) / (total_steps - 1)
    return start * (1 - alpha) + end * alpha


def extrapolate_fcfs(
    consensus: list[ConsensusYear],
    params:    DCFParams,
    from_year: int,
    to_year:   int,
) -> pd.DataFrame:
    """
    Generate FCF rows for [from_year .. to_year] with all safety guards.

    Guard 1: growth linearly decays from last-consensus growth -> rf  (floor = rf)
    Guard 2: EBIT margin linearly converges -> target_industry_margin
    Guard 3: Reinvestment = DeltaRev / sales_to_capital
    """
    sorted_con  = sorted(consensus, key=lambda c: c.year)
    last        = sorted_con[-1]
    second_last = sorted_con[-2] if len(sorted_con) >= 2 else None

    if second_last and second_last.revenue > 0:
        seed_growth = last.revenue / second_last.revenue - 1
    else:
        seed_growth = params.risk_free_rate + 0.02

    seed_margin = last.ebit_margin
    n_steps     = to_year - from_year + 1

    rows: list[dict] = []
    prev_rev = last.revenue

    for step, yr in enumerate(range(from_year, to_year + 1), start=1):
        # Guard 1: growth decay with hard floor at rf
        g = _linspace(seed_growth, params.risk_free_rate, step, n_steps)
        g = max(g, params.risk_free_rate)

        # Guard 2: margin convergence (no floor/ceiling — allows negative to converge up)
        m = _linspace(seed_margin, params.target_industry_margin, step, n_steps)

        rev   = prev_rev * (1 + g)
        ebit  = rev * m
        nopat = ebit * (1 - params.tax_rate)

        # Guard 3
        reinv = (rev - prev_rev) / params.sales_to_capital
        fcf   = nopat - reinv

        rows.append({
            "year":         yr,
            "revenue":      round(rev, 4),
            "rev_growth":   round(g * 100, 2),
            "ebit_margin":  round(m * 100, 2),
            "ebit":         round(ebit, 4),
            "nopat":        round(nopat, 4),
            "reinvestment": round(reinv, 4),
            "fcf":          round(fcf, 4),
            "source":       "extrapolated",
        })
        prev_rev = rev

    return pd.DataFrame(rows).set_index("year")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ROLLING DCF CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _pv_stream(fcf_series: pd.Series, wacc: float, base_year: int) -> float:
    """Sum of PV of each FCF discounted back to base_year end."""
    total = 0.0
    for yr, fcf in fcf_series.items():
        n = int(yr) - base_year
        if n <= 0:
            continue
        total += float(fcf) / (1 + wacc) ** n
    return total


def _terminal_value(
    fcf_y11:     float,
    wacc:        float,
    terminal_g:  float,
    base_year:   int,
    tv_fcf_year: int,
) -> float:
    """
    Guard 4: TV = FCF_Y11 / (WACC - rf), discounted to base_year.
    Raises ValueError if spread <= 0.
    """
    spread = wacc - terminal_g
    if spread <= 0:
        raise ValueError(
            f"Guard 4 violated: WACC({wacc:.4f}) <= terminal_g({terminal_g:.4f}). "
            "Increase beta or ERP."
        )
    tv = fcf_y11 / spread
    n  = tv_fcf_year - base_year
    return tv / (1 + wacc) ** n


def calculate_rolling_targets(
    actuals:   Financials,
    consensus: list[ConsensusYear],
    params:    DCFParams | None = None,
) -> pd.DataFrame:
    """
    Compute year-end target prices for 2026, 2027, 2028.

    For each base_year T in {2026, 2027, 2028}:
      Base Cash  = Cash_{T-1} + FCF_T      (rolling cash accumulation)
      EV         = PV(FCF_{T+1..T+10}) + PV(TV at Y11)   discounted to T
      Equity     = EV + Base Cash - Debt
      TP         = Equity / Shares
    """
    if params is None:
        params = DCFParams()

    wacc       = params.wacc
    terminal_g = params.terminal_growth   # Guard 4: == rf

    # build consensus FCF table
    con_df = consensus_to_fcf(consensus, actuals, params)

    # extrapolate 2029 .. 2039 (T=2026 needs up to Y11=2037; T=2028 needs up to Y11=2039)
    last_con_year = max(c.year for c in consensus)
    ext_df = extrapolate_fcfs(
        consensus  = consensus,
        params     = params,
        from_year  = last_con_year + 1,
        to_year    = 2028 + 11,
    )

    all_fcf: pd.Series = pd.concat([con_df["fcf"], ext_df["fcf"]])

    # rolling cash: Cash_T = Cash_{T-1} + FCF_T
    rolling_cash: dict[int, float] = {}
    cash_prev = actuals.cash
    for yr in sorted(c.year for c in consensus):
        fcf_t = float(all_fcf.loc[yr])
        rolling_cash[yr] = cash_prev + fcf_t
        cash_prev = rolling_cash[yr]

    results: list[dict] = []

    for base_year in ROLLING_BASES:
        proj_start  = base_year + 1
        proj_end    = base_year + 10
        tv_fcf_year = base_year + 11       # Guard 4: FCF at Y11

        window_fcf = all_fcf[
            (all_fcf.index >= proj_start) & (all_fcf.index <= proj_end)
        ]

        pv_fcfs = _pv_stream(window_fcf, wacc, base_year)

        fcf_y11 = (float(all_fcf.loc[tv_fcf_year])
                   if tv_fcf_year in all_fcf.index
                   else float(all_fcf.iloc[-1]))
        pv_tv   = _terminal_value(fcf_y11, wacc, terminal_g, base_year, tv_fcf_year)

        ev           = pv_fcfs + pv_tv
        base_cash    = rolling_cash.get(base_year, actuals.cash)
        equity_value = ev + base_cash - actuals.debt
        target_price = equity_value / actuals.shares if actuals.shares > 0 else float("nan")

        results.append({
            "valuation_date": base_year,
            "firm_type":      classify_firm(actuals)[0],
            "wacc_pct":       round(wacc * 100, 2),
            "terminal_g_pct": round(terminal_g * 100, 2),
            "proj_window":    f"{proj_start}-{proj_end}",
            "tv_fcf_year":    tv_fcf_year,
            "fcf_y11":        round(fcf_y11, 4),
            "pv_fcfs_B":      round(pv_fcfs, 3),
            "pv_tv_B":        round(pv_tv, 3),
            "ev_B":           round(ev, 3),
            "base_cash_B":    round(base_cash, 3),
            "debt_B":         round(actuals.debt, 3),
            "equity_B":       round(equity_value, 3),
            "shares_B":       round(actuals.shares, 4),
            "target_price":   round(target_price, 2),
        })

    return pd.DataFrame(results).set_index("valuation_date")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL SCHEDULE (diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def build_full_schedule(
    actuals:   Financials,
    consensus: list[ConsensusYear],
    params:    DCFParams,
) -> pd.DataFrame:
    con_df = consensus_to_fcf(consensus, actuals, params)
    ext_df = extrapolate_fcfs(
        consensus  = consensus,
        params     = params,
        from_year  = max(c.year for c in consensus) + 1,
        to_year    = 2028 + 11,
    )
    return pd.concat([con_df, ext_df])


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MOCK EXECUTION  -- extreme data (IonQ-style: 200% growth, -400% margin)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 130)

    # SCENARIO A: Hyper-growth, deeply unprofitable (IonQ-style)
    ionq_actuals = Financials(
        revenue        = 0.043,
        ebit_margin    = -4.0,     # -400% stress test
        revenue_growth = 2.0,      # 200% stress test
        cash           = 0.38,
        debt           = 0.05,
        shares         = 0.67,
    )
    ionq_consensus = [
        ConsensusYear(year=2026, revenue=0.105, ebit_margin=-2.5),
        ConsensusYear(year=2027, revenue=0.230, ebit_margin=-1.2),
        ConsensusYear(year=2028, revenue=0.420, ebit_margin=-0.4),
    ]
    ionq_params = DCFParams(
        risk_free_rate=0.044, erp=0.055, beta=2.10,
        tax_rate=0.0, target_industry_margin=0.15, sales_to_capital=0.8,
    )

    # SCENARIO B: AAPL-proxy (mature, profitable)
    aapl_actuals = Financials(
        revenue=416.16, ebit_margin=0.318, revenue_growth=0.064,
        cash=53.77, debt=97.34, shares=15.12,
    )
    aapl_consensus = [
        ConsensusYear(year=2026, revenue=486.77, ebit_margin=0.330),
        ConsensusYear(year=2027, revenue=528.50, ebit_margin=0.338),
        ConsensusYear(year=2028, revenue=573.00, ebit_margin=0.342),
    ]
    aapl_params = DCFParams(
        risk_free_rate=0.044, erp=0.055, beta=1.24,
        tax_rate=0.155, target_industry_margin=0.28, sales_to_capital=2.0,
    )

    for name, actuals, consensus, params in [
        ("IonQ-style  [STRESS: 200% growth, -400% margin]", ionq_actuals, ionq_consensus, ionq_params),
        ("AAPL-proxy  [NORMAL: 6% growth, 32% margin]",     aapl_actuals, aapl_consensus, aapl_params),
    ]:
        firm_type, _ = classify_firm(actuals)
        print("=" * 78)
        print(f"  {name}")
        print("=" * 78)
        print(f"  Firm type    : {firm_type.upper()}")
        print(f"  WACC         : {params.wacc*100:.2f}%  "
              f"(rf {params.risk_free_rate*100:.1f}% + "
              f"b{params.beta} x ERP {params.erp*100:.1f}%)")
        print(f"  Terminal g   : {params.terminal_growth*100:.2f}%  [Guard 4: = rf]")
        print(f"  Spread       : {(params.wacc-params.risk_free_rate)*100:.2f}%  [> 0 enforced]")
        print(f"  Ind. margin  : {params.target_industry_margin*100:.1f}%  [Guard 2 target]")
        print(f"  S/Capital    : {params.sales_to_capital}  [Guard 3]")
        print()

        sched = build_full_schedule(actuals, consensus, params)
        cols  = ["revenue", "rev_growth", "ebit_margin", "nopat", "reinvestment", "fcf", "source"]
        print(sched[cols].to_string())
        print()

        targets = calculate_rolling_targets(actuals, consensus, params)
        print("  Rolling Target Prices:")
        for yr, row in targets.iterrows():
            print(f"    {yr}E -> ${row['target_price']:.2f}  "
                  f"| EV ${row['ev_B']:.3f}B "
                  f"| PV(FCF) ${row['pv_fcfs_B']:.3f}B "
                  f"| PV(TV) ${row['pv_tv_B']:.3f}B "
                  f"| Cash ${row['base_cash_B']:.3f}B")
        print()
