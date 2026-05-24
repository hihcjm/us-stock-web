"""
rolling_dcf.py  (us_stock_web)
==============================
Aswath Damodaran 4-Stage Life Cycle DCF Valuation Engine
for US Stocks — USD / B-shares unit convention

Core Design Principle (v2)
--------------------------
  - All stages share a SINGLE FCFF formula:
      FCFF_t = NOPAT_t × (1 − RR_t)
      RR_t   = clip( g_t / ROIC_t,  RR_MIN, RR_MAX )
  - Stage classification adjusts ONLY the effective discount rate:
      Stage 1 (Startup)    : WACC + stage_premium  (default +3%)
      Stage 2 (High Growth): WACC
      Stage 3 (Mature)     : WACC − 0.5%  (lower risk)
      Stage 4 (Decline)    : WACC
  - Growth path is unified in _growth_path() → no discontinuity at stage boundaries

Unit Convention
---------------
  All monetary values : B-USD  (billions of dollars)
  shares              : B-shares (billions of shares)
  intrinsic_value     : USD per share = equity_value(B) / shares(B)
  Rates               : fraction (0.0 – 1.0)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Input Data Structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Financials:
    """
    Latest fiscal-year actuals snapshot.
    All monetary amounts in B-USD.
    """
    # ── Income Statement ────────────────────────────────────────────────────
    revenue:            float   # Total Revenue (B-USD)
    ebit:               float   # Operating Income / EBIT (B-USD)
    ebit_margin:        float   # EBIT margin = ebit / revenue (fraction)
    tax_rate:           float   # Effective tax rate (fraction, e.g. 0.15)

    # ── Cash Flow Statement ─────────────────────────────────────────────────
    depr_amort:         float   # D&A (B-USD)
    capex:              float   # Capital Expenditures, positive (B-USD)
    change_wc:          float   # Change in Working Capital (B-USD, increase=positive=cash outflow)

    # ── Balance Sheet ───────────────────────────────────────────────────────
    cash_st:            float   # Cash + Short-term Investments (B-USD)
    debt:               float   # Total Debt (B-USD)
    minority_interest:  float   # Non-controlling Interest (B-USD, 0 for most US firms)

    # ── Market Data ─────────────────────────────────────────────────────────
    shares:             float   # Shares Outstanding (B-shares)
                                # e.g. Apple 15.4B shares → 15.4


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DCF Engine
# ═══════════════════════════════════════════════════════════════════════════════

class DamodaranDCF:
    """
    Damodaran 4-Stage Life Cycle DCF — single FCFF formula, stage-adjusted discount rate.

    Effective WACC by Stage
    -----------------------
      Stage 1: WACC + stage_premium  (default +0.03 = +3pp)
               Startup concentration / binary risk
      Stage 2: WACC                  (base case)
      Stage 3: WACC − 0.005          (−0.5pp, lower operating risk)
      Stage 4: WACC                  (base, liquidation uncertainty)

    Common FCFF Formula (_growth_path)
    ------------------------------------
      NOPAT_t  = NOPAT_{t-1} × (1 + g_t)
      RR_t     = clip( g_t / ROIC_t,  RR_MIN, RR_MAX )
      FCFF_t   = NOPAT_t × (1 − RR_t)
      PV_t     = FCFF_t / (1 + eff_wacc)^t
    """

    _GUARD_REINV_MIN  = -0.80
    _GUARD_REINV_MAX  =  0.95
    _GUARD_ROIC_FLOOR =  0.001
    _GUARD_ROIC_CAP   =  0.80

    def __init__(
        self,
        financials:     Financials,
        rf:             float = 0.044,
        erp:            float = 0.055,
        beta:           float = 1.0,
        debt_spread:    float = 0.015,
        equity_weight:  Optional[float] = None,
    ):
        self.fin         = financials
        self.rf          = rf
        self.erp         = erp
        self.beta        = beta
        self.debt_spread = debt_spread
        self._eq_weight  = equity_weight

        self.wacc, self.coe, self.cod = self._calculate_wacc()

    # ── A. WACC ─────────────────────────────────────────────────────────────

    def _calculate_wacc(self) -> tuple[float, float, float]:
        """
        CoE = rf + β × ERP   (no CRP — US is base market)
        CoD = (rf + spread) × (1 − t)
        """
        coe      = self.rf + self.beta * self.erp
        spread   = max(self.debt_spread, 0.05) if self.fin.ebit <= 0 else self.debt_spread
        cod      = (self.rf + spread) * (1 - self.fin.tax_rate)

        if self._eq_weight is not None:
            e_w = max(0.0, min(1.0, self._eq_weight))
            d_w = 1.0 - e_w
        else:
            if self.fin.debt > 0:
                nopat_proxy = max(self.fin.ebit * (1 - self.fin.tax_rate), 1e-6)
                implied_ev  = nopat_proxy / max(coe * 0.9, 0.05)
                d_w = min(self.fin.debt / (implied_ev + self.fin.debt), 0.60)
                e_w = 1.0 - d_w
            else:
                e_w, d_w = 1.0, 0.0

        wacc = e_w * coe + d_w * cod
        return wacc, coe, cod

    # ── B. Utility ──────────────────────────────────────────────────────────

    def _nopat(self) -> float:
        return self.fin.ebit * (1 - self.fin.tax_rate)

    def _base_roic(self) -> float:
        """
        ROIC = NOPAT / IC
        IC = max(D&A × 7, Revenue × 0.3)  — conservative (larger IC → lower ROIC)
        """
        nopat = self._nopat()
        da    = max(self.fin.depr_amort, 1e-9)
        ic    = max(da * 7.0, self.fin.revenue * 0.3, 1e-9)
        roic  = nopat / ic
        return min(max(roic, self._GUARD_ROIC_FLOOR), self._GUARD_ROIC_CAP)

    def _base_reinvestment_rate(self) -> float:
        nopat = self._nopat()
        if nopat <= 0:
            return 0.50
        reinv = self.fin.capex - self.fin.depr_amort + self.fin.change_wc
        rr    = reinv / nopat
        return max(self._GUARD_REINV_MIN, min(rr, self._GUARD_REINV_MAX))

    @staticmethod
    def _pv(rate: float, t: int) -> float:
        return 1.0 / ((1.0 + rate) ** t)

    def _equity_bridge(
        self,
        ev:    float,
        fcffs: Optional[list[dict]] = None,
        extra: Optional[dict]       = None,
    ) -> dict:
        """
        Equity Bridge (US GAAP):
          Equity = EV + Cash − Debt − Minority Interest
          IV/share = Equity (B-USD) / Shares (B-shares) = USD/share
        """
        equity_value    = ev + self.fin.cash_st - self.fin.debt - self.fin.minority_interest
        price_per_share = (equity_value / self.fin.shares) if self.fin.shares > 0 else 0.0

        result = {
            "intrinsic_value": price_per_share,
            "ev":              ev,
            "equity_value":    equity_value,
            "cash_st":         self.fin.cash_st,
            "debt":            self.fin.debt,
            "minority":        self.fin.minority_interest,
            "wacc":            self.wacc,
            "coe":             self.coe,
            "cod":             self.cod,
            "rf":              self.rf,
            "erp":             self.erp,
            "beta":            self.beta,
            "fcff_schedule":   fcffs or [],
        }
        if extra:
            result.update(extra)
        return result

    # ── C. Shared Growth Path Engine ────────────────────────────────────────

    def _growth_path(
        self,
        g_base:       float,
        roic_base:    float,
        g_terminal:   float,
        eff_wacc:     float,
        phase1_years: int   = 5,
        phase2_years: int   = 5,
        nopat_start:  Optional[float] = None,
        note:         str   = "",
    ) -> tuple[list[dict], float, float]:
        """
        Unified FCFF projection engine shared by all 4 stages.

        Growth path
        -----------
          Phase 1 (t=1..phase1_years)  : g=g_base, ROIC=roic_base  (maintained)
          Phase 2 (t=+1..+phase2_years): g → g_terminal  (linear)
                                         ROIC → eff_wacc  (linear, maturity proxy)
          Terminal (t>total)            : Gordon Growth at g_terminal

        FCFF formula (same for all stages)
        -----------------------------------
          NOPAT_t  = NOPAT_{t-1} × (1 + g_t)
          RR_t     = clip( g_t / ROIC_t,  RR_MIN, RR_MAX )
          FCFF_t   = NOPAT_t × (1 − RR_t)
          PV_t     = FCFF_t / (1 + eff_wacc)^t

        Returns
        -------
        (fcffs: list[dict], pv_sum: float, pv_tv: float)
        """
        g_terminal = min(g_terminal, self.rf)
        g_terminal = max(g_terminal, 0.005)

        total_years   = phase1_years + phase2_years
        current_nopat = (nopat_start if nopat_start is not None else self._nopat())
        if current_nopat <= 0:
            current_nopat = max(self.fin.revenue * 0.01, 1e-9)

        fcffs, pv_sum = [], 0.0

        for t in range(1, total_years + 1):
            if t <= phase1_years:
                g_t    = g_base
                roic_t = roic_base
                phase  = f"phase1_{note}" if note else "phase1"
            else:
                alpha  = (t - phase1_years) / phase2_years
                g_t    = g_base    * (1 - alpha) + g_terminal * alpha
                roic_t = roic_base * (1 - alpha) + eff_wacc   * alpha
                roic_t = max(roic_t, self._GUARD_ROIC_FLOOR)
                phase  = f"phase2_{note}" if note else "phase2"

            rr_t   = max(self._GUARD_REINV_MIN, min(g_t / roic_t, self._GUARD_REINV_MAX))
            current_nopat *= (1 + g_t)
            fcf_t  = current_nopat * (1 - rr_t)
            pv_t   = fcf_t * self._pv(eff_wacc, t)
            pv_sum += pv_t

            fcffs.append({
                "year":       t,
                "growth_g":   round(g_t,           4),
                "roic":       round(roic_t,         4),
                "reinv_rate": round(rr_t,           4),
                "nopat":      round(current_nopat,  4),
                "fcf":        round(fcf_t,          4),
                "pv_fcf":     round(pv_t,           4),
                "eff_wacc":   round(eff_wacc,       4),
                "phase":      phase,
            })

        # Terminal Value (Gordon Growth)
        rr_tv    = g_terminal / max(eff_wacc, g_terminal + 0.001)
        nopat_tv = current_nopat * (1 + g_terminal)
        fcf_tv   = nopat_tv * (1 - rr_tv)
        tv       = fcf_tv / max(eff_wacc - g_terminal, 0.001)
        pv_tv    = tv * self._pv(eff_wacc, total_years)

        return fcffs, pv_sum, pv_tv

    # ── D. Public Interface ─────────────────────────────────────────────────

    def calculate_intrinsic_value(self, stage: int = 2, **kwargs) -> dict:
        """
        Dispatch to stage model.

        stage : 1 = Startup      (Top-Down, TAM-based)
                2 = High Growth  (3-Phase, ROIC Fading)
                3 = Mature       (Stable Growth, lower risk discount)
                4 = Decline      (Liquidating Cash Flow)

        Returns dict:
            intrinsic_value  : float  — USD per share
            ev               : float  — B-USD
            equity_value     : float  — B-USD
            eff_wacc         : float  — stage-adjusted discount rate
            fcff_schedule    : list
            ...
        """
        return {
            1: self._startup_valuation,
            2: self._high_growth_valuation,
            3: self._mature_valuation,
            4: self._decline_valuation,
        }.get(stage, self._high_growth_valuation)(**kwargs)

    # ── D-1. Stage 1: Startup ───────────────────────────────────────────────

    def _startup_valuation(
        self,
        tam:                 float = 0.0,
        target_share:        float = 0.10,
        target_margin:       float = 0.15,
        prob_failure:        float = 0.30,
        liquidation_val_pct: float = 0.50,
        ramp_years:          int   = 10,
        stage_premium:       float = 0.03,
        **kwargs,
    ) -> dict:
        """
        Stage 1 — Startup: TAM-based Top-Down revenue ramp.

        Same FCFF formula. Startup-specific adjustments:
          - eff_wacc = WACC + stage_premium  (default +3pp)
          - Failure probability applied to going-concern EV

        Parameters
        ----------
        tam                  : Total Addressable Market (B-USD). Default = revenue × 20.
        target_share         : Target market share at year N (default 10%)
        target_margin        : Target EBIT margin at year N (default 15%)
        prob_failure         : Probability of business failure (default 30%)
        liquidation_val_pct  : Cash recovery rate on failure (default 50%)
        ramp_years           : Projection horizon (default 10)
        stage_premium        : Additional risk premium over WACC (default 3pp)
        """
        eff_wacc  = self.wacc + stage_premium

        if tam <= 0:
            tam = self.fin.revenue * 20.0

        rev_yr0  = max(self.fin.revenue, 1e-9)
        rev_yr_n = tam * target_share

        # Year-1 NOPAT along TAM ramp path
        alpha_1  = 1.0 / ramp_years
        rev_1    = rev_yr0 * ((rev_yr_n / rev_yr0) ** alpha_1)
        margin_1 = self.fin.ebit_margin + alpha_1 * (target_margin - self.fin.ebit_margin)
        nopat_1  = max(rev_1 * margin_1 * (1 - self.fin.tax_rate), 1e-9)

        g_base   = min((rev_yr_n / rev_yr0) ** (1.0 / ramp_years) - 1.0, 0.60)
        roic_start = min(
            max(target_margin * (1 - self.fin.tax_rate) / 0.30, self._GUARD_ROIC_FLOOR),
            self._GUARD_ROIC_CAP,
        )
        g_terminal = min(0.025, self.rf)

        fcffs, pv_sum, pv_tv = self._growth_path(
            g_base       = g_base,
            roic_base    = roic_start,
            g_terminal   = g_terminal,
            eff_wacc     = eff_wacc,
            phase1_years = ramp_years // 2,
            phase2_years = ramp_years - ramp_years // 2,
            nopat_start  = nopat_1,
            note         = "startup",
        )

        going_concern_ev = pv_sum + pv_tv
        liquidation_val  = (self.fin.cash_st + self.fin.depr_amort * 3) * liquidation_val_pct
        ev               = (going_concern_ev * (1 - prob_failure)
                            + liquidation_val   *  prob_failure)

        extra = {
            "stage":             "startup",
            "eff_wacc":          round(eff_wacc,          4),
            "stage_premium":     stage_premium,
            "tam":               tam,
            "target_share":      target_share,
            "target_margin":     target_margin,
            "prob_failure":      prob_failure,
            "going_concern_ev":  round(going_concern_ev,  4),
            "liquidation_val":   round(liquidation_val,   4),
            "g_base":            round(g_base,            4),
            "terminal_g":        g_terminal,
            "pv_explicit":       round(pv_sum,            4),
            "pv_terminal_value": round(pv_tv,             4),
        }
        return self._equity_bridge(ev, fcffs, extra)

    # ── D-2. Stage 2: High Growth ───────────────────────────────────────────

    def _high_growth_valuation(
        self,
        g_override:    Optional[float] = None,
        roic_override: Optional[float] = None,
        rev_cagr:      Optional[float] = None,
        phase1_years:  int   = 5,
        phase2_years:  int   = 5,
        g_terminal:    float = 0.025,
        **kwargs,
    ) -> dict:
        """
        Stage 2 — High Growth: 3-Phase Bottom-Up.

        g_base priority
        ---------------
        1) g_override  (direct input)
        2) rev_cagr    (historical revenue CAGR — corrects low-RR bias)
        3) ROIC × RR_base  (balance-sheet implied)
        floor: max(rf×1.5, WACC×0.4)

        Parameters
        ----------
        g_override    : Growth rate override (fraction)
        roic_override : ROIC override (fraction)
        rev_cagr      : Historical revenue CAGR (fraction)
        phase1_years  : High-growth maintenance period (default 5)
        phase2_years  : Fade period (default 5)
        g_terminal    : Perpetual growth rate (default 2.5%, capped at rf)
        """
        eff_wacc  = self.wacc

        roic      = roic_override if roic_override is not None else self._base_roic()
        rr_base   = self._base_reinvestment_rate()
        g_roic    = roic * max(rr_base, 0.0)

        if g_override is not None:
            g_base = float(g_override)
        elif rev_cagr is not None and rev_cagr > 0:
            g_max  = roic * 0.90
            g_base = max(min(rev_cagr, g_max), g_roic)
        else:
            g_base = g_roic

        g_floor = max(self.rf * 1.5, self.wacc * 0.4)
        if g_base < g_floor and g_override is None:
            g_base = g_floor

        g_base = min(g_base, 0.40)

        fcffs, pv_sum, pv_tv = self._growth_path(
            g_base       = g_base,
            roic_base    = roic,
            g_terminal   = g_terminal,
            eff_wacc     = eff_wacc,
            phase1_years = phase1_years,
            phase2_years = phase2_years,
            nopat_start  = self._nopat(),
            note         = "high_growth",
        )

        pv_s1 = sum(r["pv_fcf"] for r in fcffs if r["year"] <= phase1_years)
        pv_s2 = sum(r["pv_fcf"] for r in fcffs if r["year"] >  phase1_years)

        extra = {
            "stage":             "high_growth",
            "eff_wacc":          round(eff_wacc,   4),
            "g_base":            round(g_base,     4),
            "roic_base":         round(roic,       4),
            "rr_base":           round(rr_base,    4),
            "terminal_g":        round(min(g_terminal, self.rf), 4),
            "pv_stage1":         round(pv_s1,      4),
            "pv_stage2":         round(pv_s2,      4),
            "pv_terminal_value": round(pv_tv,      4),
        }
        return self._equity_bridge(pv_sum + pv_tv, fcffs, extra)

    # ── D-3. Stage 3: Mature ────────────────────────────────────────────────

    def _mature_valuation(
        self,
        g_stable:      float = 0.025,
        g_near:        float = 0.05,
        near_years:    int   = 5,
        wacc_discount: float = 0.005,
        g_terminal:    float = 0.025,
        **kwargs,
    ) -> dict:
        """
        Stage 3 — Mature: Stable growth, reduced risk.

        eff_wacc = WACC − wacc_discount  (default −0.5pp)
        [Rule] terminal_g = min(g_terminal, g_stable, rf)

        Parameters
        ----------
        g_stable       : Perpetual growth rate (default 2.5%)
        g_near         : Near-term growth (default 5%)
        near_years     : Near-term horizon (default 5)
        wacc_discount  : Reduction applied to WACC (default 0.5pp)
        g_terminal     : Override perpetual growth rate
        """
        eff_wacc  = max(self.wacc - wacc_discount, self.rf + 0.005)
        g_tv      = min(g_stable, g_terminal, self.rf)

        roic_base = self._base_roic()
        roic_start = min(roic_base, eff_wacc * 2.5)

        fcffs, pv_sum, pv_tv = self._growth_path(
            g_base       = g_near,
            roic_base    = roic_start,
            g_terminal   = g_tv,
            eff_wacc     = eff_wacc,
            phase1_years = near_years,
            phase2_years = 5,
            nopat_start  = self._nopat(),
            note         = "mature",
        )

        pv_near = sum(r["pv_fcf"] for r in fcffs if r["year"] <= near_years)

        extra = {
            "stage":             "mature",
            "eff_wacc":          round(eff_wacc,   4),
            "wacc_discount":     wacc_discount,
            "terminal_g":        round(g_tv,        4),
            "terminal_g_capped": g_tv < g_stable,
            "rf_cap":            self.rf,
            "pv_near":           round(pv_near,    4),
            "pv_terminal_value": round(pv_tv,      4),
        }
        return self._equity_bridge(pv_sum + pv_tv, fcffs, extra)

    # ── D-4. Stage 4: Decline ───────────────────────────────────────────────

    def _decline_valuation(
        self,
        g_decline:         float = -0.05,
        capex_ratio:       float = 0.50,
        liquidation_years: int   = 10,
        terminal_multiple: float = 3.0,
        **kwargs,
    ) -> dict:
        """
        Stage 4 — Decline: Revenue contraction, asset run-off.

        Same FCFF formula with negative g_base.
        CapEx < D&A → RR < 0 → FCFF > NOPAT  (asset run-off generates cash)
        No perpetuity; residual asset liquidation value replaces terminal value.

        Parameters
        ----------
        g_decline         : Annual revenue decline (default −5%, sign auto-corrected)
        capex_ratio       : CapEx / D&A ratio (default 0.50)
        liquidation_years : Run-off period (default 10)
        terminal_multiple : Residual D&A-based asset value multiple (default 3×)
        """
        g_decline  = -abs(g_decline)
        eff_wacc   = self.wacc

        da_base       = max(self.fin.depr_amort, 1e-9)
        roic_decline  = max(self._base_roic() * capex_ratio, self._GUARD_ROIC_FLOOR)
        nopat_start   = max(self._nopat(), da_base * 0.1)

        fcffs, pv_sum, _ = self._growth_path(
            g_base       = g_decline,
            roic_base    = roic_decline,
            g_terminal   = g_decline,
            eff_wacc     = eff_wacc,
            phase1_years = liquidation_years // 2,
            phase2_years = liquidation_years - liquidation_years // 2,
            nopat_start  = nopat_start,
            note         = "decline",
        )

        residual = da_base * terminal_multiple * self._pv(eff_wacc, liquidation_years)
        ev       = pv_sum + residual

        extra = {
            "stage":          "decline",
            "eff_wacc":       round(eff_wacc,    4),
            "g_decline":      round(g_decline,   4),
            "capex_ratio":    capex_ratio,
            "pv_operating":   round(pv_sum,      4),
            "pv_liquidation": round(residual,    4),
            "terminal_g":     g_decline,
            "note":           "Terminal = residual asset liquidation (no perpetuity)",
        }
        return self._equity_bridge(ev, fcffs, extra)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Quick Sanity Check  (python api/rolling_dcf.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _sanity_check():
    """Apple FY2024 approximate figures — B-USD."""
    fin = Financials(
        revenue=391.0, ebit=123.2, ebit_margin=0.315,
        tax_rate=0.15, depr_amort=11.5, capex=9.4, change_wc=0.5,
        cash_st=65.0, debt=101.0, minority_interest=0.0, shares=15.4,
    )
    engine = DamodaranDCF(fin, rf=0.044, erp=0.055, beta=1.25)

    print("=" * 65)
    print("  Damodaran 4-Stage DCF - Single FCFF Formula")
    print("  Apple FY2024 (approx.)")
    print("=" * 65)
    print(f"  WACC : {engine.wacc*100:.2f}%   CoE: {engine.coe*100:.2f}%   rf: {engine.rf*100:.2f}%")
    print(f"  ROIC : {engine._base_roic()*100:.2f}%   Base RR: {engine._base_reinvestment_rate()*100:.2f}%")
    print("-" * 65)

    configs = {
        1: dict(tam=3000.0, target_share=0.15, target_margin=0.20,
                prob_failure=0.05, stage_premium=0.03),
        2: dict(rev_cagr=0.08, g_terminal=0.025),
        3: dict(g_stable=0.025, g_near=0.05, wacc_discount=0.005),
        4: dict(g_decline=0.05, capex_ratio=0.40),
    }
    labels = {1: "Startup", 2: "High Growth", 3: "Mature", 4: "Decline"}
    wacc_notes = {
        1: f"WACC+3%={engine.wacc*100+3:.1f}%",
        2: f"WACC={engine.wacc*100:.1f}%",
        3: f"WACC-0.5%={engine.wacc*100-0.5:.1f}%",
        4: f"WACC={engine.wacc*100:.1f}%",
    }

    for s in [1, 2, 3, 4]:
        res = engine.calculate_intrinsic_value(stage=s, **configs[s])
        print(f"  Stage {s} ({labels[s]:12s}) [{wacc_notes[s]:16s}]: "
              f"IV = ${res['intrinsic_value']:>8,.2f}/share  EV = ${res['ev']:.1f}B")
    print("=" * 65)


if __name__ == "__main__":
    _sanity_check()
