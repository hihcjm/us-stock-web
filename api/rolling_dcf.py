"""
rolling_dcf.py  (us_stock_web)
==============================
Aswath Damodaran 4-Stage Life Cycle DCF Valuation Engine
for US Stocks  — USD / B-shares unit convention

Unit Convention
---------------
  All monetary values : B-USD  (billions of dollars)
  shares              : B-shares (billions of shares)
  intrinsic_value     : USD per share  = equity_value(B) / shares(B)
  Rates               : fraction (0.0 – 1.0)

US Market Parameters (vs Korea)
--------------------------------
  ERP   : 5.5%  (Damodaran US implied ERP, no CRP needed)
  rf    : 10Y US Treasury (fetched live, fallback 4.4%)
  Tax   : 21%  (US federal statutory rate)
  WACC cap: no minority-interest deduction (US GAAP, not K-IFRS)

Usage
-----
  fin = Financials(
      revenue=383.0, ebit=114.0, ebit_margin=0.297, tax_rate=0.15,
      depr_amort=11.5, capex=11.0, change_wc=0.5,
      cash_st=29.9, debt=104.0, minority_interest=0.0,
      shares=15.4,          # B-shares  (15.4B shares)
  )
  engine = DamodaranDCF(fin, rf=0.044, erp=0.055, beta=1.25)
  result = engine.calculate_intrinsic_value(stage=2)
  print(result['intrinsic_value'])   # USD per share
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
    depr_amort:         float   # D&A  (B-USD)
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
    Damodaran 4-Stage Life Cycle DCF Engine for US Stocks.

    Parameters
    ----------
    financials    : Financials
    rf            : float   — 10Y US Treasury yield (fraction, default 4.4%)
    erp           : float   — US Equity Risk Premium (fraction, default 5.5%)
                             No CRP added (US is base market)
    beta          : float   — 5-year weekly or 1-year daily beta (default 1.0)
    debt_spread   : float   — Credit spread over rf (default 1.5%)
    equity_weight : float | None — explicit E/(D+E). None = auto-estimate.
    """

    _GUARD_REINV_MIN  = -0.5
    _GUARD_REINV_MAX  =  0.95
    _GUARD_ROIC_FLOOR =  0.001

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
        WACC for US equities.
        CoE = rf + β × ERP          (no CRP — US is base market)
        CoD = rf + spread × (1−t)   (after-tax cost of debt)
        Weights: caller-supplied or implied via NOPAT capitalisation.
        """
        coe = self.rf + self.beta * self.erp

        # Cost of Debt: widen spread for loss-making firms
        spread   = max(self.debt_spread, 0.05) if self.fin.ebit <= 0 else self.debt_spread
        cod_pre  = self.rf + spread
        cod      = cod_pre * (1 - self.fin.tax_rate)

        # Capital structure weights
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

    def _base_reinvestment_rate(self) -> float:
        nopat = self._nopat()
        if nopat <= 0:
            return 0.5
        reinv = self.fin.capex - self.fin.depr_amort + self.fin.change_wc
        rr    = reinv / nopat
        return max(self._GUARD_REINV_MIN, min(rr, self._GUARD_REINV_MAX))

    def _base_roic(self) -> float:
        """
        ROIC = NOPAT / Invested Capital
        IC = max(D&A × 5, Revenue × 0.5)
        Same logic as pvgo_web, calibrated for US firms.
        """
        nopat        = self._nopat()
        da           = max(self.fin.depr_amort, 1e-9)
        ic_da_based  = da * 5.0
        ic_rev_based = self.fin.revenue * 0.5
        ic           = max(ic_da_based, ic_rev_based, 1e-9)
        roic         = nopat / ic
        return min(max(roic, self._GUARD_ROIC_FLOOR), 0.40)

    @staticmethod
    def _pv(wacc: float, t: int) -> float:
        return 1.0 / ((1.0 + wacc) ** t)

    def _equity_bridge(
        self,
        ev:    float,
        fcffs: Optional[list[dict]] = None,
        extra: Optional[dict]       = None,
    ) -> dict:
        """
        Equity Bridge (US GAAP):
          Equity Value = EV + Cash & ST Invest − Total Debt − Minority Interest
          IV/share     = Equity Value (B-USD) / Shares (B-shares)  =  USD/share
        """
        equity_value    = ev + self.fin.cash_st - self.fin.debt - self.fin.minority_interest
        price_per_share = (equity_value / self.fin.shares) if self.fin.shares > 0 else 0.0

        result = {
            "intrinsic_value": price_per_share,       # USD per share
            "ev":              ev,                     # B-USD
            "equity_value":    equity_value,           # B-USD
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

    # ── C. Public Interface ─────────────────────────────────────────────────

    def calculate_intrinsic_value(self, stage: int = 2, **kwargs) -> dict:
        """
        Dispatch to the appropriate stage model.

        stage : 1 = Startup      (Top-Down, TAM-based)
                2 = High Growth  (3-Stage Bottom-Up, ROIC Fading)
                3 = Mature       (Gordon Growth or 2-Stage)
                4 = Decline      (Liquidating Cash Flow)

        Returns dict with keys:
            intrinsic_value  : float  — USD per share
            ev               : float  — Enterprise Value (B-USD)
            equity_value     : float  — Equity Value (B-USD)
            wacc             : float
            fcff_schedule    : list   — year-by-year FCFF detail
            stage            : str
            terminal_g       : float
            ...
        """
        dispatch = {
            1: self._startup_valuation,
            2: self._high_growth_valuation,
            3: self._mature_valuation,
            4: self._decline_valuation,
        }
        return dispatch.get(stage, self._high_growth_valuation)(**kwargs)

    # ── D-1. Stage 1: Startup (Top-Down) ───────────────────────────────────

    def _startup_valuation(
        self,
        tam:                 float = 0.0,
        target_share:        float = 0.10,
        target_margin:       float = 0.15,
        prob_failure:        float = 0.30,
        liquidation_val_pct: float = 0.50,
        ramp_years:          int   = 10,
        **kwargs,
    ) -> dict:
        """
        Stage 1 — Startup: Top-Down revenue ramp.

        Parameters
        ----------
        tam              : Total Addressable Market (B-USD). Default = revenue × 20.
        target_share     : Market share in year N (default 10%)
        target_margin    : EBIT margin at year N (default 15%)
        prob_failure     : Probability of failure (default 30%)
        liquidation_val_pct : Recovery rate on cash in distress (default 50%)
        ramp_years       : Projection horizon (default 10)
        """
        if tam <= 0:
            tam = self.fin.revenue * 20.0

        rev_yr0     = max(self.fin.revenue, 1e-9)
        rev_yr_n    = tam * target_share
        margin_yr0  = self.fin.ebit_margin
        margin_yr_n = target_margin

        rr_initial  = 0.85
        g_terminal  = min(0.025, self.rf)
        rr_terminal = g_terminal / max(self.wacc, 0.01)

        fcffs, pv_fcff = [], 0.0

        for t in range(1, ramp_years + 1):
            alpha    = t / ramp_years
            rev_t    = rev_yr0 * ((rev_yr_n / rev_yr0) ** alpha)
            margin_t = margin_yr0 + alpha * (margin_yr_n - margin_yr0)
            ebit_t   = rev_t * margin_t
            nopat_t  = ebit_t * (1 - self.fin.tax_rate)
            rr_t     = max(0.0, min(rr_initial + alpha * (rr_terminal - rr_initial), 0.95))
            fcf_t    = nopat_t * (1 - rr_t)
            pv_t     = fcf_t * self._pv(self.wacc, t)
            pv_fcff += pv_t
            fcffs.append({
                "year": t, "revenue": round(rev_t, 4), "ebit": round(ebit_t, 4),
                "nopat": round(nopat_t, 4), "reinv_rate": round(rr_t, 4),
                "fcf": round(fcf_t, 4), "pv_fcf": round(pv_t, 4), "phase": "ramp",
            })

        last_nopat = fcffs[-1]["nopat"] * (1 + g_terminal)
        fcf_term   = last_nopat * (1 - rr_terminal)
        tv         = fcf_term / max(self.wacc - g_terminal, 0.001)
        pv_tv      = tv * self._pv(self.wacc, ramp_years)

        going_concern_ev = pv_fcff + pv_tv
        liquidation_val  = (self.fin.cash_st + self.fin.depr_amort * 3) * liquidation_val_pct
        ev               = going_concern_ev * (1 - prob_failure) + liquidation_val * prob_failure

        extra = {
            "stage": "startup", "tam": tam, "target_share": target_share,
            "target_margin": target_margin, "prob_failure": prob_failure,
            "going_concern_ev": going_concern_ev, "liquidation_val": liquidation_val,
            "terminal_g": g_terminal,
            "pv_stage": round(pv_fcff, 4), "pv_terminal_value": round(pv_tv, 4),
        }
        return self._equity_bridge(ev, fcffs, extra)

    # ── D-2. Stage 2: High Growth (3-Stage Bottom-Up) ──────────────────────

    def _high_growth_valuation(
        self,
        g_override:    Optional[float] = None,
        roic_override: Optional[float] = None,
        rev_cagr:      Optional[float] = None,
        **kwargs,
    ) -> dict:
        """
        Stage 2 — High Growth: 3-Stage Bottom-Up.

        Phase 1 (t=1–5)  : High growth — current ROIC & RR maintained
        Phase 2 (t=6–10) : Transition  — ROIC fades linearly to WACC
        Phase 3 (t=11–∞) : Perpetuity  — ROIC = WACC, g = rf (terminal)

        g_override    : Override implied growth rate
        roic_override : Override ROIC estimate
        rev_cagr      : Historical revenue CAGR — used to correct for low RR bias
        """
        roic    = roic_override if roic_override is not None else self._base_roic()
        rr_base = self._base_reinvestment_rate()
        g_roic  = roic * max(rr_base, 0.0)

        if g_override is not None:
            g_base = float(g_override)
        elif rev_cagr is not None and rev_cagr > 0:
            # Use rev_cagr directly, capped so RR = g/ROIC stays ≤ 90%
            g_max_by_roic = roic * 0.90
            g_base = min(rev_cagr, g_max_by_roic)
            g_base = max(g_base, g_roic)
        else:
            g_base = g_roic

        # Floor: at minimum rf*1.5 or WACC*0.4 (growth firm baseline)
        g_floor = max(self.rf * 1.5, self.wacc * 0.4)
        if g_base < g_floor and g_override is None:
            g_base = g_floor

        g_base = min(g_base, 0.40)

        nopat_base    = self._nopat()
        current_nopat = nopat_base
        fcffs, pv_fcff, pv_s1, pv_s2 = [], 0.0, 0.0, 0.0

        for t in range(1, 11):
            if t <= 5:
                g_t, roic_t, phase = g_base, roic, "high_growth"
            else:
                alpha  = (t - 5) / 5.0
                g_t    = g_base * (1 - alpha) + self.rf * alpha
                roic_t = max(roic * (1 - alpha) + self.wacc * alpha, self._GUARD_ROIC_FLOOR)
                phase  = "transition"

            rr_t = max(0.0, min(g_t / roic_t, self._GUARD_REINV_MAX))
            current_nopat *= (1 + g_t)
            fcf_t = current_nopat * (1 - rr_t)
            pv_t  = fcf_t * self._pv(self.wacc, t)
            pv_fcff += pv_t
            pv_s1 += pv_t if t <= 5 else 0
            pv_s2 += pv_t if t >  5 else 0

            fcffs.append({
                "year": t, "growth_g": round(g_t, 4), "roic": round(roic_t, 4),
                "reinv_rate": round(rr_t, 4), "nopat": round(current_nopat, 4),
                "fcf": round(fcf_t, 4), "pv_fcf": round(pv_t, 4), "phase": phase,
            })

        # terminal_g: strictly positive, capped at rf
        g_terminal  = max(min(g_base, self.rf), 0.01)
        rr_terminal = g_terminal / max(self.wacc, 0.001)
        nopat_t11   = current_nopat * (1 + g_terminal)
        fcf_t11     = nopat_t11 * (1 - rr_terminal)
        tv          = fcf_t11 / max(self.wacc - g_terminal, 0.001)
        pv_tv       = tv * self._pv(self.wacc, 10)

        extra = {
            "stage": "high_growth", "g_base": round(g_base, 4),
            "roic_base": round(roic, 4), "rr_base": round(rr_base, 4),
            "terminal_g": round(g_terminal, 4),
            "pv_stage1": round(pv_s1, 4), "pv_stage2": round(pv_s2, 4),
            "pv_terminal_value": round(pv_tv, 4),
        }
        return self._equity_bridge(pv_fcff + pv_tv, fcffs, extra)

    # ── D-3. Stage 3: Mature ────────────────────────────────────────────────

    def _mature_valuation(
        self,
        g_stable:      float = 0.025,
        use_two_stage: bool  = True,
        g_near:        float = 0.05,
        near_years:    int   = 5,
        **kwargs,
    ) -> dict:
        """
        Stage 3 — Mature: Stable growth.

        [Rule] terminal_g = min(g_stable, rf)  — never exceeds risk-free rate.

        g_stable      : Perpetual growth rate (default 2.5%). Capped at rf.
        use_two_stage : True = near-term + perpetuity; False = Gordon Growth only.
        g_near        : Near-term growth (default 5%)
        near_years    : Near-term period length (default 5)
        """
        g_terminal  = min(g_stable, self.rf)
        nopat_base  = self._nopat()
        rr_terminal = g_terminal / max(self.wacc, 0.001)

        fcffs, pv_fcff = [], 0.0

        if use_two_stage:
            g_near_eff    = min(g_near, self.wacc)
            current_nopat = nopat_base
            for t in range(1, near_years + 1):
                rr_t = max(rr_terminal, min(g_near_eff / max(self._base_roic(), 0.001), 0.80))
                current_nopat *= (1 + g_near_eff)
                fcf_t  = current_nopat * (1 - rr_t)
                pv_t   = fcf_t * self._pv(self.wacc, t)
                pv_fcff += pv_t
                fcffs.append({
                    "year": t, "growth_g": round(g_near_eff, 4), "reinv_rate": round(rr_t, 4),
                    "nopat": round(current_nopat, 4), "fcf": round(fcf_t, 4),
                    "pv_fcf": round(pv_t, 4), "phase": "near_stable",
                })
            nopat_tv = current_nopat * (1 + g_terminal)
            fcf_tv   = nopat_tv * (1 - rr_terminal)
            tv       = fcf_tv / max(self.wacc - g_terminal, 0.001)
            pv_tv    = tv * self._pv(self.wacc, near_years)
            ev       = pv_fcff + pv_tv
        else:
            fcf_stable = nopat_base * (1 + g_terminal) * (1 - rr_terminal)
            tv         = fcf_stable / max(self.wacc - g_terminal, 0.001)
            ev         = tv
            pv_tv      = ev
            fcffs.append({
                "year": "inf", "growth_g": round(g_terminal, 4),
                "reinv_rate": round(rr_terminal, 4), "nopat": round(nopat_base, 4),
                "fcf": round(fcf_stable / (1 + g_terminal), 4),
                "pv_fcf": round(ev, 4), "phase": "gordon_growth",
            })

        extra = {
            "stage": "mature", "terminal_g": round(g_terminal, 4),
            "terminal_g_capped": g_terminal < g_stable, "rf_cap": self.rf,
            "two_stage": use_two_stage,
            "pv_near": round(pv_fcff, 4), "pv_terminal_value": round(pv_tv, 4),
        }
        return self._equity_bridge(ev, fcffs, extra)

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
        Stage 4 — Decline: Liquidating Cash Flow model.

        CapEx < D&A  →  negative reinvestment rate
        → cash generation EXCEEDS NOPAT via asset run-off.

        g_decline         : Annual revenue decline (default −5%, sign auto-corrected)
        capex_ratio       : CapEx / D&A ratio (default 0.50 → CapEx = 50% of D&A)
        liquidation_years : Runoff period (default 10)
        terminal_multiple : Residual asset value = D&A × multiple (default 3×)
        """
        g_decline = -abs(g_decline)
        nopat_base, da_base, rev_t = self._nopat(), self.fin.depr_amort, self.fin.revenue
        wc_release_rate = 0.02
        fcffs, pv_fcff = [], 0.0

        for t in range(1, liquidation_years + 1):
            rev_t  *= (1 + g_decline)
            da_t    = da_base * max(1 + g_decline * t * 0.5, 0.2)
            capex_t = da_t * capex_ratio
            wc_in   = rev_t * wc_release_rate
            margin_t = self.fin.ebit_margin * max(1 + g_decline * t * 0.3, 0.3)
            nopat_t  = rev_t * margin_t * (1 - self.fin.tax_rate)
            reinv_t  = max(-0.80, min((capex_t - da_t - wc_in) / max(abs(nopat_t), 1e-9), 0.20))
            fcf_t    = nopat_t * (1 - reinv_t)
            pv_t     = fcf_t * self._pv(self.wacc, t)
            pv_fcff += pv_t
            fcffs.append({
                "year": t, "revenue": round(rev_t, 4), "ebit": round(rev_t * margin_t, 4),
                "nopat": round(nopat_t, 4), "capex": round(capex_t, 4), "da": round(da_t, 4),
                "reinv_rate": round(reinv_t, 4), "fcf": round(fcf_t, 4),
                "pv_fcf": round(pv_t, 4), "phase": "decline",
            })

        residual = da_base * terminal_multiple * self._pv(self.wacc, liquidation_years)
        ev       = pv_fcff + residual
        extra    = {
            "stage": "decline", "g_decline": round(g_decline, 4),
            "capex_ratio": capex_ratio, "pv_operating": round(pv_fcff, 4),
            "pv_liquidation": round(residual, 4), "terminal_g": g_decline,
        }
        return self._equity_bridge(ev, fcffs, extra)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Quick Sanity Check  (python api/rolling_dcf.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _sanity_check():
    """Apple FY2024 approximate figures."""
    fin = Financials(
        revenue=391.0, ebit=123.2, ebit_margin=0.315,
        tax_rate=0.15, depr_amort=11.5, capex=9.4, change_wc=0.5,
        cash_st=65.0, debt=101.0, minority_interest=0.0, shares=15.4,
    )
    engine = DamodaranDCF(fin, rf=0.044, erp=0.055, beta=1.25)
    print("=" * 60)
    print("  Damodaran 4-Stage DCF  (Apple FY2024 approx.)")
    print("=" * 60)
    print(f"  WACC: {engine.wacc*100:.2f}%  CoE: {engine.coe*100:.2f}%  rf: {engine.rf*100:.2f}%")
    print("-" * 60)
    for stage, name in [(1,"Startup"),(2,"High Growth"),(3,"Mature"),(4,"Decline")]:
        kw = {}
        if stage == 1:
            kw = dict(tam=3000.0, target_share=0.15, target_margin=0.20, prob_failure=0.05)
        res = engine.calculate_intrinsic_value(stage=stage, **kw)
        print(f"  Stage {stage} ({name:12s}): IV = ${res['intrinsic_value']:>8,.2f}/share  EV = ${res['ev']:.1f}B")
    print("=" * 60)


if __name__ == "__main__":
    _sanity_check()
