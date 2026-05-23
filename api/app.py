from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import math
import re
import os

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
app = Flask(__name__, template_folder=template_dir)

# ── 유틸 ──────────────────────────────────────────────────────────
def safe_float(val):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return float(val)
    except:
        return None

def fmt_b(val, unit='B'):
    """단위 변환: 원본 달러 → B(십억달러) or M(백만달러)"""
    if val is None: return None
    if unit == 'B': return round(val / 1e9, 2)
    if unit == 'M': return round(val / 1e6, 2)
    return round(val, 2)

def get_series_val(series, key):
    """DataFrame 행에서 최신 non-NaN 값 반환."""
    if series is None or key not in series.index:
        return None
    row = series.loc[key]
    for v in row:
        f = safe_float(v)
        if f is not None:
            return f
    return None

# ── 1. 데이터 수집 ────────────────────────────────────────────────
def fetch_yf_data(ticker_symbol):
    """yfinance로 필요한 모든 데이터 수집."""
    t = yf.Ticker(ticker_symbol)
    info = t.info
    if not info or info.get('quoteType') not in ('EQUITY', 'ETF', None):
        # quoteType 없어도 시도
        if not info.get('shortName') and not info.get('longName'):
            return None

    income  = t.income_stmt      # 연간 손익계산서
    cashflow = t.cashflow        # 연간 현금흐름표
    balance  = t.balance_sheet   # 연간 재무상태표

    return {
        'ticker': ticker_symbol.upper(),
        'info':   info,
        'income': income,
        'cashflow': cashflow,
        'balance':  balance,
        'ticker_obj': t,
    }

# ── 2. 연간 재무 히스토리 파싱 ────────────────────────────────────
def parse_annual_history(data):
    """
    연간 실적 딕셔너리 반환.
    years: ['FY2022','FY2023','FY2024','FY2025']
    rows:  {'Revenue': [v,v,v,v], 'EPS': [...], ...}
    """
    income   = data['income']
    cashflow = data['cashflow']
    balance  = data['balance']
    info     = data['info']

    # ── 연도 라벨 수집 (income_stmt 기준) ──
    years = []
    col_dates = []
    if income is not None and not income.empty:
        for col in income.columns:
            try:
                yr = pd.Timestamp(col).year
                years.append(f"FY{yr}")
                col_dates.append(col)
            except:
                pass

    if not years:
        return None

    def get_row(df, key):
        if df is None or df.empty or key not in df.index:
            return [None] * len(years)
        row = df.loc[key]
        return [safe_float(row.get(c)) for c in col_dates]

    def get_row_first(df, *keys):
        for k in keys:
            r = get_row(df, k)
            if any(v is not None for v in r):
                return r
        return [None] * len(years)

    # ── 발행주식수 (억주) ──
    shares_list = get_row_first(income, 'Diluted Average Shares', 'Basic Average Shares')
    shares_list = [fmt_b(v, 'B') for v in shares_list]   # 십억주

    # ── 주요 손익 항목 (단위: B달러) ──
    rev   = [fmt_b(v) for v in get_row(income, 'Total Revenue')]
    op    = [fmt_b(v) for v in get_row_first(income, 'Operating Income', 'EBIT')]
    net   = [fmt_b(v) for v in get_row_first(income, 'Net Income', 'Net Income Common Stockholders')]
    eps_d = [safe_float(v) for v in get_row_first(income, 'Diluted EPS', 'Basic EPS')]
    gross = [fmt_b(v) for v in get_row(income, 'Gross Profit')]

    # ── 현금흐름 ──
    op_cf = [fmt_b(v) for v in get_row_first(cashflow,
                'Operating Cash Flow',
                'Cash Flow From Continuing Operating Activities')]
    capex = [fmt_b(abs(v)) if v is not None else None
             for v in get_row_first(cashflow, 'Capital Expenditure')]
    fcf   = []
    for i in range(len(years)):
        fcf_direct = get_row(cashflow, 'Free Cash Flow')[i]
        if fcf_direct is not None:
            fcf.append(fmt_b(fcf_direct))
        elif op_cf[i] is not None and capex[i] is not None:
            fcf.append(round(op_cf[i] - capex[i], 2))
        else:
            fcf.append(None)

    # ── 재무상태표 ──
    equity = [fmt_b(v) for v in get_row_first(balance,
                'Stockholders Equity', 'Common Stock Equity')]
    shares_bs = get_row_first(balance, 'Ordinary Shares Number')   # 십억주 단위
    bps = []
    for i in range(len(years)):
        eq = equity[i]
        sh = safe_float(shares_bs[i]) if shares_bs[i] is not None else None
        # equity는 B달러, shares는 십억주 단위
        if eq is not None and sh and sh > 0:
            bps.append(round(eq / sh, 2))   # 달러/주
        else:
            bps.append(None)

    # ── 마진 ──
    op_margin  = [round(op[i]/rev[i]*100, 1) if rev[i] and op[i] is not None else None
                  for i in range(len(years))]
    net_margin = [round(net[i]/rev[i]*100, 1) if rev[i] and net[i] is not None else None
                  for i in range(len(years))]
    fcf_margin = [round(fcf[i]/rev[i]*100, 1) if rev[i] and fcf[i] is not None else None
                  for i in range(len(years))]
    gross_margin = [round(gross[i]/rev[i]*100, 1) if rev[i] and gross[i] is not None else None
                    for i in range(len(years))]

    # ── 현재 주가 기반 배수 (히스토리용) ──
    # PER, PBR, PSR 히스토리는 trailing price 기준 역산 불가 → info에서 현재값만 제공
    # → 각 연도 실적 기반으로 역산 (현재 주가 사용)
    price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))

    per_hist = []
    pbr_hist = []
    psr_hist = []
    for i in range(len(years)):
        # EPS > 0 인 연도만
        e = eps_d[i]
        if price and e and e > 0:
            per_hist.append(round(price / e, 2))
        else:
            per_hist.append(None)
        # BPS > 0 인 연도만
        b = bps[i]
        if price and b and b > 0:
            pbr_hist.append(round(price / b, 2))
        else:
            pbr_hist.append(None)
        # PSR
        r_val = rev[i]
        sh_val = shares_list[i]
        if price and r_val and sh_val and sh_val > 0:
            sps = r_val / sh_val   # B달러 / B주 = 달러/주
            psr_hist.append(round(price / sps, 2))
        else:
            psr_hist.append(None)

    rows = {
        'Revenue':      rev,
        'Gross Profit': gross,
        'Operating Income': op,
        'Net Income':   net,
        'EPS (Diluted)': eps_d,
        'BPS':          bps,
        'Op CF':        op_cf,
        'Capex':        capex,
        'FCF':          fcf,
        'Op Margin':    op_margin,
        'Net Margin':   net_margin,
        'Gross Margin': gross_margin,
        'FCF Margin':   fcf_margin,
        'PER':          per_hist,
        'PBR':          pbr_hist,
        'PSR':          psr_hist,
    }

    return {
        'years':     years,
        'col_dates': col_dates,
        'rows':      rows,
        'price':     price,
    }

# ── 3. 컨센서스 추정치 ────────────────────────────────────────────
def get_estimates(data):
    """
    yfinance analyst estimates에서 Forward EPS·Revenue 수집.
    반환: {
      'fwd_eps': 9.6, 'fwd_rev': 478.0(B), 'fwd_pe': 32.2,
      'target_mean': 308.0, 'target_high': 400.0, 'target_low': 215.0,
      'n_analysts': 43, 'recommendation': 'Buy',
      'next_yr_eps': 10.5, 'next_yr_rev': 517.0(B),
    }
    """
    info = data['info']
    t    = data['ticker_obj']
    result = {}

    price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
    result['price'] = price

    # 현재 연도 trailing
    result['trailing_eps'] = safe_float(info.get('trailingEps'))
    result['trailing_pe']  = safe_float(info.get('trailingPE'))

    # Forward (다음 12개월)
    result['fwd_eps'] = safe_float(info.get('forwardEps'))
    result['fwd_pe']  = safe_float(info.get('forwardPE'))

    # 애널리스트 목표가
    result['target_mean']   = safe_float(info.get('targetMeanPrice'))
    result['target_high']   = safe_float(info.get('targetHighPrice'))
    result['target_low']    = safe_float(info.get('targetLowPrice'))
    result['target_median'] = safe_float(info.get('targetMedianPrice'))
    result['n_analysts']    = info.get('numberOfAnalystOpinions')

    # 투자의견
    rec_map = {1: 'Strong Buy', 1.5: 'Buy', 2: 'Buy',
               2.5: 'Hold', 3: 'Hold', 3.5: 'Underperform',
               4: 'Sell', 5: 'Strong Sell'}
    rec_mean = safe_float(info.get('recommendationMean'))
    if rec_mean:
        closest = min(rec_map.keys(), key=lambda x: abs(x - rec_mean))
        result['recommendation'] = rec_map[closest]
    else:
        result['recommendation'] = info.get('recommendationKey', '').capitalize()
    result['rec_score'] = rec_mean

    # 연도별 EPS/Revenue 추정치
    try:
        ee = t.earnings_estimate
        if ee is not None and not ee.empty:
            # 0y = 현재 회계연도, +1y = 다음 회계연도
            if '0y' in ee.index:
                result['curr_yr_eps']  = safe_float(ee.loc['0y', 'avg'])
                result['curr_yr_eps_low']  = safe_float(ee.loc['0y', 'low'])
                result['curr_yr_eps_high'] = safe_float(ee.loc['0y', 'high'])
                result['curr_yr_growth']   = safe_float(ee.loc['0y', 'growth'])
            if '+1y' in ee.index:
                result['next_yr_eps']  = safe_float(ee.loc['+1y', 'avg'])
                result['next_yr_eps_low']  = safe_float(ee.loc['+1y', 'low'])
                result['next_yr_eps_high'] = safe_float(ee.loc['+1y', 'high'])
                result['next_yr_growth']   = safe_float(ee.loc['+1y', 'growth'])
    except:
        pass

    try:
        re_ = t.revenue_estimate
        if re_ is not None and not re_.empty:
            if '0y' in re_.index:
                result['curr_yr_rev'] = fmt_b(safe_float(re_.loc['0y', 'avg']))
            if '+1y' in re_.index:
                result['next_yr_rev'] = fmt_b(safe_float(re_.loc['+1y', 'avg']))
    except:
        pass

    # PEG
    result['peg'] = safe_float(info.get('pegRatio'))

    # PSR trailing
    result['psr_trailing'] = safe_float(info.get('priceToSalesTrailing12Months'))

    # Beta / 52주
    result['beta']    = safe_float(info.get('beta'))
    result['w52_high'] = safe_float(info.get('fiftyTwoWeekHigh'))
    result['w52_low']  = safe_float(info.get('fiftyTwoWeekLow'))

    # 시가총액 (B달러)
    result['market_cap'] = fmt_b(safe_float(info.get('marketCap')))

    # 배당
    result['div_yield'] = safe_float(info.get('dividendYield'))   # 소수점 (0.035)
    result['div_rate']  = safe_float(info.get('dividendRate'))

    # ROE
    result['roe'] = safe_float(info.get('returnOnEquity'))   # 소수점 (1.41)
    result['roa'] = safe_float(info.get('returnOnAssets'))
    result['roe_pct'] = round(result['roe'] * 100, 1) if result['roe'] else None
    result['roa_pct'] = round(result['roa'] * 100, 1) if result['roa'] else None

    # Debt/Equity
    result['de_ratio'] = safe_float(info.get('debtToEquity'))

    # 섹터/산업
    result['sector']   = info.get('sector', '')
    result['industry'] = info.get('industry', '')
    result['name']     = info.get('shortName') or info.get('longName') or data['ticker']

    # 발행주식수 (주 단위 → calc_dcf에서 B주로 변환)
    result['shares_outstanding'] = safe_float(info.get('sharesOutstanding'))

    return result

# ── 4. 밸류에이션 밴드 ────────────────────────────────────────────
def calc_band(hist_annual, estimates):
    """
    PER/PBR/PSR/FCF 히스토리 밴드 계산.
    적자(음수) 연도 제외.
    """
    rows  = hist_annual['rows']
    years = hist_annual['years']
    price = hist_annual['price']

    def make_band(metric, hist_vals, fwd_val=None, fwd_base=None,
                  base_label=None, no_theory=False):
        # None, 음수 제외
        clean = [v for v in hist_vals if v is not None and v > 0]
        if len(clean) < 2:
            return {"metric": metric, "error": "데이터 부족 (2개년 미만)"}
        avg = sum(clean) / len(clean)
        std = math.sqrt(sum((v - avg)**2 for v in clean) / len(clean))
        def grade(val):
            if val is None or std == 0: return None
            z = (val - avg) / std
            if z < -2: return "Significantly Undervalued"
            if z < -1: return "Undervalued"
            if z <  1: return "Fair"
            if z <  2: return "Overvalued"
            return "Significantly Overvalued"
        def theory(base):
            return round(avg * base, 2) if base and not no_theory else None
        def diff(tp):
            if tp is None or not price: return None, None
            d = tp - price
            return f"{d:+.2f}", f"{d/price*100:+.1f}"

        tp_fwd = theory(fwd_base)
        d_fwd, dp_fwd = diff(tp_fwd)
        bands = {k: round(avg + s*std, 2)
                 for k, s in [('m2s',-2),('m1s',-1),('avg',0),('p1s',1),('p2s',2)]}
        return {
            "metric": metric, "base_label": base_label,
            "hist_vals": clean,
            "hist_avg": round(avg, 2), "hist_std": round(std, 2),
            "bands": bands,
            "fwd_val": fwd_val,
            "grade_fwd": grade(fwd_val),
            "theory_fwd": f"{tp_fwd:.2f}" if tp_fwd else None,
            "diff_fwd": d_fwd, "diff_pct_fwd": dp_fwd,
        }

    results = []
    est = estimates

    # ── PER ──
    per_hist  = [v for v, e in zip(rows['PER'], rows['EPS (Diluted)'])
                 if v is not None and e is not None and e > 0]
    fwd_pe    = est.get('fwd_pe')
    fwd_eps   = est.get('fwd_eps') or est.get('curr_yr_eps')
    results.append(make_band("P/E", per_hist, fwd_val=fwd_pe,
                              fwd_base=fwd_eps, base_label="Fwd EPS"))

    # ── PBR ──
    pbr_hist = [v for v, b in zip(rows['PBR'], rows['BPS'])
                if v is not None and b is not None and b > 0]
    # forward BPS 없으면 trailing PBR
    trailing_pbr = safe_float(est['price'] / rows['BPS'][-1]
                               if rows['BPS'] and rows['BPS'][-1] else None)
    results.append(make_band("P/B", pbr_hist, fwd_val=trailing_pbr,
                              fwd_base=rows['BPS'][-1] if rows['BPS'] else None,
                              base_label="BPS (latest)"))

    # ── PEG ──
    peg_hist = []
    eps_list = rows['EPS (Diluted)']
    for i in range(1, len(eps_list)):
        pe  = rows['PER'][i]
        e0  = eps_list[i-1]
        e1  = eps_list[i]
        if pe and e0 and e1 and e0 > 0 and e1 > 0:
            g = (e1/e0 - 1) * 100
            if g > 0:
                peg_hist.append(round(pe / g, 2))
    fwd_peg = est.get('peg')
    results.append(make_band("PEG", peg_hist, fwd_val=fwd_peg, no_theory=True))

    # ── PSR ──
    psr_hist = [v for v, r in zip(rows['PSR'], rows['Revenue'])
                if v is not None and r is not None and r > 0]
    fwd_psr  = est.get('psr_trailing')
    results.append(make_band("P/S", psr_hist, fwd_val=fwd_psr,
                              base_label="Rev/Share"))

    return results

# ── 5. DCF ────────────────────────────────────────────────────────
def calc_dcf(hist_annual, estimates, r=None):
    """
    실제 FCF 기반 DCF.
    - 과거 FCF > 0 연도 마진 평균
    - Forward Revenue × 마진 → 추정 FCF
    - 3개년 프로젝션 + 터미널 밸류
    """
    try:
        rows  = hist_annual['rows']
        years = hist_annual['years']
        price = hist_annual['price']
        est   = estimates

        # 할인율: rf(10Y Treasury) + beta × ERP(5.5%)
        beta = est.get('beta') or 1.0
        # risk-free rate
        try:
            tnx = yf.Ticker('^TNX')
            tnx_hist = tnx.history(period='5d')
            rf = float(tnx_hist['Close'].iloc[-1]) / 100 if not tnx_hist.empty else 0.044
        except:
            rf = 0.044
        if r is None:
            r = rf + beta * 0.055
        g_terminal = 0.025

        if r <= g_terminal:
            return {"error": f"Discount rate ({r*100:.1f}%) ≤ terminal growth ({g_terminal*100:.1f}%)"}

        # ── 과거 FCF 마진 (FCF > 0 연도만) ──
        hist_fcf_detail = []
        hist_margins    = []
        for i, yr in enumerate(years):
            rev = rows['Revenue'][i]
            fcf = rows['FCF'][i]
            op_cf = rows['Op CF'][i]
            capex = rows['Capex'][i]
            if rev and rev > 0 and fcf is not None:
                margin = fcf / rev
                excluded = fcf <= 0
                if not excluded:
                    hist_margins.append(margin)
                hist_fcf_detail.append({
                    'year': yr,
                    'op_cf': op_cf,
                    'capex': capex,
                    'fcf': fcf,
                    'margin': round(margin * 100, 1),
                    'src': 'Yahoo Finance',
                    'excluded': excluded,
                })

        if len(hist_margins) < 1:
            return {"error": "Not enough positive FCF history"}

        avg_margin = sum(hist_margins) / len(hist_margins)

        # ── 매출 성장률 ──
        # 우선순위: ① 컨센서스 CY→+1Y 성장률 ② 과거 CAGR+마지막1년 평균 (상한 30%)
        rev_list = [v for v in rows['Revenue'] if v and v > 0]
        curr_rev = est.get('curr_yr_rev')
        next_rev = est.get('next_yr_rev')

        if curr_rev and next_rev and curr_rev > 0:
            # 컨센서스 기반 성장률 사용
            rev_growth = min((next_rev / curr_rev) - 1, 0.30)
        elif len(rev_list) >= 2:
            cagr   = (rev_list[-1] / rev_list[0]) ** (1 / (len(rev_list) - 1)) - 1
            last_g = rev_list[-1] / rev_list[-2] - 1
            rev_growth = min((cagr + last_g) / 2, 0.30)
        else:
            rev_growth = 0.08

        # ── 프로젝션 기준 매출 (Forward Revenue 우선) ──
        base_rev = curr_rev or (rev_list[-1] if rev_list else None)

        # 발행주식수 (B주)
        # 우선순위: ① est.shares_outstanding ② EPS/NetIncome 역산(최신 흑자연도) ③ 시총/주가
        shares_b = None
        sh_raw = safe_float(est.get('shares_outstanding'))  # 주 단위
        if sh_raw and sh_raw > 1e6:
            shares_b = sh_raw / 1e9

        if not shares_b:
            for i in range(len(years)):   # 최신(인덱스 0)부터
                eps = rows['EPS (Diluted)'][i]
                net = rows['Net Income'][i]
                if eps and net and abs(eps) > 0.01 and net > 0:
                    shares_b = net / eps   # B달러 / (달러/주) = B주
                    break

        if not shares_b and price:
            mktcap_b = safe_float(est.get('market_cap'))   # B달러
            if mktcap_b:
                shares_b = mktcap_b / price   # B달러 / (달러/주) = B주

        if not shares_b:
            shares_b = 1.0

        fcf_years = []

        # FY+1 (현재 컨센서스 연도)
        label1 = f"FY+1 (Est)"
        rev1   = base_rev
        fcf1   = rev1 * avg_margin if rev1 else None
        if fcf1:
            fcf_years.append((label1, fcf1, rev1))

        # FY+2
        label2 = f"FY+2 (Est)"
        rev2   = next_rev or (base_rev * (1 + rev_growth) if base_rev else None)
        fcf2   = rev2 * avg_margin if rev2 else None
        if fcf2:
            fcf_years.append((label2, fcf2, rev2))

        # FY+3
        label3 = f"FY+3 (Est)"
        rev3   = (rev2 * (1 + rev_growth)) if rev2 else None
        fcf3   = rev3 * avg_margin if rev3 else None
        if fcf3:
            fcf_years.append((label3, fcf3, rev3))

        if not fcf_years:
            return {"error": "Cannot project FCF"}

        # ── 할인 계산 ──
        pv_fcfs  = []
        cum_pv   = 0
        for t_idx, (label, fcf_e, rev_e) in enumerate(fcf_years):
            n       = t_idx + 1
            pv      = fcf_e / (1 + r) ** n
            cum_pv += pv
            tv      = fcf_e * (1 + g_terminal) / (r - g_terminal)
            pv_tv   = tv / (1 + r) ** n
            total_pv = cum_pv + pv_tv
            # 주당 공정가치: total_pv(B달러) / shares_b(B주) = 달러/주
            fv = total_pv / shares_b if shares_b else None
            diff_v  = f"{fv - price:+.2f}"  if fv and price else None
            diff_p  = f"{(fv - price)/price*100:+.1f}" if fv and price else None
            pv_fcfs.append({
                "year":       label,
                "fcf":        round(fcf_e, 2),     # B달러
                "pv":         round(pv, 2),
                "pv_tv":      round(pv_tv, 2),
                "total_pv":   round(total_pv, 2),
                "fair_value": f"${fv:.2f}" if fv else "N/A",
                "diff":       diff_v,
                "diff_pct":   diff_p,
            })

        return {
            "avg_fcf_margin": round(avg_margin * 100, 1),
            "rev_growth":     round(rev_growth * 100, 1),
            "r":              round(r * 100, 2),
            "rf":             round(rf * 100, 2),
            "beta":           round(beta, 2),
            "g_terminal":     round(g_terminal * 100, 1),
            "shares":         round(shares_b, 3),   # B주
            "hist_fcf":       hist_fcf_detail,
            "pv_fcfs":        pv_fcfs,
        }
    except Exception as e:
        import traceback
        return {"error": f"DCF error: {traceback.format_exc()}"}

# ── 6. 재무 테이블 HTML ───────────────────────────────────────────
def build_fin_table(hist_annual):
    rows  = hist_annual['rows']
    years = hist_annual['years']

    DISPLAY = [
        ('Revenue',          'Revenue (B$)'),
        ('Gross Profit',     'Gross Profit (B$)'),
        ('Gross Margin',     'Gross Margin (%)'),
        ('Operating Income', 'Operating Income (B$)'),
        ('Op Margin',        'Op Margin (%)'),
        ('Net Income',       'Net Income (B$)'),
        ('Net Margin',       'Net Margin (%)'),
        ('EPS (Diluted)',    'EPS (Diluted, $)'),
        ('BPS',              'Book Value/Share ($)'),
        ('Op CF',            'Operating CF (B$)'),
        ('Capex',            'Capex (B$)'),
        ('FCF',              'Free Cash Flow (B$)'),
        ('FCF Margin',       'FCF Margin (%)'),
        ('PER',              'P/E (trailing)'),
        ('PBR',              'P/B (trailing)'),
        ('PSR',              'P/S (trailing)'),
    ]

    header = ('<thead><tr><th>Item</th>'
              + ''.join(f'<th>{y}</th>' for y in years)
              + '</tr></thead>')
    body = '<tbody>'
    for key, label in DISPLAY:
        if key not in rows:
            continue
        vals = rows[key]
        if all(v is None for v in vals):
            continue
        cells = ''
        for v in vals:
            if v is None:
                cells += '<td>-</td>'
            else:
                cells += f'<td>{v}</td>'
        body += f'<tr><td>{label}</td>{cells}</tr>'
    body += '</tbody>'
    return f'<table class="financial-table">{header}{body}</table>'

# ── 7. 메인 분석 함수 ────────────────────────────────────────────
def analyze_us_stock(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()

    raw = fetch_yf_data(ticker_symbol)
    if raw is None:
        return {"error": f"'{ticker_symbol}' not found on Yahoo Finance."}

    name = (raw['info'].get('shortName') or raw['info'].get('longName') or ticker_symbol)
    if not raw['info'].get('currentPrice') and not raw['info'].get('regularMarketPrice'):
        return {"error": f"No price data for '{ticker_symbol}'. Check the ticker symbol."}

    try:
        hist    = parse_annual_history(raw)
        if hist is None:
            return {"error": "Failed to parse financial history."}

        est     = get_estimates(raw)
        dcf     = calc_dcf(hist, est)
        band    = calc_band(hist, est)
        fin_tbl = build_fin_table(hist)

        return {
            "ticker":  ticker_symbol,
            "name":    name,
            "sector":  est.get('sector', ''),
            "industry": est.get('industry', ''),
            "price":   f"${est['price']:.2f}" if est.get('price') else "N/A",
            "w52_high": est.get('w52_high'),
            "w52_low":  est.get('w52_low'),
            "market_cap": est.get('market_cap'),
            "beta":    est.get('beta'),
            "div_yield": round(est['div_yield'] * 100, 2) if est.get('div_yield') else None,
            "div_rate":  est.get('div_rate'),
            "roe_pct":   est.get('roe_pct'),
            "roa_pct":   est.get('roa_pct'),
            "de_ratio":  est.get('de_ratio'),
            "raw_table": fin_tbl,
            "est":       est,
            "dcf":       dcf,
            "band":      band,
        }
    except Exception as e:
        import traceback
        return {"error": f"Analysis error: {traceback.format_exc()}"}

# ── Flask 라우트 ─────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    ticker = ""
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()
        if ticker:
            result = analyze_us_stock(ticker)
    return render_template('index.html', result=result, ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True)
