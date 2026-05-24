from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import math
import re
import os
import requests
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

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

    if not info or (not info.get('shortName') and not info.get('longName')):
        return None

    income   = t.income_stmt
    cashflow = t.cashflow
    balance  = t.balance_sheet

    return {
        'ticker':     ticker_symbol.upper(),
        'info':       info,
        'income':     income,
        'cashflow':   cashflow,
        'balance':    balance,
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

    # ── 연도 라벨 수집 (income_stmt 기준, 오래된 연도→최신 순 정렬) ──
    years = []
    col_dates = []
    if income is not None and not income.empty:
        pairs = []
        for col in income.columns:
            try:
                yr = pd.Timestamp(col).year
                pairs.append((yr, col))
            except:
                pass
        # 오름차순 (FY2022 → FY2023 → ... → FY2025)
        pairs.sort(key=lambda x: x[0])
        for yr, col in pairs:
            years.append(f"FY{yr}")
            col_dates.append(col)

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
    # equity_raw: 원달러(e.g. 7.37e10), shares_bs: 주 수(e.g. 1.47e10)
    # BPS($/주) = equity(원달러) / shares(주수)
    equity_raw = get_row_first(balance,
                'Stockholders Equity', 'Common Stock Equity')
    equity = [fmt_b(v) for v in equity_raw]   # 테이블 표시용 B달러
    shares_bs = get_row_first(balance, 'Ordinary Shares Number')
    bps = []
    for i in range(len(years)):
        eq_orig = safe_float(equity_raw[i]) if i < len(equity_raw) else None
        sh = safe_float(shares_bs[i]) if i < len(shares_bs) and shares_bs[i] is not None else None
        # equity는 원달러, shares는 주 수 → BPS = $/주
        if eq_orig is not None and sh and sh > 0:
            bps.append(round(eq_orig / sh, 2))   # 달러/주
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

# ── 3-a. Stockanalysis 장기 컨센서스 스크래핑 ────────────────────
def get_stockanalysis_estimates(ticker):
    """
    stockanalysis.com/stocks/{ticker}/forecast/ 에서
    FY2026~2027 EPS·Revenue 컨센서스(avg/high/low)와 목표주가 수집.
    반환 예:
    {
      'eps': {'FY2026': {'avg':8.94,'high':9.46,'low':8.36,'n_analysts':50}, ...},
      'rev': {'FY2026': {'avg':486.8,'high':509.5,'low':445.8}, ...},
      'target': {'low':215,'avg':308.07,'median':310,'high':400},
      'fwd_pe': {'FY2026': 34.53, 'FY2027': 31.31},
    }
    실패 시 None 반환.
    """
    try:
        url = f'https://stockanalysis.com/stocks/{ticker.lower()}/forecast/'
        hdrs = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/124.0.0.0 Safari/537.36'),
            'Accept-Language': 'en-US,en;q=0.9',
        }
        resp = requests.get(url, headers=hdrs, timeout=12)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        if len(tables) < 4:
            return None

        def parse_val(s):
            """'486.8B' → 486.8, '8.94' → 8.94, 'Upgrade'/'Pro' → None"""
            s = s.strip()
            if not s or s in ('Upgrade', 'Pro', '-', 'N/A', ''):
                return None
            s = s.replace(',', '')
            mul = 1
            if s.endswith('T'):
                mul = 1e3; s = s[:-1]
            elif s.endswith('B'):
                mul = 1; s = s[:-1]
            elif s.endswith('M'):
                mul = 1e-3; s = s[:-1]
            try:
                return round(float(s) * mul, 2)
            except:
                return None

        result = {'eps': {}, 'rev': {}, 'target': {}, 'fwd_pe': {}}

        # ── table[0]: 목표주가 ──
        t0 = tables[0]
        rows_t0 = t0.find_all('tr')
        if len(rows_t0) >= 2:
            hdrs_t0 = [th.get_text(strip=True) for th in rows_t0[0].find_all(['th','td'])]
            vals_t0 = [td.get_text(strip=True) for td in rows_t0[1].find_all(['th','td'])]
            tgt_map = {'Low':'low','Average':'avg','Median':'median','High':'high'}
            for h, v in zip(hdrs_t0, vals_t0):
                k = tgt_map.get(h)
                if k:
                    pv = parse_val(v.replace('$','').replace('%','').strip())
                    if pv: result['target'][k] = pv

        # ── table[3]: 연간 종합 (EPS·Revenue·Forward PE·애널리스트 수) ──
        t3 = tables[3]
        rows_t3 = t3.find_all('tr')
        if rows_t3:
            year_hdrs = [td.get_text(strip=True) for td in rows_t3[0].find_all(['th','td'])]
            row_map = {}
            for row in rows_t3[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(['th','td'])]
                if cells:
                    row_map[cells[0]] = cells[1:]

            for col_i, yr_label in enumerate(year_hdrs[1:], 0):
                # FY2026, FY2027 만 처리
                m = re.search(r'(\d{4})', yr_label)
                if not m:
                    continue
                yr_int = int(m.group(1))
                if yr_int < 2025:
                    continue
                fy_key = f'FY{yr_int}'

                eps_val  = parse_val(row_map.get('EPS', [''] * 20)[col_i]) if 'EPS' in row_map else None
                rev_val  = parse_val(row_map.get('Revenue', [''] * 20)[col_i]) if 'Revenue' in row_map else None
                fpe_val  = parse_val(row_map.get('Forward PE', [''] * 20)[col_i]) if 'Forward PE' in row_map else None
                n_str    = row_map.get('No. Analysts', [''] * 20)[col_i] if 'No. Analysts' in row_map else ''
                try:
                    n_val = int(n_str) if n_str and n_str not in ('Upgrade','Pro','-') else None
                except:
                    n_val = None

                if eps_val:
                    if fy_key not in result['eps']:
                        result['eps'][fy_key] = {}
                    result['eps'][fy_key]['avg'] = eps_val
                    if n_val:
                        result['eps'][fy_key]['n_analysts'] = n_val
                if rev_val:
                    if fy_key not in result['rev']:
                        result['rev'][fy_key] = {}
                    result['rev'][fy_key]['avg'] = rev_val
                if fpe_val:
                    result['fwd_pe'][fy_key] = fpe_val

        # ── table[4]: Revenue 상세(high/avg/low) ──
        if len(tables) > 4:
            t4 = tables[4]
            rows_t4 = t4.find_all('tr')
            if rows_t4:
                yr_hdrs4 = [td.get_text(strip=True) for td in rows_t4[0].find_all(['th','td'])]
                for row in rows_t4[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all(['th','td'])]
                    if not cells: continue
                    kind = cells[0].strip().lower()  # 'high','avg','low'
                    if kind not in ('high','avg','low'): continue
                    for col_i, yr_label in enumerate(yr_hdrs4[1:], 0):
                        m = re.search(r'(\d{4})', yr_label)
                        if not m: continue
                        yr_int = int(m.group(1))
                        if yr_int < 2025: continue
                        fy_key = f'FY{yr_int}'
                        v = parse_val(cells[col_i + 1]) if col_i + 1 < len(cells) else None
                        if v:
                            if fy_key not in result['rev']:
                                result['rev'][fy_key] = {}
                            result['rev'][fy_key][kind] = v

        # ── table[6]: EPS 상세(high/avg/low) ──
        if len(tables) > 6:
            t6 = tables[6]
            rows_t6 = t6.find_all('tr')
            if rows_t6:
                yr_hdrs6 = [td.get_text(strip=True) for td in rows_t6[0].find_all(['th','td'])]
                for row in rows_t6[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all(['th','td'])]
                    if not cells: continue
                    kind = cells[0].strip().lower()
                    if kind not in ('high','avg','low'): continue
                    for col_i, yr_label in enumerate(yr_hdrs6[1:], 0):
                        m = re.search(r'(\d{4})', yr_label)
                        if not m: continue
                        yr_int = int(m.group(1))
                        if yr_int < 2025: continue
                        fy_key = f'FY{yr_int}'
                        v = parse_val(cells[col_i + 1]) if col_i + 1 < len(cells) else None
                        if v:
                            if fy_key not in result['eps']:
                                result['eps'][fy_key] = {}
                            result['eps'][fy_key][kind] = v

        return result if (result['eps'] or result['rev']) else None

    except Exception:
        return None


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

    # Stockanalysis 장기 컨센서스 (FY2026~2027)
    result['sa'] = get_stockanalysis_estimates(data['ticker'])

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

    # ── PER: EPS > 0 AND PER > 0 인 연도만 ──
    per_hist = [v for v, e in zip(rows['PER'], rows['EPS (Diluted)'])
                if v is not None and v > 0 and e is not None and e > 0]
    fwd_pe   = est.get('fwd_pe')
    fwd_eps  = est.get('fwd_eps') or est.get('curr_yr_eps')
    # Forward EPS도 양수여야 이론주가 산출
    if fwd_eps is not None and fwd_eps <= 0:
        fwd_eps = None
    if fwd_pe is not None and fwd_pe <= 0:
        fwd_pe = None
    results.append(make_band("P/E", per_hist, fwd_val=fwd_pe,
                              fwd_base=fwd_eps, base_label="Fwd EPS"))

    # ── PBR: BPS > 0 AND PBR > 0 인 연도만 ──
    pbr_hist = [v for v, b in zip(rows['PBR'], rows['BPS'])
                if v is not None and v > 0 and b is not None and b > 0]
    # 최신 BPS가 양수인 경우에만 trailing PBR 계산
    latest_bps = next((b for b in rows['BPS'] if b is not None and b > 0), None)
    trailing_pbr = round(price / latest_bps, 2) if price and latest_bps else None
    results.append(make_band("P/B", pbr_hist, fwd_val=trailing_pbr,
                              fwd_base=latest_bps,
                              base_label="BPS (latest)"))

    # ── PEG: EPS 양수 성장인 연도만 ──
    peg_hist = []
    eps_list = rows['EPS (Diluted)']
    for i in range(1, len(eps_list)):
        pe  = rows['PER'][i]
        e0  = eps_list[i-1]
        e1  = eps_list[i]
        if pe and pe > 0 and e0 and e0 > 0 and e1 and e1 > 0:
            g = (e1/e0 - 1) * 100
            if g > 0:
                peg_hist.append(round(pe / g, 2))
    fwd_peg = est.get('peg')
    if fwd_peg is not None and fwd_peg <= 0:
        fwd_peg = None
    results.append(make_band("PEG", peg_hist, fwd_val=fwd_peg, no_theory=True))

    # ── PSR: Revenue > 0 AND PSR > 0 인 연도만 ──
    psr_hist = [v for v, r in zip(rows['PSR'], rows['Revenue'])
                if v is not None and v > 0 and r is not None and r > 0]
    fwd_psr  = est.get('psr_trailing')
    if fwd_psr is not None and fwd_psr <= 0:
        fwd_psr = None
    results.append(make_band("P/S", psr_hist, fwd_val=fwd_psr,
                              base_label="Rev/Share"))

    return results

# ── 5. EPS 가치평가 3종 ──────────────────────────────────────────
def calc_us_valuations(hist_annual, estimates, r):
    """
    연도별 가치평가 3종 (한국 버전과 동일 로직, 단위는 달러):
      1. EPS/r   : 내재가치 = EPS / r
      2. RIM     : 내재가치 = BPS × (ROE / r)
                   ROE는 yfinance 소수 (e.g. 1.41 = 141%) → 그대로 사용
      3. Graham  : 내재가치 = EPS × (8.5 + ROE평균%) × (1 - r)
                   ROE평균 = trailing ROE (소수, 0~30% 클램프)
    r: 요구수익률 소수 (e.g. 0.13)
    """
    rows  = hist_annual['rows']
    years = list(hist_annual['years'])   # ['FY2022','FY2023',...]
    price = hist_annual['price']
    est   = estimates

    eps_list = rows.get('EPS (Diluted)', [])
    bps_list = rows.get('BPS', [])

    # Forward EPS / BPS 추가 (컨센서스)
    fwd_eps  = est.get('fwd_eps')           # 다음 12개월
    next_eps = est.get('next_yr_eps')       # +1y
    roe_raw  = est.get('roe')               # yfinance: 소수 (e.g. 1.41)

    # ── g: 전체 연도 ROE 평균 ───────────────────────────────────────
    # yfinance는 단일 trailing ROE만 제공 → 해당 값을 g로 사용
    # 단, 0~30% 클램프
    graham_g = None
    if roe_raw is not None and roe_raw > 0:
        graham_g = max(0.0, min(roe_raw, 0.30))   # 소수, 30% 상한

    # ── 연도별 행 구성 ──────────────────────────────────────────────
    result_rows = []

    def make_row(yr, eps, bps, roe, is_fwd=False):
        row = {'year': yr, 'is_fwd': is_fwd}

        # 1. EPS/r
        val_eps_r = round(eps / r, 2) if eps and eps > 0 and r and r > 0 else None
        gap_eps_r = round((val_eps_r - price) / price * 100, 1) \
                    if val_eps_r and price and price > 0 else None

        # 2. RIM: BPS × (ROE / r)
        val_rim = None; gap_rim = None
        if bps and bps > 0 and roe and roe > 0 and r and r > 0:
            val_rim = round(bps * (roe / r), 2)
            if price and price > 0:
                gap_rim = round((val_rim - price) / price * 100, 1)

        # 3. Graham: EPS × (8.5 + ROE평균%) × (1 - r)
        # graham_g는 소수(e.g. 0.30), ×100 해서 % 정수로 사용
        val_graham = None; gap_graham = None
        if eps and eps > 0 and graham_g is not None and r and r > 0:
            val_graham = round(eps * (8.5 + graham_g * 100) * (1 - r), 2)
            if price and price > 0:
                gap_graham = round((val_graham - price) / price * 100, 1)

        row.update({
            'eps':        round(eps, 2) if eps is not None else None,
            'bps':        round(bps, 2) if bps is not None else None,
            'roe':        round(roe * 100, 1) if roe is not None else None,  # % 표시용
            'val_eps_r':  f"${val_eps_r:,.2f}" if val_eps_r is not None else '-',
            'gap_eps_r':  gap_eps_r,
            'val_rim':    f"${val_rim:,.2f}" if val_rim is not None else '-',
            'gap_rim':    gap_rim,
            'val_graham': f"${val_graham:,.2f}" if val_graham is not None else '-',
            'gap_graham': gap_graham,
        })
        return row

    # 실적 연도
    for i, yr in enumerate(years):
        eps = eps_list[i] if i < len(eps_list) else None
        bps = bps_list[i] if i < len(bps_list) else None
        if eps is None and bps is None:
            continue
        result_rows.append(make_row(yr, eps, bps, roe_raw, is_fwd=False))

    # Forward EPS (컨센서스 추정)
    if fwd_eps is not None:
        result_rows.append(make_row('Fwd (12M)', fwd_eps, None, roe_raw, is_fwd=True))
    if next_eps is not None:
        result_rows.append(make_row('Next FY', next_eps, None, roe_raw, is_fwd=True))

    return {
        'r':        round(r * 100, 2),
        'graham_g': round(graham_g * 100, 1) if graham_g is not None else None,
        'rows':     result_rows,
    }


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

        est = get_estimates(raw)

        # ── 요구수익률 r 계산 (rf + beta × ERP) ──────────────────────
        try:
            tnx = yf.Ticker('^TNX')
            tnx_hist = tnx.history(period='5d')
            rf = float(tnx_hist['Close'].iloc[-1]) / 100 if not tnx_hist.empty else 0.044
        except:
            rf = 0.044
        beta  = safe_float(est.get('beta')) or 1.0
        r_val = rf + beta * 0.055

        band       = calc_band(hist, est)
        fin_tbl    = build_fin_table(hist)
        valuation  = calc_us_valuations(hist, est, r=r_val)

        return {
            "ticker":     ticker_symbol,
            "name":       name,
            "sector":     est.get('sector', ''),
            "industry":   est.get('industry', ''),
            "price":      f"${est['price']:.2f}" if est.get('price') else "N/A",
            "price_raw":  est.get('price'),
            "w52_high":   est.get('w52_high'),
            "w52_low":    est.get('w52_low'),
            "market_cap": est.get('market_cap'),
            "beta":       est.get('beta'),
            "div_yield":  round(est['div_yield'] * 100, 2) if est.get('div_yield') else None,
            "div_rate":   est.get('div_rate'),
            "roe_pct":    est.get('roe_pct'),
            "roa_pct":    est.get('roa_pct'),
            "de_ratio":   est.get('de_ratio'),
            "r_info": {
                "rf":   f"{rf*100:.2f}",
                "beta": f"{beta:.2f}",
                "r":    f"{r_val*100:.2f}",
            },
            "raw_table":  fin_tbl,
            "est":        est,
            "sa":         est.get('sa'),
            "band":       band,
            "valuation":  valuation,
        }
    except Exception as e:
        import traceback
        return {"error": f"Analysis error: {traceback.format_exc()}"}

# ── Flask 라우트 ─────────────────────────────────────────────────
@app.route('/health')
def health():
    return 'OK', 200

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    ticker = ""
    try:
        if request.method == 'POST':
            ticker = request.form.get('ticker', '').strip().upper()
            if ticker:
                result = analyze_us_stock(ticker)
        return render_template('index.html', result=result, ticker=ticker)
    except Exception:
        import traceback
        logger.exception("Unhandled error in index route")
        result = {"error": f"서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n{traceback.format_exc()}"}
        return render_template('index.html', result=result, ticker=ticker), 500

if __name__ == '__main__':
    app.run(debug=True)
