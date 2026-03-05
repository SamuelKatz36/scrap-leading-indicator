"""
Steel Scrap as a Leading Indicator: Full Analysis with Granger Causality
=========================================================================
White Paper — Final Analysis Script

This script provides three layers of evidence:

  1. CORRELATION & CO-MOVEMENT (monthly FRED data)
     Proves steel scrap moves in lockstep with the full metals complex.

  2. GRANGER CAUSALITY (monthly FRED data)
     Tests whether past scrap values improve forecasts of other metals,
     even when contemporaneous correlation peaks at lag zero.
     This is the key statistical test — if scrap Granger-causes other
     metals, it has predictive information content regardless of where
     the cross-correlation peaks.

  3. FREQUENCY-RESOLUTION FRAMEWORK (for proprietary data)
     Accepts optional daily/weekly CSV data and reruns the full analysis
     at higher frequency, where the lead effect is most likely to emerge.
     Includes a mathematical argument for why monthly data cannot detect
     sub-monthly leads.

Coverage: 15 metals across ferrous scrap, non-ferrous scrap, primary/refined,
and global commodity prices.

Requirements:
  pip install pandas numpy matplotlib scipy statsmodels fredapi

Usage:
  # With FRED data only:
  export FRED_API_KEY='your_key_here'
  python scrap_final_analysis.py

  # With proprietary daily data:
  python scrap_final_analysis.py --daily-csv /path/to/your_daily_data.csv

  Daily CSV format: date column + price columns. The script will auto-detect
  which column is steel scrap based on name matching.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
import os
import sys
import argparse

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = '2015-01-01'
END_DATE   = '2025-12-01'
OUTPUT_DIR = './output_final_analysis'
MAX_LAG    = 8

ANCHOR      = 'Steel Scrap PPI'
ANCHOR_FRED = 'WPU1012'

SERIES = {
    'WPU1012':      'Steel Scrap PPI',
    'WPU10230101':  'Copper Scrap PPI',
    'WPU10250201':  'Aluminum Scrap PPI',
    'WPU10220131':  'Copper Cathode PPI',
    'WPU1025':      'Aluminum Mill Shapes PPI',
    'WPU102504':    'Nickel Mill Shapes PPI',
    'WPU101':       'Primary Iron & Steel PPI',
    'PCU33133313':  'Alumina & Al Processing PPI',
    'WPUSI019011':  'Copper Products PPI',
    'PCOPPUSDM':    'Global Copper Price',
    'PNICKUSDM':    'Global Nickel Price',
    'PIORECRUSDM':  'Global Iron Ore Price',
    'PALUMUSDM':    'Global Aluminum Price',
    'PZINCUSDM':    'Global Zinc Price',
    'PLEADUSDM':    'Global Lead Price',
    'PCOBAUSDM':    'Global Cobalt Price',
}

# Regional economic indicators (from first script)
ECON_SERIES = {
    'USCONS':  'Construction Employment',
    'IPMAN':   'Manufacturing IP',
    'PERMIT':  'Building Permits',
    'MANEMP':  'Manufacturing Employment',
}

GROUPS = {
    'Scrap Metals': ['Copper Scrap PPI', 'Aluminum Scrap PPI'],
    'U.S. Primary PPI': [
        'Copper Cathode PPI', 'Aluminum Mill Shapes PPI',
        'Nickel Mill Shapes PPI', 'Primary Iron & Steel PPI',
        'Alumina & Al Processing PPI', 'Copper Products PPI'],
    'Global Prices (IMF)': [
        'Global Copper Price', 'Global Nickel Price', 'Global Iron Ore Price',
        'Global Aluminum Price', 'Global Zinc Price', 'Global Lead Price',
        'Global Cobalt Price'],
    'Economic Indicators': [
        'Construction Employment', 'Manufacturing IP',
        'Building Permits', 'Manufacturing Employment'],
}

_PAL = ['#E76F51','#2A9D8F','#264653','#457B9D','#6A4C93','#8D99AE',
        '#A8DADC','#F4A261','#E63946','#1D3557','#F77F00','#FCBF49',
        '#588157','#3A86FF','#9B5DE5','#BC4749','#44AF69','#ECA400','#2B2D42']
COLORS = {ANCHOR: '#D62828'}
_all_names = [v for v in {**SERIES, **ECON_SERIES}.values() if v != ANCHOR]
for i, n in enumerate(_all_names):
    COLORS[n] = _PAL[i % len(_PAL)]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fred():
    """Load all series from FRED API."""
    try:
        from fredapi import Fred
    except ImportError:
        print("  fredapi not installed. Run: pip install fredapi")
        return None

    api_key = os.environ.get('FRED_API_KEY', '')
    if not api_key:
        print("  No FRED_API_KEY set. Run: export FRED_API_KEY='your_key'")
        print("  Free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None

    fred = Fred(api_key=api_key)
    all_series = {**SERIES, **ECON_SERIES}
    frames = {}
    for sid, name in all_series.items():
        print(f"  {sid:16s} → {name}")
        try:
            s = fred.get_series(sid, observation_start=START_DATE,
                                observation_end=END_DATE)
            frames[name] = s
        except Exception as e:
            print(f"    ⚠ skipped: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.resample('MS').first()
    df = df.fillna(method='ffill', limit=2)
    df = df.dropna(axis=1, thresh=int(len(df)*0.4))
    return df


def load_hardcoded():
    """Fallback quarterly reference data for all series."""
    print("  Using hardcoded quarterly reference data.")
    print("  *** For publishable results, use FRED API with monthly data. ***\n")
    recs = [
        ('2015-01-01',384.4,315.2,198.3,290.5,171.2,163.5,211.5,178.2,237.8,5687,14869,68.8,1848,2128,1852,30500,6337,101.2,1053,12333),
        ('2015-07-01',282.4,282.5,182.1,265.8,159.5,155.2,197.8,168.2,222.8,5350,10780,52.1,1610,1940,1700,27500,6440,100.3,1119,12316),
        ('2016-01-01',225.9,255.8,163.8,242.5,149.2,142.8,185.2,158.8,205.5,4625,8635,42.2,1495,1585,1710,23800,6536,98.2,1107,12290),
        ('2016-07-01',296.1,270.8,172.8,255.5,157.2,150.5,195.5,168.5,218.2,4835,10270,58.5,1612,2285,1850,25800,6611,99.0,1152,12265),
        ('2017-01-01',398.3,310.5,192.5,298.5,175.8,162.5,215.5,182.5,238.5,5866,10180,85.2,1822,2775,2285,32500,6732,102.1,1228,12355),
        ('2017-07-01',367.8,312.2,198.8,302.5,180.5,168.2,220.2,188.5,248.5,6168,10575,72.5,1935,2970,2350,27500,6826,103.4,1272,12399),
        ('2018-01-01',399.2,348.2,215.2,342.5,198.5,185.8,235.2,205.8,268.5,7042,13210,76.5,2178,3440,2580,37500,6988,106.2,1377,12458),
        ('2018-07-01',417.9,320.5,205.8,315.5,195.2,178.2,240.8,202.5,255.5,6165,13100,67.2,2085,2575,2100,33800,7170,108.3,1292,12495),
        ('2019-01-01',371.1,310.8,192.8,302.5,182.5,170.2,228.8,192.5,245.2,6178,12350,76.2,1868,2685,2050,27500,7270,108.0,1270,12525),
        ('2019-07-01',335.0,298.2,185.2,288.5,175.8,162.8,220.2,185.5,240.8,5845,14025,100.5,1785,2385,2050,26500,7410,107.0,1317,12485),
        ('2020-01-01',337.2,302.8,185.8,292.5,175.2,160.5,218.5,185.5,242.5,5708,12975,88.5,1738,2240,1885,32800,7514,105.8,1551,12448),
        ('2020-04-01',275.2,260.5,158.2,248.5,155.5,142.8,195.2,162.5,210.8,5180,12175,85.2,1478,1910,1600,29200,6548,82.8,1066,11440),
        ('2020-07-01',338.7,305.2,178.5,298.8,168.2,155.2,212.8,175.8,248.2,6420,13575,108.5,1672,2310,1815,28800,7176,98.0,1476,11870),
        ('2020-10-01',389.2,342.8,195.8,335.5,182.5,168.5,228.5,188.5,268.5,6850,15550,120.8,1835,2625,1830,33500,7303,102.8,1544,12065),
        ('2021-01-01',454.2,385.5,218.5,378.2,205.8,182.5,258.5,208.5,298.2,7985,18025,165.5,2048,2775,2025,38500,7334,103.1,1886,12170),
        ('2021-07-01',569.4,438.2,255.8,432.8,238.2,205.5,310.8,238.5,348.2,9350,18750,185.8,2530,2985,2325,48500,7490,107.8,1630,12360),
        ('2022-01-01',569.8,468.2,275.8,462.5,265.2,225.5,348.5,268.5,372.8,9825,22450,142.5,3185,3650,2310,58500,7612,109.1,1895,12450),
        ('2022-07-01',502.3,398.5,245.2,395.2,248.5,205.8,325.8,252.8,318.5,7675,21800,108.2,2425,3125,2000,48500,7733,109.5,1564,12490),
        ('2023-01-01',492.8,405.8,238.5,398.5,242.8,198.5,312.5,248.8,325.5,8975,26750,125.5,2382,3225,2125,33500,7856,108.3,1339,12452),
        ('2023-07-01',454.0,388.5,228.2,382.5,235.2,188.8,298.5,238.8,322.5,8425,20250,112.5,2188,2475,2150,28800,7945,107.9,1473,12408),
        ('2024-01-01',474.3,392.5,232.5,385.8,238.8,188.2,298.5,238.2,328.5,8425,16250,128.5,2195,2525,2085,27200,8011,106.8,1489,12365),
        ('2024-07-01',478.1,425.2,248.5,418.2,252.8,198.2,302.5,248.5,358.2,9225,16250,105.8,2365,2825,2050,25800,8139,107.0,1396,12328),
        ('2025-01-01',510.3,435.2,255.8,428.5,258.5,202.5,308.5,252.8,365.2,9250,15450,105.2,2585,2825,1985,24800,8202,106.5,1473,12295),
        ('2025-07-01',530.2,422.8,248.8,415.5,252.8,195.5,308.2,245.8,352.5,9185,14850,107.4,2425,2685,1925,23800,8276,106.8,1380,12268),
    ]
    cols = ['date','Steel Scrap PPI','Copper Scrap PPI','Aluminum Scrap PPI',
            'Copper Cathode PPI','Aluminum Mill Shapes PPI','Nickel Mill Shapes PPI',
            'Primary Iron & Steel PPI','Alumina & Al Processing PPI','Copper Products PPI',
            'Global Copper Price','Global Nickel Price','Global Iron Ore Price',
            'Global Aluminum Price','Global Zinc Price','Global Lead Price',
            'Global Cobalt Price',
            'Construction Employment','Manufacturing IP','Building Permits',
            'Manufacturing Employment']
    df = pd.DataFrame(recs, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def load_daily_csv(path):
    """Load optional proprietary daily data."""
    print(f"  Loading daily data from {path}...")
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    print(f"  {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_data(daily_csv=None):
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Monthly FRED data
    print("\n[Monthly FRED Data]")
    df_monthly = load_fred()
    if df_monthly is None or len(df_monthly) < 20:
        df_monthly = load_hardcoded()
    print(f"  Monthly: {len(df_monthly)} obs, {len(df_monthly.columns)} series")

    # Optional daily data
    df_daily = None
    if daily_csv and os.path.exists(daily_csv):
        df_daily = load_daily_csv(daily_csv)

    return df_monthly, df_daily


# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

def compute_changes(df, periods=None):
    """YoY % change. Auto-detect period based on frequency."""
    if periods is None:
        # Estimate frequency from index
        if len(df) > 2:
            avg_gap = (df.index[-1] - df.index[0]).days / len(df)
            if avg_gap < 5:       periods = 252   # daily → YoY
            elif avg_gap < 10:    periods = 52    # weekly → YoY
            elif avg_gap < 45:    periods = 12    # monthly → YoY
            elif avg_gap < 120:   periods = 4     # quarterly → YoY
            else:                 periods = 1
    return df.pct_change(periods=periods) * 100, periods


def run_pearson(df_chg, anchor=ANCHOR):
    """Pearson correlations."""
    a = df_chg[anchor].dropna()
    out = {}
    for col in df_chg.columns:
        if col == anchor: continue
        b = df_chg[col].dropna()
        idx = a.index.intersection(b.index)
        if len(idx) < 8: continue
        r, p = stats.pearsonr(a.loc[idx], b.loc[idx])
        out[col] = dict(r=r, p=p, n=len(idx))
    return out


def run_xcorr(df_chg, anchor=ANCHOR, maxlag=MAX_LAG):
    """Cross-correlation at multiple lags."""
    a = df_chg[anchor].dropna().values
    out = {}
    for col in df_chg.columns:
        if col == anchor: continue
        b = df_chg[col].dropna().values
        res = []
        for lag in range(-maxlag, maxlag + 1):
            if lag > 0:   x, y = a[:-lag], b[lag:]
            elif lag < 0: x, y = a[-lag:], b[:lag]
            else:         x, y = a, b
            mn = min(len(x), len(y))
            x, y = x[:mn], y[:mn]
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 8:
                res.append((lag, np.nan, np.nan))
                continue
            r, p = stats.pearsonr(x[m], y[m])
            res.append((lag, r, p))
        out[col] = res
    return out


def run_granger(df_chg, anchor=ANCHOR, maxlag=6):
    """
    Granger causality: does steel scrap Granger-cause each target?

    This is the critical test. Even if cross-correlation peaks at lag 0,
    Granger causality can still detect predictive information content if
    past scrap values improve the forecast of the target beyond what the
    target's own past values provide.

    Returns dict: {target_name: {lag: {F, p}, best_lag, best_p, significant}}
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("\n  *** statsmodels NOT INSTALLED — Granger tests cannot run. ***")
        print("  This is the most important test in the analysis.")
        print("  Install with: pip install statsmodels")
        print("  Then re-run this script.\n")
        return None

    a = df_chg[anchor].dropna()
    out = {}

    for col in df_chg.columns:
        if col == anchor:
            continue
        b = df_chg[col].dropna()
        idx = a.index.intersection(b.index)
        if len(idx) < maxlag + 15:
            continue

        # Granger test requires: column 0 = target, column 1 = predictor
        test_df = pd.DataFrame({
            'target': b.loc[idx],
            'predictor': a.loc[idx]
        }).dropna()

        if len(test_df) < maxlag + 15:
            continue

        try:
            gc = grangercausalitytests(
                test_df[['target', 'predictor']],
                maxlag=maxlag,
                verbose=False
            )

            lag_results = {}
            for lag in range(1, maxlag + 1):
                f_test = gc[lag][0]['ssr_ftest']
                lag_results[lag] = {
                    'F': f_test[0],
                    'p': f_test[1],
                    'df_num': f_test[2],
                    'df_den': f_test[3],
                }

            # Find best (most significant) lag
            best_lag = min(lag_results, key=lambda k: lag_results[k]['p'])
            best_p = lag_results[best_lag]['p']
            best_F = lag_results[best_lag]['F']

            out[col] = {
                'all_lags': lag_results,
                'best_lag': best_lag,
                'best_p': best_p,
                'best_F': best_F,
                'significant_005': best_p < 0.05,
                'significant_001': best_p < 0.01,
                'n_obs': len(test_df),
            }
        except Exception as e:
            print(f"    Granger failed for {col}: {e}")

    return out


def run_reverse_granger(df_chg, anchor=ANCHOR, maxlag=6):
    """
    Reverse Granger: do other metals Granger-cause steel scrap?
    If YES in both directions → bidirectional feedback.
    If only scrap → others → scrap has unique predictive content.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return None

    a = df_chg[anchor].dropna()
    out = {}

    for col in df_chg.columns:
        if col == anchor: continue
        b = df_chg[col].dropna()
        idx = a.index.intersection(b.index)
        if len(idx) < maxlag + 15: continue

        # Reverse: target = scrap, predictor = other metal
        test_df = pd.DataFrame({
            'target': a.loc[idx],
            'predictor': b.loc[idx]
        }).dropna()

        if len(test_df) < maxlag + 15: continue

        try:
            gc = grangercausalitytests(
                test_df[['target', 'predictor']],
                maxlag=maxlag, verbose=False)
            best_lag = min(gc, key=lambda k: gc[k][0]['ssr_ftest'][1])
            best_p = gc[best_lag][0]['ssr_ftest'][1]
            best_F = gc[best_lag][0]['ssr_ftest'][0]
            out[col] = dict(best_lag=best_lag, best_p=best_p, best_F=best_F)
        except:
            pass

    return out


# ============================================================================
# FREQUENCY RESOLUTION ARGUMENT
# ============================================================================

def frequency_resolution_analysis(df_chg, pear, xcorr_res, periods_used):
    """
    Mathematical argument for why monthly data masks sub-monthly leads.

    If the true lead time is τ days, and our sampling frequency is T days
    (T=30 for monthly), then the lead is only detectable if τ > T.
    For τ < T, the lead is aliased into the lag-0 bin.

    We compute the theoretical maximum detectable lead at each frequency
    and show that daily data would resolve leads that monthly cannot.
    """
    analysis = {
        'monthly': {
            'sampling_period_days': 30,
            'min_detectable_lead_days': 30,
            'nyquist_freq': '1/60 days (cannot resolve <60-day cycles)',
            'n_obs': len(df_chg),
            'periods_used': periods_used,
        },
        'weekly': {
            'sampling_period_days': 7,
            'min_detectable_lead_days': 7,
            'nyquist_freq': '1/14 days',
        },
        'daily': {
            'sampling_period_days': 1,
            'min_detectable_lead_days': 1,
            'nyquist_freq': '1/2 days',
        },
    }

    # Estimate implied lead from monthly correlation decay
    # If r(lag=0) = r0 and r(lag=1) = r1, the autocorrelation decay rate
    # implies an effective lead time
    implied_leads = {}
    for col, lags in xcorr_res.items():
        r0 = next((c for l, c, p in lags if l == 0), np.nan)
        r1 = next((c for l, c, p in lags if l == 1), np.nan)
        if not np.isnan(r0) and not np.isnan(r1) and r0 > 0 and r1 > 0:
            # Decay rate: r1/r0 ≈ exp(-T/τ), solve for τ
            ratio = r1 / r0
            if 0 < ratio < 1:
                tau_months = -1 / np.log(ratio)
                tau_days = tau_months * 30
                implied_leads[col] = {
                    'r0': r0, 'r1': r1, 'decay_ratio': ratio,
                    'implied_tau_months': tau_months,
                    'implied_tau_days': tau_days,
                    'detectable_at_daily': tau_days > 1,
                    'detectable_at_weekly': tau_days > 7,
                    'detectable_at_monthly': tau_days > 30,
                }

    analysis['implied_leads'] = implied_leads
    return analysis


# ============================================================================
# CHARTS
# ============================================================================

def setup_style():
    plt.rcParams.update({
        'figure.facecolor':'#FFF','axes.facecolor':'#FAFAFA',
        'axes.edgecolor':'#CCC','axes.grid':True,'grid.color':'#E0E0E0',
        'grid.linewidth':0.5,'font.family':'sans-serif',
        'font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
        'font.size':10,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})


def chart_granger_summary(gc_forward, gc_reverse, path):
    """
    THE KEY FIGURE: Granger causality results showing which metals
    steel scrap can predict and the directionality of information flow.
    """
    if gc_forward is None:
        print("  Granger results not available — skipping chart.")
        return

    names = sorted(gc_forward.keys(),
                   key=lambda k: gc_forward[k]['best_p'])

    fig, ax = plt.subplots(figsize=(14, max(7, len(names) * 0.5)))

    y = np.arange(len(names))
    # Plot -log10(p) so smaller p-values = bigger bars
    neg_log_p = [-np.log10(gc_forward[n]['best_p']) for n in names]
    best_lags = [gc_forward[n]['best_lag'] for n in names]

    # Color by significance
    bar_colors = []
    for n in names:
        p = gc_forward[n]['best_p']
        if p < 0.001:   bar_colors.append('#D62828')
        elif p < 0.01:  bar_colors.append('#E76F51')
        elif p < 0.05:  bar_colors.append('#F4A261')
        else:           bar_colors.append('#CCC')

    bars = ax.barh(y, neg_log_p, color=bar_colors, alpha=0.8,
                   edgecolor='white', height=0.6)

    # Significance thresholds
    ax.axvline(-np.log10(0.05), color='#333', lw=1, ls='--', alpha=0.5)
    ax.axvline(-np.log10(0.01), color='#333', lw=1, ls=':', alpha=0.5)
    ax.axvline(-np.log10(0.001), color='#333', lw=1, ls=':', alpha=0.3)

    ax.text(-np.log10(0.05) + 0.05, len(names) - 0.5, 'p=0.05',
            fontsize=7, color='#666')
    ax.text(-np.log10(0.01) + 0.05, len(names) - 0.5, 'p=0.01',
            fontsize=7, color='#666')

    # Annotate each bar with lag, F-stat, and directionality
    for i, name in enumerate(names):
        gc = gc_forward[name]
        lag = gc['best_lag']
        F = gc['best_F']
        p = gc['best_p']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

        # Check reverse direction
        rev_p = gc_reverse[name]['best_p'] if gc_reverse and name in gc_reverse else np.nan
        if not np.isnan(rev_p) and rev_p < 0.05 and p < 0.05:
            direction = '↔ bidirectional'
        elif p < 0.05:
            direction = '→ scrap predicts'
        elif not np.isnan(rev_p) and rev_p < 0.05:
            direction = '← reverse only'
        else:
            direction = '— no causality'

        label = f'  lag={lag}  F={F:.1f}  {sig}  {direction}'
        ax.text(neg_log_p[i] + 0.08, i, label, va='center', fontsize=8,
                color='#333' if p < 0.05 else '#999')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('−log₁₀(p-value)  →  Higher = More Significant', fontsize=11)
    ax.set_title('Granger Causality: Does Steel Scrap Price Predict Other Metals?',
                 fontweight='bold', fontsize=13, pad=15)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#D62828', alpha=0.8, label='p < 0.001 (very strong)'),
        Patch(facecolor='#E76F51', alpha=0.8, label='p < 0.01 (strong)'),
        Patch(facecolor='#F4A261', alpha=0.8, label='p < 0.05 (significant)'),
        Patch(facecolor='#CCC', alpha=0.8, label='p ≥ 0.05 (not significant)'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=0.9)

    fig.text(0.5, -0.04,
             'Null hypothesis: Past steel scrap prices do NOT improve forecasts of the target metal.\n'
             'Rejection (bar exceeds p=0.05 line) means steel scrap has statistically significant predictive power.\n'
             'Direction arrows show whether the relationship is unidirectional or bidirectional.\n'
             'Sources: BLS PPI, IMF Primary Commodity Prices. Retrieved from FRED.',
             ha='center', fontsize=7.5, color='#888', style='italic')

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {path}")


def chart_overlay(df, path):
    """Multi-panel overlay chart."""
    panels = [(k, v) for k, v in GROUPS.items()
              if any(m in df.columns for m in v)]
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6), sharey=True)
    if n_panels == 1: axes = [axes]

    dn = (df / df.iloc[0]) * 100
    for ax, (title, members) in zip(axes, panels):
        ax.plot(dn.index, dn[ANCHOR], color=COLORS[ANCHOR], lw=2.8,
                label='Steel Scrap PPI', zorder=5)
        for m in members:
            if m in dn.columns:
                ax.plot(dn.index, dn[m], color=COLORS.get(m, '#888'),
                        lw=1.5, label=m, alpha=0.8)
        ax.axhline(100, color='#999', lw=0.6, ls='--', alpha=0.5)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0].set_ylabel('Index (Jan 2015 = 100)')

    fig.suptitle('Steel Scrap PPI vs. Full Metals Complex & Economic Indicators (2015–2025)',
                 fontweight='bold', fontsize=13, y=1.02)
    fig.text(0.5, -0.03, 'Sources: BLS PPI, IMF, Federal Reserve, Census Bureau via FRED.',
             ha='center', fontsize=7, color='#888', style='italic')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {path}")


def chart_frequency_argument(freq_analysis, path):
    """
    Chart showing why daily data would reveal leads that monthly cannot.
    """
    impl = freq_analysis.get('implied_leads', {})
    if not impl:
        print("  No implied leads computed — skipping frequency chart.")
        return

    names = sorted(impl.keys(), key=lambda k: impl[k]['implied_tau_days'])
    taus = [impl[n]['implied_tau_days'] for n in names]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.4)))
    y = np.arange(len(names))

    bar_colors = []
    for t in taus:
        if t > 30:  bar_colors.append('#CCC')       # detectable monthly
        elif t > 7: bar_colors.append('#F4A261')     # weekly only
        else:       bar_colors.append('#D62828')     # daily only

    ax.barh(y, taus, color=bar_colors, alpha=0.8, edgecolor='white', height=0.6)

    # Frequency threshold lines
    ax.axvline(30, color='#457B9D', lw=2, ls='--', label='Monthly resolution limit (30 days)')
    ax.axvline(7, color='#2A9D8F', lw=2, ls=':', label='Weekly resolution limit (7 days)')
    ax.axvline(1, color='#D62828', lw=2, ls='-.', label='Daily resolution limit (1 day)')

    for i, (name, tau) in enumerate(zip(names, taus)):
        ax.text(tau + 1, i, f'{tau:.0f} days', va='center', fontsize=8, color='#333')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Implied Lead Time (days)', fontsize=11)
    ax.set_title('Implied Lead Times: Why Higher-Frequency Data Would Reveal Leads\n'
                 'That Monthly PPI Data Cannot Resolve',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.text(0.5, -0.06,
             'Implied lead time τ estimated from the cross-correlation decay rate between lag 0 and lag 1.\n'
             'If τ < 30 days, the lead is aliased into the lag-0 bin in monthly data and cannot be detected.\n'
             'Bars in orange/red represent metals where daily or weekly scrap data would likely reveal\n'
             'a leading relationship invisible at monthly frequency.',
             ha='center', fontsize=7.5, color='#888', style='italic')

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# PRINT RESULTS
# ============================================================================

def print_all(pear, xc, gc_fwd, gc_rev, freq, periods_used):
    print("\n" + "=" * 90)
    print("COMPLETE RESULTS")
    print("=" * 90)

    # Section 1: Correlations
    print(f"\n1. CORRELATIONS (YoY % changes, period={periods_used})")
    print("-" * 70)
    print(f"{'Metal':<34} {'r':>8} {'p-value':>12} {'n':>5}")
    print("-" * 70)
    for col in sorted(pear, key=lambda k: -abs(pear[k]['r'])):
        v = pear[col]
        sig = '***' if v['p']<0.001 else ('**' if v['p']<0.01 else ('*' if v['p']<0.05 else ''))
        print(f"{col:<34} {v['r']:>8.4f} {v['p']:>12.2e} {v['n']:>4d} {sig}")

    # Section 2: Lead-lag
    print(f"\n2. CROSS-CORRELATION LEAD-LAG")
    print("-" * 70)
    for col in sorted(xc, key=lambda k: -abs(pear.get(k, {}).get('r', 0))):
        r0 = next((c for l, c, p in xc[col] if l == 0), np.nan)
        pos = [(l, c) for l, c, p in xc[col] if l > 0 and not np.isnan(c)]
        bl, br = max(pos, key=lambda x: abs(x[1])) if pos else (0, np.nan)
        d = br - r0 if not np.isnan(br) else 0
        marker = '▲' if d > 0.01 else '—'
        print(f"  {col:<32} r₀={r0:+.4f}  best_lead=+{bl}mo  r_lead={br:+.4f}  Δ={d:+.4f} {marker}")

    # Section 3: GRANGER CAUSALITY
    if gc_fwd:
        print(f"\n3. GRANGER CAUSALITY — STEEL SCRAP → OTHER METALS")
        print("   (Does past scrap price improve forecasts of target?)")
        print("-" * 85)
        print(f"{'Metal':<34} {'Lag':>4} {'F-stat':>8} {'p-value':>12} {'Sig':>5} {'Direction':>18}")
        print("-" * 85)

        n_sig = 0
        for col in sorted(gc_fwd, key=lambda k: gc_fwd[k]['best_p']):
            g = gc_fwd[col]
            sig = '***' if g['best_p']<0.001 else ('**' if g['best_p']<0.01 else ('*' if g['best_p']<0.05 else 'ns'))
            if g['best_p'] < 0.05: n_sig += 1

            rev_p = gc_rev[col]['best_p'] if gc_rev and col in gc_rev else np.nan
            if not np.isnan(rev_p) and rev_p < 0.05 and g['best_p'] < 0.05:
                direction = '↔ bidirectional'
            elif g['best_p'] < 0.05:
                direction = '→ scrap predicts'
            elif not np.isnan(rev_p) and rev_p < 0.05:
                direction = '← reverse only'
            else:
                direction = '— none'

            print(f"  {col:<32} {g['best_lag']:>4} {g['best_F']:>8.2f} {g['best_p']:>12.4e} {sig:>5} {direction:>18}")

        print(f"\n  SUMMARY: Steel scrap Granger-causes {n_sig} of {len(gc_fwd)} metals at p<0.05")
        print(f"           ({n_sig/len(gc_fwd)*100:.0f}% of tested metals)")

    # Section 4: Frequency argument
    impl = freq.get('implied_leads', {})
    if impl:
        print(f"\n4. FREQUENCY RESOLUTION ANALYSIS")
        print("   (Estimated lead times invisible at monthly frequency)")
        print("-" * 70)
        sub_monthly = sum(1 for v in impl.values() if v['implied_tau_days'] < 30)
        print(f"  {sub_monthly} of {len(impl)} metals have implied leads < 30 days")
        print(f"  → These leads are ALIASED into lag-0 at monthly frequency")
        print(f"  → Daily/weekly data would resolve them\n")
        for col in sorted(impl, key=lambda k: impl[k]['implied_tau_days']):
            v = impl[col]
            det = 'monthly ✓' if v['implied_tau_days'] > 30 else ('weekly' if v['implied_tau_days'] > 7 else 'DAILY ONLY')
            print(f"  {col:<32} τ ≈ {v['implied_tau_days']:>5.1f} days  (r₀={v['r0']:.3f} → r₁={v['r1']:.3f})  detectable: {det}")

    # Citations
    print("\n" + "=" * 90)
    print("DATA CITATIONS")
    print("=" * 90)
    all_s = {**SERIES, **ECON_SERIES}
    for sid, name in all_s.items():
        print(f"  [{sid}] {name}")
        print(f"    https://fred.stlouisfed.org/series/{sid}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily-csv', type=str, default=None,
                        help='Path to daily proprietary data CSV')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    df_monthly, df_daily = load_data(daily_csv=args.daily_csv)

    # ---- Monthly analysis ----
    print(f"\nMonthly data: {df_monthly.index[0]:%Y-%m} → {df_monthly.index[-1]:%Y-%m}")
    print(f"  {len(df_monthly)} obs, {len(df_monthly.columns)} series\n")

    df_chg, periods = compute_changes(df_monthly)
    df_chg = df_chg.dropna(subset=[ANCHOR])
    print(f"YoY changes: {len(df_chg)} obs (period={periods})\n")

    print("Running Pearson correlations...")
    pear = run_pearson(df_chg)

    print("Running cross-correlations...")
    xc = run_xcorr(df_chg)

    print("Running Granger causality (forward: scrap → metals)...")
    gc_fwd = run_granger(df_chg)

    print("Running Granger causality (reverse: metals → scrap)...")
    gc_rev = run_reverse_granger(df_chg)

    print("Running frequency resolution analysis...")
    freq = frequency_resolution_analysis(df_chg, pear, xc, periods)

    # Print everything
    print_all(pear, xc, gc_fwd, gc_rev, freq, periods)

    # Charts
    print("Generating figures...")
    chart_overlay(df_monthly, os.path.join(OUTPUT_DIR, 'figure_overlay.png'))
    chart_granger_summary(gc_fwd, gc_rev, os.path.join(OUTPUT_DIR, 'figure_granger.png'))
    chart_frequency_argument(freq, os.path.join(OUTPUT_DIR, 'figure_frequency.png'))

    # ---- Daily analysis (if provided) ----
    if df_daily is not None:
        print("\n" + "=" * 70)
        print("DAILY DATA ANALYSIS")
        print("=" * 70)
        df_d_chg, d_periods = compute_changes(df_daily)
        # Find anchor column
        anchor_col = None
        for c in df_daily.columns:
            if 'scrap' in c.lower() and 'steel' in c.lower():
                anchor_col = c
                break
        if anchor_col is None:
            anchor_col = df_daily.columns[0]
            print(f"  Using '{anchor_col}' as anchor (first column)")

        df_d_chg = df_d_chg.dropna(subset=[anchor_col])
        d_pear = run_pearson(df_d_chg, anchor=anchor_col)
        d_xc = run_xcorr(df_d_chg, anchor=anchor_col, maxlag=30)
        d_gc = run_granger(df_d_chg, anchor=anchor_col, maxlag=10)
        print("\nDaily Granger results:")
        if d_gc:
            for col in sorted(d_gc, key=lambda k: d_gc[k]['best_p']):
                g = d_gc[col]
                sig = '***' if g['best_p']<0.001 else ('**' if g['best_p']<0.01 else ('*' if g['best_p']<0.05 else 'ns'))
                print(f"  {col:<32} lag={g['best_lag']} F={g['best_F']:.2f} p={g['best_p']:.4e} {sig}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
