"""
Marketplace Validation V2: Reads from local CSV files
======================================================
Run download_fred_data.py FIRST, then run this script.

Usage:
  python download_fred_data.py          # downloads data
  python scrap_marketplace_analysis_v2.py  # runs analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings, os, glob

warnings.filterwarnings('ignore')

OUTPUT_DIR = './output_marketplace'
DATA_DIR = './fred_data'

SERIES = {
    'WPU1012':      'Steel Scrap PPI',
    'WPU101':       'Primary Iron & Steel PPI',
    'WPU1025':      'Aluminum Mill Shapes PPI',
    'WPU102504':    'Nickel Mill Shapes PPI',
    'WPU10230101':  'Copper Scrap PPI',
    'WPU10250201':  'Aluminum Scrap PPI',
    'PCOPPUSDM':    'Global Copper Price',
    'PNICKUSDM':    'Global Nickel Price',
    'PIORECRUSDM':  'Global Iron Ore Price',
    'PALUMUSDM':    'Global Aluminum Price',
    'USCONS':       'Construction Employment',
    'IPMAN':        'Manufacturing IP',
    'PERMIT':       'Building Permits',
    'MANEMP':       'Manufacturing Employment',
}

def load_csvs():
    """Load all CSVs from fred_data/ directory."""
    frames = {}
    for sid, name in SERIES.items():
        path = os.path.join(DATA_DIR, f"{sid}.csv")
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        df = pd.read_csv(path, parse_dates=['DATE'], index_col='DATE')
        df.columns = ['value']
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        frames[name] = df['value']
        print(f"  {sid:16s} → {name} ({len(df)} obs)")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.resample('MS').first()
    df = df.ffill(limit=2)
    return df


def build_and_backtest(df, scrap_col='Steel Scrap PPI', target_col='Primary Iron & Steel PPI',
                        forecast_horizon=1, train_window=24):
    scrap_chg = df[scrap_col].pct_change() * 100
    target_chg = df[target_col].pct_change() * 100
    target_future = target_chg.shift(-forecast_horizon)

    results = []
    n = len(df)

    for i in range(train_window, n - forecast_horizon):
        train_x = scrap_chg.iloc[i - train_window:i].values
        train_y = target_future.iloc[i - train_window:i].values
        mask = ~(np.isnan(train_x) | np.isnan(train_y))
        if mask.sum() < 10:
            continue
        tx, ty = train_x[mask], train_y[mask]
        slope, intercept, r_value, p_value, std_err = stats.linregress(tx, ty)

        current_scrap = scrap_chg.iloc[i]
        if np.isnan(current_scrap):
            continue
        predicted_chg = intercept + slope * current_scrap
        actual_chg = target_future.iloc[i]
        if np.isnan(actual_chg):
            continue

        current_price = df[target_col].iloc[i]
        actual_next_price = df[target_col].iloc[i + forecast_horizon]
        predicted_price = current_price * (1 + predicted_chg / 100)
        naive_price = current_price

        results.append({
            'date': df.index[i],
            'current_price': current_price,
            'actual_next_price': actual_next_price,
            'predicted_price': predicted_price,
            'naive_price': naive_price,
            'predicted_chg': predicted_chg,
            'actual_chg': actual_chg,
            'naive_chg': 0,
            'scrap_chg': current_scrap,
            'model_r2': r_value**2,
        })

    return pd.DataFrame(results)


def compute_metrics(results):
    actual = results['actual_next_price'].values
    predicted = results['predicted_price'].values
    naive = results['naive_price'].values

    rmse_model = np.sqrt(np.mean((predicted - actual) ** 2))
    rmse_naive = np.sqrt(np.mean((naive - actual) ** 2))
    mae_model = np.mean(np.abs(predicted - actual))
    mae_naive = np.mean(np.abs(naive - actual))

    actual_dir = np.sign(results['actual_chg'].values)
    pred_dir = np.sign(results['predicted_chg'].values)
    dir_accuracy = np.mean(actual_dir == pred_dir) * 100

    rmse_improvement = (1 - rmse_model / rmse_naive) * 100
    mae_improvement = (1 - mae_model / mae_naive) * 100

    return {
        'rmse_model': rmse_model, 'rmse_naive': rmse_naive,
        'rmse_improvement_pct': rmse_improvement,
        'mae_model': mae_model, 'mae_naive': mae_naive,
        'mae_improvement_pct': mae_improvement,
        'directional_accuracy': dir_accuracy,
        'n_predictions': len(results),
    }


def setup():
    plt.rcParams.update({
        'figure.facecolor': '#FFF', 'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#CCC', 'axes.grid': True, 'grid.color': '#E0E0E0',
        'grid.linewidth': 0.5, 'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight'})


def chart_backtest(results, metrics, target_name, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1], sharex=True)
    dates = pd.to_datetime(results['date'])

    ax1.plot(dates, results['actual_next_price'], color='#333', lw=2,
             label='Actual (next month)', zorder=5)
    ax1.plot(dates, results['predicted_price'], color='#D62828', lw=1.5,
             ls='--', label='Scrap-based forecast', alpha=0.9)
    ax1.plot(dates, results['naive_price'], color='#CCC', lw=1,
             ls=':', label='Naive (no change)', alpha=0.7)
    ax1.fill_between(dates, results['predicted_price'], results['actual_next_price'],
                      alpha=0.1, color='#D62828')
    ax1.set_ylabel(f'{target_name} (Index)', fontsize=11)
    ax1.set_title(f'Backtest: Scrap-Based Forward Price Model for {target_name}',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.text(0.98, 0.95,
             f'Model RMSE: {metrics["rmse_model"]:.2f}\n'
             f'Naive RMSE: {metrics["rmse_naive"]:.2f}\n'
             f'Improvement: {metrics["rmse_improvement_pct"]:.1f}%\n'
             f'Direction accuracy: {metrics["directional_accuracy"]:.1f}%\n'
             f'(vs 50% random baseline)\n'
             f'n = {metrics["n_predictions"]} predictions',
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#CCC', alpha=0.9))

    model_errors = np.abs(results['predicted_price'] - results['actual_next_price'])
    naive_errors = np.abs(results['naive_price'] - results['actual_next_price'])
    ax2.bar(dates, naive_errors, width=25, color='#CCC', alpha=0.6,
            label=f'Naive error (MAE={metrics["mae_naive"]:.2f})')
    ax2.bar(dates, model_errors, width=25, color='#D62828', alpha=0.6,
            label=f'Model error (MAE={metrics["mae_model"]:.2f})')
    ax2.set_ylabel('Absolute Error', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.text(0.5, -0.03,
             f'Rolling 24-month training window. {forecast_horizon}-month ahead forecast. '
             f'Sources: BLS PPI via FRED.',
             ha='center', fontsize=8, color='#888', style='italic')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {path}")


def chart_directional(results, target_name, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    actual_dir = np.sign(results['actual_chg'])
    pred_dir = np.sign(results['predicted_chg'])
    correct = actual_dir == pred_dir

    ax.scatter(results['scrap_chg'][correct], results['actual_chg'][correct],
               color='#2A9D8F', alpha=0.6, s=50, label='Correct prediction',
               edgecolor='white', lw=0.5, zorder=3)
    ax.scatter(results['scrap_chg'][~correct], results['actual_chg'][~correct],
               color='#D62828', alpha=0.6, s=50, label='Incorrect prediction',
               edgecolor='white', lw=0.5, zorder=3)
    ax.axhline(0, color='#999', lw=0.8, ls='--')
    ax.axvline(0, color='#999', lw=0.8, ls='--')

    ax.text(0.95, 0.95, 'Scrap up, Target up\n(correct)', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#2A9D8F', fontweight='bold')
    ax.text(0.05, 0.05, 'Scrap down, Target down\n(correct)', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, color='#2A9D8F', fontweight='bold')
    ax.text(0.95, 0.05, 'Scrap up, Target down\n(wrong)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#D62828')
    ax.text(0.05, 0.95, 'Scrap down, Target up\n(wrong)', transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color='#D62828')

    dir_acc = correct.mean() * 100
    ax.set_xlabel('Steel Scrap PPI (MoM % change this month)', fontsize=11)
    ax.set_ylabel(f'{target_name} (MoM % change next month)', fontsize=11)
    ax.set_title(f'Directional Accuracy: {dir_acc:.1f}%\n'
                 f'Does Scrap Direction Predict {target_name} Direction?',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    fig.text(0.5, -0.03, 'Each dot = one month. Sources: BLS PPI via FRED.',
             ha='center', fontsize=8, color='#888', style='italic')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {path}")


def chart_value(results, target_name, path, monthly_tons=1000):
    cum_savings = []
    running = 0
    for _, row in results.iterrows():
        price_chg = row['actual_next_price'] - row['current_price']
        if row['predicted_chg'] > 0 and price_chg > 0:
            savings = price_chg * monthly_tons
        elif row['predicted_chg'] < 0 and price_chg < 0:
            savings = abs(price_chg) * monthly_tons
        elif row['predicted_chg'] > 0 and price_chg < 0:
            savings = price_chg * monthly_tons
        else:
            savings = -price_chg * monthly_tons
        running += savings
        cum_savings.append({'date': row['date'], 'monthly': savings, 'cumulative': running})

    sdf = pd.DataFrame(cum_savings)
    dates = pd.to_datetime(sdf['date'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)

    ax1.fill_between(dates, 0, sdf['cumulative'], where=sdf['cumulative'] >= 0,
                      color='#2A9D8F', alpha=0.3)
    ax1.fill_between(dates, 0, sdf['cumulative'], where=sdf['cumulative'] < 0,
                      color='#D62828', alpha=0.3)
    ax1.plot(dates, sdf['cumulative'], color='#333', lw=2)
    ax1.axhline(0, color='#999', lw=0.5)
    final = sdf['cumulative'].iloc[-1]
    ax1.set_ylabel('Cumulative Savings ($)', fontsize=11)
    ax1.set_title(f'Value of Scrap Foresight: {monthly_tons:,}-Ton/Month Buyer of {target_name}\n'
                  f'Cumulative savings: ${final:,.0f}',
                  fontweight='bold', fontsize=13)

    colors = ['#2A9D8F' if s >= 0 else '#D62828' for s in sdf['monthly']]
    ax2.bar(dates, sdf['monthly'], width=25, color=colors, alpha=0.6)
    ax2.axhline(0, color='#999', lw=0.5)
    ax2.set_ylabel('Monthly ($)', fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    avg = sdf['monthly'].mean()
    fig.text(0.5, -0.03,
             f'Avg monthly impact: ${avg:,.0f}. Sources: BLS PPI via FRED. Simplified backtest.',
             ha='center', fontsize=8, color='#888', style='italic')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup()

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: {DATA_DIR}/ not found. Run download_fred_data.py first!")
        return

    print("Loading CSV data...")
    df = load_csvs()
    print(f"\nData: {df.index[0]:%Y-%m} → {df.index[-1]:%Y-%m}, {len(df)} monthly obs\n")

    targets = [
        ('Primary Iron & Steel PPI', 1),
        ('Aluminum Mill Shapes PPI', 2),
        ('Nickel Mill Shapes PPI', 1),
        ('Global Copper Price', 1),
        ('Global Iron Ore Price', 1),
        ('Manufacturing IP', 2),
    ]

    for target_name, horizon in targets:
        if target_name not in df.columns:
            print(f"  Skipping {target_name} — not in data")
            continue

        print(f"\n{'='*60}")
        print(f"BACKTEST: Scrap → {target_name} (horizon={horizon}mo)")
        print(f"{'='*60}")

        results = build_and_backtest(df, target_col=target_name,
                                      forecast_horizon=horizon, train_window=24)

        if len(results) < 10:
            print(f"  Only {len(results)} predictions — need more data.")
            continue

        metrics = compute_metrics(results)
        print(f"  RMSE model:    {metrics['rmse_model']:.3f}")
        print(f"  RMSE naive:    {metrics['rmse_naive']:.3f}")
        print(f"  Improvement:   {metrics['rmse_improvement_pct']:.1f}%")
        print(f"  Direction acc: {metrics['directional_accuracy']:.1f}%")
        print(f"  n predictions: {metrics['n_predictions']}")

        safe = target_name.replace(' ', '_').replace('&', 'and')
        chart_backtest(results, metrics, target_name,
                       os.path.join(OUTPUT_DIR, f'backtest_{safe}.png'))
        chart_directional(results, target_name,
                          os.path.join(OUTPUT_DIR, f'directional_{safe}.png'))
        chart_value(results, target_name,
                    os.path.join(OUTPUT_DIR, f'value_{safe}.png'))

    print(f"\nAll saved to {OUTPUT_DIR}/")
    print("Done.")

# needed for chart_backtest
forecast_horizon = 1

if __name__ == '__main__':
    main()
