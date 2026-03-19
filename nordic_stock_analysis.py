"""
Nordic Stock Market Analysis & Price Prediction
=================================================
Author: Rupesh Jha
Description:
    Mathematical analysis of Nordic stock market data including
    time series analysis, volatility modelling, correlation analysis,
    Monte Carlo simulation and price prediction using statistical methods.

    Demonstrates: BSc Mathematics foundations applied to real financial data
    Tools: Python, NumPy, SciPy, Pandas, Matplotlib, Scikit-learn

Note:
    Stock data simulated to match realistic Nordic market characteristics.
    Methodology applicable to real market data via Yahoo Finance API (yfinance).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

ACCENT  = '#1F4E79'
COLORS  = ['#1F4E79', '#2E75B6', '#70AD47', '#FFC000', '#FF0000', '#7030A0']
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SIMULATE REALISTIC NORDIC STOCK DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Nordic Stock Market Analysis & Price Prediction")
print("=" * 65)
print("\nPART 1: Generating realistic Nordic market data...")

# Nordic blue chip companies
STOCKS = {
    'NOKIA.HE':   {'start': 4.50,  'drift': 0.0003, 'vol': 0.022, 'sector': 'Technology'},
    'KONE.HE':    {'start': 52.00, 'drift': 0.0005, 'vol': 0.015, 'sector': 'Industrial'},
    'NORDEA.HE':  {'start': 10.50, 'drift': 0.0004, 'vol': 0.018, 'sector': 'Finance'},
    'SAMPO.HE':   {'start': 38.00, 'drift': 0.0006, 'vol': 0.014, 'sector': 'Finance'},
    'NESTE.HE':   {'start': 28.00, 'drift': 0.0007, 'vol': 0.020, 'sector': 'Energy'},
    'WARTSILA.HE':{'start': 12.00, 'drift': 0.0004, 'vol': 0.019, 'sector': 'Industrial'},
}

# Generate 5 years of daily trading data
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='B')  # Business days only
n_days = len(dates)

price_data = {}
for ticker, params in STOCKS.items():
    # Geometric Brownian Motion — standard financial price model
    # dS = μS dt + σS dW  (where dW is Wiener process)
    daily_returns = np.random.normal(
        loc=params['drift'],
        scale=params['vol'],
        size=n_days
    )
    # Add COVID crash March 2020 and recovery
    covid_start = (dates >= '2020-02-20') & (dates <= '2020-03-23')
    covid_recovery = (dates >= '2020-03-24') & (dates <= '2020-06-30')
    daily_returns[covid_start]    -= 0.025
    daily_returns[covid_recovery] += 0.015

    # Cumulative price path
    prices = params['start'] * np.exp(np.cumsum(daily_returns))
    price_data[ticker] = prices

prices_df = pd.DataFrame(price_data, index=dates)
print(f"  Generated {n_days} trading days for {len(STOCKS)} Nordic stocks")
print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
print(f"  Stocks: {', '.join(STOCKS.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — RETURNS & STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 2: Statistical Analysis...")

# Daily log returns — mathematically preferred over simple returns
# Log returns are additive and normally distributed
log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

# Summary statistics
stats_summary = pd.DataFrame({
    'Mean Daily Return (%)':  log_returns.mean() * 100,
    'Volatility (Ann. %)':    log_returns.std() * np.sqrt(252) * 100,
    'Sharpe Ratio':           (log_returns.mean() / log_returns.std()) * np.sqrt(252),
    'Skewness':               log_returns.skew(),
    'Kurtosis':               log_returns.kurtosis(),
    'Max Drawdown (%)':       ((prices_df / prices_df.cummax()) - 1).min() * 100,
}).round(3)

print("\nStock Performance Summary:")
print(stats_summary.to_string())

# Normality test — Jarque-Bera
print("\nNormality Tests (Jarque-Bera):")
for col in log_returns.columns:
    jb_stat, jb_p = stats.jarque_bera(log_returns[col])
    normal = "Normal" if jb_p > 0.05 else "Non-normal (fat tails)"
    print(f"  {col:15s}: JB={jb_stat:.2f}, p={jb_p:.4f} → {normal}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 3: Correlation Analysis...")

correlation_matrix = log_returns.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3).to_string())

# Find highest and lowest correlations
corr_pairs = []
tickers = list(STOCKS.keys())
for i in range(len(tickers)):
    for j in range(i+1, len(tickers)):
        corr_pairs.append({
            'Pair': f"{tickers[i]} / {tickers[j]}",
            'Correlation': correlation_matrix.loc[tickers[i], tickers[j]]
        })
corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
print(f"\nHighest correlation: {corr_df.iloc[0]['Pair']} ({corr_df.iloc[0]['Correlation']:.3f})")
print(f"Lowest correlation:  {corr_df.iloc[-1]['Pair']} ({corr_df.iloc[-1]['Correlation']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 4: Technical Indicators...")

# Focus on NOKIA for detailed analysis
nokia = prices_df['NOKIA.HE'].copy()

# Moving averages
nokia_df = pd.DataFrame({'Price': nokia})
nokia_df['MA20']  = nokia.rolling(window=20).mean()   # 1 month
nokia_df['MA50']  = nokia.rolling(window=50).mean()   # 2.5 months
nokia_df['MA200'] = nokia.rolling(window=200).mean()  # 1 year

# Bollinger Bands — 2 standard deviations
nokia_df['BB_mid']   = nokia.rolling(20).mean()
nokia_df['BB_upper'] = nokia_df['BB_mid'] + 2 * nokia.rolling(20).std()
nokia_df['BB_lower'] = nokia_df['BB_mid'] - 2 * nokia.rolling(20).std()

# RSI — Relative Strength Index
delta = nokia.diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs    = gain / loss
nokia_df['RSI'] = 100 - (100 / (1 + rs))

# Volatility — rolling 30 day
nokia_df['Volatility_30d'] = log_returns['NOKIA.HE'].rolling(30).std() * np.sqrt(252) * 100

print(f"  Current price:      €{nokia.iloc[-1]:.2f}")
print(f"  MA20:               €{nokia_df['MA20'].iloc[-1]:.2f}")
print(f"  MA50:               €{nokia_df['MA50'].iloc[-1]:.2f}")
print(f"  MA200:              €{nokia_df['MA200'].iloc[-1]:.2f}")
print(f"  Current RSI:        {nokia_df['RSI'].iloc[-1]:.1f}")
print(f"  30d Volatility:     {nokia_df['Volatility_30d'].iloc[-1]:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — VALUE AT RISK (VaR)
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 5: Risk Analysis — Value at Risk (VaR)...")

# VaR — how much could we lose in worst X% of days?
confidence_levels = [0.90, 0.95, 0.99]
investment = 10000  # €10,000 investment

print(f"\nVaR Analysis — €{investment:,} investment:")
print(f"{'Stock':<15} {'VaR 90%':>10} {'VaR 95%':>10} {'VaR 99%':>10}")
print("-" * 50)

var_results = {}
for ticker in STOCKS.keys():
    returns = log_returns[ticker].dropna()
    vars = {}
    for cl in confidence_levels:
        var = np.percentile(returns, (1-cl)*100) * investment
        vars[cl] = var
    var_results[ticker] = vars
    print(f"{ticker:<15} {vars[0.90]:>9.0f}€ {vars[0.95]:>9.0f}€ {vars[0.99]:>9.0f}€")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 6: Monte Carlo Simulation...")

# Simulate 1000 possible future price paths for Nokia
n_simulations = 1000
n_forecast    = 252  # 1 year of trading days
last_price    = nokia.iloc[-1]
daily_vol     = log_returns['NOKIA.HE'].std()
daily_drift   = log_returns['NOKIA.HE'].mean()

simulations = np.zeros((n_forecast, n_simulations))
for sim in range(n_simulations):
    random_returns = np.random.normal(daily_drift, daily_vol, n_forecast)
    price_path     = last_price * np.exp(np.cumsum(random_returns))
    simulations[:, sim] = price_path

# Results
final_prices    = simulations[-1, :]
mean_final      = np.mean(final_prices)
median_final    = np.median(final_prices)
percentile_5    = np.percentile(final_prices, 5)
percentile_95   = np.percentile(final_prices, 95)

print(f"\n  Monte Carlo Results — Nokia 1-Year Forecast ({n_simulations:,} simulations):")
print(f"  Current price:      €{last_price:.2f}")
print(f"  Mean forecast:      €{mean_final:.2f}")
print(f"  Median forecast:    €{median_final:.2f}")
print(f"  5th percentile:     €{percentile_5:.2f} (bear case)")
print(f"  95th percentile:    €{percentile_95:.2f} (bull case)")
print(f"  Probability > current price: {(final_prices > last_price).mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — PRICE PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 7: Price Prediction Model...")

# Feature engineering for ML model
nokia_ml = nokia_df.copy().dropna()
nokia_ml['Return_1d']  = log_returns['NOKIA.HE']
nokia_ml['Return_5d']  = log_returns['NOKIA.HE'].rolling(5).sum()
nokia_ml['Return_20d'] = log_returns['NOKIA.HE'].rolling(20).sum()
nokia_ml['Vol_20d']    = log_returns['NOKIA.HE'].rolling(20).std()
nokia_ml['Target']     = nokia_ml['Price'].shift(-5)  # Predict 5 days ahead
nokia_ml = nokia_ml.dropna()

features = ['MA20', 'MA50', 'RSI', 'Return_1d', 'Return_5d', 'Vol_20d']
X = nokia_ml[features]
y = nokia_ml['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model  = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n  Linear Regression — 5-Day Price Prediction:")
print(f"  RMSE:     €{rmse:.3f}")
print(f"  R² Score: {r2:.3f}")
print(f"\n  Feature Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"    {feat:<15}: {coef:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — PORTFOLIO OPTIMISATION (Markowitz)
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 8: Portfolio Optimisation (Markowitz Mean-Variance)...")

# Generate random portfolios and find efficient frontier
n_portfolios = 5000
returns_annual  = log_returns.mean() * 252
cov_matrix      = log_returns.cov() * 252

portfolio_returns = []
portfolio_vols    = []
portfolio_sharpes = []
portfolio_weights = []

for _ in range(n_portfolios):
    weights = np.random.random(len(STOCKS))
    weights /= weights.sum()
    
    p_return = np.dot(weights, returns_annual)
    p_vol    = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_sharpe = p_return / p_vol
    
    portfolio_returns.append(p_return)
    portfolio_vols.append(p_vol)
    portfolio_sharpes.append(p_sharpe)
    portfolio_weights.append(weights)

portfolio_returns  = np.array(portfolio_returns)
portfolio_vols     = np.array(portfolio_vols)
portfolio_sharpes  = np.array(portfolio_sharpes)

# Maximum Sharpe Ratio portfolio
max_sharpe_idx = np.argmax(portfolio_sharpes)
max_sharpe_weights = portfolio_weights[max_sharpe_idx]

print(f"\n  Optimal Portfolio (Maximum Sharpe Ratio):")
print(f"  Expected Annual Return: {portfolio_returns[max_sharpe_idx]*100:.1f}%")
print(f"  Expected Volatility:    {portfolio_vols[max_sharpe_idx]*100:.1f}%")
print(f"  Sharpe Ratio:           {portfolio_sharpes[max_sharpe_idx]:.2f}")
print(f"\n  Optimal Weights:")
for ticker, weight in zip(STOCKS.keys(), max_sharpe_weights):
    print(f"    {ticker:<15}: {weight*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 9: Generating dashboard...")

fig = plt.figure(figsize=(22, 28))
fig.suptitle('Nordic Stock Market Analysis & Price Prediction\nRupesh Jha — Mathematical Finance Portfolio',
             fontsize=16, fontweight='bold', color=ACCENT, y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Normalised price performance ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
normalised = prices_df / prices_df.iloc[0] * 100
for i, col in enumerate(normalised.columns):
    ax1.plot(normalised.index, normalised[col], label=col.replace('.HE',''),
             color=COLORS[i], linewidth=1.5)
ax1.axvline(pd.Timestamp('2020-03-23'), color='red', linestyle='--', alpha=0.5, label='COVID Low')
ax1.set_title('Normalised Price Performance (Base=100)', fontweight='bold')
ax1.set_ylabel('Indexed Price')
ax1.legend(fontsize=8, ncol=2)

# ── Plot 2: Correlation heatmap ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
import seaborn as sns
mask = np.zeros_like(correlation_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax2, vmin=-1, vmax=1, center=0,
            xticklabels=[t.replace('.HE','') for t in STOCKS.keys()],
            yticklabels=[t.replace('.HE','') for t in STOCKS.keys()],
            linewidths=0.5)
ax2.set_title('Return Correlation Matrix', fontweight='bold')

# ── Plot 3: Nokia technical analysis ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
recent = nokia_df[nokia_df.index >= nokia_df.index[-1] - pd.DateOffset(days=365)]
ax3.plot(recent.index, recent['Price'], color=ACCENT, linewidth=1.5, label='Price')
ax3.plot(recent.index, recent['MA20'],  color='orange', linewidth=1, label='MA20', linestyle='--')
ax3.plot(recent.index, recent['MA50'],  color='green',  linewidth=1, label='MA50', linestyle='--')
ax3.fill_between(recent.index, recent['BB_upper'], recent['BB_lower'],
                 alpha=0.1, color='blue', label='Bollinger Bands')
ax3.set_title('Nokia — Technical Analysis (Last 12 Months)', fontweight='bold')
ax3.set_ylabel('Price (€)')
ax3.legend(fontsize=8)

# ── Plot 4: Monte Carlo simulation ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
future_dates = pd.date_range(start=nokia.index[-1], periods=n_forecast+1, freq='B')[1:]
for sim in range(0, n_simulations, 10):  # Plot every 10th simulation
    ax4.plot(future_dates, simulations[:, sim], alpha=0.05, color='blue', linewidth=0.5)
ax4.plot(future_dates, np.percentile(simulations, 5,  axis=1), 'r--', linewidth=2, label='5th percentile')
ax4.plot(future_dates, np.percentile(simulations, 50, axis=1), 'g-',  linewidth=2, label='Median')
ax4.plot(future_dates, np.percentile(simulations, 95, axis=1), 'g--', linewidth=2, label='95th percentile')
ax4.axhline(last_price, color='black', linestyle=':', linewidth=1.5, label=f'Current €{last_price:.2f}')
ax4.set_title(f'Monte Carlo Simulation — Nokia 1-Year Forecast\n({n_simulations:,} simulations)', fontweight='bold')
ax4.set_ylabel('Price (€)')
ax4.legend(fontsize=8)

# ── Plot 5: Return distribution with normal fit ───────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
nokia_returns = log_returns['NOKIA.HE'].dropna()
ax5.hist(nokia_returns, bins=80, density=True, alpha=0.6, color=ACCENT, label='Actual Returns')
x = np.linspace(nokia_returns.min(), nokia_returns.max(), 200)
mu, sigma = nokia_returns.mean(), nokia_returns.std()
ax5.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution Fit')
ax5.set_title('Nokia — Return Distribution vs Normal\n(Fat tails indicate higher risk than normal assumes)', fontweight='bold')
ax5.set_xlabel('Daily Log Return')
ax5.set_ylabel('Density')
ax5.legend()

# ── Plot 6: Efficient frontier ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
scatter = ax6.scatter(portfolio_vols*100, portfolio_returns*100,
                      c=portfolio_sharpes, cmap='viridis', alpha=0.3, s=5)
ax6.scatter(portfolio_vols[max_sharpe_idx]*100, portfolio_returns[max_sharpe_idx]*100,
            color='red', s=200, zorder=5, marker='*', label='Max Sharpe Portfolio')
plt.colorbar(scatter, ax=ax6, label='Sharpe Ratio')
ax6.set_title('Efficient Frontier — Portfolio Optimisation\n(Markowitz Mean-Variance)', fontweight='bold')
ax6.set_xlabel('Portfolio Volatility (%)')
ax6.set_ylabel('Expected Return (%)')
ax6.legend()

# ── Plot 7: Prediction model ──────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
ax7.scatter(y_test, y_pred, alpha=0.3, color=ACCENT, s=10)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax7.set_title(f'Price Prediction Model — Actual vs Predicted\nR²={r2:.3f}, RMSE=€{rmse:.3f}', fontweight='bold')
ax7.set_xlabel('Actual Price (€)')
ax7.set_ylabel('Predicted Price (€)')
ax7.legend()

# ── Plot 8: VaR comparison ────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
tickers_short = [t.replace('.HE','') for t in STOCKS.keys()]
var_95 = [abs(var_results[t][0.95]) for t in STOCKS.keys()]
bars = ax8.bar(tickers_short, var_95, color=COLORS, edgecolor='white')
ax8.set_title(f'Value at Risk (95%) — €{investment:,} Investment\n(Maximum expected daily loss)', fontweight='bold')
ax8.set_ylabel('VaR Amount (€)')
for bar, val in zip(bars, var_95):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'€{val:.0f}', ha='center', fontweight='bold', fontsize=9)

plt.savefig('/mnt/user-data/outputs/nordic_stock_dashboard.png', dpi=150, bbox_inches='tight')
print("Dashboard saved successfully!")
print("\n" + "=" * 65)
print("Analysis Complete!")
print("=" * 65)
print("\nFiles generated:")
print("  - nordic_stock_dashboard.png")
print("  - nordic_stock_analysis.py")
