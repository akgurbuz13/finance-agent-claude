# Portfolio Advisor v3 — Comprehensive Architecture Review

> Generated: 2026-02-12
> Scope: Full codebase review — pipelines, agent architecture, mathematical models, data flows, storage, and production readiness assessment.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Daily Lifecycle — What Happens Every Day](#2-daily-lifecycle)
3. [Weekly Lifecycle](#3-weekly-lifecycle)
4. [The Pre-Compute Pipeline — Detailed Breakdown](#4-the-pre-compute-pipeline)
5. [How Prices Are Fetched](#5-how-prices-are-fetched)
6. [How Calculations Are Performed and Stored](#6-how-calculations-are-performed-and-stored)
7. [Mathematical and Financial Models — Rigorous Assessment](#7-mathematical-and-financial-models)
8. [Agent Architecture — Why 33 Tools for the Chat Agent](#8-agent-architecture)
9. [Per-Instrument vs Batch Processing — Current State and Recommendations](#9-per-instrument-vs-batch-processing)
10. [How Insights and Recommendations Are Stored](#10-how-insights-and-recommendations-are-stored)
11. [Production Readiness Assessment](#11-production-readiness-assessment)
12. [Identified Issues and Recommendations](#12-identified-issues-and-recommendations)

---

## 1. System Overview

The Portfolio Advisor is a multi-agent portfolio advisory system built on the **OpenAI Agents SDK**. It runs as a single Python process that simultaneously operates:

- **APScheduler 4.x** (`AsyncScheduler`) with **9 scheduled jobs** for automated analysis
- **Telegram bot** (python-telegram-bot) for user interaction via 15 commands + free-text chat
- **SQLite database** (aiosqlite) with 20 tables across 3 schema versions
- **10 LLM agents** with distinct roles, models, and tool sets
- **150+ computational tools** spanning technical analysis, quantitative finance, portfolio optimization, risk management, time series analysis, and macroeconomic indicators

The system does **NOT** place trades. It produces analysis, forecasts, and recommendations delivered via Telegram.

### Core Design Principles

1. **LLM as Synthesizer, Not Calculator** — All mathematical computation (technical indicators, quant models, risk metrics) is performed programmatically in Python. The LLM receives pre-computed results and synthesizes them into narratives, recommendations, and alerts.
2. **Cache-First Architecture** — The pre-compute pipeline runs 2-3x daily and stores all results in SQLite. Chat interactions and scheduled jobs read cached data first (0 LLM tokens), only delegating to live agents when cache is stale or the request is outside scope.
3. **Agents-as-Tools Hierarchy** — The chat agent has 29 direct tools + 4 agent-as-tool delegates. Specialist agents (Technical, Quantitative, Portfolio, Research) with their 60+ collective tools are hidden behind `.as_tool()` wrappers, reducing context dilution.

---

## 2. Daily Lifecycle

Here is the exact sequence of events that occurs every trading day, in UTC:

### 06:00 — Pre-Compute Morning (`precompute_job`)

**No LLM cost. Pure Python computation.**

1. Load user preferences and watchlist from DB
2. Batch-fetch 1 year of OHLCV data for the entire watchlist via `yf.download()` (single HTTP call for all equity tickers)
3. Fetch crypto data individually via CoinGecko API
4. Also fetch SPY, IWM, IWD, IWF for factor proxy calculations
5. **For each ticker** (sequentially, in a for-loop):
   - Compute 12 technical indicators (SMA/EMA, RSI, MACD, ATR/Bollinger, S/R, Ichimoku, VWAP, OBV, ADX/DMI, Stochastic, Fibonacci, Volume Profile)
   - Synthesize overall bias via regime-conditioned weighted voting across all 12 indicators
   - Compute quant metrics: return forecast (1w/1m/3m with confidence intervals), EWMA volatility, Hurst exponent regime, factor exposures (beta/alpha/R-squared)
   - Compute advanced quant: GARCH(1,1) volatility, HMM regime detection, Kalman time-varying beta, Fama-French 3-factor model
   - Compute performance stats: Sharpe, Sortino, Calmar, skewness, kurtosis
   - Generate a plain-text narrative per ticker (e.g., "AAPL: Bullish (0.78 conf). Price above SMA50...")
   - Store all results in `technical_indicators` and `quant_metrics` tables with `snapshot_hour=6`
6. Compute portfolio-level risk: VaR(95%), Expected Shortfall, max drawdown, portfolio beta, asset class breakdown
7. Compute macro snapshot: yield curve slope proxy (TLT/IEF), VIX proxy (SPY realized vol), credit spread proxy (HYG/IEF), composite macro regime
8. Store portfolio risk + macro in `daily_risk_metrics` with `snapshot_hour=6`
9. Compute NxN correlation matrix across all watchlist tickers, diversification score, cluster assignments
10. Store in `correlation_snapshot`
11. Fetch earnings calendar for all equity tickers via `yfinance.Ticker.calendar`
12. Store/update in `earnings_calendar`

**Result**: All quantitative data for the day is computed and stored. Every subsequent job and chat query reads from this cache.

### 06:30 — Daily Monitoring (`daily_job`)

**~40-50K tokens total (Research Agent + Daily Synthesis Agent).**

1. `_build_daily_context()` loads all cached data from DB (**0 tokens**):
   - Bulk technical indicators for all watchlist tickers
   - Bulk quant metrics, merged by ticker
   - Latest risk metrics (macro + portfolio)
   - Upcoming earnings (7 days) and recently reported earnings (3 days)
   - Correlation snapshot (diversification score, top pairs)
   - Current portfolio state
   - Assembles a ~2-4 KB structured text document

2. **Research Agent** (gpt-5-mini, ~20K tokens):
   - Receives: watchlist, date, earnings hint (if any tickers report within 48h)
   - Executes web search for macro/market context
   - Returns: structured JSON with themes, ticker-specific news, impact classifications, source tiering
   - This is the **only web-connected LLM call** in the daily pipeline

3. **Daily Synthesis Agent** (gpt-5.2, ~20K tokens):
   - Receives: pre-computed analysis document + research findings
   - Synthesizes into `DailyBrief` JSON:
     - `market_summary` (2-3 paragraphs)
     - `instruments[]` (per-ticker: signal, confidence, what_happened, why_it_matters)
     - `themes[]` (macro themes with impact)
     - `risk_snapshot` (VaR, ES, drawdown, beta, concentration warnings)
     - `telegram_summary` (max 4000 chars, scannable format)
   - Stores in `daily_briefs` + `instrument_briefs` tables
   - Logs return forecasts in `forecasts_log` for each ticker

4. Send `telegram_summary` to user via Telegram

### 09:00 — News Check (`news_check_job`)

**~20K tokens (Research Agent only).**

1. Check for earnings alerts (tomorrow's reports, just-reported results)
2. Run `run_news_alert_pipeline()`:
   - Call Research Agent with watchlist and high-impact themes request
   - Parse returned themes from JSON
   - Compare against existing active themes using **similarity-based deduplication** (SequenceMatcher + keyword overlap, threshold 0.75)
   - Store genuinely new themes in `research_themes`
   - Send Telegram alerts for new HIGH impact themes only
   - Deactivate themes older than 7 days

### 13:00 — Pre-Compute Midday (`precompute_job`)

**No LLM cost. Same pipeline as 06:00.**

- Stores all results with `snapshot_hour=13`
- Morning data (`snapshot_hour=6`) is **preserved**, not overwritten
- This enables morning-vs-midday comparison in the next job

### 13:30 — Midday Update (`midday_update_job`)

**~20K tokens (Research Agent via news pipeline).**

1. Load morning signals from DB (`snapshot_hour=6`)
2. Load midday signals (just computed, `snapshot_hour=13`)
3. **Compare per-ticker**: detect signal reversals (bullish-to-bearish or vice versa) and significant confidence shifts (>0.15)
4. If changes detected, send structured Telegram alert:
   - "**AAPL**: bullish → slightly_bearish (conf 0.78 → 0.55)"
5. Run earnings alert check
6. Run news alert pipeline (same as 09:00)

### 15:00 — News Check (`news_check_job`)

Same as 09:00 — lightweight news + earnings check.

### 20:00 — Evening Summary (`evening_summary_job`)

**No LLM cost. Pure database queries + formatting.**

1. Load today's technical indicators, quant metrics, risk metrics from DB
2. Format an evening scorecard:
   - **Top 5 signals** by confidence
   - **Top 3 quant highlights** by absolute 1-week forecast magnitude
   - **Risk snapshot**: VaR(95%), Expected Shortfall, beta, max drawdown
3. Send scorecard to Telegram
4. Run earnings alert check
5. Run news alert pipeline

### 22:00 — Forecast Evaluation (`forecast_evaluation_job`)

**No LLM cost. Pure computation.**

1. Find unevaluated forecasts from 7+ days ago in `forecasts_log`
2. Fetch recent prices via yfinance (3-month lookback)
3. For each forecast:
   - Compute actual return over the predicted horizon (1w=7d, 1m=21d, 3m=63d)
   - Determine if direction was correct (predicted up/down matches actual)
   - Compute absolute error (|predicted - actual|)
4. Store evaluation results in `forecast_accuracy` table

### Daily Token Budget Summary

| Time | Job | LLM Tokens | Cost Driver |
|------|-----|-----------|-------------|
| 06:00 | Pre-compute | 0 | Pure Python |
| 06:30 | Daily monitoring | ~40-50K | Research + Synthesis agents |
| 09:00 | News check | ~20K | Research agent |
| 13:00 | Pre-compute midday | 0 | Pure Python |
| 13:30 | Midday update | ~20K | Research agent (news pipeline) |
| 15:00 | News check | ~20K | Research agent |
| 20:00 | Evening summary | ~20K | Research agent (news pipeline) |
| 22:00 | Forecast eval | 0 | Pure Python |
| **Total** | | **~120-130K** | |

Ad-hoc chat messages add ~15K tokens each (33 tools, cache-first strategy reduces most to 0 LLM cost for cache hits).

---

## 3. Weekly Lifecycle

### Sunday 18:00 — Weekly Report (`weekly_job`)

**~80-120K tokens. Most expensive single job.**

1. Build context with `token_budget_remaining=200,000`
2. **Weekly Orchestrator** (gpt-5.2, 7 tools) runs:
   - **Step 1**: Gather data
     - `get_user_preferences()` — risk tolerance, investment style, constraints
     - `get_current_portfolio()` — current allocations
     - `retrieve_daily_briefs(past 7 days)` — the week's daily analyses
     - `retrieve_weekly_reports(count=2)` — prior reports for continuity
   - **Step 2**: Summarize the week (extract consistent signals, reversals, themes, major movers)
   - **Step 3**: Dispatch **Portfolio Agent** via `.as_tool()`:
     - Portfolio Agent (gpt-5.2, 24 tools) runs full optimization:
       - Risk Parity, Mean-Variance, Max Sharpe, HRP, CVaR optimization
       - Kelly Criterion sizing, Max Diversification, Entropy optimization
       - Black-Litterman with analyst views
       - Full risk pass: concentration limits, drawdown adjustment, cash targets
       - Transaction cost analysis
     - Returns: recommended allocation deltas with rationale
   - **Step 4**: Dispatch **Reporting Agent** via `.as_tool()`:
     - Reporting Agent (gpt-5.2, 4 tools) produces the investment committee memo:
       - Executive summary (3-4 sentences)
       - Market review (3-5 paragraphs)
       - Portfolio assessment (what worked/didn't)
       - Recommended allocations (table with delta + rationale)
       - Risk assessment (3 key risks, hedging suggestions)
       - Forward outlook (1-2 week scenarios)
       - Action items (max 5, prioritized)
     - Returns: `WeeklyReport` JSON
   - **Step 5**: Store report in `weekly_reports` table
3. Send `telegram_summary` to user (max 4000 chars)

---

## 4. The Pre-Compute Pipeline — Detailed Breakdown

Located in `tools/precomputed.py`, the pipeline is the computational core of the system. Here is the exact sequence, per ticker, with algorithms:

### 4.1 Technical Indicators (12 per ticker)

| # | Indicator | Algorithm | Key Outputs |
|---|-----------|-----------|-------------|
| 1 | **SMA/EMA** | SMA(50), SMA(200), EMA(12), EMA(26). Golden/death cross detection. | sma50, sma200, ema12, ema26, trend status |
| 2 | **RSI(14)** | Wilder's smoothed average gain/loss. RSI = 100 - 100/(1 + RS). | rsi, overbought/oversold/neutral |
| 3 | **MACD** | MACD = EMA(12) - EMA(26), Signal = EMA(9 of MACD), Histogram = MACD - Signal. | macd_line, signal_line, histogram, crossover |
| 4 | **ATR/Bollinger** | ATR(14) = smoothed True Range. BB = SMA(20) +/- 2*StdDev(20). %B, bandwidth. | atr_14, bb_upper/lower, bandwidth, pct_b |
| 5 | **Support/Resistance** | Pivot = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H, R2 = P+(H-L), S2 = P-(H-L). | pivot, r1, r2, s1, s2 |
| 6 | **Ichimoku** | Tenkan(9), Kijun(26), Senkou A/B(26/52), Chikou(26). Cloud position analysis. | tenkan, kijun, senkou_a/b, chikou, cloud signal |
| 7 | **VWAP** | VWAP = cumsum(price*volume) / cumsum(volume). Session and rolling calculations. | vwap, price deviation from vwap |
| 8 | **OBV** | On-Balance Volume: cumulative sum, +volume on up days, -volume on down days. Trend divergence detection. | obv, obv_trend, price-volume divergence |
| 9 | **ADX/DMI** | +DI = smoothed(+DM/ATR), -DI = smoothed(-DM/ATR). ADX = smoothed(|+DI - -DI| / (+DI + -DI)). | adx, plus_di, minus_di, trend strength |
| 10 | **Stochastic** | %K(14) = (C - L14) / (H14 - L14) * 100. %D = SMA(3 of %K). | k, d, overbought/oversold |
| 11 | **Fibonacci** | Retracement levels: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0 from swing high to swing low over a configurable period. | levels JSON, proximity signals |
| 12 | **Volume Profile** | Price range split into N bins. Volume accumulated per bin. POC (Point of Control) = highest volume bin. | poc_price, value_area_high/low |

### 4.2 Bias Synthesis (`_synthesize_bias`)

After computing all 12 indicators, a **regime-conditioned weighted voting** system produces the overall bias:

1. Each indicator contributes a bullish or bearish vote weighted by its confidence
2. **Regime conditioning** adjusts weights based on detected market regime:
   - **Trending regime**: trend indicators (SMA/EMA, ADX, Ichimoku, MACD) get **1.5x weight**
   - **Mean-reverting regime**: oscillators (RSI, Stochastic, Bollinger %B, Fibonacci) get **1.5x weight**
   - **Volatile regime**: volatility indicators (ATR/Bollinger, VWAP, Volume Profile) get **1.5x weight**
3. Bullish ratio = weighted_bullish / (weighted_bullish + weighted_bearish)
4. Classification: bullish (>=0.70), slightly bullish (>=0.55), neutral (0.45-0.55), slightly bearish (<=0.45), bearish (<=0.30)
5. Confidence adjustments: +0.05 for strong consensus, -0.10 for neutral (indecision)
6. Divergences tracked: if RSI diverges from OBV, flagged in narrative

### 4.3 Quantitative Models (per ticker)

| Model | Algorithm | Output |
|-------|-----------|--------|
| **Return Forecast** | EWMA mean returns projected at 1w/1m/3m horizons with normal confidence intervals | return_1w/1m/3m_pct, CI low/high |
| **Vol Forecast** | EWMA volatility (lambda=0.94), 252-day annualization, percentile ranking | ewma_vol, vol_regime, vol_percentile |
| **Regime Detection** | Hurst exponent (R/S analysis): H>0.55 = trending, H<0.45 = mean-reverting, else = random walk | hurst, regime, confidence |
| **Factor Exposures** | OLS: r_asset = alpha + beta * r_SPY. CAPM beta, alpha, R-squared. | beta, alpha, r_squared |
| **GARCH(1,1)** | `arch` library. Conditional volatility forecast at 1d/5d/21d horizons. Persistence = alpha + beta. Half-life = ln(2)/-ln(persistence). | garch_vol (annualized), persistence, forecast_vol |
| **HMM Regime** | Gaussian HMM with 3 states (bull/bear/transition). States labeled by mean return. Transition matrix + expected duration. | current_state, state_probability |
| **Kalman Beta** | State-space model: [alpha, beta] follow random walk. Recursive predict-update filter. Beta trend over 30 days. | current_beta, beta_trend, CI |
| **Fama-French 3-Factor** | r_i = alpha + beta_mkt*r_m + beta_smb*SMB + beta_hml*HML. SMB = IWM - SPY, HML = IWD - IWF (ETF proxies). Style classification. | beta_market, beta_smb, beta_hml, style |

### 4.4 Portfolio Risk (aggregate)

| Metric | Formula | Notes |
|--------|---------|-------|
| **VaR(95%)** | 5th percentile of weighted portfolio returns | Historical VaR, 1-day horizon |
| **Expected Shortfall** | Mean of returns below VaR threshold | Coherent risk measure (CVaR) |
| **Max Drawdown** | min((cumulative / running_max) - 1) | Peak-to-trough measure |
| **Portfolio Beta** | cov(port, SPY) / var(SPY) | Systematic risk exposure |
| **Asset Class Breakdown** | Equity/Bond/Commodity/Crypto/Cash % | Based on hard-coded ticker sets |

### 4.5 Macro Snapshot

| Indicator | Method | Interpretation |
|-----------|--------|----------------|
| **Yield Curve Slope** | (IEF 20d return - TLT 20d return) * 100 | Proxy; negative = inversion signal |
| **VIX Level** | SPY 21-day realized vol * sqrt(252) * 100 | <15 low, 15-20 normal, 20-30 elevated, >30 extreme |
| **Credit Spread** | (IEF 20d return - HYG 20d return) * 100 | Widening = stress |
| **Macro Regime** | Composite of VIX regime + yield curve + credit | Expansion / Slowdown / Contraction / Recovery |

### 4.6 Correlation Matrix

- NxN correlation matrix from aligned daily returns (Pearson)
- Top 10 most correlated pairs (by absolute value)
- Diversification score = 1 - average absolute correlation
- Cluster assignments via correlation threshold (>0.7 = same cluster)

### 4.7 Earnings Calendar

- `yfinance.Ticker(t).calendar` for next earnings date
- `yfinance.Ticker(t).earnings_dates` for historical EPS estimates/actuals
- Surprise calculation: ((actual - estimate) / |estimate|) * 100
- Crypto tickers skipped (no earnings)

---

## 5. How Prices Are Fetched

### Equities (Stocks + ETFs)

- **Source**: yfinance (`yf.download()`)
- **Batch fetch**: All equity tickers + SPY + IWM + IWD + IWF in a single `yf.download()` call
- **Period**: 1 year of daily OHLCV data
- **Minimum data requirement**: 30 bars (tickers with less are skipped)
- **Format**: DataFrame with date index, columns: open, high, low, close, volume
- **Frequency**: 2-3x daily via pre-compute pipeline

### Crypto (BTC, ETH, SOL, AVAX)

- **Source**: CoinGecko API (`/coins/{id}/ohlc`)
- **Individual fetch**: Each crypto ticker fetched separately (no batch API)
- **Period**: 365 days
- **Rate limiting**: CoinGecko free tier (10-30 calls/min)
- **Mapping**: `CRYPTO_MAP = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "AVAX": "avalanche-2"}`

### On-Demand (Chat Agent)

- When the chat agent needs live data (stale cache or non-watchlist ticker):
  - `fetch_ohlcv(ticker, period)` — yfinance single-ticker fetch
  - `fetch_crypto_data(ticker, days)` — CoinGecko single-ticker fetch
- These are direct tool calls, not batch

### Price Cache

- Historical prices are **not** cached in the `price_cache` table by the pre-compute pipeline (the pipeline computes indicators directly from fetched data)
- The `price_cache` table exists but is primarily used by on-demand tools when a chat user queries data

---

## 6. How Calculations Are Performed and Stored

### Two-Layer Tool Pattern

Every computational tool follows a strict two-layer pattern:

```
_raw() function (pure Python) ←── called by pre-compute pipeline
    ↓ wraps
@function_tool wrapper (agents SDK) ←── called by LLM agents
```

**Example:**
```python
# Layer 1: Pure computation (no LLM, no DB, no side effects)
def compute_rsi_raw(df: pd.DataFrame, period: int = 14) -> dict:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return {"rsi": float(rsi.iloc[-1]), "interpretation": "overbought" if rsi.iloc[-1] > 70 ...}

# Layer 2: Agent-callable wrapper
@function_tool
async def compute_rsi(ctx: RunContextWrapper[AppContext], ticker: str, prices_json: str) -> str:
    df = _prices_to_series(prices_json)
    raw = compute_rsi_raw(df)
    return json.dumps({"ticker": ticker, **raw})
```

**Why this matters:**
- Pre-compute pipeline calls `_raw()` directly — zero LLM overhead, no JSON serialization round-trips
- Agents call `@function_tool` wrappers — get the same computation with proper context handling
- All math is in Python; the LLM never performs arithmetic

### Storage Schema

Results are stored in SQLite across 20 tables. The key storage tables for computed data:

**`technical_indicators`** — 1 row per (ticker, date, snapshot_hour):
- 5 core indicators: sma50, sma200, ema12, ema26, rsi_14, macd_line/signal/histogram, atr_14, bb_upper/lower/bandwidth/pct_b, pivot/r1/r2/s1/s2
- 7 advanced indicators: ichimoku (5 lines), vwap, obv, adx, stochastic k/d, fib_levels (JSON), volume_profile (not stored as individual columns)
- Synthesis: overall_bias, overall_confidence, narrative (human-readable summary)

**`quant_metrics`** — 1 row per (ticker, date, snapshot_hour):
- Forecasts: return_1w/1m/3m_pct with CI bounds
- Volatility: ewma_vol, vol_regime, vol_percentile
- Regime: hurst, regime, regime_confidence
- Factor: beta, alpha, r_squared
- Advanced: garch_vol, hmm_state, kalman_beta, ff3_betas (JSON)
- Performance: sharpe, sortino, calmar, skewness, kurtosis

**`daily_risk_metrics`** — 1 row per (date, snapshot_hour):
- Portfolio: var_95, es_95, max_drawdown, current_drawdown, portfolio_beta, asset_class_pcts (JSON)
- Macro: yield_curve_slope, yield_curve_inverted, vix_level, vix_regime, credit_spread, macro_regime

**`correlation_snapshot`** — 1 row per date:
- correlation_matrix (JSON: all pair correlations)
- top_correlations (JSON: top 10 pairs)
- diversification_score, cluster_assignments (JSON)

**`earnings_calendar`** — 1 row per (ticker, earnings_date):
- eps_estimate, revenue_estimate, eps_actual, revenue_actual, eps_surprise_pct, status

---

## 7. Mathematical and Financial Models — Rigorous Assessment

### 7.1 Technical Analysis — Correct and Standard

The 12 technical indicators are implemented using **standard, well-known formulas**:

- **RSI**: Uses Wilder's exponential smoothing (not simple average), which is the correct industry standard
- **MACD**: Standard 12/26/9 parameters
- **Bollinger Bands**: 20-period SMA with 2 standard deviations — standard
- **Ichimoku**: Full 5-line system with correct periods (9/26/52)
- **Stochastic**: Standard %K(14) with %D = SMA(3)
- **ADX/DMI**: Proper +DM/-DM calculation with Wilder's smoothing
- **Fibonacci**: Standard retracement levels (0.236, 0.382, 0.5, 0.618, 0.786)

**Assessment**: Technically correct. The regime-conditioned weighting system for bias synthesis is a thoughtful addition — adjusting indicator weights based on detected market conditions (trending vs mean-reverting) is sound practice.

### 7.2 Quantitative Models — Academically Grounded

| Model | Implementation Quality | Notes |
|-------|----------------------|-------|
| **GARCH(1,1)** | Correct. Uses `arch` library (industry standard). Returns are scaled by 100 for numerical stability. Persistence, half-life, and multi-horizon forecasts are properly computed. | EGARCH variant also supported for asymmetric volatility |
| **HMM Regime** | Correct. Uses `hmmlearn` GaussianHMM with 3 states. States labeled by mean return (bull/bear/transition). Transition matrix and expected durations properly computed. | Standard approach in quantitative finance |
| **Kalman Filter Beta** | Correct. State-space model [alpha, beta] with random walk transition. Proper predict-update recursion. Observation noise = 0.5*var(asset). Transition covariance: alpha (1e-7, slow-moving), beta (1e-5, faster). 95% CI from posterior covariance. | This is a legitimate and useful technique for time-varying CAPM |
| **Fama-French 3-Factor** | Correct formulation. Uses ETF proxies (IWM-SPY for SMB, IWD-IWF for HML) rather than Kenneth French data library. OLS regression with proper t-statistics and adjusted R-squared. Style classification based on factor loadings. | **Note**: ETF proxies are an approximation. Academic research typically uses the Ken French factor data (available freely). However, the ETF proxy approach avoids an external dependency and is acceptable for advisory purposes. |
| **Hurst Exponent** | Uses R/S (Rescaled Range) analysis — the classical method. Proper implementation with log-log regression. Interpretation: H>0.55 trending, H<0.45 mean-reverting. | Standard, though DFA (Detrended Fluctuation Analysis) is sometimes preferred for robustness |
| **Return Forecasts** | EWMA mean returns with normal CI. This is a basic statistical forecast (not a structural model). | Honest about limitations — the confidence intervals acknowledge high uncertainty |

### 7.3 Portfolio Optimization — Comprehensive and Correct

The system implements **7 core + 5 advanced** optimization methods:

| Method | Mathematical Basis | Implementation Notes |
|--------|-------------------|---------------------|
| **Risk Parity** | Minimize Sigma(risk_contrib_i - target)^2. Target = portfolio_vol/n. Uses SLSQP. | Correctly equalizes risk contributions |
| **Mean-Variance** | Maximize mu'w - (gamma/2) * w'Sigma*w. Risk aversion gamma = 4/2/0.5 by risk tolerance. | Standard Markowitz; gamma calibration is reasonable |
| **Max Sharpe** | Maximize (mu'w - r_f) / sqrt(w'Sigma*w). r_f = 5%. | r_f=5% is appropriate for current rate environment |
| **Black-Litterman** | BL posterior = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]. tau=0.05, delta=2.5. | Correct BL formulation. Omega derived from view confidence. |
| **CVaR Optimization** | Rockafellar-Uryasev: minimize alpha + (1/(1-beta)) * mean(max(0, -r'w - alpha)). | Correct LP relaxation for CVaR minimization |
| **HRP** | Lopez de Prado's method: correlation distance -> hierarchical clustering -> quasi-diagonalization -> recursive bisection with inverse-variance weighting. | Correct implementation of the original paper |
| **Kelly Criterion** | Full Kelly: f* = Sigma^-1 * mu. Half Kelly recommended for practical use. Expected log growth rate computation. | Correctly warns about full Kelly volatility; half-Kelly recommendation is standard practice |
| **Max Diversification** | DR = (w'sigma) / sqrt(w'Sigma*w). Maximize diversification ratio. | Correct formulation |
| **Max Entropy** | Shannon entropy H = -sum(w_i * ln(w_i)). Maximize subject to return target. | Correct optimization |
| **Efficient Frontier** | 15-point frontier via constrained optimization at each target return level. | Standard approach |

**Risk Controls** (applied after optimization):
- Concentration limits (max 15% per position by default)
- Drawdown-aware scaling (if DD < -10%, scale equity by factor)
- Cash target enforcement
- Transaction cost analysis (turnover, one-way cost in bps)

**Assessment**: This is an **above-average** optimization suite. The inclusion of CVaR, HRP, Kelly, and Max Diversification alongside traditional MV/Risk Parity puts it on par with institutional quantitative advisory tools. The multi-method approach (blending RP/MV/Sharpe/HRP by risk tolerance) is a sound practical choice.

### 7.4 Risk Metrics — Comprehensive

| Metric | Method | Assessment |
|--------|--------|------------|
| **Historical VaR** | Empirical percentile | Standard, no distribution assumptions |
| **Expected Shortfall** | Mean of tail losses | Coherent risk measure — correct |
| **Cornish-Fisher VaR** | CF expansion adjusting for skew and kurtosis | Correct 4th-order expansion; captures fat tails better than Gaussian VaR |
| **EVT VaR** | Generalized Pareto Distribution (GPD) fit to tail exceedances | Correct POT (Peaks Over Threshold) approach with shape/scale estimation |
| **Monte Carlo VaR** | 10,000 simulations, normal or Student-t | Student-t calibrated from excess kurtosis: nu = 6/kurtosis + 4 |
| **Stress Testing** | 4 historical scenarios (2008, 2020, 2022, Flash Crash) with beta-adjusted shocks | Reasonable; could expand to user-defined scenarios |
| **Tail Dependence** | Empirical joint exceedance: P(X<q, Y<q) / P(X<q) | Captures co-crash risk beyond linear correlation |

### 7.5 Time Series Analysis

| Tool | Method | Assessment |
|------|--------|------------|
| **Autocorrelation** | ACF/PACF with Durbin-Levinson algorithm | Standard |
| **Stationarity** | Augmented Dickey-Fuller with AIC lag selection | Standard; correctly tests both log-prices and returns |
| **Seasonal Decomposition** | Additive decomposition via centered moving average | Standard STL-like approach |
| **Cointegration** | Engle-Granger 2-step with ADF on residuals | Standard; includes half-life and z-score trading signal |
| **Granger Causality** | F-test comparing restricted vs unrestricted VAR models | Standard |
| **Change Points** | CUSUM + Binary Segmentation | Standard structural break detection |
| **Spectral Analysis** | FFT with Hann window on centered returns | Standard; identifies dominant frequencies |
| **ARCH Test** | Engle's LM test on squared returns | Standard prerequisite for GARCH |

### 7.6 Advanced Analytics

| Tool | Method | Assessment |
|------|--------|------------|
| **PCA** | Eigenvalue decomposition of standardized returns covariance | Standard |
| **Clustering** | Ward linkage with silhouette-like quality metric | Standard |
| **Style Analysis** | Sharpe RBSA: constrained regression with non-negative weights summing to 1. Rolling 63-day window. | Standard; correct Sharpe 1992 formulation |
| **Brinson Attribution** | Brinson-Fachler: allocation + selection + interaction effects | Standard performance attribution |
| **Entropy & Mutual Information** | Shannon entropy, Herfindahl index, histogram-based MI | Standard information-theoretic measures |

### 7.7 Overall Mathematical Assessment

**The system accurately uses advanced mathematics and finance.** The implementations are:
- Mathematically correct (formulas match academic references)
- Numerically stable (scaling, minimum data requirements, error handling)
- Practically sound (half-Kelly, multi-method optimization blending, regime conditioning)

**One area worth noting**: The return forecast model is relatively simple (EWMA extrapolation). In a production setting with more resources, one could integrate more sophisticated models (ARIMA-GARCH, VAR, or even ensemble ML models). However, the current approach is honest about its limitations and the confidence intervals appropriately reflect forecast uncertainty.

---

## 8. Agent Architecture — Why 33 Tools for the Chat Agent

### The Problem

The chat agent is the primary user interface. It must handle any question the user asks — from "what's my portfolio?" (instant DB lookup) to "run a full optimization" (complex multi-agent task). This requires access to diverse tool categories.

### The 33 Tools, Explained

The chat agent's 33 tools break down into **29 direct tools + 4 agent delegates**:

**10 Pre-Computed Cache Tools (instant, 0 LLM cost)**
These are the most frequently used tools. When a user asks "how is AAPL doing?", the agent reads cached analysis instead of recomputing:

| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `check_data_freshness` | Is cache stale? | Routing decision — must be instant |
| `get_cached_technical` | Cached technical indicators for a ticker | Most common query type |
| `get_cached_quant` | Cached quant metrics for a ticker | Most common query type |
| `get_cached_bulk_summary` | All tickers at once | Portfolio overview requests |
| `get_signal_history` | How has the signal changed over days? | Trend analysis |
| `get_intraday_changes` | Morning vs midday comparison | "What changed today?" |
| `get_indicator_trend` | Specific indicator over time | "Show me RSI trend for AAPL" |
| `get_daily_analysis_snapshot` | Full pre-computed narrative | "Give me today's analysis" |
| `get_cached_macro` | Macro regime, VIX, yield curve | "What's the macro environment?" |
| `get_cached_correlations` | Correlation matrix, diversification score | "How diversified am I?" |

These 10 tools handle **~60-70% of all user queries** at zero LLM cost per tool call. Moving them behind an agent delegate would add unnecessary latency and tokens.

**2 Market Data Tools**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `fetch_ohlcv` | Live price fetch for any ticker | User asks about non-watchlist ticker |
| `fetch_crypto_data` | Live crypto prices | Crypto-specific queries |

**4 Macro/Economic Tools**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `fetch_fred_series` | FRED economic data (CPI, unemployment, etc.) | User asks specific economic question |
| `get_yield_curve` | Live yield curve snapshot | Treasury-specific question |
| `get_economic_calendar` | Major upcoming economic events | "What's on the calendar?" |
| `compute_macro_regime` | Live macro regime assessment | "What regime are we in?" (if cached is stale) |

**2 Earnings Tools**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `get_upcoming_earnings` | Watchlist earnings calendar | "When does AAPL report?" |
| `get_earnings_results` | Historical earnings with surprise data | "How did MSFT earnings go?" |

**3 Portfolio State Tools**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `get_current_portfolio` | Current allocations | "Show my portfolio" |
| `update_portfolio` | Modify allocations | "I bought more NVDA" |
| `get_portfolio_history` | Historical snapshots | "How has my portfolio changed?" |

**3 User Preference Tools**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `get_user_preferences` | Risk tolerance, watchlist, etc. | Context for any recommendation |
| `update_user_preference` | Change a setting | "Set my risk to aggressive" |
| `update_watchlist` | Add/remove tickers | "Add GOOG to my watchlist" |

**3 Database Query Tools**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `retrieve_daily_briefs` | Past daily analyses | "Show me yesterday's brief" |
| `retrieve_weekly_reports` | Past weekly reports | "What did last week's report say?" |
| `query_forecasts_log` | Forecast accuracy tracking | "How accurate have my forecasts been?" |

**1 Web Search Tool**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `WebSearchTool` | Real-time web search | "What's happening with NVDA?" |

**1 Usage Tracking Tool**
| Tool | Purpose | Why Direct |
|------|---------|-----------|
| `get_usage_summary` | Token cost tracking | "How much have I used?" |

**4 Agent-as-Tool Delegates (each hides a specialist with many tools)**
| Delegate | Hidden Agent | Hidden Tools | When Used |
|----------|-------------|-------------|-----------|
| `run_technical_analysis` | Technical Agent | 14 tools | Cache stale, non-watchlist ticker, "run live analysis" |
| `run_quantitative_analysis` | Quantitative Agent | 23 tools | Deep quant analysis, advanced time series |
| `run_portfolio_analysis` | Portfolio Agent | 24 tools | "Optimize my portfolio", "stress test", "what's my VaR" |
| `run_market_research` | Research Agent | 1 tool (web) | "Latest news on...", event research |

### Why Not Fewer Tools?

Reducing below ~28-33 tools would require moving frequently-used tools behind agent delegates, which would:
- **Add latency**: Every delegation adds a full LLM round-trip (~2-5 seconds)
- **Add token cost**: The delegate agent must be invoked with its own instructions + tool schemas
- **Reduce accuracy**: More abstraction layers mean more opportunities for the model to misinterpret the request

The current 33 tools is within OpenAI's recommended range for production systems (they suggest 15-30, but note that well-organized tools with clear descriptions can work up to ~40). The key optimization was removing 62 specialist tools that are now hidden behind 4 delegates.

### The Before/After Comparison

| Metric | Before (v2, 86 tools) | After (v3, 33 tools) |
|--------|----------------------|---------------------|
| Tool schema tokens per message | ~25,800 | ~8,400 |
| Model tool selection accuracy | Degraded above ~30 tools | Within optimal range |
| Annual tool schema cost (3 msg/day) | ~$1,620 | ~$515 |
| Deep analysis capability | Same — tools still exist, just hidden | Same |

---

## 9. Per-Instrument vs Batch Processing — Current State and Recommendations

### Current State: Hybrid Approach

The system uses **different strategies at different stages**:

#### Pre-Compute Pipeline: Per-Instrument Computation

The pre-compute pipeline processes instruments **one by one** in a sequential for-loop:

```python
# tools/precomputed.py, line 294
for ticker in watchlist:
    df = ticker_dfs.get(ticker)
    # ... compute 12 technical indicators
    # ... compute quant metrics (GARCH, HMM, Kalman, FF3)
    # ... store in DB
```

However, the **data fetch** is batched:
```python
# Single HTTP call for all equity tickers
bulk_df = yf.download(all_equity, period="1y", ...)
```

This is the correct approach — batch the I/O, compute individually. Each ticker's indicators depend only on that ticker's price series (plus SPY/factor proxies for beta/FF3).

#### Daily Synthesis: ALL Tickers in One LLM Call

The daily synthesis agent receives **all watchlist tickers' data in a single prompt**:

```python
# scheduler/jobs.py, line 328-337
prompt = (
    f"=== PRE-COMPUTED ANALYSIS ===\n{analysis_context}\n\n"  # Contains ALL tickers
    f"=== MARKET RESEARCH ===\n{research_output}\n\n"
    f"Produce the DailyBrief JSON with telegram_summary."
)
result = await Runner.run(starting_agent=get_daily_synthesis_agent(), input=prompt, ...)
```

The `analysis_context` is a ~2-4 KB document containing narratives for all watchlist tickers. For a 22-ticker watchlist, this is approximately:
- ~100-150 chars per ticker narrative
- ~500 chars for macro/risk/earnings/correlation sections
- Total: ~3-4 KB (well within context limits)

#### Research Agent: ALL Tickers in One Call

```python
# scheduler/jobs.py, line 312-321
research_result = await Runner.run(
    starting_agent=get_research_agent(),
    input=f"Research macro/market context for today. Watchlist: {', '.join(ctx.watchlist)}.",
    context=ctx,
)
```

The research agent receives the full watchlist and runs web searches to cover as many tickers as relevant.

#### Weekly Orchestrator: ALL Tickers in One Flow

The weekly pipeline passes all daily briefs (7 days, all tickers) to the portfolio agent and reporting agent as a single flow.

### Assessment

**The current approach is reasonable for the typical watchlist size (20-30 tickers).** Here's why:

1. **Pre-computed narratives are compact**: Each ticker narrative is ~100-150 chars. 30 tickers = ~4 KB. GPT-5.2's context window easily handles this.

2. **The LLM's job is synthesis, not computation**: The daily synthesis agent doesn't compute anything — it combines pre-computed numbers with research findings into a coherent narrative. This is inherently a cross-ticker task (identifying themes, comparing relative signals, assessing portfolio-level risk).

3. **Research benefits from cross-ticker context**: A single research call with the full watchlist produces better-connected findings than 22 isolated calls would. The agent can identify themes that affect multiple tickers.

### Where Per-Instrument Would Help

If the watchlist grows beyond ~50 tickers, the pre-computed analysis context would exceed ~7-8 KB and start consuming significant prompt space. At that point, you would want to:
- Split the daily synthesis into sector/asset-class groups
- Process 8-10 tickers per synthesis call
- Add a final aggregation step

However, **for 20-30 tickers, the current all-at-once approach is more efficient** — it avoids redundant LLM round-trips and produces more coherent cross-ticker analysis.

**The one exception** where per-instrument processing would currently improve quality is the **research agent**. With 22 tickers in a single web search session, the agent cannot realistically search for ticker-specific news for each one. It tends to focus on the top 5-10 most newsworthy tickers and provide general macro context for the rest. If thorough per-ticker research is desired, running the research agent in batches of 5-7 tickers would produce more comprehensive coverage — though at 3-4x the token cost.

---

## 10. How Insights and Recommendations Are Stored

### Daily Insights

**`daily_briefs` table** — 1 row per day:
- `content_json` — Full DailyBrief JSON (market_summary, instruments[], themes[], risk_snapshot)
- `market_summary` — 2-3 paragraph macro narrative
- `telegram_summary` — Scannable summary sent to user

**`instrument_briefs` table** — 1 row per (ticker, day):
- `signal` — bullish/bearish/neutral
- `confidence` — 0.0 to 1.0
- `what_happened` — 1-2 sentence summary
- `why_it_matters` — Interpretation for portfolio context

**`forecasts_log` table** — 1 row per (ticker, forecast_date, horizon):
- `predicted_value` — Expected return %
- `predicted_direction` — up/down
- Later filled: `actual_value`, `was_correct`

### Weekly Recommendations

**`weekly_reports` table** — 1 row per week:
- `content_json` — Full WeeklyReport JSON (executive_summary, market_review, allocations[], risk_assessment, outlook, action_items)
- `allocations_json` — Recommended asset allocation with deltas and rationale
- `executive_summary` — 3-4 sentence high-level recommendation
- `telegram_summary` — Formatted for Telegram delivery

### Research Themes

**`research_themes` table** — 1 row per detected theme:
- `theme` — Title (e.g., "Fed signaling rate cut acceleration")
- `summary` — 2-3 sentence explanation
- `impact` — high/medium/low
- `affected_tickers` — JSON list
- `sources` — JSON list of source URLs
- `source_tier` — Tier 1 (official)/Tier 2 (major media)/Tier 3 (analyst)/Tier 4 (social)
- `is_active` — Deactivated after 7 days

### Forecast Accuracy

**`forecast_accuracy` table** — 1 row per evaluated forecast:
- `predicted_direction` / `actual_direction`
- `predicted_return_pct` / `actual_return_pct`
- `absolute_error`
- `is_direction_correct` (boolean)

This enables tracking forecast quality over time: directional accuracy %, mean absolute error, by ticker and horizon.

---

## 11. Production Readiness Assessment

### Strengths

1. **Separation of Compute and Synthesis**: All mathematical computation is programmatic. The LLM never does arithmetic. This eliminates the most common source of AI errors in quantitative systems.

2. **Robust Error Handling**: Every tool has try/except blocks. The pre-compute pipeline continues on per-ticker failures. Scheduler jobs log exceptions and send Telegram alerts on critical failures.

3. **Data Integrity**: SQLite with WAL mode, UNIQUE constraints preventing duplicate entries, `ON CONFLICT DO UPDATE` for upserts. The `snapshot_hour` system prevents intraday data loss.

4. **Token Cost Management**: The v3 architecture reduced daily LLM costs by ~64% (from ~344K to ~124K tokens/day). Cache-first strategy means most chat queries cost 0 tokens for data retrieval.

5. **Configuration Flexibility**: All critical parameters are configurable via environment variables (models, schedule times, watchlist, budgets, thresholds). Per-agent model selection allows cost optimization.

6. **Monitoring**: `analysis_runs` table tracks every pipeline execution. `token_usage` tracks LLM costs. `forecast_accuracy` tracks prediction quality. Evening scorecard provides daily checkpoint.

### Concerns for Production

1. **Single-Process Architecture**: The entire system runs in one Python process. If the process dies, all scheduled jobs stop, the Telegram bot goes offline, and no alerts are sent. A production deployment would benefit from:
   - Process supervision (systemd, Docker with restart policy)
   - Health check endpoint
   - Separate scheduler and bot processes for isolation

2. **SQLite Concurrency**: SQLite allows only one writer at a time. With scheduled jobs and Telegram handlers potentially writing simultaneously, contention is possible. For a single-user system this is acceptable, but scaling to multiple users would require PostgreSQL or similar.

3. **No Authentication Beyond Telegram Chat ID**: The system trusts that the `telegram_chat_id` matches the authorized user. There's no API key rotation, rate limiting on the Telegram side, or session management.

4. **yfinance Reliability**: yfinance is an unofficial API that scrapes Yahoo Finance. It can break without notice, has no SLA, and occasionally returns stale or missing data. The pre-compute pipeline handles this gracefully (skips tickers with insufficient data), but a production system should consider a paid data provider (Polygon.io, Alpha Vantage, IEX Cloud) as primary or fallback.

5. **CoinGecko Rate Limiting**: The free CoinGecko API has a 10-30 calls/min rate limit. With 4 crypto tickers fetched individually, this is fine, but adding more crypto tickers could hit limits. Sequential fetching with no explicit rate limiting is a risk.

6. **No Backpressure / Circuit Breaker**: If the OpenAI API is slow or down, scheduled jobs will hang until timeout. There's no circuit breaker pattern, retry with exponential backoff, or job deadline enforcement.

7. **Global Singletons for Agents**: All agents use module-level singleton pattern (`_agent: Agent | None = None`). This works for single-process but complicates testing (must reset globals between tests) and prevents per-request customization.

8. **Macro Proxies**: The yield curve slope, VIX, and credit spread are computed from ETF price ratios (TLT/IEF, SPY realized vol, HYG/IEF). These are proxies, not actual rates/spreads. While reasonable for direction and regime classification, they can diverge from actual values. A production system with a FRED API key would get actual treasury yields, VIX index values, and credit spread data.

### Verdict

**The system is production-worthy for its intended use case** (single-user portfolio advisory via Telegram). The architecture is sound, the mathematics are correct, and the error handling is reasonable. The key risks are operational (process management, data source reliability) rather than architectural.

For scaling beyond a single user or for managing real capital decisions, the concerns listed above would need to be addressed — particularly data source reliability, process isolation, and database concurrency.

---

## 12. Identified Issues and Recommendations

### Critical (Should Fix)

1. **Research Agent Context Dilution for Large Watchlists**: The research agent receives all 22 tickers in a single prompt and must search for news on all of them. In practice, it focuses on the top 5-10 most newsworthy and gives superficial coverage to the rest. For watchlists above 15 tickers, consider batching the research agent into groups of 7-8 tickers per call.

2. **Yield Curve Proxy Accuracy**: The TLT/IEF price ratio is a directional proxy for yield curve slope but can produce false signals during periods of unusual bond market activity (e.g., Fed operations, liquidity events). If `PA_FRED_API_KEY` is set, the system should prefer actual treasury yield data from FRED over ETF proxies. Currently, `compute_macro_regime()` in `economic_data.py` does use FRED when available, but the pre-compute pipeline's `_compute_macro_snapshot()` always uses ETF proxies. These two code paths should be unified.

### Important (Should Consider)

3. **Fama-French Factor Proxies vs Academic Data**: The SMB and HML factors are computed as (IWM-SPY) and (IWD-IWF). These are reasonable proxies but deviate from the academic Ken French factors, particularly during factor rotation periods. The Ken French data library (freely available at mba.tuck.dartmouth.edu) could be fetched periodically for more accurate factor exposure analysis. This is a data quality improvement, not a correctness issue.

4. **Return Forecast Model Simplicity**: The current return forecast is EWMA mean extrapolation with normal confidence intervals. This is honest and defensible, but it doesn't capture mean-reversion, momentum, or structural breaks. A practical upgrade would be a simple ARMA model or a regime-switching forecast that adjusts expectations based on the HMM state (which is already computed).

5. **No Survivorship Bias Handling**: The system uses whatever data yfinance returns. If a ticker was delisted, it may return incomplete data without warning. The 30-bar minimum check helps, but there's no explicit delisting detection.

### Minor (Nice to Have)

6. **Correlation Clustering**: The current clustering algorithm (greedy correlation threshold >0.7) is simple but can produce inconsistent clusters depending on ticker ordering. The Ward linkage clustering already exists in `advanced_analytics.py` (`compute_clustering_raw`) and could be reused in the pre-compute pipeline for more robust cluster assignments.

7. **Evening Summary Could Include Forecast Tracking**: The evening scorecard shows today's signals and quant highlights but doesn't compare morning predictions to actual close prices. Adding a "forecast vs actual" section would provide immediate feedback on model quality.

8. **Transaction Cost Model**: The current model uses a flat 10 bps cost. A production system could use ticker-specific spread estimates (larger for small-caps, crypto; tighter for SPY) for more realistic rebalancing cost analysis.

---

*End of Architecture Review*
