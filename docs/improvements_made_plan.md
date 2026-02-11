# Portfolio Advisor v2 — Comprehensive System Redesign

## Context

The initial system (v1) is fully implemented and working: 8 agents, 45 tools, SQLite DB, Telegram bot, APScheduler. A thorough review revealed the system scores ~6/10 on tool sophistication, has architectural inefficiencies (serial agent dispatch, no pre-computed caching, 45-tool context bloat on chat agent), and is missing critical financial analysis capabilities. This redesign addresses:

1. **Pre-computed analysis pipeline** — run technical/quant analysis 2-3x daily, cache results for instant LLM queries
2. **Model upgrade** — switch from gpt-5-mini to gpt-5.2 (free daily tokens) for all analysis agents; mini only for trivial tasks
3. **Advanced financial tools** — graduate-level quant: GARCH, HMM, Kalman, Fama-French, CVaR optimization, HRP, Kelly Criterion, entropy-based methods, EVT, stress testing, Granger causality, PCA, copulas, economic data
4. **Onboarding flow** — guided new-user setup via Telegram conversation
5. **Multi-schedule + news alerts** — morning/midday/evening runs, intraday high-impact event alerts
6. **Expanded preferences** — from 7 to 20 fields, validation, richer user customization

---

## Phase 1: Foundation — DB Schema, Config, Preferences, Model Upgrade

**Goal**: Data/config groundwork. Zero regressions — existing pipelines keep working.

### 1A. New Database Tables (`db/schema.py`)

Append 8 new `CREATE TABLE IF NOT EXISTS` statements:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `technical_indicators` | Pre-computed per-ticker per-date | `ticker, indicator_date, run_id, sma50, sma200, ema12, ema26, rsi_14, macd_line/signal/histogram, atr_14, bb_upper/lower/bandwidth/pct_b, pivot/r1/r2/s1/s2, ichimoku_* (nullable), vwap, obv, adx, stochastic_k/d, fib_levels, overall_bias, overall_confidence, narrative` — UNIQUE(ticker, indicator_date) |
| `quant_metrics` | Pre-computed per-ticker per-date | `ticker, metric_date, run_id, return_1w/1m/3m_pct + CIs, ewma_vol, vol_regime/percentile, hurst, regime/confidence, beta/alpha/r_squared, skewness/kurtosis, sharpe/sortino/calmar, garch_vol (nullable), hmm_state (nullable), kalman_beta (nullable), ff3_betas (nullable), cornish_fisher_var (nullable), evt_var (nullable)` — UNIQUE(ticker, metric_date) |
| `daily_risk_metrics` | Portfolio-level risk by date | `risk_date, run_id, var_95, es_95, max_drawdown, current_drawdown, portfolio_beta, asset_class_pcts, stress_test_results (nullable), diversification_ratio, entropy_score` — UNIQUE(risk_date) |
| `research_themes` | Queryable themes with impact | `theme_date, theme, summary, impact (high/medium/low), affected_tickers (JSON), sources (JSON), source_tier, is_active` |
| `forecast_accuracy` | Evaluated predictions | `forecast_id (FK), evaluation_date, predicted/actual direction, predicted/actual return, absolute_error, is_direction_correct` — UNIQUE(forecast_id) |
| `onboarding_state` | Tracks onboarding progress | `current_step (welcome/risk/horizon/watchlist/portfolio/done), steps_completed (JSON), started_at, completed_at` — single row |
| `chat_history` | Persistent conversation memory | `chat_id, role (user/assistant), content, created_at` — indexed on (chat_id, created_at) |
| `analysis_runs` | Tracks pre-compute freshness | `run_id (UUID PK), run_type, started_at, completed_at, tickers_processed (JSON), status (running/completed/failed), duration_seconds` |

### 1B. New Database Queries (`db/queries.py`)

Add ~20 functions:
- `store_technical_indicators(db, data)` / `get_technical_indicators(db, ticker, date?)` / `get_bulk_technical_indicators(db, tickers, date?)`
- `store_quant_metrics(db, data)` / `get_quant_metrics(db, ticker, date?)` / `get_bulk_quant_metrics(db, tickers, date?)`
- `create_analysis_run(db, run_id, run_type, tickers)` / `complete_analysis_run(db, run_id, status, error?)` / `get_latest_analysis_run(db, run_type)`
- `store_chat_message(db, chat_id, role, content)` / `get_chat_history(db, chat_id, limit=20)`
- `evaluate_forecast(db, forecast_id, actual_return_pct)` — backfills `forecast_accuracy` + `forecasts_log`
- `store_research_theme(db, theme)` / `get_active_research_themes(db, days=7)`
- `get_signal_trend(db, ticker, days=7)` — signal history for a ticker
- `get_forecast_accuracy_summary(db, days=30)` — aggregate MAPE, directional accuracy
- `store_daily_risk_metrics(db, data)` / `get_risk_metrics_history(db, days=30)`
- Onboarding state CRUD: `get_onboarding_state(db)` / `update_onboarding_step(db, step)`

### 1C. Config Expansion (`config.py`)

Add to `Settings`:
```python
# Multi-schedule (UTC hours)
morning_run_hour: int = 6
midday_run_hour: int = 13
evening_run_hour: int = 20

# Per-agent model assignments
model_orchestrator: str = "gpt-5.2"
model_technical: str = "gpt-5.2"
model_quantitative: str = "gpt-5.2"
model_portfolio: str = "gpt-5.2"
model_reporting: str = "gpt-5.2"
model_chat: str = "gpt-5.2"
model_research: str = "gpt-5-mini"    # just web search relay
model_onboarding: str = "gpt-5-mini"  # simple conversation

# Pre-computation
precompute_enabled: bool = True
precompute_stale_hours: float = 8.0

# FRED API
fred_api_key: str = ""

# Onboarding
onboarding_enabled: bool = True
```

### 1D. Expanded Preferences (`models/preferences.py`)

Add 13 fields to `UserPreferences`:
- `investment_style` (passive/active/tactical)
- `rebalance_frequency` (weekly/biweekly/monthly/quarterly)
- `max_crypto_pct`, `min_bond_pct`, `max_single_sector_pct`
- `preferred_sectors` (list)
- `esg_filter`, `dividend_preference` (growth/income/neutral)
- `tax_aware`, `notification_level` (low/medium/high)
- `analysis_depth` (brief/detailed/exhaustive)
- `benchmark` (default SPY)
- `notes` (free-form)

Add migration logic in `db/connection.py` to `ALTER TABLE ADD COLUMN` for each new field with defaults.

### 1E. Model Upgrade Across All Agents

Convert all agent files to use lazy initialization with config-based model selection:

```python
_agent: Agent[AppContext] | None = None
def get_technical_agent() -> Agent[AppContext]:
    global _agent
    if _agent is None:
        _agent = Agent[AppContext](name="...", model=get_settings().model_technical, ...)
    return _agent
```

Apply to: `technical.py`, `quantitative.py`, `portfolio.py`, `reporting.py`, `chat.py`, `orchestrator.py`, `research.py`. Update orchestrator to call `get_*_agent()` factory functions.

**Files modified**: `db/schema.py`, `db/queries.py`, `db/connection.py`, `config.py`, `models/preferences.py`, all 7 agent files

---

## Phase 2: Pre-Computation Pipeline + Multi-Schedule

**Goal**: Compute all indicators 2-3x daily, store in DB, chat agent queries cache.

### 2A. Pre-Computed Analysis Tool (`tools/precomputed.py` — NEW)

Core pipeline function (not a @function_tool — called directly by scheduler):
- `run_precompute_pipeline(ctx: AppContext)` — batch fetches OHLCV for entire watchlist via `yf.download(tickers)`, then loops each ticker computing all technical indicators + quant metrics, stores in `technical_indicators` and `quant_metrics` tables. Tracks progress via `analysis_runs`.

**Key refactoring**: Extract raw computation logic from existing `@function_tool` wrappers into pure functions (e.g., `_compute_rsi_raw(close_series, period=14) -> float`). Tool wrappers call pure functions + return JSON. Pre-compute pipeline calls pure functions directly without `RunContextWrapper`.

Agent-facing tools (5 @function_tools):
- `get_cached_technical(ctx, ticker)` — returns latest pre-computed technical indicators, includes `is_fresh` flag (age < stale_hours)
- `get_cached_quant(ctx, ticker)` — same for quant metrics
- `get_cached_bulk_summary(ctx, tickers)` — condensed multi-ticker view (signal, confidence, RSI, regime, vol)
- `check_data_freshness(ctx)` — when was last pre-compute run, is it fresh?
- `get_signal_history(ctx, ticker, days)` — signal trend over time

### 2B. Multi-Schedule Support (`scheduler/runner.py`)

Expand from 2 scheduled jobs to 6:

| Job | Schedule | Function |
|-----|----------|----------|
| `precompute_morning` | 06:00 UTC | `precompute_job()` — full indicator computation |
| `daily_monitoring` | 06:30 UTC | `daily_job()` — orchestrator synthesis (uses cached data) |
| `precompute_midday` | 13:00 UTC | `precompute_job()` — refresh indicators |
| `midday_update` | 13:30 UTC | `midday_update_job()` — delta-focused update |
| `evening_summary` | 20:00 UTC | `evening_summary_job()` — day scorecard |
| `weekly_report` | Sun 18:00 UTC | `weekly_job()` — unchanged |
| `forecast_eval` | 22:00 UTC daily | `forecast_evaluation_job()` — backfill actuals |

### 2C. New Job Functions (`scheduler/jobs.py`)

- `precompute_job()` — calls `run_precompute_pipeline(ctx)`
- `midday_update_job()` — runs precompute, then compares morning vs midday indicators for signal changes, sends Telegram if significant changes detected
- `evening_summary_job()` — summarizes day: morning signals vs actual close, forecast accuracy, sends day scorecard
- `forecast_evaluation_job()` — finds unevaluated forecasts from 7+ days ago, computes actual returns, backfills `forecast_accuracy`

### 2D. Chat Agent Cache Integration (`agents/chat.py`)

Add pre-computed tools to chat agent. Update instructions with new top-priority section:

> **Always check cached data first**: Call `check_data_freshness` → if fresh, use `get_cached_technical`/`get_cached_quant`/`get_cached_bulk_summary`. Only fall through to live computation if cache is stale or missing.

### 2E. Persistent Chat History (`telegram_bot/chat_handler.py`)

Replace `context.user_data` chat history with `chat_history` table reads/writes. Load last 20 messages from DB on each request. Store both user and assistant messages.

**Files modified**: `scheduler/runner.py`, `scheduler/jobs.py`, `agents/chat.py`, `telegram_bot/chat_handler.py`
**Files created**: `tools/precomputed.py`

---

## Phase 3: Advanced Technical Indicators

**Goal**: Add 7 new indicators for deeper market structure analysis.

### New File: `tools/advanced_technical.py`

| Function | What It Does | Signal Logic |
|----------|-------------|--------------|
| `compute_ichimoku_cloud(ctx, ticker, prices_json)` | Tenkan/Kijun/Senkou A&B/Chikou. Cloud = dynamic S/R zone | Price above cloud = bullish; TK cross = entry signal; cloud twist = reversal |
| `compute_vwap(ctx, ticker, prices_json)` | Rolling 20-day VWAP (institutional benchmark) | Price > VWAP = bullish institutional flow |
| `compute_obv(ctx, ticker, prices_json)` | Cumulative On-Balance Volume + divergence detection | OBV diverging from price = reversal warning |
| `compute_adx_dmi(ctx, ticker, prices_json)` | ADX trend strength + +DI/-DI direction | ADX > 25 = trending; +DI > -DI = uptrend |
| `compute_stochastic_oscillator(ctx, ticker, prices_json)` | %K/%D with crossover detection | %K < 20 = oversold; bullish crossover below 20 = buy signal |
| `compute_fibonacci_retracements(ctx, ticker, prices_json)` | Key levels from swing high/low (23.6%, 38.2%, 50%, 61.8%, 78.6%) | Price at 38.2% or 61.8% = potential reversal zone |
| `compute_volume_profile(ctx, ticker, prices_json)` | Volume-at-Price histogram, POC, HVN/LVN zones | HVN = support/resistance; POC = fair value |

Update `agents/technical.py` to include these tools. Update pre-compute pipeline to populate the nullable advanced columns in `technical_indicators`.

**Files created**: `tools/advanced_technical.py`
**Files modified**: `agents/technical.py`, `agents/chat.py`, `tools/precomputed.py`

---

## Phase 4: Advanced Quantitative Models

**Goal**: Graduate-level quant models for volatility, regime, and factor analysis.

### New Dependencies (`pyproject.toml`)
```
arch>=7.0        # GARCH/EGARCH
hmmlearn>=0.3    # Hidden Markov Models
statsmodels>=0.14  # Econometric tests
scikit-learn>=1.3  # PCA, clustering
```

### New File: `tools/advanced_quant.py`

| Function | Math | Key Output |
|----------|------|------------|
| `compute_garch_volatility(ctx, ticker, prices_json, model_type="GARCH")` | GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. EGARCH adds leverage. Uses `arch` library. | Conditional vol forecast (1d/5d/21d), persistence (α+β), half-life of vol shocks |
| `detect_regime_hmm(ctx, ticker, prices_json, n_states=3)` | Gaussian HMM fit to returns. 3 states: bull/bear/transition. Uses `hmmlearn`. | Current state + probabilities, transition matrix, expected state duration |
| `compute_kalman_filter(ctx, ticker, prices_json, benchmark="SPY")` | Kalman filter with 2-state [α, β] random walk. Adaptive beta that tracks regime changes. | Time-varying beta + alpha with 95% CIs, beta trend over 30d |
| `compute_fama_french_3factor(ctx, ticker, prices_json)` | r_i - r_f = α + β_mkt·(r_m - r_f) + β_smb·SMB + β_hml·HML. Uses ETF proxies (SPY, IWM, IWD, IWF). | Market/size/value factor betas, alpha, R² |

Update `agents/quantitative.py` to include these tools. Update pre-compute pipeline to populate the nullable GARCH/HMM/Kalman/FF3 columns.

**Files created**: `tools/advanced_quant.py`
**Files modified**: `agents/quantitative.py`, `agents/chat.py`, `tools/precomputed.py`, `pyproject.toml`

---

## Phase 5: Advanced Portfolio, Risk, Time Series, Analytics, and Economic Data

**Goal**: Professional-grade portfolio construction, risk management, and macro analysis. Largest phase.

### 5A. Advanced Portfolio (`tools/advanced_portfolio.py` — NEW)

| Function | Math | Purpose |
|----------|------|---------|
| `optimize_cvar(ctx, tickers, ...)` | Minimize CVaR via Rockafellar-Uryasev LP: min α + 1/(1-β)·1/T·Σmax(0, -rₜ·w - α) | Tail-risk-aware optimization |
| `optimize_hrp(ctx, tickers, prices_dict)` | Lopez de Prado HRP: correlation clustering → quasi-diagonalization → recursive bisection with inverse-variance | No optimizer instability; robust to estimation error |
| `compute_kelly_criterion(ctx, tickers, ...)` | Full Kelly: f* = Σ⁻¹·μ. Half-Kelly: f*/2. Optimal geometric growth sizing | Position sizing guide |
| `optimize_max_diversification(ctx, tickers, ...)` | Maximize DR = (w'·σ) / √(w'·Σ·w). Ratio of weighted-avg vol to portfolio vol | Maximum diversification benefit |
| `optimize_entropy_weighted(ctx, tickers, ...)` | Maximize Shannon entropy H = -Σwᵢ·ln(wᵢ) subject to return target | Information-theoretic diversification |
| `compute_transaction_costs(ctx, current, target, prices, cost_bps=10)` | TC = Σ|w_target - w_current|·price·cost_rate | Practical rebalancing cost |

### 5B. Advanced Risk (`tools/advanced_risk.py` — NEW)

| Function | Math | Purpose |
|----------|------|---------|
| `compute_cornish_fisher_var(ctx, weights, prices, conf=0.95)` | z_cf = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36 | Skew/kurtosis-adjusted VaR |
| `compute_evt_var(ctx, weights, prices, conf=0.99)` | Fit GPD to tail losses. Shape ξ > 0 = heavy tails. VaR_q = u + (β/ξ)·((n/N_u·(1-q))^{-ξ} - 1) | Extreme tail risk |
| `compute_monte_carlo_var(ctx, weights, prices, n_sims=10000)` | Simulate N paths from fitted distribution (normal or Student-t). VaR = percentile of simulated P&L | Forward-looking risk |
| `run_stress_test(ctx, weights, prices, scenario="2008")` | Historical factor shocks: 2008 (SPY -38%, TLT +20%), COVID (SPY -34%), 2022 (TLT -31%, SPY -19%). Apply via factor betas | Scenario analysis |
| `compute_tail_dependence(ctx, tickers, prices)` | λ_L = lim P(U<u|V<u) as u→0. Empirical: count joint exceedances below q-percentile | Co-crash risk |

### 5C. Advanced Time Series (`tools/advanced_time_series.py` — NEW)

| Function | Math | Purpose |
|----------|------|---------|
| `compute_granger_causality(ctx, ticker1, ticker2, prices, max_lag=5)` | F-test: does lagged X improve Y prediction? VAR model with restricted/unrestricted comparison | Lead-lag relationships |
| `detect_change_points(ctx, ticker, prices, method="cusum")` | CUSUM: cumulative sum of deviations from mean. Change at max|Sₖ|. Also binary segmentation | Structural break detection |
| `compute_spectral_analysis(ctx, ticker, prices)` | FFT periodogram: P(f) = |FFT(x)|²/N. Report dominant frequencies | Cyclical patterns |
| `test_arch_effects(ctx, ticker, prices)` | Engle's ARCH LM test: regress ε² on lagged ε². LM stat ~ χ². Precondition for GARCH | GARCH applicability |

### 5D. Advanced Analytics (`tools/advanced_analytics.py` — NEW)

| Function | Math | Purpose |
|----------|------|---------|
| `compute_pca_returns(ctx, tickers, prices, n_components=5)` | SVD on standardized returns. Eigenvalues, variance explained, loadings | Factor extraction |
| `compute_hierarchical_clustering(ctx, tickers, prices)` | Distance = √(2·(1-ρ)). Ward/single linkage. Dendrogram + cluster assignments | Asset grouping |
| `compute_style_analysis(ctx, ticker, prices, factors)` | Sharpe (1992) RBSA: constrained regression rᵢ = Σwⱼ·r_factor + ε, Σwⱼ=1, wⱼ≥0. Rolling window | Time-varying style exposures |
| `compute_brinson_attribution(ctx, portfolio, benchmark, prices)` | Brinson-Fachler: Total = Allocation + Selection + Interaction | Performance attribution |
| `compute_information_entropy(ctx, weights)` | Shannon: H = -Σwᵢ·ln(wᵢ). KL divergence from market: D_KL = Σwᵢ·ln(wᵢ/w_mkt) | Diversification measure |
| `compute_mutual_information(ctx, tickers, prices)` | MI(X,Y) = H(X) + H(Y) - H(X,Y) via KNN estimation. Captures non-linear dependencies | Beyond-correlation dependency |

### 5E. Economic Data (`tools/economic_data.py` — NEW)

| Function | Data Source | Purpose |
|----------|------------|---------|
| `fetch_fred_series(ctx, series_id, start, end)` | FRED API | CPI, GDP, unemployment, Fed funds, treasury yields, HY spreads |
| `get_yield_curve(ctx)` | FRED: DGS1/2/5/10/30 | Slope (10Y-2Y), curvature, inversion detection |
| `get_economic_calendar(ctx)` | Web search | Upcoming FOMC, CPI, NFP, GDP dates with expected impact |
| `compute_macro_regime(ctx)` | Composite | Yield curve slope + credit spread + unemployment trend → expansion/slowdown/recession/recovery |

**New dependency**: `fredapi>=0.5`

**Files created**: `tools/advanced_portfolio.py`, `tools/advanced_risk.py`, `tools/advanced_time_series.py`, `tools/advanced_analytics.py`, `tools/economic_data.py`
**Files modified**: `agents/portfolio.py`, `agents/quantitative.py`, `agents/chat.py`, `pyproject.toml`

---

## Phase 6: Onboarding Flow + News Alert Pipeline

**Goal**: Guided new-user setup and intraday high-impact event alerts.

### 6A. Onboarding Agent (`agents/onboarding.py` — NEW)

Conversational agent (gpt-5-mini) with 5-step guided flow:

1. **Welcome** — explain system capabilities, ask if ready to set up
2. **Risk Profile** — ask risk tolerance (with explanations: conservative = capital preservation, moderate = balanced, aggressive = growth-focused) + time horizon
3. **Watchlist Setup** — suggest default sectors (US equities, bonds, commodities, crypto), let user pick/add/remove. Show what each covers
4. **Initial Portfolio** — option to input existing positions (e.g., "I hold 50% SPY, 30% QQQ, 20% cash") or start from 100% cash
5. **Preferences** — notification level, analysis depth, benchmark, any exclusions
6. **Complete** — confirm all settings, trigger first pre-compute run + daily analysis

Tools: `update_user_preference`, `update_watchlist`, `update_portfolio`, `set_onboarding_step` (new tool for state management)

### 6B. Telegram Integration

In `telegram_bot/commands.py` — modify `cmd_start`:
- Check `onboarding_state` table
- If `current_step != "done"`, start/resume onboarding flow
- If already done, show normal welcome

In `telegram_bot/chat_handler.py`:
- Check onboarding state before routing to chat agent
- If onboarding incomplete, route to onboarding agent instead

### 6C. News Alert Pipeline (`scheduler/alerts.py` — NEW)

Runs as part of midday/evening jobs (not a separate agent):

1. Call research agent with current watchlist
2. Compare returned themes against `research_themes` table
3. If NEW theme with `impact = "high"` found → send immediate Telegram alert
4. Update `research_themes` (mark old themes inactive, insert new ones)

Format for alerts:
```
**Market Alert**
[THEME]: [summary]
Impact: HIGH | Affected: AAPL, NVDA, SPY
Source: [url]
```

**Files created**: `agents/onboarding.py`, `scheduler/alerts.py`
**Files modified**: `telegram_bot/commands.py`, `telegram_bot/chat_handler.py`

---

## Summary of All Changes

### New Files (13)
| File | Phase | Purpose |
|------|-------|---------|
| `tools/precomputed.py` | 2 | Pre-compute pipeline + cache query tools |
| `tools/advanced_technical.py` | 3 | Ichimoku, VWAP, OBV, ADX, Stochastic, Fibonacci, Volume Profile |
| `tools/advanced_quant.py` | 4 | GARCH, HMM, Kalman Filter, Fama-French 3-factor |
| `tools/advanced_portfolio.py` | 5 | CVaR, HRP, Kelly, Max Diversification, Entropy, Transaction Costs |
| `tools/advanced_risk.py` | 5 | Cornish-Fisher VaR, EVT, Monte Carlo VaR, Stress Testing, Tail Dependence |
| `tools/advanced_time_series.py` | 5 | Granger Causality, Change Points, Spectral Analysis, ARCH Test |
| `tools/advanced_analytics.py` | 5 | PCA, Clustering, Style Analysis, Brinson Attribution, Entropy, Mutual Info |
| `tools/economic_data.py` | 5 | FRED integration, Yield Curve, Economic Calendar, Macro Regime |
| `agents/onboarding.py` | 6 | Guided new-user setup agent |
| `scheduler/alerts.py` | 6 | News alert pipeline |

### Modified Files (15)
| File | Phases | Changes |
|------|--------|---------|
| `db/schema.py` | 1 | +8 tables |
| `db/queries.py` | 1 | +20 query functions |
| `db/connection.py` | 1 | Migration logic for new preference columns |
| `config.py` | 1 | +15 settings |
| `models/preferences.py` | 1 | +13 preference fields |
| `agents/technical.py` | 1,3 | Model upgrade + advanced technical tools |
| `agents/quantitative.py` | 1,4 | Model upgrade + advanced quant tools |
| `agents/portfolio.py` | 1,5 | Model upgrade + advanced portfolio tools |
| `agents/reporting.py` | 1 | Lazy init (already gpt-5.2) |
| `agents/orchestrator.py` | 1 | Model upgrade + lazy init |
| `agents/chat.py` | 1,2,3,4,5 | Model upgrade + cached tools + all new tools |
| `agents/research.py` | 1 | Lazy init (stays gpt-5-mini) |
| `scheduler/runner.py` | 2 | 6 scheduled jobs (was 2) |
| `scheduler/jobs.py` | 2 | +4 new job functions |
| `telegram_bot/commands.py` | 6 | Onboarding in /start |
| `telegram_bot/chat_handler.py` | 2,6 | Persistent history + onboarding routing |
| `pyproject.toml` | 4,5 | +5 dependencies |

### New Dependencies
- `arch>=7.0` (GARCH)
- `hmmlearn>=0.3` (HMM)
- `statsmodels>=0.14` (econometrics)
- `scikit-learn>=1.3` (PCA, clustering)
- `fredapi>=0.5` (economic data)

---

## Verification Plan

After each phase:

1. **Phase 1**: `python -c "from portfolio_advisor.db.schema import ..."` — verify new DDL parses. Run `init_db()` and check all tables exist via `.tables`. Verify all agent lazy init works.
2. **Phase 2**: Force-trigger `precompute_job()` → verify `technical_indicators` and `quant_metrics` populated. Query via `get_cached_technical()`. Start bot, send chat message → verify it uses cached data.
3. **Phase 3**: Run each new indicator on SPY test data → verify JSON output. Run pre-compute → verify advanced columns populated.
4. **Phase 4**: Run GARCH on 1Y SPY data → verify persistence + forecast. Run HMM → verify 3 states detected. Run Kalman → verify adaptive beta.
5. **Phase 5**: Run CVaR optimization → verify weights sum to 100%. Run stress test → verify scenario impacts. Run PCA → verify variance explained. Test FRED API.
6. **Phase 6**: Fresh DB → `/start` → verify onboarding flow. Complete onboarding → verify preferences stored. Trigger midday job → verify alert sent for new theme.

Full end-to-end: Force daily run → verify pre-compute + orchestrator + Telegram output. Force weekly run → verify report uses cached data + advanced metrics.
