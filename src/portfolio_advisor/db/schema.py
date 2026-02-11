"""Database schema DDL."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    risk_tolerance TEXT NOT NULL DEFAULT 'moderate',
    time_horizon TEXT NOT NULL DEFAULT 'medium',
    excluded_assets TEXT NOT NULL DEFAULT '[]',
    allowed_regions TEXT NOT NULL DEFAULT '["US","EU","APAC","EM"]',
    cash_target_pct REAL NOT NULL DEFAULT 10.0,
    max_position_pct REAL NOT NULL DEFAULT 15.0,
    watchlist TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS portfolio_state (
    ticker TEXT PRIMARY KEY,
    weight_pct REAL NOT NULL,
    asset_class TEXT NOT NULL DEFAULT 'equity',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    state_json TEXT NOT NULL,
    trigger TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daily_briefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_date TEXT NOT NULL UNIQUE,
    content_json TEXT NOT NULL,
    market_summary TEXT,
    telegram_summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_daily_briefs_date ON daily_briefs(brief_date);

CREATE TABLE IF NOT EXISTS instrument_briefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal TEXT,
    confidence REAL,
    what_happened TEXT,
    why_it_matters TEXT,
    technical_json TEXT,
    quant_json TEXT,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(brief_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_instrument_briefs_date_ticker
    ON instrument_briefs(brief_date, ticker);

CREATE TABLE IF NOT EXISTS weekly_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_ending TEXT NOT NULL UNIQUE,
    content_json TEXT NOT NULL,
    executive_summary TEXT,
    allocations_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_weekly_reports_date ON weekly_reports(week_ending);

CREATE TABLE IF NOT EXISTS price_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    bar_date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT NOT NULL DEFAULT 'yfinance',
    UNIQUE(ticker, bar_date)
);

CREATE TABLE IF NOT EXISTS forecasts_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    forecast_type TEXT NOT NULL,
    horizon TEXT,
    predicted_value TEXT,
    actual_value TEXT,
    was_correct INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_forecasts_ticker_date
    ON forecasts_log(ticker, forecast_date);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usage_date TEXT NOT NULL,
    model TEXT NOT NULL,
    run_type TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── v2 Tables ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    indicator_date TEXT NOT NULL,
    snapshot_hour INTEGER DEFAULT 6,
    run_id TEXT,
    sma50 REAL,
    sma200 REAL,
    ema12 REAL,
    ema26 REAL,
    rsi_14 REAL,
    macd_line REAL,
    macd_signal REAL,
    macd_histogram REAL,
    atr_14 REAL,
    bb_upper REAL,
    bb_lower REAL,
    bb_bandwidth REAL,
    bb_pct_b REAL,
    pivot REAL,
    r1 REAL,
    r2 REAL,
    s1 REAL,
    s2 REAL,
    ichimoku_tenkan REAL,
    ichimoku_kijun REAL,
    ichimoku_senkou_a REAL,
    ichimoku_senkou_b REAL,
    ichimoku_chikou REAL,
    vwap REAL,
    obv REAL,
    adx REAL,
    stochastic_k REAL,
    stochastic_d REAL,
    fib_levels TEXT,
    overall_bias TEXT,
    overall_confidence REAL,
    narrative TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, indicator_date, snapshot_hour)
);

CREATE TABLE IF NOT EXISTS quant_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    metric_date TEXT NOT NULL,
    snapshot_hour INTEGER DEFAULT 6,
    run_id TEXT,
    return_1w_pct REAL,
    return_1w_ci_low REAL,
    return_1w_ci_high REAL,
    return_1m_pct REAL,
    return_1m_ci_low REAL,
    return_1m_ci_high REAL,
    return_3m_pct REAL,
    return_3m_ci_low REAL,
    return_3m_ci_high REAL,
    ewma_vol REAL,
    vol_regime TEXT,
    vol_percentile REAL,
    hurst REAL,
    regime TEXT,
    regime_confidence REAL,
    beta REAL,
    alpha REAL,
    r_squared REAL,
    skewness REAL,
    kurtosis REAL,
    sharpe REAL,
    sortino REAL,
    calmar REAL,
    garch_vol REAL,
    hmm_state TEXT,
    kalman_beta REAL,
    ff3_betas TEXT,
    cornish_fisher_var REAL,
    evt_var REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, metric_date, snapshot_hour)
);

CREATE TABLE IF NOT EXISTS daily_risk_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_date TEXT NOT NULL,
    snapshot_hour INTEGER DEFAULT 6,
    run_id TEXT,
    var_95 REAL,
    es_95 REAL,
    max_drawdown REAL,
    current_drawdown REAL,
    portfolio_beta REAL,
    asset_class_pcts TEXT,
    stress_test_results TEXT,
    diversification_ratio REAL,
    entropy_score REAL,
    yield_curve_slope REAL,
    yield_curve_inverted INTEGER,
    vix_level REAL,
    vix_regime TEXT,
    credit_spread REAL,
    macro_regime TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(risk_date, snapshot_hour)
);

CREATE TABLE IF NOT EXISTS research_themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_date TEXT NOT NULL,
    theme TEXT NOT NULL,
    summary TEXT,
    impact TEXT CHECK (impact IN ('high', 'medium', 'low')),
    affected_tickers TEXT DEFAULT '[]',
    sources TEXT DEFAULT '[]',
    source_tier TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_research_themes_date ON research_themes(theme_date);
CREATE INDEX IF NOT EXISTS idx_research_themes_active ON research_themes(is_active);

CREATE TABLE IF NOT EXISTS forecast_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER NOT NULL,
    evaluation_date TEXT NOT NULL,
    predicted_direction TEXT,
    actual_direction TEXT,
    predicted_return_pct REAL,
    actual_return_pct REAL,
    absolute_error REAL,
    is_direction_correct INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(forecast_id),
    FOREIGN KEY (forecast_id) REFERENCES forecasts_log(id)
);

CREATE TABLE IF NOT EXISTS onboarding_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_step TEXT NOT NULL DEFAULT 'welcome',
    steps_completed TEXT NOT NULL DEFAULT '[]',
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chat_history_chat_created
    ON chat_history(chat_id, created_at);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    tickers_processed TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    duration_seconds REAL,
    error_message TEXT
);

-- ── v3 Tables ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    earnings_time TEXT,
    eps_estimate REAL,
    revenue_estimate REAL,
    eps_actual REAL,
    revenue_actual REAL,
    eps_surprise_pct REAL,
    revenue_surprise_pct REAL,
    status TEXT NOT NULL DEFAULT 'upcoming',
    source TEXT DEFAULT 'yfinance',
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, earnings_date)
);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date ON earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker ON earnings_calendar(ticker);

CREATE TABLE IF NOT EXISTS correlation_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    run_id TEXT,
    correlation_matrix TEXT NOT NULL,
    top_correlations TEXT,
    diversification_score REAL,
    cluster_assignments TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(snapshot_date)
);
"""
