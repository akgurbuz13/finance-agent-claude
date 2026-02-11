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
"""
