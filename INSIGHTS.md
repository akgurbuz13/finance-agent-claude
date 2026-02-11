# Portfolio Advisor v2 — Architecture Review & Insights

**Review Date:** 2026-02-11
**System Version:** Phase 6 Complete (86 chat tools, 18 DB tables, 7 scheduled jobs)

This document synthesizes findings from a comprehensive 4-agent deep-dive analysis of the Portfolio Advisor system architecture, focusing on tool efficiency, data freshness, news pipeline responsiveness, and agent dispatch performance.

---

## Executive Summary

The Portfolio Advisor system is a sophisticated multi-agent financial analysis platform with strong foundational patterns (pre-compute pipeline, agents-as-tools orchestrators, cache-first strategy) but three critical inefficiencies emerged:

1. **86-tool flat architecture** in chat agent (4-5× OpenAI's recommended maximum)
2. **Intraday snapshot loss** due to database overwrite behavior
3. **6-10 hour news blind spots** between scheduled runs

**Quick Win Potential:** Refactoring the chat agent to hierarchical architecture (like orchestrators already do) would reduce token overhead by 86%, improve tool selection accuracy, and save ~$1,620/year in API costs.

---

## 1. The 86-Tool Problem (P0 - Critical)

### Current State
- Chat agent has 86 tools directly attached (flat architecture)
- Generates ~25,800 tokens of tool schemas per request
- OpenAI's sweet spot: 15-25 tools; degradation beyond 50

### Impact
- **Context dilution**: 25k tokens of schemas crowds out conversation history
- **Selection confusion**: 4 VaR tools, 3 regime detectors, 8 optimizers
- **Cost**: $1,620/year wasted on redundant tool schemas (100 requests/day)
- **Latency**: LLM processes 86 descriptions on every turn

### Tool Redundancy Clusters

| Cluster | Redundant Tools | User Intent |
|---------|----------------|-------------|
| **VaR** | `compute_var`, `cornish_fisher_var`, `evt_var`, `monte_carlo_var` | "What's my downside risk?" |
| **Regime** | `detect_regime`, `detect_regime_hmm`, `compute_macro_regime` | "What's the market regime?" |
| **Volatility** | `compute_vol_forecast`, `compute_garch_volatility` | "How volatile is this?" |
| **Optimization** | 8 different `optimize_*` methods | "How should I allocate?" |
| **Time Series** | 9 academic tools (ACF, ADF, spectral, Granger...) | Rarely used in chat |
| **Analytics** | PCA, clustering, Brinson attribution, entropy, mutual info | Institutional overkill |

### Solution: Hierarchical Architecture

The orchestrators already demonstrate the correct pattern. Apply it to chat:

```
Chat Agent (~25 tools)
├── Technical Agent.as_tool()     → 14 tools hidden
├── Quantitative Agent.as_tool()  → 23 tools hidden
├── Portfolio Agent.as_tool()     → 24 tools hidden
├── Research Agent.as_tool()      → web search
├── Cache tools (5)
├── Market data (2)
├── Portfolio state (3)
├── User preferences (3)
├── Database queries (3)
├── Economic data (4)
└── Token tracking (1)
```

**Benefits:**
- 86% token reduction per request (~25,800 → ~3,600 tokens)
- Better tool selection (clear delegation signals)
- Improved maintainability (add tools to specialists, not chat agent)
- Proven pattern (orchestrators work this way already)

**Effort:** 2-3 hours refactor, 2-3 hours testing
**Risk:** Low (pattern proven by orchestrators)
**Annual Savings:** ~$1,620 (at 100 requests/day, gpt-5.2 pricing)

---

## 2. Data Freshness & Historical Comparison Gaps (P1)

### Market Data Reality
- **NOT real-time**: yfinance = 15-20 min delayed, CoinGecko = 5-10 min delayed
- **Batch refresh**: 2-3x daily (06:00, 13:00, 20:00 UTC)
- **Staleness threshold**: 8 hours (configurable via `PA_PRECOMPUTE_STALE_HOURS`)

### Critical Gap: Intraday Snapshot Loss

**The Problem:**
Tables use `UNIQUE(ticker, indicator_date)` constraint with `ON CONFLICT DO UPDATE`:
- Morning precompute (06:00): Inserts row with today's date
- Midday precompute (13:00): **Overwrites** morning row with same date

**What's Lost:**
- "RSI moved from 65 at 6am to 72 at 1pm" — impossible to answer
- "How did the technical bias change during the day?" — data gone
- "Show me intraday signal progression" — not stored

**Workaround (Temporary):**
The `midday_update_job` reads morning signals into memory before overwriting, compares to midday, and sends Telegram alerts for reversals. But once the job finishes, morning data is lost forever.

### Day-Over-Day Works, But Barely

**Existing Tool:** `get_signal_history(ticker, days=7)`
- Returns: overall bias + confidence per day
- Does NOT return: individual indicator values (RSI, MACD, beta, vol)

**Can answer:** "Signal changed from bullish to bearish between yesterday and today"
**Cannot answer:** "RSI went from 65 yesterday to 72 today" (requires custom query)

### Price Cache: Written But Never Read

- `price_cache` table has no TTL, grows indefinitely
- `fetch_ohlcv` always hits yfinance API, caches result
- But there's no cache-first check — never queries `price_cache` before fetching
- Wasted API calls, potential rate limits

### Solutions

**Short-term (Small effort):**
1. Add `get_technical_indicator_history(ticker, indicator_name, days)` tool
2. Add `get_quant_metric_history(ticker, metric_name, days)` tool
3. Implement cache-first logic for `price_cache`

**Medium-term (Moderate refactor):**
1. Change UNIQUE constraint to `(ticker, indicator_date, computed_at_hour)`
2. Store 3 snapshots per day (morning, midday, evening) instead of overwriting
3. Add DB cleanup job to prune data beyond 90 days

---

## 3. News Pipeline: 6-10 Hour Blind Spots (P1)

### Current Architecture

**Scheduled Analysis:** 3x daily
- Morning (06:30 UTC): Daily orchestrator calls research agent
- Midday (13:30 UTC): Standalone news alert pipeline
- Evening (20:00 UTC): Standalone news alert pipeline

**Research Agent (Sophisticated):**
- 6-search strategy (macro, market, sector, ticker-specific)
- Source tiering (Tier 1: Reuters/Bloomberg, Tier 2: CNBC, Tier 3: Twitter)
- Impact classification (High/Medium/Low)
- Structured JSON output with themes, affected tickers, sources
- Theme storage in `research_themes` table with deduplication

**Chat Agent (Basic):**
- Has WebSearchTool but uses different workflow
- Max 3 searches per response (budget constraint)
- No theme storage, no impact classification
- Results not tracked for novelty

### What Gets Missed

**Blind spots between scheduled runs:**
- Fed rate decision (14:00 UTC) → 6.5 hour gap until evening job
- After-hours earnings (16:00 UTC) → 14.5 hour gap until next morning
- Overnight geopolitical event (02:00 UTC) → 4.5 hour gap until morning
- Flash crash (11:00 UTC) → 2.5 hour gap until midday job

**No infrastructure for:**
- Real-time breaking news detection
- Event-driven triggers (webhooks, RSS polling)
- User-triggered full research pipeline (chat gets generic search, not research agent)

### Theme Deduplication Is Primitive

- Simple case-insensitive string matching on theme titles
- Semantically similar themes with different wording treated as new
- No embedding-based similarity or clustering

### Solutions

**Short-term (No architecture change):**
1. Add `/news` command to trigger research agent on-demand
2. Increase scheduled frequency (add 09:00, 15:00 UTC runs → 3-4 hour gaps)
3. Use sentence embeddings for theme similarity (flag "related" vs "new")

**Medium-term (Moderate refactor):**
1. RSS feed polling (Reuters/Bloomberg every 15 min) → trigger research agent on watchlist matches
2. WebSocket integration for large price moves (>2% in 5 min) → trigger research agent
3. Let chat agent call `get_research_agent().as_tool()` for on-demand systematic analysis

**Long-term (Architecture change):**
1. Event-driven architecture (webhooks for external news services)
2. Dedicated news monitoring microservice (continuous polling, shared DB)
3. Streaming news API integration (Bloomberg Terminal, Reuters Elektron)

---

## 4. Architecture Patterns: What Works Well

### ✅ Strong Patterns (Keep These)

1. **Pre-compute Pipeline**
   - Batch compute 2-3x daily, cache-first chat strategy
   - Pure `_raw()` functions + `@function_tool` wrappers for reusability
   - Freshness checking via `analysis_runs` table

2. **Orchestrators Use Hierarchical Agents-as-Tools**
   - Daily/Weekly orchestrators have 5-7 tools (not 86)
   - Call specialist agents (Technical, Quant, Portfolio, Research) via `.as_tool()`
   - Proven pattern, works well

3. **Research Agent Prompt Design**
   - Source tiering (Tier 1/2/3), impact classification (High/Medium/Low)
   - Structured JSON output, systematic search strategy
   - Better than generic web search

4. **Database Schema Design**
   - Appropriate use of UNIQUE constraints for daily snapshots
   - Indexed by date columns for time-series queries
   - JSON columns for flexible nested data (fib_levels, ff3_betas)

5. **Config-Driven Model Selection**
   - Per-agent model assignment via pydantic settings
   - Lazy initialization pattern for all agents
   - Centralized in `config.py`

### ⚠️ Anti-Patterns (Fix These)

1. **Chat Agent Flat Tool Architecture**
   - 86 tools directly attached (should be hierarchical like orchestrators)
   - Violates OpenAI best practices (4-5× recommended max)

2. **Database Overwrite Without History**
   - `ON CONFLICT DO UPDATE` loses intraday snapshots
   - No historical comparison for detailed metrics

3. **Unused Cache Infrastructure**
   - `price_cache` written but never read
   - Always fetches from API instead of checking cache

4. **Research Agent Not Accessible from Chat**
   - Chat has WebSearchTool but can't trigger full research pipeline
   - Ad-hoc searches don't store themes or apply impact classification

---

## 5. Prioritized Recommendations

| Priority | Issue | Impact | Effort | Annual Savings/Benefit |
|----------|-------|--------|--------|------------------------|
| **P0** | Refactor chat agent to hierarchical | 86% token reduction, better accuracy | 4-6 hours | ~$1,620 + better UX |
| **P1** | Store intraday snapshots (don't overwrite) | Enable "RSI 65→72" comparisons | 2-3 hours | Better insights |
| **P1** | Add day-over-day metric comparison tools | Detailed historical analysis | 2-3 hours | Better insights |
| **P1** | Let chat agent trigger research pipeline | On-demand news with theme storage | 2-3 hours | Real-time capability |
| **P2** | Implement cache-first for price_cache | Reduce API calls, faster responses | 1-2 hours | Lower costs, speed |
| **P2** | Increase news check frequency (5x daily) | Reduce blind spots to 3-4 hours | 30 min | Better coverage |
| **P3** | Add DB cleanup job for old data | Prevent database bloat | 1-2 hours | Maintenance |
| **P3** | Semantic theme deduplication | Better novelty detection | 2-3 hours | Fewer false alerts |

---

## 6. Implementation Roadmap

### Phase 1: Hierarchical Chat Agent (Week 1)

**Goal:** Reduce chat agent from 86 to ~25 tools using agents-as-tools pattern

**Steps:**
1. Read `src/portfolio_advisor/agents/orchestrator.py` to understand `.as_tool()` pattern
2. Modify `src/portfolio_advisor/agents/chat.py`:
   - Import 4 specialist agent factories
   - Replace 86-tool list with 4 agent tools + 21 direct tools
   - Update `CHAT_AGENT_INSTRUCTIONS` with delegation strategy
3. Test delegation:
   - "Technical analysis of AAPL" → should call Technical Agent
   - "Expected return for QQQ" → should call Quant Agent
   - "Optimize my portfolio" → should call Portfolio Agent
   - "Market news" → should call Research Agent
4. Test direct tools:
   - "Show my portfolio" → should use `get_current_portfolio` directly
   - "Add TSLA to watchlist" → should use `update_watchlist` directly

**Success Metrics:**
- Token overhead drops from ~25,800 to ~3,600 per request
- Tool selection accuracy improves (manual review of 20 test queries)
- Response quality maintained or improved

### Phase 2: Data Freshness Improvements (Week 2)

**Goal:** Enable intraday and day-over-day metric comparisons

**Steps:**
1. Add timestamp column to `technical_indicators` and `quant_metrics`:
   ```sql
   ALTER TABLE technical_indicators ADD COLUMN computed_at TEXT;
   CREATE INDEX idx_technical_computed ON technical_indicators(ticker, indicator_date, computed_at);
   ```
2. Change UNIQUE constraint to `(ticker, indicator_date, computed_at_hour)` to allow 3 snapshots/day
3. Add query functions in `db/queries.py`:
   - `get_technical_indicator_history(ticker, indicator_name, days)`
   - `get_quant_metric_history(ticker, metric_name, days)`
4. Add tools in `tools/precomputed.py`:
   - `get_indicator_trend` (wraps query, returns trend analysis)
   - `compare_metrics_day_over_day` (side-by-side comparison)
5. Implement cache-first for `fetch_ohlcv`:
   ```python
   # Check price_cache first
   cached = await get_cached_prices(db, ticker, start, end)
   if cached and len(cached) >= expected_days:
       return cached
   # Otherwise fetch from yfinance and cache
   ```

### Phase 3: News Pipeline Enhancements (Week 3)

**Goal:** Reduce news blind spots and enable on-demand research

**Steps:**
1. Add 2 more scheduled news checks (09:00, 15:00 UTC) → 5x daily, 3-4 hour gaps
2. Add `/news` command in `telegram_bot/bot.py`:
   ```python
   @_auth
   async def cmd_news(update, context):
       ctx = await _build_context()
       result = await run_news_alert_pipeline(ctx)
       themes = parse_themes(result.final_output)
       msg = format_themes_for_telegram(themes)
       await update.message.reply_text(msg)
   ```
3. Expose research agent as tool to chat agent:
   ```python
   get_research_agent().as_tool(
       tool_name="run_market_research",
       tool_description="..."
   )
   ```
4. Update chat agent instructions to delegate news queries to research agent
5. Add semantic theme deduplication:
   - Use sentence-transformers for embedding-based similarity
   - Flag themes as "related" if cosine similarity > 0.85

---

## 7. Key Files Reference

### Agent Architecture
- `src/portfolio_advisor/agents/chat.py` — Chat agent (needs refactor from 86 → 25 tools)
- `src/portfolio_advisor/agents/orchestrator.py` — Good hierarchical pattern (reference this)
- `src/portfolio_advisor/agents/technical.py` — 14 tools (specialist)
- `src/portfolio_advisor/agents/quantitative.py` — 23 tools (specialist)
- `src/portfolio_advisor/agents/portfolio.py` — 24 tools (specialist)
- `src/portfolio_advisor/agents/research.py` — 1 tool (specialist, sophisticated prompt)

### Pre-Compute Pipeline
- `src/portfolio_advisor/tools/precomputed.py` — Core pipeline + cache tools
- `src/portfolio_advisor/scheduler/jobs.py` — 7 scheduled jobs (precompute, daily, midday, evening, weekly, forecast eval)
- `src/portfolio_advisor/scheduler/runner.py` — APScheduler configuration

### Data Layer
- `src/portfolio_advisor/db/schema.py` — 18 tables, UNIQUE constraints
- `src/portfolio_advisor/db/queries.py` — Typed CRUD, upsert logic
- `src/portfolio_advisor/tools/market_data.py` — OHLCV fetching + caching

### News Pipeline
- `src/portfolio_advisor/scheduler/alerts.py` — News alert pipeline (theme parsing, deduplication, Telegram alerts)
- `src/portfolio_advisor/agents/research.py` — Research agent with 6-search strategy

---

## 8. Decision Log

### Why Hierarchical Over More Tools?

**Considered alternatives:**
1. Keep 86 flat tools, improve descriptions → Still 25k tokens/request
2. Remove redundant tools → Gets to ~50 tools, still 2× recommended max
3. Hierarchical agents-as-tools → Gets to ~25 tools, aligns with best practices

**Chose hierarchical because:**
- Orchestrators already prove the pattern works
- 86% token reduction is massive
- Easier to maintain (add tools to specialists, not chat agent)
- Better delegation signals in user queries

### Why Not Remove Redundant VaR/Regime/Optimization Tools?

**We will, but in specialist agents, not chat agent:**
- Portfolio Agent can decide which VaR method to use internally
- Quant Agent can decide between HMM and Hurst-based regime detection
- Chat agent doesn't need to know about these alternatives

**This is the hierarchical design advantage:**
- Chat agent: "I need portfolio risk analysis" → delegates to Portfolio Agent
- Portfolio Agent: "User wants downside risk, I'll use Monte Carlo VaR for this scenario"

### Why Store Intraday Snapshots vs. Just Cache Deltas?

**Considered alternatives:**
1. Store only changes in `signal_changes` table → Requires reconstruction to answer queries
2. Store 24 hourly snapshots → Overkill, storage bloat
3. Store 3 snapshots per day (morning, midday, evening) → Goldilocks

**Chose 3 snapshots because:**
- Aligns with existing precompute schedule (06:00, 13:00, 20:00)
- Enables "RSI at 6am vs 1pm vs 8pm" comparisons
- Small storage overhead (3× current size, manageable)
- No reconstruction logic needed

---

## 9. Metrics to Track Post-Implementation

### Chat Agent Refactor
- **Token overhead per request**: Target <5,000 (currently ~25,800)
- **Tool selection accuracy**: Manual review of 50 queries (% correct on first try)
- **Response latency**: Average time to first token (expect slight increase from delegation)
- **User satisfaction**: Qualitative feedback on response quality

### Data Freshness
- **Intraday comparison queries**: Track usage of new `get_indicator_trend` tool
- **Cache hit rate**: % of `fetch_ohlcv` calls served from `price_cache`
- **API call reduction**: Track yfinance API calls before/after cache-first

### News Pipeline
- **News check frequency**: 3x → 5x daily (confirmed by scheduler logs)
- **Average blind spot**: Target <4 hours (currently 6-10 hours)
- **On-demand news usage**: Track `/news` command usage
- **Theme novelty rate**: % of themes flagged as new (expect decrease with semantic dedup)

---

## 10. Conclusion

The Portfolio Advisor system has a solid foundation with sophisticated agents, a well-designed pre-compute pipeline, and good separation of concerns. The three critical inefficiencies (86-tool flat chat agent, intraday snapshot loss, news blind spots) are all solvable with low-to-moderate effort.

**The P0 item (hierarchical chat agent) is the clear winner:**
- Highest impact (86% token reduction, better accuracy)
- Proven pattern (orchestrators already work this way)
- Low risk (4-6 hours of work, thorough testing plan)
- Immediate benefits ($1,620/year savings + better UX)

**Next steps:**
1. Review this document and prioritize which items to tackle first
2. Create feature branch for P0 refactor
3. Implement Phase 1 (hierarchical chat agent)
4. Test thoroughly with representative user queries
5. Deploy and monitor metrics
6. Move to Phase 2 (data freshness) if Phase 1 succeeds

---

**Document Version:** 1.0
**Last Updated:** 2026-02-11
**Next Review:** After Phase 1 implementation
