"""News alert pipeline — detects new high-impact research themes and sends Telegram alerts."""

from __future__ import annotations

import json
import logging
from datetime import date
from difflib import SequenceMatcher

from agents import Runner

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.agents.research import get_research_agent
from portfolio_advisor.config import get_settings
from portfolio_advisor.db import queries
from portfolio_advisor.db.connection import get_db
from portfolio_advisor.telegram_bot.bot import send_message

logger = logging.getLogger(__name__)


async def run_news_alert_pipeline(ctx: AppContext) -> dict:
    """Run the research agent and alert on new high-impact themes.

    Steps:
    1. Pre-fetch structured news from Massive (if available) for context.
    2. Call the research agent with the current watchlist (batched if >15 tickers).
    3. Parse returned themes from the agent output.
    4. Compare against existing themes in the research_themes table.
    5. If new HIGH impact themes are found, send immediate Telegram alerts.
    6. Store new themes and deactivate old ones.

    Returns a summary dict with counts of new/existing/alerted themes.
    """
    settings = get_settings()
    today = date.today().isoformat()

    # 1. Pre-fetch structured news for context enrichment (Phase 2.3)
    news_context = ""
    if ctx.providers is not None:
        try:
            articles = await ctx.providers.fetch_news(ctx.watchlist[:20], limit=15)
            if articles:
                news_parts = []
                for a in articles[:10]:
                    sentiments = a.get("sentiments", {})
                    sent_str = ""
                    if sentiments:
                        sent_parts = [
                            f"{t}={s['sentiment']}" for t, s in sentiments.items()
                        ]
                        sent_str = f" [Sentiment: {', '.join(sent_parts)}]"
                    news_parts.append(f"- {a.get('title', '')}{sent_str}")
                news_context = (
                    "\n\nPre-fetched news with sentiment:\n" + "\n".join(news_parts)
                )
        except Exception as e:
            logger.debug(f"News pre-fetch failed (non-critical): {e}")

    # 2. Run research agent (batched for large watchlists — Phase 2.4)
    all_raw_outputs = []
    batch_size = 8
    watchlist = ctx.watchlist

    if len(watchlist) > 15:
        # Batch into groups of 8
        batches = [watchlist[i:i + batch_size] for i in range(0, len(watchlist), batch_size)]
    else:
        batches = [watchlist]

    for batch in batches:
        prompt = (
            f"Research current market-moving news and macro developments for: "
            f"{', '.join(batch)}\n"
            f"Date: {today}. Focus on high-impact themes that change the investment thesis."
            f"{news_context}"
        )

        try:
            result = await Runner.run(
                starting_agent=get_research_agent(),
                input=prompt,
                context=ctx,
            )
            raw_output = result.final_output or ""
            all_raw_outputs.append(raw_output)
        except Exception as e:
            logger.error(f"Research agent failed in alert pipeline: {e}")

    if not all_raw_outputs:
        return {"error": "All research batches failed", "new_themes": 0, "alerts_sent": 0}

    raw_output = all_raw_outputs[0]  # For backward compat with single-batch path

    # 3. Parse themes from all batch outputs (dedup across batches)
    themes = []
    for output in all_raw_outputs:
        batch_themes = _parse_themes(output)
        themes.extend(batch_themes)
    if not themes:
        logger.info("News alert pipeline: no themes parsed from research output")
        return {"new_themes": 0, "alerts_sent": 0, "parse_failed": not raw_output}

    # 3. Get existing themes to detect novelty (with similarity dedup)
    async with get_db(settings.db_path) as db:
        existing = await queries.get_active_research_themes(db, days=3)
        existing_titles = [t["theme"].lower().strip() for t in existing]

        # 4. Identify genuinely new themes (similarity-based dedup)
        new_themes = []
        for theme in themes:
            title = theme.get("theme", "").lower().strip()
            if not title:
                continue
            if _is_similar_to_existing(title, existing_titles):
                continue
            new_themes.append(theme)
            # Add to existing_titles so subsequent themes in this batch
            # are also deduped against each other
            existing_titles.append(title)

        # 5. Store new themes
        for theme in new_themes:
            theme_data = {
                "theme_date": today,
                "theme": theme.get("theme", ""),
                "summary": theme.get("summary", ""),
                "impact": theme.get("impact", "medium"),
                "affected_tickers": theme.get("affected_tickers", []),
                "sources": theme.get("sources", []),
                "source_tier": theme.get("source_tier", ""),
                "is_active": 1,
            }
            await queries.store_research_theme(db, theme_data)

        # Deactivate stale themes
        await queries.deactivate_old_themes(db, days=7)

    # 6. Send alerts for new HIGH impact themes
    alerts_sent = 0
    for theme in new_themes:
        if theme.get("impact", "").lower() == "high":
            affected = theme.get("affected_tickers", [])
            tickers_str = ", ".join(affected[:8]) if affected else "broad market"
            sources = theme.get("sources", [])
            source_str = f"\nSource: {sources[0]}" if sources else ""

            alert_msg = (
                f"**Market Alert**\n\n"
                f"**{theme.get('theme', 'New Development')}**\n"
                f"{theme.get('summary', '')}\n\n"
                f"Impact: HIGH | Affected: {tickers_str}"
                f"{source_str}"
            )
            try:
                await send_message(alert_msg)
                alerts_sent += 1
            except Exception as e:
                logger.warning(f"Failed to send alert: {e}")

    logger.info(
        f"News alert pipeline: {len(themes)} themes found, "
        f"{len(new_themes)} new, {alerts_sent} alerts sent"
    )

    return {
        "total_themes": len(themes),
        "new_themes": len(new_themes),
        "alerts_sent": alerts_sent,
    }


def _parse_themes(raw_output: str) -> list[dict]:
    """Extract themes list from research agent output.

    The research agent returns JSON with a 'themes' key, but the output
    may also contain markdown or other text wrapping.
    """
    if not raw_output:
        return []

    # Try direct JSON parse
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "themes" in data:
            return data["themes"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code fences
    for marker in ("```json", "```"):
        if marker in raw_output:
            try:
                start = raw_output.index(marker) + len(marker)
                end = raw_output.index("```", start)
                block = raw_output[start:end].strip()
                data = json.loads(block)
                if isinstance(data, dict) and "themes" in data:
                    return data["themes"]
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

    # Try to find a JSON object anywhere in the text
    brace_start = raw_output.find("{")
    if brace_start >= 0:
        # Find the matching closing brace by tracking depth
        depth = 0
        for i in range(brace_start, len(raw_output)):
            if raw_output[i] == "{":
                depth += 1
            elif raw_output[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw_output[brace_start:i + 1])
                        if isinstance(data, dict) and "themes" in data:
                            return data["themes"]
                    except json.JSONDecodeError:
                        pass
                    break

    return []


# ── Theme similarity helpers ─────────────────────────────────────────────────

_SIMILARITY_THRESHOLD = 0.75


def _theme_similarity(a: str, b: str) -> float:
    """Compute similarity between two theme titles.

    Uses SequenceMatcher (stdlib) for fuzzy string matching, plus
    keyword overlap for additional signal. Returns a score 0.0-1.0.
    """
    # Sequence-level similarity
    seq_ratio = SequenceMatcher(None, a, b).ratio()

    # Keyword overlap (handles reworded but semantically identical themes)
    words_a = set(a.split())
    words_b = set(b.split())
    # Remove very common words that don't carry meaning
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "is", "are", "at"}
    words_a -= stopwords
    words_b -= stopwords
    if words_a and words_b:
        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    else:
        overlap = 0.0

    # Weighted combination: sequence similarity is primary, keyword overlap is secondary
    return 0.6 * seq_ratio + 0.4 * overlap


def _is_similar_to_existing(title: str, existing_titles: list[str]) -> bool:
    """Check if a theme title is similar enough to any existing theme to be a duplicate."""
    for existing in existing_titles:
        if _theme_similarity(title, existing) >= _SIMILARITY_THRESHOLD:
            return True
    return False
