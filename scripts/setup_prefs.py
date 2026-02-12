"""One-time script: skip onboarding and set user preferences directly."""
import asyncio
import aiosqlite


async def setup():
    db_path = "/opt/finance-agent-claude/data/portfolio.db"
    async with aiosqlite.connect(db_path) as db:
        # Mark onboarding as done
        await db.execute(
            "INSERT OR REPLACE INTO onboarding_state "
            "(id, current_step, steps_completed, completed_at) "
            "VALUES (1, 'done', "
            "'[\"risk\",\"watchlist\",\"portfolio\",\"preferences\",\"confirm\"]', "
            "datetime('now'))"
        )

        # Set user preferences for active weekly signal trading
        await db.execute(
            "INSERT OR REPLACE INTO user_preferences "
            "(id, risk_tolerance, time_horizon, investment_style, rebalance_frequency, "
            "cash_target_pct, max_position_pct, max_crypto_pct, min_bond_pct, "
            "notification_level, analysis_depth, notes) "
            "VALUES (1, 'aggressive', 'medium', 'active', 'weekly', "
            "5.0, 20.0, 20.0, 0.0, "
            "'high', 'detailed', "
            "'Weekly signal bot: max growth while preserving principal. "
            "Rebalance weekly based on buy/sell signals.')"
        )

        await db.commit()

        async with db.execute(
            "SELECT current_step, completed_at FROM onboarding_state"
        ) as c:
            row = await c.fetchone()
            print(f"Onboarding: step={row[0]}, completed={row[1]}")
        async with db.execute(
            "SELECT risk_tolerance, time_horizon, investment_style, "
            "rebalance_frequency, notes FROM user_preferences"
        ) as c:
            row = await c.fetchone()
            print(f"Prefs: risk={row[0]}, horizon={row[1]}, "
                  f"style={row[2]}, rebalance={row[3]}")
            print(f"Notes: {row[4]}")


asyncio.run(setup())
