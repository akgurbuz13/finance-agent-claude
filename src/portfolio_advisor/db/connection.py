"""Async SQLite connection manager."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

from portfolio_advisor.db.schema import SCHEMA_SQL


async def init_db(db_path: str) -> None:
    """Create tables if they don't exist and seed defaults."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        # Seed default user_preferences row if missing
        cursor = await db.execute("SELECT COUNT(*) FROM user_preferences")
        (count,) = await cursor.fetchone()
        if count == 0:
            await db.execute(
                "INSERT INTO user_preferences (id) VALUES (1)"
            )
        await db.commit()


@asynccontextmanager
async def get_db(db_path: str) -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection with row_factory enabled."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()
