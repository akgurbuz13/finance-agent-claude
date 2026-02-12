"""Tests for SQLite concurrency behavior — WAL mode, concurrent reads/writes."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from portfolio_advisor.db.connection import get_db, init_db


@pytest.fixture
async def db_path(tmp_path):
    """Create a temporary database with schema initialized."""
    path = str(tmp_path / "concurrency.db")
    await init_db(path)
    return path


async def test_wal_mode_enabled(db_path):
    """After init_db, the journal_mode should be WAL."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "wal"


async def test_concurrent_reads_during_write(db_path):
    """Start a writer task doing a slow INSERT, simultaneously start 3 readers.

    With WAL mode, readers should not block on a concurrent writer.
    """
    # Use a simple table for testing
    async with get_db(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS concurrency_test "
            "(id INTEGER PRIMARY KEY, value TEXT)"
        )
        await db.commit()

    write_started = asyncio.Event()
    write_done = asyncio.Event()
    reader_results = []

    async def writer():
        async with get_db(db_path) as db:
            # Insert rows one at a time with a small delay to simulate a slow write
            for i in range(10):
                await db.execute(
                    "INSERT INTO concurrency_test (value) VALUES (?)",
                    (f"row_{i}",),
                )
                if i == 0:
                    write_started.set()
                await asyncio.sleep(0.01)
            await db.commit()
        write_done.set()

    async def reader(reader_id: int):
        await write_started.wait()
        # Read while writer is active
        try:
            async with get_db(db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM concurrency_test")
                row = await cursor.fetchone()
                reader_results.append({"reader": reader_id, "count": row[0], "error": None})
        except Exception as e:
            reader_results.append({"reader": reader_id, "count": None, "error": str(e)})

    # Run writer and 3 readers concurrently
    tasks = [
        asyncio.create_task(writer()),
        asyncio.create_task(reader(1)),
        asyncio.create_task(reader(2)),
        asyncio.create_task(reader(3)),
    ]

    await asyncio.gather(*tasks)

    # All readers should succeed without SQLITE_BUSY
    assert len(reader_results) == 3
    for result in reader_results:
        assert result["error"] is None, f"Reader {result['reader']} got error: {result['error']}"
        assert result["count"] is not None


async def test_busy_timeout_prevents_lock_errors(db_path):
    """Two concurrent writers should both succeed due to busy_timeout."""
    async with get_db(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS busy_test "
            "(id INTEGER PRIMARY KEY, value TEXT)"
        )
        await db.commit()

    results = []

    async def writer(writer_id: int, count: int):
        try:
            async with get_db(db_path) as db:
                for i in range(count):
                    await db.execute(
                        "INSERT INTO busy_test (value) VALUES (?)",
                        (f"writer_{writer_id}_row_{i}",),
                    )
                    await asyncio.sleep(0.005)
                await db.commit()
            results.append({"writer": writer_id, "success": True, "error": None})
        except Exception as e:
            results.append({"writer": writer_id, "success": False, "error": str(e)})

    # Run two writers concurrently
    await asyncio.gather(
        writer(1, 20),
        writer(2, 20),
    )

    # Both should succeed
    assert len(results) == 2
    for result in results:
        assert result["success"], f"Writer {result['writer']} failed: {result['error']}"

    # Verify all rows were inserted
    async with get_db(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM busy_test")
        row = await cursor.fetchone()

    assert row[0] == 40
