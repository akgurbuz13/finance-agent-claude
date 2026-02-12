"""Root conftest — loads .env so live tests can find API keys."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any test collection
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)
