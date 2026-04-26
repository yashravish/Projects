import os
import pathlib

# Local pytest uses SQLite; CI provides DATABASE_URL (Postgres) via the workflow.
if "DATABASE_URL" not in os.environ:
    _ROOT = pathlib.Path(__file__).resolve().parent.parent
    _DB = _ROOT / ".pytest.sqlite"
    if _DB.exists():
        _DB.unlink()
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB.as_posix()}"
