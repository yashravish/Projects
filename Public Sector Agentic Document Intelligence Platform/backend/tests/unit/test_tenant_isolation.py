"""Static guarantee: every public service function in the listed service
modules accepts `organization_id` as a keyword.

This is the AST-level rail that prevents an "I forgot to filter by tenant" bug.
If a future contributor adds a public function without the parameter the test
fails loudly — the code reviewer can then either accept the exception (rare —
add the function name to `EXEMPT_FUNCTIONS`) or fix the function.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET_FILES = [
    ROOT / "app" / "services" / "document_service.py",
    ROOT / "app" / "services" / "ingestion_service.py",
    ROOT / "app" / "services" / "query_service.py",
    ROOT / "app" / "services" / "evaluation_service.py",
    ROOT / "app" / "services" / "training_service.py",
    ROOT / "app" / "services" / "audit_service.py",
]
EXEMPT_FUNCTIONS = {
    # Internal mutators that always operate on an already-tenant-scoped row.
    "_set_status",
    # Tenant-agnostic peek at a static dataset (no DB access).
    "get_default_dataset_view",
}


def _public_async_functions(tree: ast.AST) -> list[ast.AsyncFunctionDef]:
    out: list[ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            if node.name in EXEMPT_FUNCTIONS:
                continue
            out.append(node)
    return out


def test_public_service_functions_require_organization_id() -> None:
    failures: list[str] = []
    for path in TARGET_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in _public_async_functions(tree):
            kw_names = {a.arg for a in fn.args.kwonlyargs}
            pos_names = {a.arg for a in fn.args.args}
            if "organization_id" not in kw_names and "organization_id" not in pos_names:
                failures.append(f"{path.name}::{fn.name} is missing 'organization_id'")
    assert not failures, "tenant-isolation rail violations:\n  " + "\n  ".join(failures)
