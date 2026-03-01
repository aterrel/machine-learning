# Code Standards

This document defines code quality standards for this project.

> **Stack-specific note**: This document covers Python/FastAPI standards by default.
> If your project uses a different stack, adapt or replace the examples.
> The `bootstrap` command will update CLAUDE.md with stack-appropriate commands.
> Add language-specific standards files (e.g., `typescript-standards.md`) as needed.

---

## Formatting and Linting

All backend code must pass:
```bash
ruff check src/     # linting
ruff format src/    # formatting (line-length 100)
```

Update these commands in CLAUDE.md to match your actual source directory.

**Recommended ruff config** (add to `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "S"]
ignore = ["E501"]
```

---

## Type Hints

All public functions must have type hints:

```python
# Good
async def get_item(item_id: int, db: AsyncSession) -> Item | None: ...

# Bad
async def get_item(item_id, db): ...
```

---

## Naming Conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_leading_underscore`
- Module-private helpers: `_leading_underscore`

---

## Imports

```python
# Standard library first
from __future__ import annotations
from typing import Any

# Third-party second
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

# Local last
from app.models.item import Item
from app.services.item_service import ItemService
```

---

## Documentation

All public functions must have docstrings:

```python
async def process_item(item: Item) -> ProcessedItem:
    """Process an item and return the result.

    Parameters
    ----------
    item : Item
        The item to process.

    Returns
    -------
    ProcessedItem
        The processed result.

    Raises
    ------
    ProcessingError
        If processing fails and no fallback is available.
    """
```

---

## Error Handling

### Use custom exception types for domain errors:

```python
# src/exceptions.py
class AppError(Exception):
    """Base exception for this application."""

class ValidationError(AppError):
    """Raised when input validation fails."""

class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""
```

### Rules:
- Never use bare `except:`
- Always use specific exception types
- Raise with informative messages
- Use `from exc` when chaining exceptions:
  ```python
  try:
      result = external_api.call()
  except ExternalAPIError as exc:
      raise ServiceError(f"External API call failed") from exc
  ```
- API layer should use `HTTPException` with appropriate status codes

---

## Async Standards

- All database operations must use `AsyncSession` and `await`
- All HTTP calls to external services must use async clients
- Never call blocking I/O in an async context — use `asyncio.to_thread()` if needed

```python
# Good — async database query
async def get_item(db: AsyncSession, item_id: int) -> Item | None:
    result = await db.execute(select(Item).where(Item.id == item_id))
    return result.scalar_one_or_none()
```

---

## Testing Standards

### Test file structure
```
tests/
├── conftest.py           # Shared fixtures
├── api/                  # API endpoint tests
│   └── test_items.py
├── services/             # Service layer tests
│   └── test_item_service.py
└── models/               # Model tests
    └── test_models.py
```

### pytest conventions
```python
import pytest


class TestItemAPI:
    """Tests for /api/v1/items endpoints."""

    @pytest.mark.asyncio
    async def test_create_item_returns_201(self, async_client, test_db):
        response = await async_client.post("/api/v1/items", json={"name": "test"})
        assert response.status_code == 201
        assert response.json()["name"] == "test"

    @pytest.mark.asyncio
    async def test_get_item_not_found_returns_404(self, async_client):
        response = await async_client.get("/api/v1/items/99999")
        assert response.status_code == 404
```

---

## Security Standards

Non-negotiable for any project:

- **Never log sensitive data** — passwords, tokens, PII
- **Validate all inputs** at API boundaries using Pydantic models
- **Never expose internal errors** to API clients — log internally, return safe messages
- **Use parameterized queries** — never string-format SQL

---

## Code Quality Checklist

Before committing:

- [ ] `ruff check src/` passes
- [ ] `ruff format src/` applied (or no changes needed)
- [ ] `pytest tests/` passes
- [ ] All new public functions have docstrings
- [ ] All new public functions have type hints
- [ ] No bare `except:` blocks
- [ ] No hardcoded secrets or credentials
- [ ] Import order follows conventions
- [ ] Commit message uses correct agent prefix

---

## Module Organization

Standard module structure:
```python
"""Module docstring describing purpose and main exports.

References:
    - REQ-NNNN: relevant requirement
    - ARCH-NNN: relevant architecture doc
"""

from __future__ import annotations

# Standard library
from typing import Any

# Third-party
from sqlalchemy.ext.asyncio import AsyncSession

# Local
from app.models.item import Item

# Module-level constants
DEFAULT_PAGE_SIZE = 50


class ItemService:
    ...


async def get_items() -> list[Item]:
    ...


# Public API
__all__ = ["ItemService", "get_items"]
```
