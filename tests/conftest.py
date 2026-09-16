"""Shared test setup.

`app/api.py` builds its FastAPI instance at module scope (`app = create_app()`),
and `create_app()` now refuses to run without a gateway token. So the token has
to exist before any test module imports `app.api` — a fixture would run too
late. conftest is imported first, which makes this the right place.
"""

from __future__ import annotations

import os

os.environ.setdefault("GATEWAY_TOKEN", "test-gateway-token")
# Never let a test touch a real profile directory, even if one env var is
# missed further down; individual tests still point this at their own tmp_path.
os.environ.setdefault("HERMES_USERS_HOME", os.path.join(os.getcwd(), ".pytest_cache", "users"))
