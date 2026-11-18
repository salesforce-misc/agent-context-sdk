# Copyright (c) 2026, Salesforce, Inc.,
# All rights reserved.
# For full license text, see the LICENSE.txt file

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from harness import LiveConfig, build_live_config, env_file_applied, live_enabled


@pytest.fixture(scope="module")
def live_config() -> Iterator[LiveConfig]:
    """Credentials for the live tier; skips the test when it isn't enabled.

    Module-scoped so a skip or failure for one adapter doesn't stick to the
    rest, and so the env file is only in os.environ while a live module runs.
    """
    if not live_enabled():
        pytest.skip("live tier disabled; run with SMOKE_LIVE=1 (or run_smoke.py --tier live)")
    with env_file_applied():
        yield build_live_config(os.environ)
