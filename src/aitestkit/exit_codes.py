"""Process exit codes for the aitestkit CLI.

Documented in README and `aitestkit run --help`; CI relies on these.
"""

from __future__ import annotations

import enum


class ExitCode(enum.IntEnum):
    """Well-known aitestkit exit codes."""

    OK = 0
    TEST_FAILURES = 1
    CONFIG_ERROR = 2
    BUDGET_EXCEEDED = 3
    ENFORCEMENT_ABORT = 4
    GATE_FAILURE = 5
