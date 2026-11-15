"""Python assertion helpers for banking/payment SUT testing.

Each function raises ``AssertionError`` with a descriptive message on
violation — usable from pytest, ``assert_test`` wrappers, or any runner.
All helpers accept plain dicts/lists or aitestkit model objects.
"""

from assertions.agent_assertions import (
    assert_confirmation_precedes,
    assert_max_tool_repeats,
    assert_terminated,
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_order,
)
from assertions.business_assertions import (
    assert_idempotent_retry,
    assert_positive_amount,
    assert_sufficient_funds,
    assert_valid_transition,
    assert_within_transfer_limit,
)
from assertions.schema_assertions import (
    assert_iban,
    assert_iso_currency,
    assert_json_schema,
    assert_no_pii,
    assert_required_fields,
)
from assertions.semantic_assertions import (
    assert_cosine_similarity,
    assert_mentions_all,
    assert_mentions_any,
    assert_no_placeholders,
    assert_refusal,
)
from assertions.workflow_assertions import (
    assert_agent_participated,
    assert_handoffs,
    assert_no_tool_errors,
    assert_step_budget,
)

__all__ = [
    "assert_agent_participated",
    "assert_confirmation_precedes",
    "assert_cosine_similarity",
    "assert_handoffs",
    "assert_iban",
    "assert_idempotent_retry",
    "assert_iso_currency",
    "assert_json_schema",
    "assert_max_tool_repeats",
    "assert_mentions_all",
    "assert_mentions_any",
    "assert_no_pii",
    "assert_no_placeholders",
    "assert_no_tool_errors",
    "assert_positive_amount",
    "assert_refusal",
    "assert_required_fields",
    "assert_step_budget",
    "assert_sufficient_funds",
    "assert_terminated",
    "assert_tool_called",
    "assert_tool_not_called",
    "assert_tool_order",
    "assert_valid_transition",
    "assert_within_transfer_limit",
]
