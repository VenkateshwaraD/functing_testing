"""OpenTelemetry exporter — spans per case + child spans per metric.

Requires ``aitestkit[otel]``. Sends to the OTLP HTTP endpoint from
``OTEL_EXPORTER_OTLP_ENDPOINT`` (default http://localhost:4318). Jaeger,
Tempo, Phoenix, and Langfuse all ingest OTLP, so one exporter covers
the observability platforms.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING

from aitestkit.exceptions import ConfigError

if TYPE_CHECKING:
    from pathlib import Path

    from aitestkit.models import TestRun


class OtelExporter:
    """Emit the run as a parent span with case/metric children."""

    name = "otel"

    def __init__(self, *, endpoint: str | None = None, service: str = "aitestkit") -> None:
        if importlib.util.find_spec("opentelemetry") is None:
            raise ConfigError(
                "otel exporter requires: pip install 'aitestkit[otel]'"
            )
        self.endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
        )
        self.service = service

    def export(self, test_run: TestRun, out_dir: Path) -> str | None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": self.service,
                "aitestkit.run_id": test_run.run_id,
                "aitestkit.suite": test_run.suite_name,
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{self.endpoint}/v1/traces"))
        )
        tracer = provider.get_tracer("aitestkit")

        with tracer.start_as_current_span(
            f"aitestkit run {test_run.suite_name}"
        ) as run_span:
            run_span.set_attribute("aitestkit.run_id", test_run.run_id)
            run_span.set_attribute("aitestkit.cases", len(test_run.results))
            for result in test_run.results:
                with tracer.start_as_current_span(
                    f"case {result.case_id}"
                ) as case_span:
                    case_span.set_attribute("aitestkit.case_id", result.case_id)
                    case_span.set_attribute("aitestkit.status", result.status.value)
                    case_span.set_attribute("aitestkit.latency_ms", result.latency_ms)
                    case_span.set_attribute(
                        "aitestkit.cost_usd", result.sut_usage.cost_usd
                    )
                    if result.error:
                        case_span.set_attribute("aitestkit.error", result.error)
                    for metric in result.metrics:
                        with tracer.start_as_current_span(
                            f"metric {metric.metric}"
                        ) as metric_span:
                            metric_span.set_attribute(
                                "aitestkit.metric_status", metric.status.value
                            )
                            if metric.score is not None:
                                metric_span.set_attribute(
                                    "aitestkit.score", metric.score
                                )
                            if metric.reason:
                                metric_span.set_attribute(
                                    "aitestkit.reason", metric.reason[:512]
                                )
        provider.force_flush()
        provider.shutdown()
        return f"exported {len(test_run.results)} case spans to {self.endpoint}"
