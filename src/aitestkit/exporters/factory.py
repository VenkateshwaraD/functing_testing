"""Exporter factory — resolve ``--export`` names to exporter instances."""

from __future__ import annotations

from aitestkit.exceptions import ConfigError
from aitestkit.exporters.base import Exporter


def build_exporters(
    names: list[str],
    *,
    otel_endpoint: str | None = None,
) -> list[Exporter]:
    exporters: list[Exporter] = []
    for name in names:
        if name == "otel":
            from aitestkit.exporters.otel import OtelExporter

            exporters.append(OtelExporter(endpoint=otel_endpoint))
        elif name == "allure":
            from aitestkit.exporters.allure import AllureExporter

            exporters.append(AllureExporter())
        else:
            raise ConfigError(
                f"unknown exporter {name!r} (available: otel, allure)"
            )
    return exporters
