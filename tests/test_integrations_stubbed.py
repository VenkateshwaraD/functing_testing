"""Optional-integration coverage: langgraph, deepeval/ragas, otel, vector DBs.

All external SDKs are stubbed through sys.modules so adapter logic runs
without installing extras or touching the network.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from aitestkit.exceptions import ConfigError
from aitestkit.models import MetricStatus, TargetOutput, TestCase

pytestmark = pytest.mark.anyio

_REAL_FIND_SPEC = importlib.util.find_spec


@contextmanager
def stub_modules(modules: dict[str, types.ModuleType]):
    sentinel = object()
    old = {name: sys.modules.get(name, sentinel) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, prev in old.items():
            if prev is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _spec_for(*names: str):
    """A find_spec stand-in that claims only ``names`` are installed."""

    def fake(name: str, *a: Any, **k: Any) -> ModuleSpec | None:
        if name in names:
            return ModuleSpec(name, loader=None)
        return _REAL_FIND_SPEC(name, *a, **k)

    return fake


class _FakeEmbedder:
    name = "fake"

    async def embed(
        self, texts: list[str], *, model: str
    ) -> tuple[list[list[float]], Any]:
        from aitestkit.models import Usage

        vectors = [
            [float(len(t) % 7) + 0.1, float(len(t) % 3) + 0.1] for t in texts
        ]
        return vectors, Usage(provider="fake", model=model)


# --------------------------------------------------------------------------
# LangGraph adapter + target — duck-typed, no stubs needed


class _FakeGraph:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.seen_input: Any = None

    async def astream_events(self, input: Any, **kwargs: Any):
        self.seen_input = input
        for event in self._events:
            yield event


_GRAPH_EVENTS = [
    {"event": "on_chain_start", "name": "router", "data": {"input": {}}},
    {"event": "on_chat_model_start", "name": "gpt", "data": {"name": "gpt"}},
    {"event": "on_chat_model_stream", "name": "gpt", "data": {}},
    {
        "event": "on_tool_start",
        "name": "search",
        "data": {"input": {"q": "x"}},
    },
    {"event": "on_tool_end", "name": "search", "data": {"output": "found"}},
    {"event": "on_chain_end", "name": "router", "data": {"output": "handed"}},
    {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"output": "final answer"}},
    },
]


def test_langgraph_trajectory_conversion() -> None:
    from aitestkit.agentic.langgraph_adapter import langgraph_trajectory

    trajectory = langgraph_trajectory(_GRAPH_EVENTS)
    kinds = [e.type for e in trajectory.events]
    assert "tool_call" in kinds
    assert "observation" in kinds
    assert "handoff" in kinds
    assert trajectory.finish_reason == "done"


async def test_invoke_langgraph_and_target() -> None:
    from aitestkit.agentic.langgraph_adapter import invoke_langgraph
    from aitestkit.targets.langgraph_target import LangGraphTarget, _final_answer

    graph = _FakeGraph(_GRAPH_EVENTS)
    final, trajectory = await invoke_langgraph(graph, {"input": "hi"})
    assert final == {"output": "final answer"}
    assert trajectory.finish_reason == "done"

    mod = types.ModuleType("fake_graph_mod")
    mod.graph = lambda: _FakeGraph(_GRAPH_EVENTS)  # type: ignore[attr-defined]
    with stub_modules({"fake_graph_mod": mod}):
        target = LangGraphTarget(spec="fake_graph_mod:graph")
        output = await target.invoke(TestCase(input="hi"))
    assert output.output == "final answer"
    assert output.trajectory is not None
    assert target._graph.seen_input == {"input": "hi"}

    assert _final_answer("plain") == "plain"
    assert _final_answer({"messages": [SimpleNamespace(content="last")]}) == "last"
    assert _final_answer({"other": 1}) == "{'other': 1}"
    assert _final_answer(SimpleNamespace(content="c")) == "c"
    assert _final_answer(None) is None


async def test_langgraph_rejects_bad_graph() -> None:
    from aitestkit.agentic.langgraph_adapter import invoke_langgraph
    from aitestkit.targets.langgraph_target import LangGraphTarget

    with pytest.raises(ConfigError):
        await invoke_langgraph(object(), {})

    mod = types.ModuleType("bad_graph_mod")
    mod.graph = lambda: object()  # type: ignore[attr-defined]
    with stub_modules({"bad_graph_mod": mod}), pytest.raises(ConfigError):
        LangGraphTarget(spec="bad_graph_mod:graph")


async def test_langgraph_stream_error_wrapped() -> None:
    from aitestkit.exceptions import TargetError
    from aitestkit.targets.langgraph_target import LangGraphTarget

    class _Boom:
        async def astream_events(self, input: Any, **kwargs: Any):
            raise RuntimeError("graph blew up")
            yield  # pragma: no cover — makes it an async gen

    mod = types.ModuleType("boom_graph_mod")
    mod.graph = lambda: _Boom()  # type: ignore[attr-defined]
    with stub_modules({"boom_graph_mod": mod}):
        target = LangGraphTarget(spec="boom_graph_mod:graph")
        with pytest.raises(TargetError):
            await target.invoke(TestCase(input="hi"))


# --------------------------------------------------------------------------
# Metric backends — find_spec + module stubs


def _metric_ctx() -> Any:
    from aitestkit.metrics.base import MetricContext

    return MetricContext()


async def test_ragas_backend_scored() -> None:
    from aitestkit.metrics.backends.ragas_backend import RagasMetric

    ragas = types.ModuleType("ragas")
    metrics = types.ModuleType("ragas.metrics")
    schema = types.ModuleType("ragas.dataset_schema")

    class SingleTurnSample:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Faithfulness:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def single_turn_ascore(self, sample: Any) -> float:
            assert sample.kwargs["user_input"] == "q"
            return 0.9

    metrics.Faithfulness = Faithfulness  # type: ignore[attr-defined]
    schema.SingleTurnSample = SingleTurnSample  # type: ignore[attr-defined]
    ragas.metrics = metrics  # type: ignore[attr-defined]

    with (
        patch("importlib.util.find_spec", side_effect=_spec_for("ragas")),
        stub_modules(
            {"ragas": ragas, "ragas.metrics": metrics, "ragas.dataset_schema": schema}
        ),
    ):
        metric = RagasMetric(metric="faithfulness", threshold=0.8)
        result = await metric.measure(
            TestCase(input="q", expected_output="e"),
            TargetOutput(output="a", retrieval_context=["c"]),
            _metric_ctx(),
        )
    assert result.status == MetricStatus.SCORED
    assert result.score == pytest.approx(0.9)
    assert result.passed is True
    assert result.metric == "ragas_faithfulness"


async def test_ragas_error_paths() -> None:
    from aitestkit.metrics.backends.ragas_backend import RagasMetric

    metric = RagasMetric(metric="faithfulness")
    # not installed -> error, not a quality failure
    result = await metric.measure(TestCase(input="q"), TargetOutput(), _metric_ctx())
    assert result.status == MetricStatus.ERROR
    assert "ragas" in (result.reason or "")

    # unknown metric name
    with patch("importlib.util.find_spec", side_effect=_spec_for("ragas")):
        bad = RagasMetric(metric="nope")
        result = await bad.measure(TestCase(input="q"), TargetOutput(), _metric_ctx())
    assert result.status == MetricStatus.ERROR
    assert "unknown ragas metric" in (result.reason or "")


async def test_deepeval_backend_scored_and_errors() -> None:
    from aitestkit.metrics.backends.deepeval_backend import DeepEvalMetric

    deepeval = types.ModuleType("deepeval")
    metrics = types.ModuleType("deepeval.metrics")
    test_case_mod = types.ModuleType("deepeval.test_case")

    class LLMTestCase:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FaithfulnessMetric:
        def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
            self.threshold = threshold

        def measure(self, tc: Any) -> None:
            self.score = 0.91
            self.success = True
            self.reason = "all claims supported"
            self.verbose_logs = "logs"

    metrics.FaithfulnessMetric = FaithfulnessMetric  # type: ignore[attr-defined]
    test_case_mod.LLMTestCase = LLMTestCase  # type: ignore[attr-defined]
    deepeval.metrics = metrics  # type: ignore[attr-defined]

    with (
        patch("importlib.util.find_spec", side_effect=_spec_for("deepeval")),
        stub_modules(
            {
                "deepeval": deepeval,
                "deepeval.metrics": metrics,
                "deepeval.test_case": test_case_mod,
            }
        ),
    ):
        metric = DeepEvalMetric(metric="faithfulness", threshold=0.8)
        result = await metric.measure(
            TestCase(input="q"), TargetOutput(output="a"), _metric_ctx()
        )
        assert result.status == MetricStatus.SCORED
        assert result.score == pytest.approx(0.91)
        assert result.passed is True
        assert result.details["backend"] == "deepeval"

        # scoring exception -> error status
        class _BoomMetric(FaithfulnessMetric):
            def measure(self, tc: Any) -> None:
                raise RuntimeError("deepeval blew up")

        metrics.FaithfulnessMetric = _BoomMetric  # type: ignore[attr-defined]
        result = await metric.measure(TestCase(input="q"), TargetOutput(), _metric_ctx())
        assert result.status == MetricStatus.ERROR

        # missing class in module -> error
        del metrics.FaithfulnessMetric  # type: ignore[attr-defined]
        result = await metric.measure(TestCase(input="q"), TargetOutput(), _metric_ctx())
        assert result.status == MetricStatus.ERROR
        assert "no FaithfulnessMetric" in (result.reason or "")

    # unknown metric + not-installed paths
    metric = DeepEvalMetric(metric="unknown-metric")
    with patch("importlib.util.find_spec", side_effect=_spec_for("deepeval")):
        result = await metric.measure(TestCase(input="q"), TargetOutput(), _metric_ctx())
    assert "unknown deepeval metric" in (result.reason or "")

    metric = DeepEvalMetric(metric="faithfulness")
    result = await metric.measure(TestCase(input="q"), TargetOutput(), _metric_ctx())
    assert result.status == MetricStatus.ERROR


# --------------------------------------------------------------------------
# OTel exporter — stub the sdk modules


def _otel_modules(recorded: list[dict[str, Any]]) -> dict[str, types.ModuleType]:
    from contextlib import contextmanager as cm

    names = [
        "opentelemetry",
        "opentelemetry.exporter",
        "opentelemetry.exporter.otlp",
        "opentelemetry.exporter.otlp.proto",
        "opentelemetry.exporter.otlp.proto.http",
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.sdk",
        "opentelemetry.sdk.resources",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
    ]
    mods = {name: types.ModuleType(name) for name in names}

    class OTLPSpanExporter:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Resource:
        @classmethod
        def create(cls, attrs: dict[str, Any]) -> Resource:
            return cls()

    class BatchSpanProcessor:
        def __init__(self, exporter: Any) -> None:
            self.exporter = exporter

    class _Span:
        def __init__(self, name: str) -> None:
            self.name = name
            self.attrs: dict[str, Any] = {}

        def set_attribute(self, key: str, value: Any) -> None:
            self.attrs[key] = value

    class _Tracer:
        @cm
        def start_as_current_span(self, name: str):
            span = _Span(name)
            recorded.append({"name": name, "attrs": span.attrs})
            yield span

    class TracerProvider:
        def __init__(self, resource: Any = None) -> None:
            self.resource = resource

        def add_span_processor(self, processor: Any) -> None:
            pass

        def get_tracer(self, name: str) -> _Tracer:
            return _Tracer()

        def force_flush(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

    mods["opentelemetry.exporter.otlp.proto.http.trace_exporter"].OTLPSpanExporter = (
        OTLPSpanExporter  # type: ignore[attr-defined]
    )
    mods["opentelemetry.sdk.resources"].Resource = Resource  # type: ignore[attr-defined]
    mods["opentelemetry.sdk.trace"].TracerProvider = TracerProvider  # type: ignore[attr-defined]
    mods["opentelemetry.sdk.trace.export"].BatchSpanProcessor = (  # type: ignore[attr-defined]
        BatchSpanProcessor
    )
    return mods


def test_otel_exporter_emits_spans(tmp_path: Any) -> None:
    from aitestkit.exporters.otel import OtelExporter
    from aitestkit.models import CaseResult, CaseStatus, MetricResult, TestRun

    recorded: list[dict[str, Any]] = []
    with (
        patch("importlib.util.find_spec", side_effect=_spec_for("opentelemetry")),
        stub_modules(_otel_modules(recorded)),
    ):
        exporter = OtelExporter(endpoint="http://x:4318")
        run = TestRun(suite_name="s", run_id="r1")
        result = CaseResult(
            case_id="c1",
            input="q",
            status=CaseStatus.PASSED,
            latency_ms=12.0,
            metrics=[
                MetricResult(
                    metric="contains",
                    status=MetricStatus.SCORED,
                    score=1.0,
                    passed=True,
                )
            ],
        )
        run.results.append(result)
        message = exporter.export(run, tmp_path)
    assert "1 case spans" in (message or "")
    span_names = [r["name"] for r in recorded]
    assert "case c1" in span_names
    assert "metric contains" in span_names


def test_otel_requires_extra() -> None:
    from aitestkit.exporters.otel import OtelExporter

    with pytest.raises(ConfigError):
        OtelExporter()


def test_exporter_factory(tmp_path: Any) -> None:
    from aitestkit.exporters.factory import build_exporters

    exporters = build_exporters(["allure"])
    assert exporters[0].name == "allure"
    with pytest.raises(ConfigError):
        build_exporters(["bogus"])
    with pytest.raises(ConfigError):
        build_exporters(["otel"])  # extra missing


# --------------------------------------------------------------------------
# Vector stores — stub chromadb / faiss / qdrant / psycopg / pinecone


class _ChromaCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def upsert(self, ids: Any, embeddings: Any, documents: Any, metadatas: Any) -> None:
        for cid, emb, doc, meta in zip(
            ids, embeddings, documents, metadatas, strict=False
        ):
            self.rows[cid] = {"embedding": emb, "document": doc, "metadata": meta}

    def query(self, query_embeddings: Any, n_results: int) -> dict[str, Any]:
        ids = list(self.rows)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.rows[i]["document"] for i in ids]],
            "distances": [[0.1 for _ in ids]],
            "metadatas": [[self.rows[i]["metadata"] for i in ids]],
        }

    def count(self) -> int:
        return len(self.rows)

    def get(self, include: Any) -> dict[str, Any]:
        ids = list(self.rows)
        return {
            "ids": ids,
            "embeddings": [self.rows[i]["embedding"] for i in ids],
            "documents": [self.rows[i]["document"] for i in ids],
            "metadatas": [self.rows[i]["metadata"] for i in ids],
        }


async def test_chroma_store() -> None:
    from aitestkit.rag.chunking import Chunk
    from aitestkit.vectordb.chroma_store import ChromaVectorStore

    collection = _ChromaCollection()
    chromadb = types.ModuleType("chromadb")

    class _Client:
        def get_or_create_collection(self, name: str, metadata: Any) -> Any:
            return collection

    chromadb.EphemeralClient = _Client  # type: ignore[attr-defined]
    chromadb.PersistentClient = _Client  # type: ignore[attr-defined]

    with stub_modules({"chromadb": chromadb}):
        store = ChromaVectorStore(_FakeEmbedder())
        chunks = [
            Chunk(id="c1", text="alpha text", metadata={"src": "a"}),
            Chunk(id="c2", text="beta"),
        ]
        await store.upsert(chunks)
        assert await store.count() == 2
        hits = await store.query("alpha", top_k=1)
        assert len(hits) == 1
        assert hits[0][0].id == "c1"
        assert hits[0][1] == pytest.approx(0.9)  # 1 - 0.1 distance
        drift = await store.reembed_sample()
        assert len(drift) == 2  # deterministic sample covers the 2 rows


async def test_faiss_store(tmp_path: Any) -> None:
    import numpy as np

    from aitestkit.rag.chunking import Chunk
    from aitestkit.vectordb.faiss_store import FaissVectorStore

    faiss = types.ModuleType("faiss")

    class _Index:
        def __init__(self, dims: int) -> None:
            self.dims = dims
            self.vectors: list[Any] = []

        def add(self, array: Any) -> None:
            self.vectors.extend(array.tolist())

        def search(self, array: Any, k: int) -> tuple[Any, Any]:
            query = np.asarray(array)[0]
            stored = np.asarray(self.vectors)
            scores = stored @ query
            order = np.argsort(-scores)[:k]
            return np.array([scores[order]]), np.array([order])

        def reconstruct(self, i: int) -> Any:
            return np.asarray(self.vectors[i])

        @property
        def ntotal(self) -> int:
            return len(self.vectors)

    written: dict[str, Any] = {}
    tmp_faiss_path = tmp_path / "idx"  # -> idx.faiss + idx.json

    def _write_index(index: Any, path: str) -> None:
        written[path] = index
        Path(path).write_text("stub-index")

    faiss.IndexFlatIP = _Index  # type: ignore[attr-defined]
    faiss.write_index = _write_index  # type: ignore[attr-defined]
    faiss.read_index = lambda p: written[p]  # type: ignore[attr-defined]

    with stub_modules({"faiss": faiss}):
        store = FaissVectorStore(_FakeEmbedder(), dims=2)
        await store.upsert([Chunk(id="c1", text="alpha"), Chunk(id="c2", text="bb")])
        assert await store.count() == 2
        hits = await store.query("alpha", top_k=1)
        assert hits[0][0].id == "c1"
        drift = await store.reembed_sample()
        assert all(score > 0.99 for _, score in drift)

        # persistence round-trip via path=
        store2 = FaissVectorStore(
            _FakeEmbedder(), dims=2, path=str(tmp_faiss_path)
        )
        await store2.upsert([Chunk(id="p1", text="persisted")])
        assert written  # write_index was invoked
        store3 = FaissVectorStore(
            _FakeEmbedder(), dims=2, path=str(tmp_faiss_path)
        )
        assert await store3.count() == 1
        assert (await store3.query("persisted", top_k=1))[0][0].id == "p1"


async def test_qdrant_store() -> None:
    from aitestkit.rag.chunking import Chunk
    from aitestkit.vectordb.remote import QdrantVectorStore

    qdrant_client = types.ModuleType("qdrant_client")
    models = types.ModuleType("qdrant_client.models")

    class Distance:
        COSINE = "Cosine"

    class VectorParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class PointStruct:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.points: list[Any] = []
            self.exists = False

        def collection_exists(self, name: str) -> bool:
            return self.exists

        def create_collection(self, **kwargs: Any) -> None:
            self.exists = True

        def upsert(self, collection_name: str, points: list[Any]) -> None:
            self.points.extend(points)

        def search(self, **kwargs: Any) -> list[Any]:
            return [
                SimpleNamespace(
                    id=1,
                    payload={"chunk_id": "c1", "text": "alpha", "src": "s"},
                    score=0.7,
                )
            ]

        def count(self, name: str) -> Any:
            return SimpleNamespace(count=len(self.points))

    qdrant_client.QdrantClient = _Client  # type: ignore[attr-defined]
    qdrant_client.models = models  # type: ignore[attr-defined]
    models.Distance = Distance  # type: ignore[attr-defined]
    models.VectorParams = VectorParams  # type: ignore[attr-defined]
    models.PointStruct = PointStruct  # type: ignore[attr-defined]

    with stub_modules({"qdrant_client": qdrant_client, "qdrant_client.models": models}):
        store = QdrantVectorStore(
            _FakeEmbedder(), host="h", port=6333, collection="col", model="m"
        )
        await store.upsert([Chunk(id="c1", text="alpha", metadata={"src": "s"})])
        assert await store.count() == 1
        hits = await store.query("q", top_k=1)
        assert hits[0][0].id == "c1"
        assert hits[0][0].metadata == {"src": "s"}
        assert hits[0][1] == pytest.approx(0.7)
        with pytest.raises(ConfigError):
            await store.reembed_sample()


async def test_pgvector_store() -> None:
    from aitestkit.rag.chunking import Chunk
    from aitestkit.vectordb.remote import PgVectorStore

    psycopg = types.ModuleType("psycopg")
    queries: list[tuple[str, Any]] = []

    class _Cursor:
        def fetchall(self) -> list[Any]:
            return [("c1", "alpha", {"src": "s"}, 0.8)]

        def fetchone(self) -> Any:
            return (2,)

    class _Conn:
        def __enter__(self) -> _Conn:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def execute(self, sql: str, params: Any = None) -> _Cursor:
            queries.append((sql, params))
            return _Cursor()

    psycopg.connect = lambda dsn: _Conn()  # type: ignore[attr-defined]

    with stub_modules({"psycopg": psycopg}):
        store = PgVectorStore(
            _FakeEmbedder(), dsn="postgres://x", table="t_1", dims=2, model="m"
        )
        await store.upsert([Chunk(id="c1", text="alpha", metadata={"src": "s"})])
        hits = await store.query("q", top_k=1)
        assert hits[0][0].id == "c1"
        assert hits[0][1] == pytest.approx(0.8)
        assert await store.count() == 2
        assert any("CREATE TABLE" in sql for sql, _ in queries)

        with pytest.raises(ConfigError):
            PgVectorStore(
                _FakeEmbedder(), dsn="d", table='bad"name', dims=2, model="m"
            )._ident('bad"name')
        with pytest.raises(ConfigError):
            await store.reembed_sample()


async def test_pinecone_store(monkeypatch: Any) -> None:
    from aitestkit.rag.chunking import Chunk
    from aitestkit.vectordb.remote import PineconeVectorStore

    pinecone = types.ModuleType("pinecone")

    class ServerlessSpec:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Index:
        def __init__(self) -> None:
            self.upserted: list[Any] = []

        def upsert(self, vectors: list[Any]) -> None:
            self.upserted.extend(vectors)

        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "matches": [
                    {
                        "id": "c1",
                        "score": 0.6,
                        "metadata": {"text": "alpha", "src": "s"},
                    }
                ]
            }

        def describe_index_stats(self) -> dict[str, Any]:
            return {"total_vector_count": 3}

    class Pinecone:
        def __init__(self, api_key: str) -> None:
            self._index = _Index()

        def has_index(self, name: str) -> bool:
            return True

        def Index(self, name: str) -> _Index:  # noqa: N802 — SDK surface
            return self._index

    pinecone.Pinecone = Pinecone  # type: ignore[attr-defined]
    pinecone.ServerlessSpec = ServerlessSpec  # type: ignore[attr-defined]

    monkeypatch.setenv("PINECONE_API_KEY", "k")
    with stub_modules({"pinecone": pinecone}):
        store = PineconeVectorStore(
            _FakeEmbedder(),
            api_key_env="PINECONE_API_KEY",
            index_name="idx",
            dims=2,
            model="m",
        )
        await store.upsert([Chunk(id="c1", text="alpha", metadata={"src": "s"})])
        hits = await store.query("q", top_k=1)
        assert hits[0][0].id == "c1"
        assert hits[0][0].metadata == {"src": "s"}
        assert await store.count() == 3
        with pytest.raises(ConfigError):
            await store.reembed_sample()


def test_remote_store_factory_and_missing_deps() -> None:
    from aitestkit.vectordb.remote import build_remote_store

    with pytest.raises(ConfigError):
        build_remote_store("bogus", {}, embedder=_FakeEmbedder())
    with pytest.raises(ConfigError):
        build_remote_store("qdrant", {}, embedder=_FakeEmbedder())
    with pytest.raises(ConfigError):
        build_remote_store("pgvector", {"dsn": "d"}, embedder=_FakeEmbedder())
    with pytest.raises(ConfigError):
        build_remote_store("pinecone", {}, embedder=_FakeEmbedder())


# --------------------------------------------------------------------------
# Smaller uncovered modules


def test_trajectory_recorder_roundtrip(tmp_path: Any) -> None:
    from aitestkit.models import Trajectory
    from aitestkit.tracing.recorder import load_trajectory, save_trajectory

    emitter_path = tmp_path / "golden.json"
    trajectory = Trajectory(finish_reason="done")
    save_trajectory(trajectory, emitter_path)
    loaded = load_trajectory(emitter_path)
    assert loaded.finish_reason == "done"

    with pytest.raises(ConfigError):
        load_trajectory(tmp_path / "missing.json")

    emitter_path.write_text('{"schema_version": 9999, "events": []}')
    with pytest.raises(ConfigError):
        load_trajectory(emitter_path)


def test_golden_set_roundtrip(tmp_path: Any) -> None:
    from aitestkit.rag.golden import load_golden, save_golden

    golden_path = tmp_path / "golden.yaml"
    golden = load_golden(__import__("pathlib").Path("examples/golden/retrieval.yaml"))
    save_golden(golden, golden_path)
    reloaded = load_golden(golden_path)
    assert len(reloaded.queries) == len(golden.queries)

    golden_path.write_text("schema_version: 2\nqueries: []\n")
    with pytest.raises(ValueError, match="version"):
        load_golden(golden_path)


def test_attribution_labels() -> None:
    from aitestkit.models import CaseResult, CaseStatus, MetricResult, TestRun
    from aitestkit.reporting.attribution import apply_attribution, attribute_failure

    def case(metrics: list[tuple[str, float]]) -> CaseResult:
        return CaseResult(
            case_id="c",
            input="q",
            status=CaseStatus.FAILED,
            metrics=[
                MetricResult(
                    metric=name,
                    status=MetricStatus.SCORED,
                    score=score,
                    passed=False,
                )
                for name, score in metrics
            ],
        )

    assert attribute_failure(case([("ingestion_parse", 0.1)])) == "ingestion/chunking"
    assert attribute_failure(case([("recall_at_k", 0.2)])) == "retrieval"
    assert attribute_failure(case([("rerank_delta", -0.5)])) == "rerank"
    assert attribute_failure(case([("contextual_precision", 0.1)])) == "context_window"
    assert attribute_failure(case([("recall_at_k", 0.9), ("faithfulness", 0.1)])) == "generation"
    assert attribute_failure(case([("contains", 0.0)])) is None
    assert attribute_failure(case([])) is None

    run = TestRun(suite_name="s", run_id="r")
    run.results.append(case([("faithfulness", 0.1)]))
    assert apply_attribution(run) == 1
    assert run.results[0].attribution == "generation"
    assert apply_attribution(run) == 1  # already-labeled still counted


async def test_dataquality_gx_and_edge_cases() -> None:
    from aitestkit.dataquality.metric import DataQuality

    metric = DataQuality()
    ctx = _metric_ctx()

    # no rows -> skipped
    result = await metric.measure(
        TestCase(input="q", metadata={}), TargetOutput(), ctx
    )
    assert result.status == MetricStatus.SKIPPED

    # no expectations -> skipped
    result = await metric.measure(
        TestCase(input="q", metadata={"rows": [{"a": 1}]}), TargetOutput(), ctx
    )
    assert result.status == MetricStatus.SKIPPED

    # gx backend -> delegates to bridge; missing extra is a config error
    with pytest.raises(ConfigError):
        await metric.measure(
            TestCase(
                input="q",
                metadata={
                    "rows": [{"a": 1}],
                    "expectations": [
                        {"expect_column_values_to_not_be_null": {"column": "a"}}
                    ],
                    "backend": "gx",
                },
            ),
            TargetOutput(),
            ctx,
        )
