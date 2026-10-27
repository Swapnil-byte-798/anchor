"""Offline protocol-level checks for the optional adapter, including real stdio."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from anchor.agent import build_tools  # noqa: E402
from anchor.context import AppContext, build_context  # noqa: E402
from anchor.errors import ConfigError  # noqa: E402
from anchor.ingest import ingest_corpus  # noqa: E402
from anchor.mcp_server import create_server  # noqa: E402
from anchor.settings import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How long are gateway request logs retained?"


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[AppContext]:
    corpus = tmp_path / "corpus"
    shutil.copytree(ROOT / "corpus", corpus)
    data = tmp_path / "data"
    settings = Settings.load(
        config_file="",
        paths={
            "data_dir": str(data),
            "corpus_dir": str(corpus),
            "db_path": str(data / "index.db"),
            "vectors_path": str(data / "vectors.npy"),
            "metrics_csv": str(corpus / "metrics.csv"),
        },
        embedding={"provider": "hashing"},
        llm={"provider": "stub"},
        judge={"provider": "heuristic"},
    )
    context = build_context(settings)
    try:
        ingest_corpus(
            settings, store=context.store, vectors=context.vectors, embedder=context.embedder
        )
        yield context
    finally:
        context.close()


def call(ctx: AppContext, name: str, arguments: dict[str, Any]) -> Any:
    async def request() -> Any:
        async with Client(create_server(ctx)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(request())


def test_discovery_schema_and_read_only_annotations(ctx: AppContext) -> None:
    async def request() -> Any:
        async with Client(create_server(ctx)) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(request()).tools}
    assert set(tools) == {"search_docs", "calculator", "lookup_metric"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False
    schema = tools["search_docs"].input_schema["properties"]
    assert schema["k"]["minimum"] == 1
    assert schema["k"]["maximum"] == 20
    assert schema["query"]["maxLength"] == 2000


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_docs", {"query": QUESTION, "k": 3}),
        ("calculator", {"expression": "(120 + 45) / 2"}),
        ("lookup_metric", {"metric": "list"}),
        ("lookup_metric", {"metric": "p95_latency_ms", "month": "2025-06"}),
    ],
)
def test_results_match_existing_tools(
    ctx: AppContext, name: str, arguments: dict[str, Any]
) -> None:
    tool = build_tools(ctx)[name]
    expected = tool.run(tool.args_model.model_validate(arguments))
    result = call(ctx, name, arguments)
    assert not result.is_error
    assert result.content[0].text == expected
    if name == "search_docs":
        assert "[S1]" in expected
        assert "90 days" in expected


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_docs", {"query": QUESTION, "k": 0}),
        ("search_docs", {"query": QUESTION, "k": 21}),
        ("search_docs", {"query": ""}),
        ("search_docs", {"query": "x" * 2001}),
        ("search_docs", {"query": "   "}),
        ("calculator", {"expression": "__import__('os').system('echo unsafe')"}),
        ("calculator", {"expression": "9**99"}),
        ("lookup_metric", {"metric": "not_a_real_metric"}),
    ],
)
def test_invalid_arguments_return_protocol_errors(
    ctx: AppContext, name: str, arguments: dict[str, Any]
) -> None:
    assert call(ctx, name, arguments).is_error


def test_configured_question_policy_is_preserved(ctx: AppContext) -> None:
    ctx.settings.eval.max_question_chars = 10
    result = call(ctx, "search_docs", {"query": QUESTION})
    assert result.is_error
    assert "10-character limit" in result.content[0].text


def test_disabled_tools_are_not_exposed(ctx: AppContext) -> None:
    ctx.settings.agent.tools = ["calculator", "final_answer"]

    async def request() -> None:
        async with Client(create_server(ctx)) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["calculator"]
            assert (await client.call_tool("search_docs", {"query": QUESTION})).is_error

    asyncio.run(request())


def test_no_model_or_judge_calls(ctx: AppContext) -> None:
    with (
        patch.object(type(ctx.llm), "complete", side_effect=AssertionError("LLM called")),
        patch.object(type(ctx.judge), "score", side_effect=AssertionError("judge called")),
    ):
        assert not call(ctx, "search_docs", {"query": QUESTION}).is_error
        assert not call(ctx, "calculator", {"expression": "2+2"}).is_error
        assert not call(ctx, "lookup_metric", {"metric": "list"}).is_error


def test_remote_embeddings_are_rejected(ctx: AppContext) -> None:
    ctx.settings.embedding.provider = "openai"
    with pytest.raises(ConfigError, match="requires embedding.provider=hashing"):
        create_server(ctx)


def test_real_stdio_subprocess(ctx: AppContext, tmp_path: Path) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": ctx.settings.paths.model_dump(),
                # Ordinary API settings must not make the adapter require a paid provider.
                "llm": {"provider": "openai", "api_key": None},
                "judge": {"provider": "llm"},
            }
        ),
        encoding="utf-8",
    )
    # Simulate a desktop host starting outside the repository. Config paths are absolute.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "anchor.mcp_server", "--config", str(config)],
        cwd=tmp_path,
    )

    async def request() -> None:
        async with Client(params, read_timeout_seconds=15) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 3
            search = await client.call_tool("search_docs", {"query": QUESTION})
            assert not search.is_error
            assert "90 days" in search.content[0].text
            result = await client.call_tool("calculator", {"expression": "(120 + 45) / 2"})
            assert not result.is_error
            assert "82.5" in result.content[0].text

    asyncio.run(request())
