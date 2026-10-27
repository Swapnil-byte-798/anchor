# Optional local MCP tools

This add-on lets an MCP host call Anchor's existing tools. It adds a separate
stdio entry point; it does not modify the API, agent loop, normal CLI, default
configuration, corpus, golden questions, thresholds or saved results.

## What is exposed

| Tool | Existing implementation | Input limits |
| --- | --- | --- |
| `search_docs` | `SearchDocsTool`, including the retrieval abstention gate | Query: 1-2000 characters, also subject to the configured question limit; `k`: 1-20 |
| `calculator` | `CalculatorTool`, including its arithmetic AST allowlist | Expression: 1-600 characters; existing depth, literal and exponent limits apply |
| `lookup_metric` | `LookupMetricTool`, reading the configured metrics CSV | Metric: 1-200 characters; optional month: at most 32 characters |

Only tools enabled by `agent.tools` are registered. `final_answer` is an internal
agent-loop signal and is never exposed. There are no ingestion, file-writing,
arbitrary-file-reading, shell, generation or remote-provider tools.

The tools are advertised as read-only and closed-world. These annotations are
hints for the host, not an authorization boundary. Search/metric results can
contain local data: configure a corpus you intend the host to access and treat
retrieved text as untrusted data. The adapter serializes calls on its shared
context and returns the original tools' bounded text observations.

## Install and run

From the repository root, using Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mcp]'
anchor --config config/default.yaml ingest
python -m anchor.mcp_server --config config/default.yaml
```

The last command waits for an MCP host on stdin/stdout; it is not an interactive
chat prompt. Logs go to stderr. It opens no listening network port.

The SDK is optional: `pip install -e .` still installs the ordinary project
without MCP. To run all tests with the add-on, install `.[dev,mcp]` and run
`python -m pytest`. A base installation skips the optional integration tests.

## Connect a host

Use your host's stdio-server configuration. Replace both example paths with
absolute paths on your machine:

```json
{
  "mcpServers": {
    "anchor": {
      "command": "/absolute/path/to/anchor/.venv/bin/python",
      "args": [
        "-m", "anchor.mcp_server",
        "--config", "/absolute/path/to/anchor/config/mcp-local.yaml"
      ]
    }
  }
}
```

Anchor resolves paths against the process working directory. Desktop hosts may
start elsewhere. Create a separate local YAML with absolute paths (or configure
your host's working directory to the repository root):

```yaml
paths:
  data_dir: /absolute/path/to/anchor/data
  corpus_dir: /absolute/path/to/anchor/corpus
  db_path: /absolute/path/to/anchor/data/index.db
  vectors_path: /absolute/path/to/anchor/data/vectors.npy
  metrics_csv: /absolute/path/to/anchor/corpus/metrics.csv
embedding:
  provider: hashing
```

Ingest using that same configuration before connecting the host:

```bash
anchor --config /absolute/path/to/anchor/config/mcp-local.yaml ingest
```

Try `search_docs` with `{"query":"How long are gateway request logs retained?"}`,
`calculator` with `{"expression":"(120 + 45) / 2"}`, and `lookup_metric` with
`{"metric":"list"}`.

## Cost and scope

The local adapter needs no API key, subscription, GPU or hosted service. It uses
the official open-source Python MCP SDK and the existing local hashing index.
It rejects remote embedding configurations rather than switching embedding
spaces silently. Use a separate index if your ordinary API uses remote embeddings.
The MCP process constructs a stub generator and heuristic judge only; it never
calls either provider. Your host or its chosen model may have separate charges.

Search returns snippets, not the full answer-generation/verification pipeline.
The original agent's episode budgets do not govern an external MCP host: the
host controls tool-call count, timeouts, permissions and answer generation.
This feature does not claim to improve the existing retrieval scores.

Implementation: `src/anchor/mcp_server.py`. Tests use the official SDK client
in memory and over a real stdio subprocess, check parity with existing tools,
disabled tools, invalid inputs, input policy, and blocked provider calls.
