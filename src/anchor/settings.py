"""Configuration: the ONLY module in anchor that reads ``os.environ``.

Everything else takes a ``Settings`` instance. Precedence is
init kwargs > env > .env > YAML file > defaults, and ``fingerprint()`` gives a
short, secret-free digest of the effective configuration that is stamped onto
every Trace and MetricReport — so a number in a report can always be traced back
to the config that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from anchor.types import RetrievalMode


class PathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str = "data"
    corpus_dir: str = "corpus"
    db_path: str = "data/index.db"
    vectors_path: str = "data/vectors.npy"
    reports_dir: str = "eval/reports"
    golden_path: str = "eval/golden/qa.jsonl"
    authored_seed_path: str = "eval/golden/authored_seed.jsonl"
    human_labels_path: str = "eval/golden/human_labels.jsonl"
    thresholds_path: str = "eval/thresholds.yaml"
    metrics_csv: str = "corpus/metrics.csv"
    readme_path: str = "README.md"
    #: doc_ids under corpus_dir that are ABOUT the corpus rather than part of it.
    #: `scripts/build_golden.py` has always excluded these from the question
    #: builder; ingest excludes them too, so the served corpus and the evaluated
    #: corpus are the same set of documents.
    corpus_exclude: list[str] = ["readme"]


class ChunkingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    min_chunk_tokens: int = 80


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: RetrievalMode = "hybrid"
    k_lexical: int = 30
    k_dense: int = 30
    rrf_k: int = 60
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    top_k: int = 5
    context_token_budget: int = 1400
    neighbour_window: int = 0
    max_candidates_recorded: int = 20


class GateSettings(BaseModel):
    """One scale-free feature: max lexical overlap over the top 5 candidates."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    min_lexical_overlap: float = 0.34


class GroundingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    support_threshold: float = 0.45
    grounded_ratio: float = 0.85
    partial_ratio: float = 0.60


class JudgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["heuristic", "llm"] = "heuristic"
    grounded_threshold: float = 0.80
    partial_threshold: float = 0.50


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    planner: Literal["rules", "llm"] = "rules"
    max_steps: int = 4
    tool_timeout_s: float = 10.0
    wall_clock_budget_s: float = 30.0
    max_total_tokens: int = 8000
    tools: list[str] = Field(
        default_factory=lambda: ["search_docs", "calculator", "lookup_metric", "final_answer"]
    )


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["stub", "openai", "azure_openai", "anthropic", "bedrock"] = "stub"
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str = "2024-06-01"
    aws_region: str = "us-east-1"
    temperature: float = 0.0
    max_tokens: int = 700
    timeout_s: float = 30.0
    max_retries: int = 2
    price_per_1k_input_usd: float = 0.0
    price_per_1k_output_usd: float = 0.0


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["hashing", "openai", "azure_openai", "bedrock"] = "hashing"
    model: str = "text-embedding-3-small"
    dim: int = 256
    api_key: SecretStr | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str = "2024-06-01"
    aws_region: str = "us-east-1"
    batch_size: int = 64
    timeout_s: float = 30.0
    max_retries: int = 2


class EvalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worst_n: int = 12
    threshold_slack: float = 0.05
    max_question_chars: int = 2000
    ablation_modes: list[str] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "hybrid_rerank"]
    )
    ablation_chunk_configs: list[list[int]] = Field(
        default_factory=lambda: [[256, 0], [256, 64], [512, 0], [512, 64], [1024, 0], [1024, 128]]
    )
    refusal_sweep_points: list[float] = Field(
        default_factory=lambda: [0.10, 0.20, 0.30, 0.34, 0.40, 0.50, 0.60]
    )


class ObsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_level: str = "INFO"
    log_json: bool = True
    max_body_bytes: int = 65536


class Settings(BaseSettings):
    """Precedence: init kwargs > env > .env > YAML file > defaults.

    Env vars are prefixed and nested with a double underscore:
      ANCHOR_LLM__PROVIDER=openai   ANCHOR_LLM__API_KEY=sk-...
      ANCHOR_RETRIEVAL__MODE=bm25   ANCHOR_CHUNKING__CHUNK_TOKENS=256
    """

    model_config = SettingsConfigDict(
        env_prefix="ANCHOR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "anchor"
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 1337

    paths: PathSettings = Field(default_factory=PathSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    obs: ObsSettings = Field(default_factory=ObsSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        path = _ACTIVE_CONFIG_FILE
        if path and Path(path).is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)

    @classmethod
    def load(cls, config_file: str | None = None, **overrides: Any) -> "Settings":
        """Load settings, merging a YAML file under env vars and init kwargs."""
        global _ACTIVE_CONFIG_FILE
        _ACTIVE_CONFIG_FILE = (
            config_file
            if config_file is not None
            else os.environ.get("ANCHOR_CONFIG_FILE", "config/default.yaml")
        )
        return cls(**overrides)

    def public_dict(self) -> dict[str, Any]:
        """model_dump with every SecretStr replaced by a redacted marker."""
        data = self.model_dump(mode="json")
        for section in ("llm", "embedding"):
            key = data.get(section, {}).get("api_key")
            data[section]["api_key"] = "set" if key else None
        return data

    def fingerprint(self) -> str:
        """12-hex digest of the effective, secret-free configuration."""
        blob = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def resolve(self, path_value: str) -> Path:
        """Resolve a configured relative path against the working directory."""
        return Path(path_value).expanduser()


_ACTIVE_CONFIG_FILE: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Call get_settings.cache_clear() in tests."""
    return Settings.load()
