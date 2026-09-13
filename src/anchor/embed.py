"""Embedders. The default one downloads nothing, ever.

``HashingEmbedder`` is the zero-key, zero-network default: a keyed feature-hashing
vectoriser over word tokens and character n-grams. Real providers (OpenAI, Azure
OpenAI, Bedrock) switch on through config when credentials exist, and a provider
requested without its credentials raises :class:`ProviderNotConfigured` naming the
environment variable — never a silent fallback to the stub.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Sequence

import numpy as np

from anchor.errors import ProviderError, ProviderNotConfigured
from anchor.settings import EmbeddingSettings, Settings
from anchor.textutil import normalize_text, tokenize
from anchor.types import Embedder

__all__ = ["HashingEmbedder", "RemoteEmbedder", "get_embedder"]

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero rather than becoming NaN."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


class HashingEmbedder:
    """A deterministic, dependency-free, LEXICAL stand-in for a semantic model.

    Be clear about what this is: it is NOT a semantic embedding model. It hashes
    word tokens and character 3/4-grams of the normalised text into a fixed number
    of buckets with sublinear term weighting. It therefore captures morphological
    and lexical similarity ("rate limit" ~ "rate limits") and nothing else — no
    synonymy, no paraphrase, no world knowledge.

    It earns its place for three reasons: it needs no key and no network, it is
    byte-reproducible across machines and runs (keyed BLAKE2b, never Python's
    per-process salted ``hash()``), and it makes the dense leg of the retriever
    exercisable offline so the hybrid path is a real code path in CI rather than a
    branch nobody runs.

    The honest consequence is that on the default profile the BM25 leg carries most
    of the offline retrieval quality. That is not an assertion in a README — it is
    exactly what the ``bm25`` vs ``dense`` vs ``hybrid`` rows of ``make ablate``
    measure, and the table is the evidence.
    """

    def __init__(self, dim: int = 256, seed: int = 1337) -> None:
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim}")
        self.name = "hashing"
        self.dim = int(dim)
        self.seed = int(seed)
        # BLAKE2b takes a key of up to 64 bytes; 8 bytes of seed is plenty and
        # makes the mapping stable across processes, machines and Python builds.
        self._key = self.seed.to_bytes(8, "big", signed=False)
        self._bucket_cache: dict[str, int] = {}

    def _bucket(self, feature: str) -> int:
        cached = self._bucket_cache.get(feature)
        if cached is not None:
            return cached
        digest = hashlib.blake2b(feature.encode("utf-8"), key=self._key, digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % self.dim
        if len(self._bucket_cache) < 200_000:
            self._bucket_cache[feature] = bucket
        return bucket

    @staticmethod
    def _features(text: str) -> list[str]:
        """Word tokens plus character 3-grams and 4-grams of the normalised text."""
        features = [f"w:{t}" for t in tokenize(text)]
        normalised = normalize_text(text)
        for n in (3, 4):
            if len(normalised) >= n:
                features.extend(
                    f"c{n}:{normalised[i : i + n]}" for i in range(len(normalised) - n + 1)
                )
        return features

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[int, int] = {}
        for feature in self._features(text):
            bucket = self._bucket(feature)
            counts[bucket] = counts.get(bucket, 0) + 1
        for bucket, count in counts.items():
            # Sublinear term frequency: a term repeated 20 times is not 20x the
            # evidence of a term seen once.
            vec[bucket] = 1.0 + math.log(count)
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._vector(t) for t in items]).astype(np.float32, copy=False)
        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vector(text).reshape(1, -1))[0]


class RemoteEmbedder:
    """OpenAI / Azure OpenAI / Bedrock embeddings over raw httpx (or boto3).

    Batched, retried with exponential backoff and jitter on 429/5xx/timeout, and
    normalised locally so callers can rely on cosine == dot product regardless of
    what the provider returns.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.provider = settings.provider
        self.name = f"{settings.provider}:{settings.model}"
        self.dim = int(settings.dim)
        self._dim_confirmed = False
        self._client: Any = None
        self._bedrock: Any = None

        if self.provider == "bedrock":
            try:
                import boto3  # noqa: F401
            except ImportError as exc:
                raise ProviderNotConfigured(
                    "embedding.provider='bedrock' needs boto3, which is not installed. "
                    "Install it with: pip install 'anchor[bedrock]'"
                ) from exc
            import boto3

            self._bedrock = boto3.client("bedrock-runtime", region_name=settings.aws_region)
            return

        key = settings.api_key.get_secret_value() if settings.api_key else ""
        if not key:
            raise ProviderNotConfigured(
                f"embedding.provider={self.provider!r} requires an API key. "
                "Set ANCHOR_EMBEDDING__API_KEY (or switch back to the offline default with "
                "ANCHOR_EMBEDDING__PROVIDER=hashing)."
            )
        if self.provider == "azure_openai" and not settings.azure_endpoint:
            raise ProviderNotConfigured(
                "embedding.provider='azure_openai' requires an endpoint. "
                "Set ANCHOR_EMBEDDING__AZURE_ENDPOINT."
            )
        if self.provider == "azure_openai" and not settings.azure_deployment:
            raise ProviderNotConfigured(
                "embedding.provider='azure_openai' requires a deployment name. "
                "Set ANCHOR_EMBEDDING__AZURE_DEPLOYMENT."
            )

        import httpx

        if self.provider == "azure_openai":
            base = str(settings.azure_endpoint).rstrip("/")
            self._url = (
                f"{base}/openai/deployments/{settings.azure_deployment}/embeddings"
                f"?api-version={settings.azure_api_version}"
            )
            headers = {"api-key": key, "content-type": "application/json"}
        else:
            base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
            self._url = f"{base}/embeddings"
            headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        self._client = httpx.Client(timeout=settings.timeout_s, headers=headers)

    # -- transport ------------------------------------------------------------

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last: str = "no attempt was made"
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self._client.post(self._url, json=payload)
                if response.status_code in _RETRYABLE_STATUS:
                    last = f"HTTP {response.status_code}: {response.text[:200]}"
                elif response.status_code >= 400:
                    raise ProviderError(
                        f"{self.name} embeddings failed with HTTP {response.status_code}: "
                        f"{response.text[:400]}"
                    )
                else:
                    return dict(response.json())
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.settings.max_retries:
                delay = (2.0**attempt) * 0.5 + random.uniform(0.0, 0.25)
                time.sleep(delay)
        raise ProviderError(
            f"{self.name} embeddings failed after {self.settings.max_retries + 1} attempts: {last}"
        )

    def _embed_bedrock(self, texts: list[str]) -> list[list[float]]:
        import json

        out: list[list[float]] = []
        for text in texts:
            body = json.dumps({"inputText": text})
            try:
                response = self._bedrock.invoke_model(modelId=self.settings.model, body=body)
                payload = json.loads(response["body"].read())
            except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
                raise ProviderError(f"{self.name} embeddings failed: {exc}") from exc
            vector = payload.get("embedding")
            if not isinstance(vector, list):
                raise ProviderError(f"{self.name} returned no 'embedding' field")
            out.append([float(v) for v in vector])
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "bedrock":
            return self._embed_bedrock(texts)
        payload: dict[str, Any] = {"input": texts, "model": self.settings.model}
        data = self._post_with_retries(payload)
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ProviderError(
                f"{self.name} returned {len(rows) if isinstance(rows, list) else 'no'} embeddings "
                f"for {len(texts)} inputs"
            )
        ordered = sorted(rows, key=lambda r: int(r.get("index", 0)))
        return [[float(v) for v in r["embedding"]] for r in ordered]

    # -- Embedder protocol ----------------------------------------------------

    def _embed_all(self, texts: Sequence[str]) -> np.ndarray:
        items = [t if t.strip() else " " for t in texts]
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        size = max(1, int(self.settings.batch_size))
        for start in range(0, len(items), size):
            vectors.extend(self._embed_batch(items[start : start + size]))
        if not self._dim_confirmed and vectors:
            self.dim = len(vectors[0])
            self._dim_confirmed = True
        matrix = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(matrix)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed_all(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_all([text])[0]


def get_embedder(settings: Settings) -> Embedder:
    """Build the configured embedder. No silent fallback: an unconfigured provider
    raises :class:`ProviderNotConfigured` naming the env var it needs."""
    provider = settings.embedding.provider
    if provider == "hashing":
        return HashingEmbedder(dim=settings.embedding.dim, seed=settings.seed)
    return RemoteEmbedder(settings.embedding)
