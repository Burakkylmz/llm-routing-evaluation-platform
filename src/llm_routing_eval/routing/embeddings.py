"""Stable offline lexical feature hashing, not semantic model embeddings.

This implementation uses SHA-256 rather than process-randomized Python hash,
with no model downloads, network access, or raw request logging.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class HashingEmbedding:
    """A deterministic lightweight development adapter implementing EmbeddingPort."""

    def __init__(self, *, dimensions: int = 256) -> None:
        if not 16 <= dimensions <= 4096:
            raise ValueError("Embedding dimensions must be within [16, 4096]")
        self._dimensions = dimensions

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        result: list[tuple[float, ...]] = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in _TOKEN_PATTERN.findall(text.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self._dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise ValueError("Cannot embed text without lexical tokens")
            result.append(tuple(value / norm for value in vector))
        return tuple(result)
