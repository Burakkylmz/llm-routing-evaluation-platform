"""Safe parsing of HTTP 429 ``Retry-After`` hints (shared by Groq and NVIDIA).

Only the numeric delay-seconds form is accepted (what Groq and NVIDIA send).
HTTP-date values, negative numbers, exponents, NaN/inf and malformed text are
ignored, so they become "no hint" and the orchestrator's own backoff applies.
Raw header values are never stored, logged or propagated.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

# Up to 32 integer digits and an optional fraction of up to 3 digits.
_NUMERIC_SECONDS = re.compile(r"^(?:0|[1-9][0-9]{0,31})(?:\.[0-9]{1,3})?$")
# A hint larger than this is saturated (not dropped), so it still exceeds every
# allowed retry-wait budget and can never cause an *earlier* retry.
MAX_RETRY_AFTER_SECONDS = 86_400.0


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Return the numeric ``Retry-After`` seconds, or ``None`` if absent/invalid."""
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        value = headers.get("retry-after")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if _NUMERIC_SECONDS.fullmatch(text) is None:
        return None
    return min(float(text), MAX_RETRY_AFTER_SECONDS)


def checked_hint(value: float | None) -> float | None:
    """Defensive re-check in providers: an injected transport could return any float."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not 0 <= value <= MAX_RETRY_AFTER_SECONDS:
        return None
    return float(value)
