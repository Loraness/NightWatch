import asyncio
import logging
import random
import time
from collections import deque
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from app.config import settings

logger = logging.getLogger("nightwatch.llm")

RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    TimeoutError,
    ConnectionError,
)


class AsyncRateLimiter:
    """Sliding-window rate limiter allowing bursts up to RPM within each window."""

    def __init__(self, requests_per_minute: int, window_seconds: float = 60.0) -> None:
        self._limit = requests_per_minute
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._limit <= 0:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._timestamps and (now - self._timestamps[0]) >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(now)
                    return
                wait_for = self._window - (now - self._timestamps[0])
                await asyncio.sleep(max(wait_for, 0.0))


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = settings.GEMINI_API_KEY,
        model: str = settings.GEMINI_MODEL,
        rpm_limit: int = settings.GEMINI_RPM_LIMIT,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._rate_limiter = AsyncRateLimiter(rpm_limit)

    async def generate_content(
        self,
        contents: Any,
        config: types.GenerateContentConfig | None = None,
        model: str | None = None,
    ) -> types.GenerateContentResponse:
        target_model = model or self._model
        for attempt in range(self._max_retries):
            try:
                await self._rate_limiter.acquire()
                return await self._client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                )
            except (errors.APIError, *RETRYABLE_NETWORK_ERRORS) as exc:
                is_fatal = (
                    isinstance(exc, errors.APIError)
                    and exc.code not in RETRYABLE_STATUS_CODES
                )
                if is_fatal or attempt == self._max_retries - 1:
                    raise
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Gemini attempt %s failed (%s); retrying in %.2fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly.")

    async def generate_text(
        self,
        prompt: str,
        config: types.GenerateContentConfig | None = None,
    ) -> str:
        response = await self.generate_content(contents=prompt, config=config)
        return (response.text or "").strip()
