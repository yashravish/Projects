"""
Base HTTP client wrapping httpx with retry logic and structured logging.
"""
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import IntegrationError
from app.utils.retry import retry_request

logger = logging.getLogger(__name__)


class BaseHttpClient:
    """
    Reusable synchronous HTTP client with:
      - configurable base URL
      - automatic retry on transient failures
      - structured logging per request
      - IntegrationError wrapping
    """

    def __init__(self, base_url: str, source_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.source_name = source_name
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        logger.info(
            "http_request",
            extra={"source": self.source_name, "method": "GET", "url": url, "params": params},
        )
        try:
            response = retry_request(self._client.get, path, params=params)
            logger.info(
                "http_response",
                extra={
                    "source": self.source_name,
                    "url": url,
                    "status_code": response.status_code,
                },
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise IntegrationError(
                self.source_name,
                f"HTTP {exc.response.status_code} from {url}",
            ) from exc
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise IntegrationError(
                self.source_name,
                f"Connection failed to {url}: {exc}",
            ) from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BaseHttpClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
