"""
A1D Image Upscaler API Client (Production-ready)

- Loads API key from .env (no hardcoding)
- Uses requests with timeouts
- Strong typing and clear error handling
- Reusable client class + convenience function

Install:
    pip install requests python-dotenv

Setup:
    Create a .env file in the same directory as this script with:
        A1D_API_KEY=your_real_api_key_here

Run:
    python a1d_image_upscaler.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

DEFAULT_ENDPOINT = "https://api.a1d.ai/api/image-upscaler"


class A1DUpscalerError(RuntimeError):
    """Raised when the A1D Image Upscaler API returns an error or request fails."""


@dataclass(frozen=True)
class A1DUpscalerClient:
    """
    Reusable client for the A1D Image Upscaler API.

    Environment:
        A1D_API_KEY: API key loaded via python-dotenv from .env
    """

    api_key: str
    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: int = 60

    @classmethod
    def from_env(
        cls,
        *,
        env_var_name: str = "A1D_API_KEY",
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: int = 60,
    ) -> "A1DUpscalerClient":
        """
        Create a client using API key from environment (supports .env via python-dotenv).
        """
        load_dotenv()  # Loads .env into environment variables (if present)
        api_key = os.getenv(env_var_name)

        if not api_key or not api_key.strip():
            raise ValueError(
                f"Missing API key. Set {env_var_name} in your environment or .env file."
            )

        return cls(
            api_key=api_key.strip(),
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
        )

    def upscale_image(
        self,
        image_url: str,
        *,
        scale: int = 2,
        source: str = "api",
    ) -> Dict[str, Any]:
        """
        Upscale an image by URL.

        Args:
            image_url: Publicly accessible image URL to upscale.
            scale: Scale factor (e.g., 2 or 4). Defaults to 2.
            source: Optional source field. Defaults to "api".

        Returns:
            Parsed JSON response as a dict.

        Raises:
            A1DUpscalerError: For network issues, invalid responses, or non-2xx API errors.
            ValueError: For invalid input arguments.
        """
        if not image_url or not image_url.strip():
            raise ValueError("image_url must be a non-empty string.")

        if not isinstance(scale, int) or scale <= 0:
            raise ValueError("scale must be a positive integer (e.g., 2 or 4).")

        # Required headers (Authorization format must be: "KEY {API_KEY}")
        headers: Dict[str, str] = {
            "Authorization": f"KEY {self.api_key}",
            "Content-Type": "application/json",
        }

        # Request body
        payload: Dict[str, Any] = {
            "imageUrl": image_url.strip(),
            "scale": scale,
            "source": source,
        }

        try:
            # Perform POST request with timeout
            resp = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )

            # Raise for HTTP error codes (4xx/5xx), but include API's exact message/body
            self._raise_for_status_with_details(resp)

            # Parse JSON response
            return self._safe_json(resp)

        except Timeout as exc:
            raise A1DUpscalerError(
                f"Request timed out after {self.timeout_seconds}s: {exc}"
            ) from exc
        except ConnectionError as exc:
            raise A1DUpscalerError(f"Network connection error: {exc}") from exc
        except HTTPError as exc:
            # _raise_for_status_with_details already included details in the message
            raise A1DUpscalerError(str(exc)) from exc
        except RequestException as exc:
            # Catch-all for requests-related exceptions
            raise A1DUpscalerError(f"Request failed: {exc}") from exc
        except ValueError as exc:
            # JSON parsing errors or our validation errors
            raise A1DUpscalerError(f"Invalid response or input: {exc}") from exc

    @staticmethod
    def _safe_json(resp: Response) -> Dict[str, Any]:
        """
        Safely parse JSON response.
        """
        try:
            data = resp.json()
        except ValueError:
            body_preview = (resp.text or "").strip()
            raise ValueError(
                f"API returned non-JSON response (status {resp.status_code}): {body_preview}"
            )

        if not isinstance(data, dict):
            raise ValueError(
                f"Unexpected JSON type: {type(data).__name__}; expected object/dict."
            )

        return data

    @staticmethod
    def _raise_for_status_with_details(resp: Response) -> None:
        """
        If non-2xx, raise HTTPError that includes the API's exact returned error message/body.
        """
        if 200 <= resp.status_code < 300:
            return

        detail: Optional[str]
        try:
            parsed = resp.json()
            detail = json.dumps(parsed, ensure_ascii=False)
        except ValueError:
            detail = (resp.text or "").strip()

        message = f"HTTP {resp.status_code} Error from A1D API: {detail}"
        raise HTTPError(message)


def upscale_image(image_url: str, scale: int = 2) -> Dict[str, Any]:
    """
    Convenience function (no need to manually instantiate client).

    Loads API key from .env / environment variable A1D_API_KEY.
    """
    client = A1DUpscalerClient.from_env()
    return client.upscale_image(image_url, scale=scale)


if __name__ == "__main__":
    # Sample usage for quick testing:
    # Replace with a real, publicly accessible image URL.
    sample_image_url = "https://example.com/your-image.jpg"
    sample_scale = 2

    try:
        result = upscale_image(sample_image_url, scale=sample_scale)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except A1DUpscalerError as exc:
        # Print the exact error message (including API returned message/body)
        print(f"Upscale failed: {exc}")
