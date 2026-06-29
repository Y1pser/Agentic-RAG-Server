"""Abstract base class for Vision LLM providers.

This module defines the pluggable interface for Vision Language Model providers,
enabling seamless switching between different backends (Azure Vision, GPT-4V,
etc.) through configuration-driven instantiation.

Vision LLMs take an image (bytes, path, or base64) and return a text description,
which is used by downstream components like ImageCaptioner (B2.7) to generate
captions for images extracted from documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers.

    All Vision LLM implementations must inherit from this class and implement
    the describe() method. This ensures a consistent interface across different
    providers (Azure Vision, GPT-4V, etc.).

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Accepts optional TraceContext for observability integration.
    - Config-Driven: Instances are created via factory based on settings.

    The describe() method accepts images in multiple formats:
    - bytes: Raw image binary data (PNG, JPEG, etc.)
    - str/Path: File path to an image on disk
    - str (base64): Base64-encoded image string with optional data URI prefix
    """

    @abstractmethod
    def describe(
        self,
        image: Union[bytes, str, Path],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text description of the given image.

        Args:
            image: The image to describe. Accepts:
                - bytes: Raw image binary data.
                - str (path): File path to an image on disk.
                - Path: File path to an image on disk.
                - str (base64): Base64-encoded image (optionally with data URI prefix,
                  e.g. "data:image/png;base64,...").
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters (max_tokens, detail_level, etc.).

        Returns:
            A text description of the image content.

        Raises:
            ValueError: If the image is empty, malformed, or in an unsupported format.
            FileNotFoundError: If the image path does not exist on disk.
            RuntimeError: If the Vision LLM provider call fails.
        """
        ...

    def _resolve_image_bytes(self, image: Union[bytes, str, Path]) -> bytes:
        """Resolve various image input formats to raw bytes.

        This is a convenience helper for subclasses. It handles:
        - bytes: Pass through directly.
        - str (path-like): Read from file.
        - Path: Read from file.
        - str (base64): Strip optional data URI prefix and decode.

        Args:
            image: The image in one of the supported formats.

        Returns:
            Raw image bytes.

        Raises:
            ValueError: If image is empty or base64 decoding fails.
            FileNotFoundError: If the file path does not exist.
        """
        if isinstance(image, bytes):
            if len(image) == 0:
                raise ValueError("Image bytes cannot be empty")
            return image

        if isinstance(image, Path):
            if not image.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            data = image.read_bytes()
            if len(data) == 0:
                raise ValueError(f"Image file is empty: {image}")
            return data

        # str: could be a path or base64
        if isinstance(image, str):
            # Check for base64 data URI prefix (definitive)
            if image.startswith("data:"):
                return _decode_base64_image(image)

            # Try base64 decode first for non-path-looking strings.
            # A valid base64 image payload is typically much longer than
            # a plausible file path.
            if _is_likely_base64(image):
                return _decode_base64_image(image)

            # Treat as file path
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            data = path.read_bytes()
            if len(data) == 0:
                raise ValueError(f"Image file is empty: {image}")
            return data

        raise ValueError(
            f"Unsupported image type: {type(image).__name__}. "
            f"Expected bytes, str (path or base64), or Path."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_likely_base64(s: str) -> bool:
    """Heuristic: a string with no path separators is likely base64.

    File paths typically contain directory separators (``/`` or ``\\``),
    start with ``.`` (relative), or start with a drive letter (Windows).
    A pure base64 string has none of these.
    """
    # Must not look like a path
    if "/" in s or "\\" in s:
        return False
    if s.startswith(".") or s.startswith("/"):
        return False
    # Windows absolute path: e.g. "C:\..." — check for colon in first 3 chars
    if len(s) > 2 and s[1:3] == ":\\":
        return False
    # Base64 is almost always longer than 16 chars for any real image
    return len(s) > 16


def _decode_base64_image(s: str) -> bytes:
    """Decode a base64-encoded image string to bytes.

    Handles optional data URI prefix like ``data:image/png;base64,...``.
    """
    import base64

    payload = s
    if s.startswith("data:"):
        # Strip data URI prefix: "data:image/png;base64,<payload>"
        try:
            payload = s.split(",", 1)[1]
        except IndexError:
            raise ValueError(
                "Malformed data URI: missing comma separator after MIME info"
            )

    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 image data: {exc}") from exc

    if len(data) == 0:
        raise ValueError("Decoded image data is empty")

    return data
