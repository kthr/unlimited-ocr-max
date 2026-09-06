"""baidu/Unlimited-OCR as a MAX custom architecture.

``ARCHITECTURES`` is what ``max serve --custom-architectures <this dir>`` reads.
It is resolved lazily so the stdlib-only console script (:mod:`.cli`) never
imports MAX; MAX reads the attribute through ``__getattr__`` like any other.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ARCHITECTURES", "unlimited_ocr_arch"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .arch import unlimited_ocr_arch

        return [unlimited_ocr_arch] if name == "ARCHITECTURES" else unlimited_ocr_arch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
