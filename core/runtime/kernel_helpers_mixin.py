from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence


class KernelHelpersMixin:
    @staticmethod
    def _tokens(values: Optional[Iterable[Any]]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in values or []:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    @classmethod
    def _merge_tokens(cls, *groups: Optional[Iterable[Any]]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for group in groups:
            for item in cls._tokens(group):
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
        return merged

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value in {None, ""}:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except Exception:
            return None
