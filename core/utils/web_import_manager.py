"""Compatibility shim for ImportTaskManager.

Implementation lives in ``core.utils.web_import``.
"""

from .web_import import (
    CHUNK_STATUS,
    FILE_STATUS,
    FILE_WARNING_KEEP_LIMIT,
    ImportChunkRecord,
    ImportFileRecord,
    ImportTaskManager,
    ImportTaskRecord,
    TASK_STATUS,
)

__all__ = [
    "ImportTaskManager",
    "ImportChunkRecord",
    "ImportFileRecord",
    "ImportTaskRecord",
    "TASK_STATUS",
    "FILE_STATUS",
    "CHUNK_STATUS",
    "FILE_WARNING_KEEP_LIMIT",
]
