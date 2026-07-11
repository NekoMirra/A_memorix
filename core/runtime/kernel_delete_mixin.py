from __future__ import annotations

from .kernel_delete_action_mixin import KernelDeleteActionMixin
from .kernel_delete_support_mixin import KernelDeleteSupportMixin


class KernelDeleteMixin(
    KernelDeleteActionMixin,
    KernelDeleteSupportMixin,
):
    """Compatibility aggregate for delete admin helpers."""
