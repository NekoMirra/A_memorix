from __future__ import annotations

from .kernel_admin_domain_mixin import KernelAdminDomainMixin
from .kernel_admin_ops_mixin import KernelAdminOpsMixin


class KernelAdminMixin(
    KernelAdminDomainMixin,
    KernelAdminOpsMixin,
):
    """Compatibility aggregate for memory admin entrypoints."""
