from __future__ import annotations

from .kernel_graph_mutation_mixin import KernelGraphMutationMixin
from .kernel_graph_query_mixin import KernelGraphQueryMixin


class KernelGraphMixin(
    KernelGraphMutationMixin,
    KernelGraphQueryMixin,
):
    """Compatibility aggregate for graph serialization and mutation helpers."""
