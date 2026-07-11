from __future__ import annotations

from .kernel_feedback_apply_mixin import KernelFeedbackApplyMixin
from .kernel_feedback_config_mixin import KernelFeedbackConfigMixin
from .kernel_feedback_task_mixin import KernelFeedbackTaskMixin


class KernelFeedbackMixin(
    KernelFeedbackApplyMixin,
    KernelFeedbackTaskMixin,
    KernelFeedbackConfigMixin,
):
    """Compatibility aggregate for feedback correction helpers."""
