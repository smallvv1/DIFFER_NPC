"""独立的 NDP 诊断组件。

本包只提供局部身份参考库和局部间隔保护损失，不修改 GPC、NPC 或正式训练入口。
"""

from .local_margin import (
    build_train_only_reference_library,
    local_margin_protection_loss,
    summarize_local_geometry,
)

__all__ = [
    "build_train_only_reference_library",
    "local_margin_protection_loss",
    "summarize_local_geometry",
]
