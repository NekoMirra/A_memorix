"""MetadataStore 领域拆分包（mixin 组合，保持单一 SQLite 连接）。"""

from .base import MetadataStoreBase
from .fts import MetadataFtsMixin
from .paragraphs import MetadataParagraphsMixin
from .entities import MetadataEntitiesMixin
from .relations import MetadataRelationsMixin
from .external_refs import MetadataExternalRefsMixin
from .operations import MetadataOperationsMixin
from .stats import MetadataStatsMixin
from .gc import MetadataGcMixin
from .person_profile import MetadataPersonProfileMixin
from .episodes import MetadataEpisodesMixin
from .feedback import MetadataFeedbackMixin
from .constants import RUNTIME_AUTO_MIGRATION_MIN_SCHEMA_VERSION, SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "RUNTIME_AUTO_MIGRATION_MIN_SCHEMA_VERSION",
    "MetadataStoreBase",
    "MetadataFtsMixin",
    "MetadataParagraphsMixin",
    "MetadataEntitiesMixin",
    "MetadataRelationsMixin",
    "MetadataExternalRefsMixin",
    "MetadataOperationsMixin",
    "MetadataStatsMixin",
    "MetadataGcMixin",
    "MetadataPersonProfileMixin",
    "MetadataEpisodesMixin",
    "MetadataFeedbackMixin",
]
