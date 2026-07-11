# -*- coding: utf-8 -*-
"""元数据存储模块

基于SQLite的元数据管理，存储段落、实体、关系等信息。

实现已按领域拆分为 core.storage.metadata.* mixin，本文件保留 MetadataStore 门面。
"""

from .metadata.constants import RUNTIME_AUTO_MIGRATION_MIN_SCHEMA_VERSION, SCHEMA_VERSION
from .metadata.fts import MetadataFtsMixin
from .metadata.paragraphs import MetadataParagraphsMixin
from .metadata.entities import MetadataEntitiesMixin
from .metadata.relations import MetadataRelationsMixin
from .metadata.external_refs import MetadataExternalRefsMixin
from .metadata.operations import MetadataOperationsMixin
from .metadata.stats import MetadataStatsMixin
from .metadata.gc import MetadataGcMixin
from .metadata.person_profile import MetadataPersonProfileMixin
from .metadata.episodes import MetadataEpisodesMixin
from .metadata.feedback import MetadataFeedbackMixin
from .metadata.base import MetadataStoreBase


class MetadataStore(
    MetadataFtsMixin,
    MetadataParagraphsMixin,
    MetadataEntitiesMixin,
    MetadataRelationsMixin,
    MetadataExternalRefsMixin,
    MetadataOperationsMixin,
    MetadataStatsMixin,
    MetadataGcMixin,
    MetadataPersonProfileMixin,
    MetadataEpisodesMixin,
    MetadataFeedbackMixin,
    MetadataStoreBase,
):
    """元数据存储类

    功能：
    - SQLite数据库管理
    - 段落/实体/关系元数据存储
    - 增删改查操作
    - 事务支持
    - 索引优化

    参数：
        data_dir: 数据目录
        db_name: 数据库文件名（默认metadata.db）
    """


__all__ = [
    "MetadataStore",
    "SCHEMA_VERSION",
    "RUNTIME_AUTO_MIGRATION_MIN_SCHEMA_VERSION",
]
