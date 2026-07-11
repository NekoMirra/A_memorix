from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Sequence
from src.common.logger import get_logger

logger = get_logger("A_Memorix.SDKMemoryKernel")


class KernelProfileMixin:
    @staticmethod
    def _empty_person_profile_response(*, person_id: str = "", person_name: str = "") -> Dict[str, Any]:
        return {
            "summary": "",
            "traits": [],
            "evidence": [],
            "person_id": str(person_id or "").strip(),
            "person_name": str(person_name or "").strip(),
            "profile_source": "",
            "has_manual_override": False,
        }

    async def _query_person_profile_with_feedback_refresh(
        self,
        *,
        person_id: str = "",
        person_keyword: str = "",
        limit: int = 10,
        force_refresh: bool = False,
        source_note: str,
    ) -> Dict[str, Any]:
        assert self.metadata_store is not None
        assert self.person_profile_service is not None

        pid = str(person_id or "").strip()
        if not pid and person_keyword:
            pid = self.person_profile_service.resolve_person_id(str(person_keyword or "").strip())

        dirty_request = self.metadata_store.get_person_profile_refresh_request(pid) if pid else None
        should_force_refresh = bool(force_refresh)
        if (
            pid
            and self._feedback_cfg_profile_refresh_enabled()
            and self._feedback_cfg_profile_force_refresh_on_read()
            and isinstance(dirty_request, dict)
            and str(dirty_request.get("status", "") or "").strip().lower() in {"pending", "running", "failed"}
        ):
            should_force_refresh = True

        profile = await self.person_profile_service.query_person_profile(
            person_id=pid,
            person_keyword=str(person_keyword or "").strip(),
            top_k=max(1, int(limit or 10)),
            force_refresh=should_force_refresh,
            source_note=source_note,
        )
        payload = profile if isinstance(profile, dict) else {"success": False, "error": "invalid profile payload"}
        if dirty_request:
            payload["feedback_refresh_request"] = dirty_request
        if should_force_refresh and dirty_request and not bool(payload.get("success")):
            payload.setdefault("error", "feedback_refresh_failed")
            payload["feedback_refresh_failed"] = True
        return payload

    def _build_person_profile_response(
        self,
        profile: Dict[str, Any],
        *,
        requested_person_id: str,
        limit: int,
    ) -> Dict[str, Any]:
        assert self.metadata_store is not None
        if not bool(profile.get("success")):
            return self._empty_person_profile_response(
                person_id=str(profile.get("person_id", "") or requested_person_id),
                person_name=str(profile.get("person_name", "") or ""),
            )

        evidence: List[Dict[str, Any]] = []
        evidence_limit = max(1, int(limit or 10))
        for hash_value in profile.get("evidence_ids", [])[:evidence_limit]:
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is not None:
                evidence.append(
                    {
                        "hash": hash_value,
                        "content": str(paragraph.get("content", "") or "")[:220],
                        "metadata": paragraph.get("metadata", {}) or {},
                        "type": "paragraph",
                    }
                )
                continue

            relation = self.metadata_store.get_relation(hash_value)
            if relation is not None:
                evidence.append(
                    {
                        "hash": hash_value,
                        "content": " ".join(
                            [
                                str(relation.get("subject", "") or "").strip(),
                                str(relation.get("predicate", "") or "").strip(),
                                str(relation.get("object", "") or "").strip(),
                            ]
                        ).strip(),
                        "metadata": {
                            "confidence": relation.get("confidence"),
                            "source_paragraph": relation.get("source_paragraph"),
                        },
                        "type": "relation",
                    }
                )

        evidence = self._filter_user_visible_hits(evidence)
        text = str(profile.get("profile_text", "") or "").strip()
        traits = [line.strip("- ").strip() for line in text.splitlines() if line.strip()][:8]
        return {
            "summary": text,
            "traits": traits,
            "evidence": evidence,
            "person_id": str(profile.get("person_id", "") or requested_person_id),
            "person_name": str(profile.get("person_name", "") or ""),
            "profile_source": str(profile.get("profile_source", "") or "auto_snapshot"),
            "has_manual_override": bool(profile.get("has_manual_override", False)),
        }

    async def get_person_profile(self, *, person_id: str, chat_id: str = "", limit: int = 10) -> Dict[str, Any]:
        del chat_id
        await self.initialize()
        assert self.metadata_store is not None
        assert self.person_profile_service is not None
        self._mark_person_active(person_id)
        profile = await self._query_person_profile_with_feedback_refresh(
            person_id=person_id,
            limit=max(4, int(limit or 10)),
            source_note="sdk_memory_kernel.get_person_profile",
        )
        return self._build_person_profile_response(profile, requested_person_id=person_id, limit=limit)

    async def refresh_person_profile(self, person_id: str, limit: int = 10, *, mark_active: bool = True) -> Dict[str, Any]:
        await self.initialize()
        assert self.person_profile_service
        if mark_active:
            self._mark_person_active(person_id)
        profile = await self.person_profile_service.query_person_profile(
            person_id=person_id,
            top_k=max(4, int(limit or 10)),
            force_refresh=True,
            source_note="sdk_memory_kernel.refresh_person_profile",
        )
        return profile if isinstance(profile, dict) else {}
