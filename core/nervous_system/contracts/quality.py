from __future__ import annotations

from .base import ContractModel
from .enums import DataQualitySeverity


class LineageRef(ContractModel):
    source_id: str
    content_hash: str
    record_locator: str | None = None


class DataQualityIssue(ContractModel):
    code: str
    severity: DataQualitySeverity
    component: str
    message: str
    fallback_used: str | None = None


class DataQualitySummary(ContractModel):
    issues: tuple[DataQualityIssue, ...] = ()

    @property
    def is_usable(self) -> bool:
        return not any(
            issue.severity in {DataQualitySeverity.ERROR, DataQualitySeverity.CRITICAL}
            for issue in self.issues
        )
