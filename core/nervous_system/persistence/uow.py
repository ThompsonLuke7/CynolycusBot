"""Explicit transaction boundary for nervous-system persistence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from .repositories.decision import DecisionRepository
from .repositories.execution import ExecutionRepository
from .repositories.operations import OperationsRepository
from .repositories.registry import RegistryRepository
from .repositories.state import StateRepository


class UnitOfWork:
    """Bind all repositories to one session and require explicit commit."""

    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory
        self.session: Session
        self._committed = False
        self._rolled_back = False

    def __enter__(self) -> "UnitOfWork":
        self.session = self._session_factory()
        self._committed = False
        self._rolled_back = False
        self.states = StateRepository(self.session)
        self.decisions = DecisionRepository(self.session)
        self.executions = ExecutionRepository(self.session)
        self.registry = RegistryRepository(self.session)
        self.operations = OperationsRepository(self.session)
        return self

    def commit(self) -> None:
        self.session.commit()
        self._committed = True

    def rollback(self) -> None:
        self.session.rollback()
        self._rolled_back = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc_type is not None or (not self._committed and not self._rolled_back):
                self.rollback()
        finally:
            self.session.close()
