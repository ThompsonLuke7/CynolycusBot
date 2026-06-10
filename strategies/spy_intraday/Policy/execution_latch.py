from __future__ import annotations

from dataclasses import dataclass


def _to_pos(value: int | float) -> int:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if v > 0.0:
        return 1
    if v < 0.0:
        return -1
    return 0


@dataclass(frozen=True)
class LatchUpdate:
    raw_pos: int
    executed_pos: int
    changed: bool
    pending_target: int | None
    pending_count: int
    status: str


class DirectionExecutionLatch:
    """
    Decouple noisy model direction from executed position:
    - Entry requires `entry_confirm_bars` consecutive non-flat signals while flat.
    - Exit/flip requires `exit_confirm_bars` consecutive non-aligned signals while in-position.
    - If signal re-aligns with current position, pending exit/flip is canceled.
    """

    def __init__(
        self,
        *,
        entry_confirm_bars: int = 1,
        exit_confirm_bars: int = 2,
        initial_position: int = 0,
    ) -> None:
        self.entry_confirm_bars = max(1, int(entry_confirm_bars))
        self.exit_confirm_bars = max(1, int(exit_confirm_bars))
        self._executed_pos = _to_pos(initial_position)
        self._pending_target: int | None = None
        self._pending_count = 0

    @property
    def executed_pos(self) -> int:
        return int(self._executed_pos)

    def set_position(self, pos: int | float) -> None:
        self._executed_pos = _to_pos(pos)
        self._pending_target = None
        self._pending_count = 0

    def snapshot(self) -> dict[str, int | None]:
        return {
            "executed_pos": int(self._executed_pos),
            "pending_target": (int(self._pending_target) if self._pending_target is not None else None),
            "pending_count": int(self._pending_count),
            "entry_confirm_bars": int(self.entry_confirm_bars),
            "exit_confirm_bars": int(self.exit_confirm_bars),
        }

    def _clear_pending(self) -> None:
        self._pending_target = None
        self._pending_count = 0

    def step(self, raw_pos: int | float) -> LatchUpdate:
        raw = _to_pos(raw_pos)
        changed = False
        status = "hold"

        if self._executed_pos == 0:
            if raw == 0:
                self._clear_pending()
                status = "flat_no_signal"
            else:
                if self._pending_target == raw:
                    self._pending_count += 1
                else:
                    self._pending_target = raw
                    self._pending_count = 1
                if self._pending_count >= self.entry_confirm_bars:
                    self._executed_pos = raw
                    changed = True
                    self._clear_pending()
                    status = "entry_confirmed"
                else:
                    status = "entry_pending"
        else:
            if raw == self._executed_pos:
                if self._pending_target is not None:
                    status = "pending_canceled_realign"
                else:
                    status = "hold_aligned"
                self._clear_pending()
            else:
                desired = raw
                if self._pending_target == desired:
                    self._pending_count += 1
                else:
                    self._pending_target = desired
                    self._pending_count = 1
                if self._pending_count >= self.exit_confirm_bars:
                    prev = int(self._executed_pos)
                    self._executed_pos = desired
                    changed = True
                    self._clear_pending()
                    if desired == 0:
                        status = "exit_confirmed"
                    elif desired == -prev:
                        status = "flip_confirmed"
                    else:
                        status = "reposition_confirmed"
                else:
                    status = "exit_pending" if desired == 0 else "flip_pending"

        return LatchUpdate(
            raw_pos=raw,
            executed_pos=int(self._executed_pos),
            changed=bool(changed),
            pending_target=(int(self._pending_target) if self._pending_target is not None else None),
            pending_count=int(self._pending_count),
            status=status,
        )
