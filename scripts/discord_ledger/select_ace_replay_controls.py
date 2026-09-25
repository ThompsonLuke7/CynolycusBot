"""Select one deterministic matched control per source alert for engine replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

from scripts.discord_ledger.build import read_jsonl, write_jsonl


def _key(row: dict) -> tuple[str, str, int, tuple[str, ...]]:
    return (str(row["symbol"]), str(row["session_date_et"]), int(row["direction_sign"]),
            tuple(sorted(str(value) for value in row["source_message_ids"])))


def select(alerts: list[dict], controls: list[dict]) -> list[dict]:
    """Hash-order alerts, then assign distinct controls before reading outcomes."""
    grouped: dict[tuple[str, str, int, tuple[str, ...]], list[dict]] = defaultdict(list)
    alert_groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for control in controls:
        grouped[_key(control)].append(control)
    for alert in alerts:
        alert_groups[(str(alert["symbol"]), str(alert["session_date_et"]), int(alert["direction_sign"]))].append(alert)
    selected = []
    for group_key, group_alerts in alert_groups.items():
        # The original control file repeats the same eligible decision times
        # once for each source alert. Collapse those repetitions before pairing.
        candidate_by_time = {}
        for alert in group_alerts:
            for control in grouped[_key(alert)]:
                candidate_by_time[control["decision_at_utc"]] = control
        candidates = [candidate_by_time[time] for time in sorted(candidate_by_time)]
        if len(candidates) < len(group_alerts):
            raise ValueError(f"Insufficient distinct controls for {group_key}: {len(candidates)} < {len(group_alerts)}")
        ordered_alerts = sorted(group_alerts, key=lambda alert: hashlib.sha256(
            "|".join((*_key(alert)[3], group_key[0], group_key[1], str(group_key[2]))).encode("utf-8")
        ).hexdigest())
        # A hashed circular offset prevents one-alert groups from always taking
        # the earliest (typically morning) eligible control.  The subsequent
        # rank assignment is without replacement because group_alerts <=
        # candidates was checked above.
        offset = int(hashlib.sha256("|".join((*group_key[0:2], str(group_key[2]))).encode("utf-8")).hexdigest(), 16) % len(candidates)
        for rank, alert in enumerate(ordered_alerts):
            control = dict(candidates[(offset + rank) % len(candidates)])
            control["control_selection_method"] = "sha256-order source alerts plus sha256 circular offset, then assign distinct sorted eligible controls within symbol/session/direction"
            control["control_selection_rank"] = (offset + rank) % len(candidates)
            control["control_candidate_count"] = len(candidates)
            control["paired_source_alert_decision_at_utc"] = alert["decision_at_utc"]
            # Restore the paired alert's source evidence; it is the research
            # identity carried into the state-machine replay.
            control["source_message_ids"] = alert["source_message_ids"]
            control["trade_ids"] = alert["trade_ids"]
            control["source_quote"] = alert["source_quote"]
            control["source_tags"] = alert["source_tags"]
            control["exclusive_bucket"] = alert["exclusive_bucket"]
            selected.append(control)
    identities = {(row["symbol"], row["decision_at_utc"], row["direction_sign"]) for row in selected}
    if len(identities) != len(selected):
        raise ValueError("Control selection yielded duplicate engine-replay identities")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    rows = select(read_jsonl(args.alerts), read_jsonl(args.controls))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print({"selected_controls": len(rows), "out": str(args.out)})


if __name__ == "__main__":
    main()
