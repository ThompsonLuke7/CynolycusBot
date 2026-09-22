"""Freeze relevant local evidence, retaining source path/line and hashes."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/profit_investigation_2026-09-19/data/local'

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = []
    for p in sorted((ROOT / 'Data/inference').glob('*/*.jsonl')):
        if p.name not in {'closed_trades.jsonl', 'live_signal_audit.jsonl'}:
            continue
        target = OUT / (p.parent.name + '__' + p.name)
        data = p.read_bytes()
        target.write_bytes(data)
        manifest.append({'source': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    kinds = {'order_submitted', 'position_opened', 'position_closed', 'signal', 'confirmation',
             'signal_policy_entry_decision', 'signal_policy_decision', 'entry_skipped'}
    with (OUT / 'swing_events.jsonl').open('x') as f:
        for p in sorted((ROOT / 'UI/swing_audit').glob('swing_session_*.jsonl')):
            h = hashlib.sha256()
            for i, line in enumerate(p.open('rb'), 1):
                h.update(line)
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('type') in kinds:
                    f.write(json.dumps({'source': str(p.relative_to(ROOT)), 'line': i, 'record': r}) + '\n')
            manifest.append({'source': str(p.relative_to(ROOT)), 'sha256': h.hexdigest()})
    p = ROOT / 'Data/inference/account_snapshots/broker_equity_20260918_paper.jsonl'
    shutil.copyfile(p, OUT / p.name)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('snapshotted', len(manifest), 'files')

if __name__ == '__main__':
    main()
