"""Read-only paper broker snapshot for the September 19 profit investigation."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

OUT = ROOT / 'research/profit_investigation_2026-09-19/data'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = AlpacaOptionsClient(env_file='.env#PAPER', timeout_sec=30)
    assert client._trading_base == 'https://paper-api.alpaca.markets'
    # Exclusive creation preserves prior snapshots and research outputs.
    for name in ['activities.jsonl', 'positions.json']:
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)
    cursor = '2020-01-01T00:00:00Z'
    seen = set()
    # Resume an interrupted download append-only. The endpoint can repeat its
    # boundary order despite the documented exclusive timestamp cursor.
    if (OUT / 'orders.jsonl').exists():
        previous = [json.loads(line) for line in (OUT / 'orders.jsonl').open()]
        seen = {r['id'] for r in previous}
        cursor = previous[-1].get('submitted_at') or previous[-1]['created_at']
    with (OUT / 'orders.jsonl').open('a') as f:
        while True:
            batch = client.get_orders(status='all', limit=500, direction='asc',
                                      after=cursor, until='2026-09-19T04:00:00Z', nested='false')
            assert isinstance(batch, list)
            if not batch:
                break
            for row in batch:
                if row['id'] in seen:
                    continue
                seen.add(row['id'])
                f.write(json.dumps(row) + '\n')
            nxt = batch[-1].get('submitted_at') or batch[-1]['created_at']
            if nxt == cursor and len(batch) < 500:
                break
            assert nxt != cursor, 'Pagination stalled'
            cursor = nxt
            print('orders', len(seen), cursor, flush=True)
    token = None
    seen = set()
    with (OUT / 'activities.jsonl').open('x') as f:
        while True:
            batch = client.get_account_activities(after='2020-01-01T00:00:00Z',
                until='2026-09-19T04:00:00Z', direction='asc', page_size=100, page_token=token)
            assert isinstance(batch, list)
            if not batch:
                break
            for row in batch:
                assert row['id'] not in seen, 'Activity pagination duplicate'
                seen.add(row['id'])
                f.write(json.dumps(row) + '\n')
            token = batch[-1]['id']
            print('activities', len(seen), flush=True)
    with (OUT / 'positions.json').open('x') as f:
        json.dump(client.get_positions(), f)
    (OUT / 'snapshot_metadata.json').write_text(json.dumps({
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'environment': 'paper', 'requested_after': '2020-01-01T00:00:00Z',
        'exclusive_until': '2026-09-19T04:00:00Z',
        'positions_are_capture_time_not_historical': True,
    }, indent=2))

if __name__ == '__main__':
    main()
