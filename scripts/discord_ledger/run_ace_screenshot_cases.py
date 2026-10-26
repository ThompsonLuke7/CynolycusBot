"""Replay screenshot-motivated level families on actual October price paths.

Screenshot publication clocks remain null until a capture date/display timezone
is supplied. Model events are independently generated from prior daily bars.
Original screenshot zones are price-path comparisons, not assumed fills or
causally available model inputs. Stores the exact validated daily input values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.discord_ledger.build import digest, write_jsonl
from scripts.discord_ledger.run_ace_sequence_replay import (
    DAILY, Policy, completed_candles, prepare_daily_bars, replay_symbol, session_bounds,
    session_daily_features, utc,
)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    daily_dir=args.out/'normalized_daily_inputs';daily_dir.mkdir()
    events,cases=[],[]
    policies=[Policy('sma5_touch_reaction_swing','daily',2,('sma_5',)),
              Policy('ema21_touch_reaction_swing','daily',2,('ema_21',))]
    for symbol in ('ARM','INTC','NBIS','ALAB','COHR','MU'):
        d=prepare_daily_bars(pd.read_parquet(DAILY/f'{symbol}.parquet'))
        d.to_parquet(daily_dir/f'{symbol}.parquet',index=False)
        ss=[]
        for day in ('2026-10-06','2026-10-07','2026-10-08'):
            p=args.bars_dir/f'{day}.parquet';raw=pd.read_parquet(p)
            bars,coverage=completed_candles(raw,symbol,day)
            features=session_daily_features(d,day,bars,str(p))
            ss.append((day,bars,features))
            cases.append({'symbol':symbol,'session_date_et':day,'daily_feature_status':features['daily_feature_status'],
                          'image_message_available_at_utc':None,'coverage':coverage})
        for policy in policies:
            es,_,_=replay_symbol(symbol,ss,policy,[])
            events+=es
    write_jsonl(args.out/'algorithm_events.jsonl',events)
    write_jsonl(args.out/'case_coverage.jsonl',cases)
    result={'schema':'ace_screenshot_level_family_cases_v1','research_only':True,'event_timestamp_labels_missing':True,
            'source_daily_sha256':{p.name:digest(p) for p in daily_dir.glob('*.parquet')},
            'source_minute_sha256':{str(args.bars_dir/f'{d}.parquet'):digest(args.bars_dir/f'{d}.parquet') for d in ('2026-10-06','2026-10-07','2026-10-08')},
            'script_sha256':digest(Path(__file__)),'algorithm_event_count':len(events)}
    (args.out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Screenshot-specific SMA5 and EMA21 case replay','',
           'Actual October 6–8 SIP paths, causal daily-level inputs and the same two-session management policy. The screenshot notification timestamps remain unscored. Each timestamp below is a generated signal, not a reported ACE fill.','',
           '| Symbol | Session | Policy | First long model entry (ET) |','| --- | --- | --- | --- |']
    for case in cases:
        for policy in policies:
            es=[e for e in events if e['symbol']==case['symbol'] and e['policy']==policy.name and e['kind']=='entry'
                and e['direction_sign']==1 and utc(e['decision_at_utc']).tz_convert('America/New_York').date().isoformat()==case['session_date_et']]
            stamp=utc(es[0]['decision_at_utc']).tz_convert('America/New_York').strftime('%H:%M') if es else 'no generated entry'
            lines.append(f"| {case['symbol']} | {case['session_date_et']} | {policy.name} | {stamp} |")
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(args.out),'algorithm_events':len(events)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
