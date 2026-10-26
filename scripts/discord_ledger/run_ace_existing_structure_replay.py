"""Replay the existing Intraday Structure engine independently of ACE alert clocks.

Only the frozen primary test symbol/session panel is given to the engine.
Both directions are admitted at the session open with the existing research
placeholder score 0.8. Price detectors, target logic and exit logic are the
current module's. Broker execution and unavailable dealer context are disabled.
Transition clocks are bar-start timestamps; score them at bar end (+1 minute).
Terminal archival records are not used as exit timing labels: use each live
path's first terminal transition after RUNNING.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from scripts.discord_ledger.build import digest, read_jsonl, write_jsonl
from scripts.discord_ledger.run_ace_sequence_replay import entry_metrics, lifecycle_metrics, shifted_null, session_bounds, utc
from strategies.intraday_structure.config import load_config
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction


def run_session(job: tuple[str,str,str]) -> tuple[list[dict],list[dict],dict]:
    symbol,day,path=job
    start,end=session_bounds(day)
    frame=pd.read_parquet(path)
    frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True)
    frame=frame[frame.symbol.isin({symbol,'SPY','QQQ'})&(frame.timestamp>=start)&(frame.timestamp<end)]
    if frame.duplicated(['symbol','timestamp']).any():raise ValueError('duplicate minute input')
    frame=frame.sort_values(['timestamp','symbol'],kind='stable')
    config=load_config()
    config=replace(config,execution=replace(config.execution,enabled=False),dealer_plate=replace(config.dealer_plate,enabled=False))
    engine=IntradayStructureEngine(config)
    active={};events=[];transitions=[];seq=0;seen=0
    for r in frame.itertuples(index=False):
        t=utc(r.timestamp)
        # Deterministic refresh schedule, independent of ACE's alert clock.
        if r.symbol==symbol and int((t-start).total_seconds()/60)%15==0:
            for direction in (Direction.LONG,Direction.SHORT):
                engine.register_candidate(Candidate(symbol,t.to_pydatetime(),direction,('ace_research_session_panel',),
                    .8,available_at=t.to_pydatetime()),persist=False)
        engine.on_bar(Bar.from_mapping(r._asdict()))
        for tr in engine.transitions[seen:]:
            row=tr.to_dict();row['decision_at_utc']=(utc(row['timestamp'])+pd.Timedelta(minutes=1)).isoformat()
            transitions.append(row)
            sid=row['setup_id'];sign=1 if ':long:' in sid else -1
            if row['to_state']=='RUNNING' and row['from_state']=='CONFIRMED':
                seq+=1;pid=f'existing_structure:{symbol}:{day}:{seq}';active[sid]=pid
                events.append({'kind':'entry','decision_at_utc':row['decision_at_utc'],'symbol':symbol,
                    'direction_sign':sign,'position_id':pid,'policy':'existing_structure_price_only','reason':row['reason'],'setup_type':row['setup_type']})
            elif sid in active and row['to_state']=='TARGET_REACHED':
                events.append({'kind':'trim','decision_at_utc':row['decision_at_utc'],'symbol':symbol,
                    'direction_sign':sign,'position_id':active[sid],'policy':'existing_structure_price_only','reason':row['reason'],'setup_type':row['setup_type']})
            elif sid in active and row['to_state'] in {'INVALIDATED','EXHAUSTED','CLOSED'}:
                events.append({'kind':'exit','decision_at_utc':row['decision_at_utc'],'symbol':symbol,
                    'direction_sign':sign,'position_id':active.pop(sid),'policy':'existing_structure_price_only','reason':row['reason'],'setup_type':row['setup_type']})
        seen=len(engine.transitions)
    return events,transitions,{'symbol':symbol,'session_date_et':day,'input_minute_rows':len(frame),
        'model_positions_unclosed_at_end':len(active),'candidate_refresh_minutes':15,
        'index_context_symbols_observed':sorted(set(frame.symbol)&{'SPY','QQQ'})}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-replay',type=Path,required=True)
    parser.add_argument('--bars-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    source=json.loads((args.source_replay/'results.json').read_text())
    dates=source['chronological_split_dates']['test']
    entries=[r for r in read_jsonl(args.source_replay/'entry_labels.jsonl') if r['primary_cohort'] and r['session_date_et'] in dates]
    management=[r for r in read_jsonl(args.source_replay/'management_labels.jsonl') if r['primary_cohort'] and r['session_date_et'] in dates]
    panels=sorted({(r['symbol'],r['session_date_et']) for r in entries})
    jobs=[(s,d,str(args.bars_dir/f'{d}.parquet')) for s,d in panels]
    events,transitions,coverage=[],[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for es,ts,cov in pool.map(run_session,jobs):
            events+=es;transitions+=ts;coverage.append(cov)
            print(json.dumps({'completed_panel':cov['symbol']+' '+cov['session_date_et'],'entries':sum(e['kind']=='entry' for e in es)}),flush=True)
    config=load_config();config=replace(config,execution=replace(config.execution,enabled=False),dealer_plate=replace(config.dealer_plate,enabled=False))
    grids={}
    for s,d in panels:
        f=pd.read_parquet(args.bars_dir/f'{d}.parquet');start,end=session_bounds(d)
        f=f[(f.symbol==s)&(f.timestamp>=start)&(f.timestamp<end)]
        grids[(s,d)]=[(utc(t)+pd.Timedelta(minutes=1)).isoformat() for t in sorted(f.timestamp)]
    entry_events=[e for e in events if e['kind']=='entry']
    result={'schema':'ace_existing_structure_replay_v1','research_only':True,'current_module':'strategies.intraday_structure',
        'config_snapshot':asdict(config),'data_coverage':coverage,'source_entries':len(entries),
        'entries':{str(k):entry_metrics(entries,entry_events,k) for k in (1,2,3)},
        'management':lifecycle_metrics(entries,management,events,2),'shifted_null':shifted_null(entries,entry_events,grids),
        'transition_counts':dict(Counter(t['to_state'] for t in transitions)),
        'candidate_source':'observed symbol/session panel, both directions refreshed every 15min, placeholder research score 0.8',
        'decision_time_rule':'recorded one-minute bar start plus one minute; first terminal transition after modeled RUNNING is the exit',
        'limitations':['No candidate selection/sector/theme reconstruction; available SPY/QQQ context only.',
                      'Terminal ledger archival timestamps are not used as first-exit decision times.',
                      'Multiple detector setups are opportunity paths; no portfolio/order/net-return claim.'],
        'inputs_sha256':{str(args.source_replay/'results.json'):digest(args.source_replay/'results.json'),
            **{str(args.bars_dir/f'{d}.parquet'):digest(args.bars_dir/f'{d}.parquet') for _,d in panels}}}
    args.out.mkdir(parents=True,exist_ok=False)
    write_jsonl(args.out/'algorithm_events.jsonl',events);write_jsonl(args.out/'transitions.jsonl',transitions)
    (args.out/'results.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    (args.out/'report.md').write_text('# Existing Intraday Structure: ACE timing benchmark\n\n'+
        'The current price detectors and target/exit state machine were scanned independently of ACE alert times. Candidates in both directions refresh every 15 minutes on the observed test symbol/session panel. Execution is disabled.\n\n'+
        '| Tolerance | Source entries | Algorithm entries | Entry matches | Recall |\n| --- | ---: | ---: | ---: | ---: |\n'+
        '\n'.join(f"| ±{k} five-minute candles | {m['source_entries']} | {m['algorithm_entries']} | {m['matched_entries']} | {100*m['entry_recall']:.1f}% |" for k,m in result['entries'].items())+
        '\n\nEntry and management confluence, shifted baseline, source coverage and complete config are in results.json. Bar-close availability is used for matching; archival CLOSED transitions cannot overwrite an earlier invalidation/exhaustion time. Historical comparison only.\n')
    print(json.dumps({'out':str(args.out),'entries':result['entries'],'management':result['management'],'null':result['shifted_null']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
