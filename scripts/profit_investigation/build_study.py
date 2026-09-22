"""Broker-evidenced realized outcomes and causal entry-context joins.

Research only. Weighted-average inventory cost within each flat-to-flat symbol
lifecycle. Option exercises transfer premium basis into delivered shares.
No option-bar marks, fabricated settlements, model fitting, or trading actions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.execution_quality.stage1_trade_spine import occ_meta, underlying_of

DATA = ROOT / 'research/profit_investigation_2026-09-19/data'
LOCAL = DATA / 'local'
END = pd.Timestamp('2026-09-19T04:00:00Z')
START_MONTH = pd.Timestamp('2026-08-19T04:00:00Z')
FOUR = {'momentum_expansion', 'multi_ticker_swing_htf', 'meta_ranker', 'dealer_ranker'}

def ts(x):
    return pd.to_datetime(x, utc=True) if x else None

def num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def read(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]

def available(bar, module):
    b = ts(bar)
    if b is None:
        return None
    # Recorded period is July-September, all EDT; production 4H labels are UTC.
    if module != 'dealer_ranker' and (b.hour, b.minute) in [(14, 0), (18, 0)]:
        return b.replace(hour=18 if b.hour == 14 else 20)
    return b

def scalar_fields(d, prefix='x_'):
    return {prefix+k: v for k,v in (d or {}).items() if isinstance(v, (int,float,bool))}

def quote_fields(q, prefix='entry_'):
    q = q or {}
    return {prefix+'mid': num(q.get('mid')), prefix+'spread': num(q.get('spread_pct_mid')),
            prefix+'quote_ts': q.get('quote_timestamp')}

def indexes(orders):
    owners = defaultdict(set)
    ledger = []
    plans = defaultdict(list)
    signals = {}
    entry_context = {}
    exit_context = {}
    swing_positions = defaultdict(list)
    for p in LOCAL.glob('*__closed_trades.jsonl'):
        mod = p.name.split('__')[0]
        for r in read(p):
            r['_module'] = mod
            ledger.append(r)
            for key in ['order_id','entry_order_id']:
                if r.get(key) in orders:
                    owners[r[key]].add(mod)
    for p in LOCAL.glob('*__live_signal_audit.jsonl'):
        mod = p.name.split('__')[0]
        if mod not in FOUR:
            continue
        for line,r in enumerate(read(p),1):
            a = available(r.get('bar'),mod)
            if a is None or a >= END or r.get('event') not in ['order_plan','signal_decision']:
                continue
            for ticker,sa in (r.get('signal_audits') or {}).items():
                key=(mod,ticker,str(ts(r['bar'])))
                # Prefer order_plan over the duplicated signal_decision; both
                # refer to the same decision bar, never a later bar.
                sr={'module':mod,'ticker':ticker,'signal_bar':str(ts(r['bar'])),
                    'available_at':str(a),'score':num(sa.get('score')),'rank':num(sa.get('rank')),
                    'rank_pct':num(sa.get('rank_pct')),'submit_enabled':bool(r.get('submit')),
                    'source':p.name,'source_line':line, **scalar_fields(sa.get('extra'))}
                if key not in signals or r.get('event')=='order_plan':
                    signals[key]=sr
            for plan in (r.get('planned') or r.get('plan') or []):
                if plan.get('side')!='buy' or not str(plan.get('reason','')).startswith('entry'):
                    continue
                symbol=plan['symbol'];ticker=underlying_of(symbol)
                sa=(r.get('signal_audits') or {}).get(ticker,{})
                oa=(r.get('order_audits') or {}).get(symbol,{})
                fields={'module':mod,'signal_bar':str(ts(r['bar'])),'available_at':str(a),
                    'score':num(sa.get('score')),'rank':num(sa.get('rank')),
                    'rank_pct':num(sa.get('rank_pct')), **scalar_fields(sa.get('extra')),
                    'entry_mid':num(oa.get('mid_price')),'entry_reference_price':num(oa.get('reference_price')),
                    'entry_underlying':num(oa.get('underlying_price')),
                    'entry_delta':num(oa.get('delta')),'breakeven_move_pct':num(oa.get('breakeven_move_pct')),
                    'source':p.name,'source_line':line,'join_type':'symbol_causal_plan',
                    'submit_enabled':bool(r.get('submit'))}
                plans[symbol].append(fields)
    for wrapped in read(LOCAL/'swing_events.jsonl'):
        r=wrapped['record'];p=r.get('payload') or {};kind=r.get('type')
        if kind=='order_submitted':
            ver=p.get('verification') or {};resp=p.get('response') or {}
            oid=(ver.get('order') or {}).get('id') or ver.get('order_id') or resp.get('id')
            if oid in orders:
                owners[oid].add('multi_ticker_swing')
                if p.get('side')=='sell':
                    exit_context[oid]={**quote_fields(p.get('close_quote'),'exit_'),
                                      'exit_underlying':num(p.get('exit_price'))}
        elif kind=='position_opened':
            if p.get('option_symbol') and p.get('entry_time'):
                swing_positions[p['option_symbol']].append((ts(p['entry_time']),r,p))
    for wrapped in read(LOCAL/'spy_events.jsonl'):
        r=wrapped['record'];p=r.get('payload') or {};result=p.get('result') or {};state=p.get('policy_state') or {}
        for wrapper in result.get('orders') or []:
            resp=wrapper.get('response') or {};ver=resp.get('verification') or {};raw=resp.get('response') or {}
            oid=(ver.get('order') or {}).get('id') or ver.get('order_id') or raw.get('id')
            if oid not in orders or resp.get('simulated'):
                continue
            owners[oid].add('spy_daytrader')
            if orders[oid]['side']=='buy':
                side=wrapper.get('side_key') or ('short' if occ_meta(orders[oid]['symbol']).get('right')=='P' else 'long')
                sig=state.get(side+'_entry_signal_time');diagnostic=(state.get('missed_trade_diagnostics') or {}).get(side,{})
                entry_context[oid]={'module':'spy_daytrader','score':num(diagnostic.get('prob')),
                    'signal_bar':sig,'available_at':str(ts(sig)+pd.Timedelta(minutes=10)) if sig else None,
                    'entry_underlying':num(result.get('close')),'entry_mid':num(wrapper.get('contract_price')),
                    'x_setup_peak_prob':num(state.get(side+'_intrabar_setup_peak_prob')),
                    'join_type':'exact_order_postfill_snapshot','source':wrapped['source'],'source_line':wrapped['line']}
    for oid,o in orders.items():
        if o['side']!='buy':
            continue
        submit=ts(o.get('submitted_at') or o['created_at'])
        near=[p for p in plans.get(o['symbol'],[]) if p['submit_enabled'] and
              ts(p['available_at'])<=submit and submit-ts(p['available_at'])<=pd.Timedelta(days=4)]
        if owners[oid]:
            near=[p for p in near if p['module'] in owners[oid]]
        mods={p['module'] for p in near}
        if len(mods)==1:
            best=max(near,key=lambda x:ts(x['available_at']))
            owners[oid].add(best['module'])
            entry_context.setdefault(oid,best)
        # Time match is deliberately tight; restored position records carry
        # entry metadata but are not treated as newly observed predictions.
        if owners[oid]=={'multi_ticker_swing'}:
            near=[x for x in swing_positions[o['symbol']] if abs((x[0]-ts(o.get('filled_at') or submit)).total_seconds())<120]
            if near:
                _,record,p=min(near,key=lambda x:ts(x[1]['ts']))
                meta=p.get('option_entry_meta') or {}; policy=meta.get('signal_policy') or {}
                inputs=policy.get('inputs') or {}
                entry_context[oid]={'module':'multi_ticker_swing','score':num(inputs.get('p_dir')),
                    'entry_underlying':num(p.get('entry_price')),'entry_delta':num(meta.get('selected_abs_delta')),
                    'moneyness_pct':num(meta.get('moneyness_pct')),**quote_fields(meta.get('entry_quote')),
                    **scalar_fields(inputs),**scalar_fields(meta.get('confirmation_metrics'),'confirm_'),
                    'policy_action':policy.get('action'),'policy_reason':policy.get('reason'),
                    'join_type':'entry_metadata_symbol_time','source':'swing_events.jsonl',
                    'restored_context':bool(p.get('restored_from_broker'))}
    # Exact entry IDs in intraday ledger establish ownership even if another
    # engine later adopted and sold the same position.
    return owners,entry_context,exit_context,ledger,list(signals.values())

def reconstruct(activities,orders,owners,entry_context,exit_context):
    active={};cycles=[];exits=[];unmatched=[];transfers=[]
    grouped=defaultdict(dict)
    for a in activities:
        if a.get('group_id'):
            grouped[a['group_id']][a['activity_type']]=a
    events=[]
    for a in activities:
        kind=a['activity_type']
        if kind=='FILL':events.append((ts(a['transaction_time']),0,a))
        elif kind in ['OPEXP','OPEXC']:
            # Event session date for P&L attribution; created_at for inventory
            # chronology (settlements are posted later that evening).
            events.append((ts(a.get('created_at') or a['date']),1,a))
    def buy(sym,q,px,t,oid,extra=None):
        mult=100 if occ_meta(sym)['is_option'] else 1
        if sym not in active:
            c={'cycle_id':f'{sym}:{t.isoformat()}','symbol':sym,'ticker':underlying_of(sym),
                'route':'option' if mult==100 else 'equity','entry_time':str(t),'entry_order_id':oid,
                'qty':0.,'basis':0.,'total_cost':0.,'realized_pnl':0.,'sold_cost':0.,
                'entries':[],'exits':[],'owners':set(),'closed':False,'exercise_transfer':False,
                **{f'opt_{k}':v for k,v in occ_meta(sym).items() if k!='is_option'}}
            active[sym]=c;cycles.append(c)
        c=active[sym];c['qty']+=q;c['basis']+=q*px*mult;c['total_cost']+=q*px*mult
        c['entries'].append(oid)
        c['owners'].update(owners.get(oid,set()))
        if extra:
            c['exercise_transfer']=True;c['owners'].update(extra['owners'])
        return c
    def sell(sym,q,px,t,oid,kind,session=None):
        if sym not in active or active[sym]['qty']+1e-6<q:
            unmatched.append({'symbol':sym,'qty':q,'ts':str(t),'id':oid,'kind':kind})
            raise ValueError(f'Unfunded sell/settlement {sym} {q}')
        c=active[sym];mult=100 if c['route']=='option' else 1
        basis=c['basis']*q/c['qty'];proceeds=q*px*mult
        c['qty']-=q;c['basis']-=basis;c['realized_pnl']+=proceeds-basis;c['sold_cost']+=basis
        c['exits'].append(oid)
        er={'cycle_id':c['cycle_id'],'symbol':sym,'ticker':c['ticker'],'route':c['route'],
            'exit_id':oid,'entry_time':c['entry_time'],'exit_time':str(t),
            'session':session or str(t.tz_convert('America/New_York').date()),'qty':q,
            'cost':basis,'proceeds':proceeds,'pnl':proceeds-basis,'return':proceeds/basis-1 if basis else None,
            'kind':kind,**exit_context.get(oid,{})}
        exits.append(er)
        if c['qty']<1e-6:
            c['closed']=True;c['exit_time']=str(t);c['exit_session']=er['session'];del active[sym]
        return c,basis
    for t,_,a in sorted(events,key=lambda x:(x[0],x[1])):
        kind=a['activity_type'];sym=a['symbol'];q=abs(float(a['qty']))
        if kind=='FILL':
            oid=a['order_id'];px=float(a['price'])
            if a['side']=='buy':buy(sym,q,px,t,oid)
            else:sell(sym,q,px,t,oid,'FILL')
        elif kind=='OPEXP':
            sell(sym,q,0,t,a['id'],'OPEXP',a['date'])
        else:
            stock=grouped[a['group_id']].get('OPTRD')
            assert stock and float(stock['qty'])>0 and occ_meta(sym)['right']=='C'
            c=active[sym];basis=c['basis']*q/c['qty']
            source,b=sell(sym,q,basis/q/100,t,a['id'],'BASIS_TRANSFER',a['date'])
            source['exercise_transfer']=True
            qty=float(stock['qty']);assert abs(qty-q*100)<1e-6
            buy(stock['symbol'],qty,float(stock['price'])+b/qty,t,stock['id'],source)
            transfers.append({'option':sym,'equity':stock['symbol'],'premium_basis':b,'qty':qty,'date':a['date']})
    for c in cycles:
        entry_ids=list(dict.fromkeys(c['entries']));exit_ids=list(dict.fromkeys(c['exits']))
        # Entry provenance wins over an exit by an adopting module. If no
        # entry is attributable, a unique exact exit owner is a weaker fallback.
        if not c['owners']:
            mods=set().union(*(owners.get(i,set()) for i in exit_ids))
            c['owners']=mods;c['ownership_source']='exit_order_fallback'
        else:c['ownership_source']='entry_order'
        c['module']=next(iter(c['owners'])) if len(c['owners'])==1 else ('mixed' if c['owners'] else 'unattributed')
        if entry_context.get(c['entry_order_id'],{}).get('join_type')=='symbol_causal_plan':
            c['ownership_source']='causal_plan'
            exitmods=set().union(*(owners.get(i,set()) for i in exit_ids))
            if exitmods and not exitmods.issubset(c['owners']):
                c['owners'].update(exitmods);c['module']='mixed'
        c['owners']=sorted(c['owners']);c['entries']=entry_ids;c['exits']=exit_ids
        c['entry_orders']=len(entry_ids);c['exit_orders']=len(exit_ids)
        ctx=entry_context.get(c['entry_order_id'],{})
        c.update({k:v for k,v in ctx.items() if k!='module'})
        if c['module']=='mixed':
            c['score']=None
        c['return']=c['realized_pnl']/c['sold_cost'] if c['sold_cost'] else None
        c['entry_date']=str(ts(c['entry_time']).tz_convert('America/New_York').date())
        c['entry_hour_et']=ts(c['entry_time']).tz_convert('America/New_York').hour
        c['holding_days']=(ts(c['exit_time'])-ts(c['entry_time'])).total_seconds()/86400 if c.get('exit_time') else None
        c['entry_premium']=float(orders[c['entry_order_id']]['filled_avg_price']) if c['entry_order_id'] in orders else None
        if c.get('opt_expiry'):
            c['dte']=(pd.Timestamp(c['opt_expiry'])-pd.Timestamp(c['entry_date'])).days
        o=orders.get(c['entry_order_id'],{})
        submit=ts(o.get('submitted_at'));fill=ts(c['entry_time'])
        c['submit_time']=str(submit) if submit is not None else None
        c['submit_to_first_fill_s']=(fill-submit).total_seconds() if submit is not None else None
        if c.get('available_at') and submit is not None:
            c['signal_to_submit_min']=(submit-ts(c['available_at'])).total_seconds()/60
            c['signal_to_fill_min']=(fill-ts(c['available_at'])).total_seconds()/60
        if c.get('entry_mid') and c.get('entry_premium'):
            c['entry_vs_mid']=c['entry_premium']/c['entry_mid']-1
        if c.get('entry_reference_price') and c.get('entry_premium') and c['route']=='equity':
            c['entry_vs_reference']=c['entry_premium']/c['entry_reference_price']-1
    bycycle={c['cycle_id']:c for c in cycles}
    for e in exits:e['module']=bycycle[e['cycle_id']]['module']
    return cycles,exits,transfers,active

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='run1');args=ap.parse_args()
    out=DATA.parent/args.out;out.mkdir(exist_ok=False)
    orders={o['id']:o for o in read(DATA/'orders.jsonl')};activities=read(DATA/'activities.jsonl')
    owners,context,xc,ledger,signals=indexes(orders)
    cycles,exits,transfers,active=reconstruct(activities,orders,owners,context,xc)
    # Deferred entries can have an empty executable plan. An exact filled-exit
    # ledger's entry_bar is an alternative causal key, not a nearest-bar guess.
    sigidx={(r['module'],r['ticker'],r['signal_bar']):r for r in signals}
    for c in cycles:
        if c.get('score') is not None:
            continue
        hits=[r for r in ledger if r['_module']==c['module'] and r.get('order_id') in c['exits'] and r.get('entry_bar')]
        bars={str(ts(r['entry_bar'])) for r in hits}
        if len(bars)==1:
            sr=sigidx.get((c['module'],c['ticker'],bars.pop()))
            if sr and ts(sr['available_at'])<=ts(c['entry_time']):
                c.update({k:v for k,v in sr.items() if k not in ['module','ticker']})
                c['join_type']='exact_exit_ledger_entry_bar'
                if c.get('submit_time'):
                    c['signal_to_submit_min']=(ts(c['submit_time'])-ts(sr['available_at'])).total_seconds()/60
                    c['signal_to_fill_min']=(ts(c['entry_time'])-ts(sr['available_at'])).total_seconds()/60
    for name,rows in [('lifecycles',cycles),('realized_exits',exits),('signals',signals),('basis_transfers',transfers)]:
        with (out/f'{name}.jsonl').open('x') as f:
            for r in rows:f.write(json.dumps(r,default=str)+'\n')
    df=pd.DataFrame(cycles);ef=pd.DataFrame(exits)
    df.drop(columns=['owners','entries','exits']).to_csv(out/'lifecycles.csv',index=False)
    ef.to_csv(out/'realized_exits.csv',index=False)
    # Compare quantity and notional from every execution to the order VWAP.
    fillq=defaultdict(float);filln=defaultdict(float)
    for a in activities:
        if a['activity_type']=='FILL':
            fillq[a['order_id']]+=float(a['qty']);filln[a['order_id']]+=float(a['qty'])*float(a['price'])
    qdiff=[i for i,q in fillq.items() if i not in orders or abs(q-float(orders[i]['filled_qty']))>1e-6]
    ndiff=[i for i,n in filln.items() if abs(n-float(orders[i]['filled_qty'])*float(orders[i]['filled_avg_price']))>.05]
    assert not qdiff and not ndiff
    # The frozen 20:05 ET snapshot is historical; capture-time positions can
    # differ after overnight expiry processing and are not the P&L endpoint.
    snap=read(LOCAL/'broker_equity_20260918_paper.jsonl')[-1]
    pos=snap['positions'];pos=list(pos.values()) if isinstance(pos,dict) else pos
    brokerqty={p['symbol']:float(p['qty']) for p in pos}
    qtymismatch=[{'symbol':s,'reconstructed':active.get(s,{}).get('qty',0),'snapshot':brokerqty.get(s,0)}
                 for s in set(active)|set(brokerqty) if abs(active.get(s,{}).get('qty',0)-brokerqty.get(s,0))>1e-6]
    fills_cash=sum((1 if a['side']=='sell' else -1)*float(a['qty'])*float(a['price'])*(100 if occ_meta(a['symbol'])['is_option'] else 1)
                   for a in activities if a['activity_type']=='FILL')
    othercash=sum(float(a.get('net_amount') or 0) for a in activities if a['activity_type']!='FILL')
    cash=fills_cash+othercash
    realized=ef.loc[ef.kind!='BASIS_TRANSFER','pnl'].sum();fees=sum(float(a.get('net_amount') or 0) for a in activities if a['activity_type']=='FEE')
    # Ledger rows are audited, never used as the source of P&L.
    lcheck=[]
    for r in ledger:
        oid=r.get('order_id');o=orders.get(oid)
        status='missing_order_id' if not oid else ('order_absent' if not o else ('not_filled' if float(o['filled_qty'])<=0 else 'filled'))
        lcheck.append({'module':r['_module'],'symbol':r.get('order_symbol'),'ts':r.get('ts'),
                       'ledger_pnl':r.get('realized_pnl'),'order_id':oid,'status':status})
    pd.DataFrame(lcheck).to_csv(out/'ledger_audit.csv',index=False)
    checks={'orders':len(orders),'activity_counts':dict(Counter(a['activity_type'] for a in activities)),
        'quantity_reconciliation_failures':qdiff,'notional_reconciliation_failures':ndiff,
        'earliest_fill':min(a['transaction_time'] for a in activities if a['activity_type']=='FILL'),
        'latest_fill':max(a['transaction_time'] for a in activities if a['activity_type']=='FILL'),
        'lifecycles':len(df),'closed':int(df.closed.sum()),'module_counts':df.module.value_counts().to_dict(),
        'open_quantity_mismatches':qtymismatch,'calculated_cash':cash,
        'snapshot_cash':float(snap['account']['cash']),'cash_difference':cash-float(snap['account']['cash']),
        'realized_before_fees':realized,'account_fees':fees,'open_cost_basis':sum(c['basis'] for c in active.values()),
        'broker_equity':float(snap['account']['equity']),
        'ledger_status':dict(Counter(r['status'] for r in lcheck)),
        'score_coverage_closed':int(df.loc[df.closed,'score'].notna().sum())}
    (out/'checks.json').write_text(json.dumps(checks,indent=2,default=str))
    print(json.dumps(checks,indent=2,default=str))

if __name__=='__main__':main()
