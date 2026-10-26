from __future__ import annotations

import pandas as pd

from scripts.discord_ledger.run_ace_sequence_replay import Policy, completed_candles, one_to_one_matches, replay_symbol, lifecycle_metrics, published_plans, shifted_null


def test_matching_never_reuses_one_signal_or_matches_wrong_direction():
    labels=[{'label_id':name,'kind':'entry','symbol':'SPY','direction_sign':sign,
             'decision_at_utc':time,'session_date_et':'2026-01-05','source_message_ids':[name]}
            for name,sign,time in [('a',1,'2026-01-05T15:00:00Z'),('b',1,'2026-01-05T15:01:00Z'),('c',-1,'2026-01-05T15:00:00Z')]]
    signals=[{'kind':'entry','symbol':'SPY','direction_sign':1,'decision_at_utc':'2026-01-05T15:00:00Z','position_id':'p'}]
    result=one_to_one_matches(labels,signals,10)
    assert len(result)==1
    assert result[0]['label_id']=='a'
    assert result[0]['position_id']=='p'


def test_five_minute_candle_is_not_constructed_across_a_missing_minute():
    frame=pd.DataFrame({'symbol':'SPY','timestamp':pd.date_range('2026-01-05T14:30Z',periods=10,freq='min'),
                        'open':100.,'high':101.,'low':99.,'close':100.,'volume':10.,'vwap':100.})
    frame=frame.drop(index=2)
    bars,coverage=completed_candles(frame,'SPY','2026-01-05')
    assert len(bars)==1
    assert bars.iloc[0].decision_at_utc==pd.Timestamp('2026-01-05T14:40Z')
    assert coverage['gaps_5m']==77


def _sessions():
    records=[(102,102,99.95,100.5),(102,104,102,103),(103,104,101,103.5),
             (104,111,102.5,110),(110,111,102,103)]
    rows=[{'timestamp':pd.Timestamp('2026-01-05T14:30Z')+pd.Timedelta(minutes=5*i),
           'decision_at_utc':pd.Timestamp('2026-01-05T14:35Z')+pd.Timedelta(minutes=5*i),
           'open':o,'high':h,'low':l,'close':c,'volume':10.,'session_vwap':100.}
          for i,(o,h,l,c) in enumerate(records)]
    daily={'daily_feature_status':'ok','atr_14':5.,'sma_5':100.,'ema_21':90.,'sma_34':80.,'sma_55':70.,
           'prior_close':101.,'prior_high_20':110.,'prior_low_20':60.,'source_session_file':'test_fixture'}
    return [('2026-01-05',pd.DataFrame(rows),daily)]


def test_touch_is_retained_and_stop_adjustment_is_next_candle_effective():
    events,trades,traces=replay_symbol('SPY',_sessions(),Policy('test','daily'),[])
    entries=[e for e in events if e['kind']=='entry']
    trims=[e for e in events if e['kind']=='trim']
    exits=[e for e in events if e['kind']=='exit']
    assert entries[0]['decision_at_utc']==pd.Timestamp('2026-01-05T14:40Z').isoformat()
    assert trims[0]['decision_at_utc']==pd.Timestamp('2026-01-05T14:50Z').isoformat()
    assert exits[0]['decision_at_utc']==pd.Timestamp('2026-01-05T14:55Z').isoformat()
    assert entries[0]['position_id']==trims[0]['position_id']==exits[0]['position_id']
    assert trades[0]['censored'] is False


def test_future_candles_cannot_change_prior_entry_signals():
    short=_sessions()
    short[0]=(short[0][0],short[0][1].iloc[:3].copy(),short[0][2])
    early_events,_,_=replay_symbol('SPY',short,Policy('test','daily'),[])
    full_events,_,_=replay_symbol('SPY',_sessions(),Policy('test','daily'),[])
    assert [e for e in full_events if e['decision_at_utc']<=pd.Timestamp('2026-01-05T14:45Z').isoformat()]==early_events


def test_unavailable_next_session_censors_swing_instead_of_inventing_exit():
    ss=_sessions();ss[0]=(ss[0][0],ss[0][1].iloc[:3].copy(),ss[0][2])
    events,trades,_=replay_symbol('SPY',ss,Policy('test','daily',2),[])
    assert not any(e['kind']=='exit' for e in events)
    assert trades[0]['censored'] is True
    assert trades[0]['gross_underlying_return_pct'] is None


def test_exit_overlap_without_matching_entry_is_not_a_lifecycle_match():
    entry={'label_id':'entry:m:SPY:1','kind':'entry','symbol':'SPY','direction_sign':1,
           'decision_at_utc':'2026-01-05T15:00:00Z','session_date_et':'2026-01-05','source_message_ids':['m']}
    close={**entry,'label_id':'exit:x','kind':'exit','decision_at_utc':'2026-01-05T16:00:00Z',
           'source_message_ids':['x'],'entry_label_ids':[entry['label_id']],'link_confidence':'high'}
    signals=[{'kind':'exit','symbol':'SPY','direction_sign':1,'decision_at_utc':close['decision_at_utc'],'position_id':'p'}]
    result=lifecycle_metrics([entry],[close],signals,2)
    assert result['exit:high_medium']['entry_and_management_matched']==0


def test_option_entry_premium_cannot_become_an_underlying_zone(tmp_path,monkeypatch):
    import json
    import scripts.discord_ledger.run_ace_sequence_replay as module
    p=tmp_path/'messages.jsonl'
    rows=[{'author_name':'ACE','message_id':'option','text':'ORCL 195c 1/16\nEntry: 1.55',
           'timestamp_utc':'2026-01-15T15:00:00Z','is_edited':False},
          {'author_name':'ACE','message_id':'stock','text':'IREN\nEntry Zone: 39.50–42.50',
           'timestamp_utc':'2026-01-15T15:00:00Z','is_edited':False}]
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    monkeypatch.setattr(module,'MESSAGES',p)
    plans,excluded=published_plans()
    assert [p['symbol'] for p in plans]==['IREN']
    assert excluded[0]['status']=='option_premium_is_not_underlying_entry_zone'


def test_no_signal_policy_has_no_estimated_zero_mde():
    row={'label_id':'a','kind':'entry','symbol':'SPY','direction_sign':1,
         'decision_at_utc':'2026-01-05T15:00:00Z','session_date_et':'2026-01-05','source_message_ids':['a']}
    result=shifted_null([row],[])
    assert result['session_mde_pp'] is None
    assert result['session_bootstrap_95pct_ci_pp'] is None
