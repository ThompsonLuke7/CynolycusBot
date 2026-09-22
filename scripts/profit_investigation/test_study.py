"""Checks for accounting and temporal errors that would invalidate the study."""
import pytest
import pandas as pd
from scripts.profit_investigation.build_study import reconstruct, available
from scripts.profit_investigation.enrich import observed_price, prior_features

def fill(oid,side,qty,price,symbol='ABC',time='2026-08-03T15:00:00Z'):
    return {'activity_type':'FILL','transaction_time':time,'order_id':oid,
            'symbol':symbol,'side':side,'qty':str(qty),'price':str(price)}

def test_partial_exits_then_add_do_not_use_future_entry_basis():
    a=[fill('b','buy',10,10),fill('s1','sell',5,12,time='2026-08-03T15:01:00Z'),
       fill('b2','buy',5,20,time='2026-08-03T15:02:00Z'),fill('s2','sell',10,16,time='2026-08-03T15:03:00Z')]
    c,e,_,active=reconstruct(a,{}, {},{}, {})
    assert [x['pnl'] for x in e]==[10,10]
    assert c[0]['realized_pnl']==20 and not active and c[0]['closed']

def test_exercise_transfers_premium_basis_instead_of_booking_false_loss():
    sym='ABC260807C00010000'
    a=[fill('b','buy',2,1,sym),
       {'activity_type':'OPEXC','symbol':sym,'qty':'-2','date':'2026-08-07','created_at':'2026-08-08T01:00:00Z','id':'exc','group_id':'g'},
       {'activity_type':'OPTRD','symbol':'ABC','qty':'200','price':'10','date':'2026-08-07','id':'stock','group_id':'g'},
       fill('s','sell',200,12,'ABC','2026-08-10T15:00:00Z')]
    c,e,t,active=reconstruct(a,{}, {},{}, {})
    assert sum(x['pnl'] for x in e)==200
    assert t[0]['premium_basis']==200 and not active

def test_expiration_requires_quantity_and_real_settlement_record():
    sym='ABC260807C00010000'
    a=[fill('b','buy',2,1,sym),{'activity_type':'OPEXP','symbol':sym,'qty':'-2',
       'date':'2026-08-07','created_at':'2026-08-08T01:00:00Z','id':'exp'}]
    c,e,_,active=reconstruct(a,{}, {},{}, {})
    assert e[0]['pnl']==-200 and not active
    with pytest.raises(ValueError):reconstruct([fill('s','sell',1,2)],{}, {},{}, {})

def test_four_hour_label_is_not_availability():
    assert available('2026-08-03T14:00:00Z','meta_ranker')==pd.Timestamp('2026-08-03T18:00:00Z')
    assert available('2026-08-03T18:00:00Z','meta_ranker')==pd.Timestamp('2026-08-03T20:00:00Z')
    assert available('2026-08-03T14:00:00Z','dealer_ranker')==pd.Timestamp('2026-08-03T14:00:00Z')

def test_minute_close_not_available_at_left_label():
    d=pd.DataFrame({'close':[10,99]},index=pd.to_datetime(['2026-08-03T15:00Z','2026-08-03T15:01Z'],utc=True))
    assert observed_price(d,pd.Timestamp('2026-08-03T15:01:30Z'))==10
    assert observed_price(d,pd.Timestamp('2026-08-03T15:06:00Z')) is None

def test_features_use_previous_session():
    f=prior_features('SPY','2026-09-18T15:00:00Z')
    assert f['feature_last_session']=='2026-09-17'
