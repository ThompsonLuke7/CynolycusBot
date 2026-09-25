"""RECALL / PRECISION / LIFT for moving-average support bounces.

The user's question, stated precisely: of all the big up-moves that happen,
how many are preceded by a touch of a rising long-term MA (RECALL)? And of all
MA touches, how many lead to a big up-move (PRECISION)?

The number that decides whether a rule is worth anything is PRECISION vs the
BASE RATE (lift). Recall alone is free: a rule that fires every day has 100%
recall and zero value.
"""
import glob, os
import numpy as np, pandas as pd

FILES=sorted(glob.glob('Data/shared/bars/1d/*.parquet'))
START,END=pd.Timestamp('2026-01-01').date(),pd.Timestamp('2026-08-20').date()
recs=[]
for p in FILES:
    t=os.path.basename(p)[:-8]
    try: df=pd.read_parquet(p,columns=['timestamp','high','low','close','volume'])
    except Exception: continue
    if len(df)<300: continue
    df=df.sort_values('timestamp').reset_index(drop=True)
    c,h,lo,v=df['close'],df['high'],df['low'],df['volume']
    if (c.tail(120)*v.tail(120)).median()<5e6: continue
    pc=c.shift(1); tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e100=c.ewm(span=100,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    s200=c.rolling(200).mean()
    rh,rl=h.rolling(20).max(),lo.rolling(20).min()
    d=pd.DataFrame({'ticker':t,'date':df['timestamp'].dt.tz_convert('UTC').dt.date,
        'atr_pct':atr/c,
        'e50_atr':(c-e50)/atr.replace(0,np.nan),
        'e100_atr':(c-e100)/atr.replace(0,np.nan),
        'e200_atr':(c-e200)/atr.replace(0,np.nan),
        's200_atr':(c-s200)/atr.replace(0,np.nan),
        'e100_slope':(e100-e100.shift(20))/atr.replace(0,np.nan),
        's200_slope':(s200-s200.shift(20))/atr.replace(0,np.nan),
        'range_pos':((c-rl)/(rh-rl).replace(0,np.nan)).clip(0,1),
        # the "bounce" outcome: did it actually go up a lot from here?
        'fwd20':c.shift(-20)/c-1.0,
        'fwd_max20':h.shift(-1).rolling(20).max().shift(-19)/c-1.0})
    recs.append(d)
U=pd.concat(recs,ignore_index=True)
U=U[(U.date>=START)&(U.date<=END)].dropna(
    subset=['e100_atr','e100_slope','range_pos','fwd20','fwd_max20','atr_pct','s200_atr','s200_slope'])
print(f"universe: {U.ticker.nunique()} tickers / {len(U):,} stock-days  {U.date.min()}..{U.date.max()}\n")

# THE EVENT we are trying to catch: a NBIS-style run
for thr in (0.15, 0.20):
    U[f'big{int(thr*100)}']=U.fwd_max20>=thr
EVENTS={'big move: +15% max gain within 20d':'big15',
        'big move: +20% max gain within 20d':'big20'}

SIGNALS={
 'touch of RISING EMA100 (|dist|<=1 ATR)':      (U.e100_slope>0)&(U.e100_atr.abs()<=1.0),
 '  + pulled back in range (range_pos<=0.35)':  (U.e100_slope>0)&(U.e100_atr.abs()<=1.0)&(U.range_pos<=0.35),
 'touch of RISING SMA200 (|dist|<=1 ATR)':      (U.s200_slope>0)&(U.s200_atr.abs()<=1.0),
 'touch of RISING EMA50  (|dist|<=1 ATR)':      (U.e100_slope>0)&(U.e50_atr.abs()<=1.0),
 'ANY of the three MA touches (union)':         ((U.e100_slope>0)&(U.e100_atr.abs()<=1.0))|
                                                ((U.s200_slope>0)&(U.s200_atr.abs()<=1.0))|
                                                ((U.e100_slope>0)&(U.e50_atr.abs()<=1.0)),
 'ALL of EMA50+EMA100 touch (intersection)':    (U.e100_slope>0)&(U.e100_atr.abs()<=1.0)&(U.e50_atr.abs()<=1.0),
 '[reference] extended >3 ATR above EMA100':    (U.e100_slope>0)&(U.e100_atr>3.0),
}
for ev_label,ev in EVENTS.items():
    base=U[ev].mean()
    print(f"=== EVENT: {ev_label}  |  BASE RATE = {base*100:.2f}% of all stock-days ===")
    print(f"{'signal':<46} {'fires':>7} {'recall':>8} {'precision':>10} {'lift':>7}")
    for name,mask in SIGNALS.items():
        s=U[mask]
        if len(s)<300: continue
        fires=len(s)/len(U)*100
        recall=(U[mask&U[ev]].shape[0]/max(U[ev].sum(),1))*100
        prec=s[ev].mean()*100
        print(f"{name:<46} {fires:>6.2f}% {recall:>7.2f}% {prec:>9.2f}% {prec/(base*100):>6.2f}x")
    print()
print("READ THIS COLUMN: 'lift'. 1.00x = the signal tells you NOTHING beyond the base rate.")
print("'fires' is the share of all stock-days the rule triggers on — recall is roughly")
print("bounded by it, so a rule that fires often gets high recall for free.")
