"""Pre-specified variant grid. FOUR tests, stated up front, no post-hoc mining.
All measured as excess over the SAME same-day/same-ATR-decile matched control.
Multiple comparisons: with 4 tests at alpha=.05 the chance of one false positive
is ~19%, so treat any single winner as a hypothesis, not a finding."""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

FILES=sorted(glob.glob('Data/shared/bars/1d/*.parquet'))
START,END=pd.Timestamp('2026-01-01').date(),pd.Timestamp('2026-08-20').date()
recs=[]
for p in FILES:
    t=os.path.basename(p)[:-8]
    try: df=pd.read_parquet(p,columns=['timestamp','open','high','low','close','volume'])
    except Exception: continue
    if len(df)<260: continue
    df=df.sort_values('timestamp').reset_index(drop=True)
    c,h,lo,v=df['close'],df['high'],df['low'],df['volume']
    if (c.tail(120)*v.tail(120)).median()<5e6: continue
    pc=c.shift(1); tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean(); e100=c.ewm(span=100,adjust=False).mean()
    rh,rl=h.rolling(20).max(),lo.rolling(20).min()
    hi252=h.rolling(252,min_periods=200).max()
    recs.append(pd.DataFrame({'ticker':t,'date':df['timestamp'].dt.tz_convert('UTC').dt.date,
        'atr_pct':atr/c,'ema_atr':(c-e100)/atr.replace(0,np.nan),
        'slope':(e100-e100.shift(20))/atr.replace(0,np.nan),
        'range_pos':((c-rl)/(rh-rl).replace(0,np.nan)).clip(0,1),
        'ret20':c.pct_change(20),'ret120':c.pct_change(120),
        'pct_of_52w_high':c/hi252,
        'fwd10':c.shift(-10)/c-1.0,'fwd20':c.shift(-20)/c-1.0}))
U=pd.concat(recs,ignore_index=True).dropna(
    subset=['ema_atr','slope','range_pos','fwd10','fwd20','atr_pct','ret120','pct_of_52w_high'])
U=U[(U.date>=START)&(U.date<=END)]
U['dec']=U.groupby('date')['atr_pct'].transform(lambda s: pd.qcut(s,10,labels=False,duplicates='drop'))
# cross-sectional relative strength, computed within each date (causal)
U['rs_rank']=U.groupby('date')['ret120'].rank(pct=True)
print(f"universe {U.ticker.nunique()} tickers / {len(U):,} stock-days\n")

rng=np.random.default_rng(11)
def control(s):
    picks=[]
    for (dt,dec),n in s.groupby(['date','dec']).size().items():
        pool=U[(U.date==dt)&(U.dec==dec)]
        if len(pool): picks.append(pool.sample(n=min(n*5,len(pool)),random_state=int(rng.integers(1e6))))
    return pd.concat(picks,ignore_index=True)

VARIANTS={
 '1. generic pullback to rising EMA100':
    (U.slope>0)&(U.ema_atr.between(-0.5,1.5))&(U.range_pos<=0.35),
 '2. + strong relative strength (top 25% 120d)':
    (U.slope>0)&(U.ema_atr.between(-0.5,1.5))&(U.range_pos<=0.35)&(U.rs_rank>=0.75),
 '3. + near 52w high (>=85%) as well':
    (U.slope>0)&(U.ema_atr.between(-0.5,1.5))&(U.range_pos<=0.35)&(U.rs_rank>=0.75)&(U.pct_of_52w_high>=0.85),
 "4. momentum's OWN zone: extended >3 ATR above rising EMA100":
    (U.slope>0)&(U.ema_atr>3.0)&(U.range_pos>=0.60),
}
rowsout=[]
for name,mask in VARIANTS.items():
    s=U[mask]
    if len(s)<200: print(f"{name}: only {len(s)} rows, skipped"); continue
    C=control(s)
    e10=(s.fwd10.mean()-C.fwd10.mean())*100; e20=(s.fwd20.mean()-C.fwd20.mean())*100
    t20,p20=stats.ttest_ind(s.fwd20,C.fwd20,equal_var=False)
    # month-sign stability
    s2=s.copy(); C2=C.copy()
    s2['m']=pd.to_datetime(s2.date).values.astype('datetime64[M]')
    C2['m']=pd.to_datetime(C2.date).values.astype('datetime64[M]')
    signs=[]
    for m in sorted(s2.m.unique()):
        a,b=s2[s2.m==m],C2[C2.m==m]
        if len(a)>=20 and len(b)>=20: signs.append(np.sign(a.fwd20.mean()-b.fwd20.mean()))
    pos=int(sum(1 for x in signs if x>0))
    rowsout.append((name,len(s),e10,e20,p20,f"{pos}/{len(signs)}"))
print(f"{'variant':<52} {'n':>7} {'ex10':>7} {'ex20':>7} {'p(20d)':>10} {'mo+':>6}")
for r in rowsout:
    print(f"{r[0]:<52} {r[1]:>7,} {r[2]:>+6.2f}pp {r[3]:>+6.2f}pp {r[4]:>10.3g} {r[5]:>6}")
print("\nex10/ex20 = excess forward return vs same-day same-ATR-decile control.")
print("mo+ = months (of 8) where the excess was positive. 4/8 is a coin flip.")
