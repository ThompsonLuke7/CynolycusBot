"""The steelman: a pullback entry isn't claimed to raise MEAN return, it's
claimed to improve GEOMETRY — stop just under the EMA, upside left to run.
So measure MFE/MAE over the next 20 sessions, not the mean.
Same same-day/same-ATR-decile matched control. Excursions in ATR units so
high-vol and low-vol names are comparable."""
import glob, os
import numpy as np, pandas as pd

FILES=sorted(glob.glob('Data/shared/bars/1d/*.parquet'))
START,END=pd.Timestamp('2026-01-01').date(),pd.Timestamp('2026-08-20').date()
recs=[]
for p in FILES:
    t=os.path.basename(p)[:-8]
    try: df=pd.read_parquet(p,columns=['timestamp','high','low','close','volume'])
    except Exception: continue
    if len(df)<260: continue
    df=df.sort_values('timestamp').reset_index(drop=True)
    c,h,lo,v=df['close'],df['high'],df['low'],df['volume']
    if (c.tail(120)*v.tail(120)).median()<5e6: continue
    pc=c.shift(1); tr=pd.concat([h-lo,(h-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean(); e100=c.ewm(span=100,adjust=False).mean()
    rh,rl=h.rolling(20).max(),lo.rolling(20).min()
    fwd_max=h.shift(-1).rolling(20).max().shift(-19)   # highest HIGH over next 20 bars
    fwd_min=lo.shift(-1).rolling(20).min().shift(-19)  # lowest LOW over next 20 bars
    recs.append(pd.DataFrame({'ticker':t,'date':df['timestamp'].dt.tz_convert('UTC').dt.date,
        'atr_pct':atr/c,'ema_atr':(c-e100)/atr.replace(0,np.nan),
        'slope':(e100-e100.shift(20))/atr.replace(0,np.nan),
        'range_pos':((c-rl)/(rh-rl).replace(0,np.nan)).clip(0,1),
        'mfe_atr':(fwd_max-c)/atr.replace(0,np.nan),
        'mae_atr':(c-fwd_min)/atr.replace(0,np.nan)}))
U=pd.concat(recs,ignore_index=True).dropna()
U=U[(U.date>=START)&(U.date<=END)]
U['dec']=U.groupby('date')['atr_pct'].transform(lambda s: pd.qcut(s,10,labels=False,duplicates='drop'))
rng=np.random.default_rng(23)
def control(s):
    picks=[]
    for (dt,dec),n in s.groupby(['date','dec']).size().items():
        pool=U[(U.date==dt)&(U.dec==dec)]
        if len(pool): picks.append(pool.sample(n=min(n*5,len(pool)),random_state=int(rng.integers(1e6))))
    return pd.concat(picks,ignore_index=True)

sets={'PULLBACK to rising EMA100':(U.slope>0)&(U.ema_atr.between(-0.5,1.5))&(U.range_pos<=0.35),
      'EXTENDED >3 ATR above rising EMA100':(U.slope>0)&(U.ema_atr>3.0)&(U.range_pos>=0.60)}
print(f"{'cohort':<40} {'n':>8} {'MFE(ATR)':>9} {'MAE(ATR)':>9} {'MFE/MAE':>8} {'P(MFE>2*MAE)':>13}")
for name,mask in sets.items():
    s=U[mask]; C=control(s)
    for lbl,d in ((name,s),(f'  control for {name[:22]}',C)):
        r=d.mfe_atr.median()/max(d.mae_atr.median(),1e-9)
        asym=(d.mfe_atr>2*d.mae_atr).mean()*100
        print(f"{lbl:<40} {len(d):>8,} {d.mfe_atr.median():>9.2f} {d.mae_atr.median():>9.2f} {r:>8.2f} {asym:>12.1f}%")
print("\nMFE/MAE = median max-favourable / median max-adverse excursion over the next 20 sessions, in ATRs.")
print("A pullback entry EARNS its keep only if its MFE/MAE beats its own matched control.")
