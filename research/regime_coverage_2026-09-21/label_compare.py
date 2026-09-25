"""Does our weighted-sum label already contain triple-barrier?

User's argument: expansion_survival_score already weights alpha, forward return,
persistence and DRAWDOWN -- the same ingredients a triple barrier uses.

That is true about the INGREDIENTS. The question is the OPERATOR. The weighted
label takes MFE and MAE as separate AGGREGATES over the window. A triple barrier
asks which came FIRST. Two rows with identical fwd_max_alpha and fwd_max_drawdown
can be a win and a stop-out depending on order. This measures how often that bites.
"""
import glob, os
import numpy as np, pandas as pd

FILES=sorted(glob.glob('Data/shared/bars/1d/*.parquet'))
START,END=pd.Timestamp('2026-01-01').date(),pd.Timestamp('2026-08-01').date()
H=20                      # forward window (bars)
TGT_ATR, STOP_ATR = 2.0, 1.5   # realistic target / stop in ATR
recs=[]
for p in FILES:
    t=os.path.basename(p)[:-8]
    try: df=pd.read_parquet(p,columns=['timestamp','high','low','close','volume'])
    except Exception: continue
    if len(df)<300: continue
    df=df.sort_values('timestamp').reset_index(drop=True)
    c,h,lo,v=df['close'].values,df['high'].values,df['low'].values,df['volume'].values
    if np.median(c[-120:]*v[-120:])<5e6: continue
    pc=np.concatenate([[c[0]],c[:-1]])
    tr=np.maximum(h-lo,np.maximum(np.abs(h-pc),np.abs(lo-pc)))
    atr=pd.Series(tr).ewm(alpha=1/14,adjust=False).mean().values
    n=len(c); dates=df['timestamp'].dt.tz_convert('UTC').dt.date.values
    for i in range(250,n-H):
        if not np.isfinite(atr[i]) or atr[i]<=0: continue
        if not (START<=dates[i]<=END): continue
        e=c[i]; up=e+TGT_ATR*atr[i]; dn=e-STOP_ATR*atr[i]
        fh,fl,fc=h[i+1:i+1+H],lo[i+1:i+1+H],c[i+1:i+1+H]
        # FIRST-PASSAGE: which barrier is touched first
        hit_up=np.argmax(fh>=up) if (fh>=up).any() else 10**6
        hit_dn=np.argmax(fl<=dn) if (fl<=dn).any() else 10**6
        if hit_up==10**6 and hit_dn==10**6: tb=0
        elif hit_up<hit_dn: tb=1
        else: tb=-1
        # WEIGHTED-SUM ingredients (aggregates over the SAME window, order-blind)
        mfe=(fh.max()-e)/e
        mae=(fl.min()-e)/e
        atr_adj=(fc[-1]-e)/(atr[i])
        persistence=float((fc>e).mean())
        recs.append((t,dates[i],mfe,mae,atr_adj,persistence,tb,
                     (fc[-1]/e-1.0)))
D=pd.DataFrame(recs,columns=['ticker','date','mfe','mae','atr_adj','persist','tb','fwd_close'])
print(f"{len(D):,} rows  |  {D.ticker.nunique()} tickers  |  target {TGT_ATR}ATR / stop {STOP_ATR}ATR / {H} bars\n")

# reconstruct the weighted label with momentum's deployed weights, cross-sectionally ranked
for col,asc in (('mfe',True),('atr_adj',True),('persist',True),('mae',True)):
    D[col+'_r']=D.groupby('date')[col].rank(pct=True,ascending=asc)
D['weighted']=(0.40*D.mfe_r + 0.25*D.atr_adj_r + 0.20*D.persist_r + 0.15*D.mae_r)

print("=== triple-barrier outcome distribution ===")
print(D.tb.value_counts(normalize=True).rename({1:'target first (win)',-1:'STOP first (loss)',0:'neither (timeout)'}).mul(100).round(1).to_string())

print("\n=== do they agree? outcome of the rows the WEIGHTED label loves ===")
D['wdec']=D.groupby('date')['weighted'].rank(pct=True)
for lo_,hi_,lbl in ((0.9,1.01,'TOP decile of weighted label'),(0.8,0.9,'8th decile'),(0.0,0.1,'BOTTOM decile')):
    s=D[(D.wdec>=lo_)&(D.wdec<hi_)]
    if not len(s): continue
    print(f"  {lbl:<30} n={len(s):>7,}  target-first {(s.tb==1).mean()*100:>5.1f}%  "
          f"STOPPED-first {(s.tb==-1).mean()*100:>5.1f}%  timeout {(s.tb==0).mean()*100:>5.1f}%")
print(f"  {'(whole sample)':<30} n={len(D):>7,}  target-first {(D.tb==1).mean()*100:>5.1f}%  "
      f"STOPPED-first {(D.tb==-1).mean()*100:>5.1f}%  timeout {(D.tb==0).mean()*100:>5.1f}%")

print("\n=== THE ORDERING BLIND SPOT ===")
# rows that reach BOTH barriers at some point in the window
both=D[(D.mfe>=0) & (D.mae<=0)].copy()
reached=D[( D.tb!=0 )]
amb=D[(D.mfe>0)&(D.mae<0)]
# among rows whose MFE looks great, how many were stopped first?
good_mfe=D[D.mfe_r>=0.90]
print(f"rows in the TOP DECILE of forward max-gain (MFE): {len(good_mfe):,}")
print(f"   of those, {(good_mfe.tb==-1).mean()*100:.1f}% hit the STOP FIRST -- the weighted label")
print(f"   scores them highly, a triple barrier calls them losses.")
print(f"\ncorrelation(weighted label, triple-barrier outcome) = {D.weighted.corr(D.tb.astype(float)):.3f}")
print(f"correlation(weighted label, forward close return)   = {D.weighted.corr(D.fwd_close):.3f}")
