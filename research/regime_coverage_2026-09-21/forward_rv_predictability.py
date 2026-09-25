"""Is FORWARD realized vol predictable from bars we ALREADY own?
Forward RV is a LABEL computed from underlying bars -- no broker API involved.
HAR-RV (Corsi 2009): forward RV regressed on trailing RV at 1d/5d/22d."""
import glob, os
import numpy as np, pandas as pd
np.seterr(all='ignore')

FILES=sorted(glob.glob('Data/shared/bars/1d/*.parquet'))
recs=[]
for p in FILES:
    t=os.path.basename(p)[:-8]
    try: df=pd.read_parquet(p,columns=['timestamp','open','high','low','close','volume'])
    except Exception: continue
    if len(df)<320: continue
    df=df.sort_values('timestamp').reset_index(drop=True)
    c,h,lo,o,v=df['close'],df['high'],df['low'],df['open'],df['volume']
    if (c.tail(120)*v.tail(120)).median()<5e6: continue
    if (c<=0).any() or (lo<=0).any() or (o<=0).any(): continue
    gk=(0.5*np.log(h/lo)**2-(2*np.log(2)-1)*np.log(c/o)**2).clip(lower=1e-10)
    ann=lambda s: np.sqrt(s*252)
    fwd=gk.iloc[::-1].rolling(20).mean().iloc[::-1].shift(-1)   # mean gk over t+1..t+20
    recs.append(pd.DataFrame({'ticker':t,'date':df['timestamp'].dt.tz_convert('UTC').dt.date,
        'rv1':ann(gk),'rv5':ann(gk.rolling(5).mean()),'rv22':ann(gk.rolling(22).mean()),
        'fwd_rv20':ann(fwd),'fwd_ret20':c.shift(-20)/c-1.0}))
U=pd.concat(recs,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
U=U[(U.rv1>.02)&(U.rv1<5)&(U.rv5>.02)&(U.rv5<5)&(U.rv22>.02)&(U.rv22<5)&(U.fwd_rv20>.02)&(U.fwd_rv20<5)]
U=U.sort_values('date').reset_index(drop=True)
print(f"{len(U):,} stock-days | {U.ticker.nunique()} tickers | {U.date.min()}..{U.date.max()}\n")
def r2(y,yh): return 1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
split=int(len(U)*0.7)
X=np.column_stack([np.log(U.rv1),np.log(U.rv5),np.log(U.rv22),np.ones(len(U))])
y=np.log(U.fwd_rv20.values)
b,*_=np.linalg.lstsq(X[:split],y[:split],rcond=None)
pred=X[split:]@b; yte=y[split:]
print("=== FORWARD 20d REALIZED VOL — out-of-sample (chronological 70/30) ===")
print(f"  HAR-RV (rv1,rv5,rv22)   R^2 = {r2(yte,pred):>6.3f}   corr = {np.corrcoef(pred,yte)[0,1]:.3f}")
print(f"  naive: fwd = rv22       R^2 = {r2(yte,np.log(U.rv22.values[split:])):>6.3f}")
print(f"  naive: fwd = rv5        R^2 = {r2(yte,np.log(U.rv5.values[split:])):>6.3f}")
print(f"  HAR betas: rv1={b[0]:+.3f} rv5={b[1]:+.3f} rv22={b[2]:+.3f}")
# contrast: same predictors on forward RETURN
yr=U.fwd_ret20.values
br,*_=np.linalg.lstsq(X[:split],yr[:split],rcond=None)
print(f"\n=== CONTRAST — forward 20d RETURN, same predictors, same split ===")
print(f"  R^2 = {r2(yr[split:],X[split:]@br):>6.3f}   <-- returns are ~unforecastable; VOL is not")
