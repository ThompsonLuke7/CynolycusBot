import sys, numpy as np, pandas as pd, yfinance as yf
out = sys.argv[1]
raw = yf.download(['^VIX','SPY'], start='2000-01-01', end='2026-11-03', auto_adjust=True, progress=False)['Close'].dropna()
raw.to_parquet(out)  # snapshot of exactly what was tested
v, s = raw['^VIX'], raw['SPY']
vch = v.pct_change(); r = s.pct_change()
up1, up2 = (r.shift(-1)>0), (r.shift(-2)>0)
any_up = (up1|up2); cum2 = (s.shift(-2)/s-1)
any_dn = ((r.shift(-1)<0)|(r.shift(-2)<0))
df = pd.DataFrame({'vch':vch,'any_up':any_up,'any_dn':any_dn,'cum2':cum2,'next1':r.shift(-1)}).iloc[1:-2]
def block(name, d, base):
    n=len(d)
    if n==0: return
    p = d.mean(); se=np.sqrt(base*(1-base)/n)
    print(f'{name:<52} n={n:5d}  hit={p:6.1%}  base={base:6.1%}  lift={p-base:+6.1%}  z={(p-base)/se:+5.2f}  MDE80%~{2.8*se:5.1%}')
for era,(a,b) in {'2000-2026':('2000','2026-11'),'2015-2026':('2015','2026-11'),'2021-2026':('2021','2026-11')}.items():
    d = df.loc[a:b]; print(f'--- {era}  ({len(d)} sessions)')
    spk, crush = d[d.vch>=0.10], d[d.vch<=-0.10]
    block('VIX>=+10% -> any SPY up day in next 2', spk.any_up, d.any_up.mean())
    block('VIX>=+10% -> SPY 2-day return > 0', spk.cum2>0, (d.cum2>0).mean())
    block('VIX>=+10% -> next-day SPY up', spk.next1>0, (d.next1>0).mean())
    block('VIX<=-10% -> any SPY down day in next 2', crush.any_dn, d.any_dn.mean())
    block('VIX<=-10% -> SPY 2-day return < 0', crush.cum2<0, (d.cum2<0).mean())
    block('VIX<=-10% -> next-day SPY down', crush.next1<0, (d.next1<0).mean())
    print(f'   mean 2d ret after spike {spk.cum2.mean():+.3%} vs all {d.cum2.mean():+.3%}; after crush {crush.cum2.mean():+.3%}')
print('last VIX date', v.index[-1].date())
