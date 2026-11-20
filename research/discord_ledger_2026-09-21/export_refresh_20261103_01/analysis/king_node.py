import sys, numpy as np, pandas as pd
from scipy.stats import norm
import json
ROOT = sys.argv[1]
# Cards: [{"base_date": "2026-10-23", "symbol", "expiry", "gex_king", "vex_king"}]; default = the 10-26 watchlist card (IMG_6538).
claims = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else [
    {'base_date':'2026-10-23','symbol':s,'expiry':e,'gex_king':g,'vex_king':v} for s,e,g,v in
    [('IBM','2026-10-30',260,260),('IBM','2026-11-20',275,275),('ZETA','2026-10-30',35,37),('CRWD','2026-10-30',270,270),('GILD','2026-10-30',160,132)]]
chains = {}
def chain(day):
    if day not in chains:
        c = pd.read_parquet(f"{ROOT}/Data/dealer_positioning/iv_surface/{day.replace('-','')}/iv_contracts.parquet")
        c['expiration'] = pd.to_datetime(c['expiration']).dt.date.astype(str); chains[day] = c
    return chains[day]
def greeks(S,K,T,iv):
    d1 = (np.log(S/K)+0.5*iv**2*T)/(iv*np.sqrt(T)); d2=d1-iv*np.sqrt(T)
    return norm.pdf(d1)/(S*iv*np.sqrt(T)), -norm.pdf(d1)*d2/iv   # gamma, vanna
hits = {'gex': [0, 0], 'vex': [0, 0]}
for cl in claims:
    sym, exp, gk, vk = cl['symbol'], cl['expiry'], cl['gex_king'], cl['vex_king']
    asof = pd.Timestamp(cl['base_date'] + ' 16:00'); c = chain(cl['base_date'])
    s = c[(c.symbol==sym)&(c.expiration==exp)&(~c.non_standard.astype(bool))].copy()
    if s.empty: print(sym,exp,'no chain'); continue
    S = float(s.spot.iloc[0]); T = max((pd.Timestamp(exp+' 16:00')-asof).days,1)/365
    iv = (s.volatility_pct/100).clip(lower=0.01)
    s['gamma'],s['vanna'] = greeks(S,s.strike.values,T,iv.values)
    sign = np.where(s.option_type.str.upper().str.startswith('C'),1,-1)
    s['gex'] = sign*s.gamma*s.open_interest*100*S*S*0.01
    s['vex'] = sign*s.vanna*s.open_interest*100
    s['gex_abs'] = s.gamma*s.open_interest*100*S*S*0.01
    by = s.groupby('strike').agg(net_gex=('gex','sum'),abs_gex=('gex_abs','sum'),net_vex=('vex','sum'),oi=('open_interest','sum'))
    out = {'king_|net_gex|':by.net_gex.abs().idxmax(),'king_abs_gex':by.abs_gex.idxmax(),'king_|net_vex|':by.net_vex.abs().idxmax(),
           'king_oi':by.oi.idxmax(),'call_oi_max':s[sign==1].groupby('strike').open_interest.sum().idxmax()}
    print(f'{sym} {exp} spot={S:.2f}  ACE GEX King={gk} VEX King={vk}  ->', {k:float(v) for k,v in out.items()})
    if gk is not None: hits['gex'][0] += out['king_|net_gex|'] == gk; hits['gex'][1] += 1
    if vk is not None: hits['vex'][0] += out['king_|net_vex|'] == vk; hits['vex'][1] += 1
print('matches (argmax |net GEX| / |net VEX|):', {k: f'{a}/{b}' for k, (a, b) in hits.items()})
