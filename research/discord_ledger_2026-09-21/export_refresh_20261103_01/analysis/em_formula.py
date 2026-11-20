import sys, numpy as np, pandas as pd
ROOT = sys.argv[1]
# Ace weekly EM 10/26-30 and Monday 10/26 tables (transcribed from IMG_6518 / IMG_6519)
weekly = {'AAPL':(356.97,338.49),'TSLA':(390.50,353.94),'AMZN':(269.76,226.24),'NVDA':(220.84,201.08),'AMD':(553.60,479.98),
          'META':(674.10,611.14),'MSFT':(555.51,493.51),'GOOG':(338.57,295.99),'GME':(26.64,23.00),'DJT':(6.90,5.90),
          'SOFI':(16.20,13.04),'PLTR':(208.59,184.91),'AVGO':(339.19,304.17),'JPM':(340.94,322.22)}
daily = {'AAPL':(350.88,344.58),'TSLA':(378.76,365.68),'AMZN':(251.61,244.39),'NVDA':(214.53,207.39),'AMD':(529.42,504.16),
         'META':(654.14,631.10),'MSFT':(529.81,519.21),'AVGO':(327.91,315.45)}
c = pd.read_parquet(f'{ROOT}/Data/dealer_positioning/iv_surface/20261023/iv_contracts.parquet')
c['expiration'] = pd.to_datetime(c['expiration']).dt.date.astype(str)
rows = []
for sym,(up,dn) in weekly.items():
    s = c[(c.symbol==sym)&(~c.non_standard.astype(bool))]
    if s.empty: rows.append({'sym':sym,'note':'missing'}); continue
    spot = float(s.spot.iloc[0]); center=(up+dn)/2; half=(up-dn)/2
    e = s[s.expiration=='2026-10-30']
    k = e.strike.iloc[(e.strike-spot).abs().argsort()].iloc[0]
    atm = e[e.strike==k]
    iv = atm.volatility_pct.mean()/100
    strad = atm.mark.sum()
    r = {'sym':sym,'spot':spot,'center':center,'center_minus_spot':center-spot,'half_pct':100*half/center,
         'iv_atm':iv,'em_iv_7cal':center*iv*np.sqrt(7/365),'em_iv_5trd':center*iv*np.sqrt(5/252),
         'straddle':strad,'straddle85':0.85*strad,'half':half}
    if sym in daily:
        du,dd = daily[sym]; dh=(du-dd)/2
        r.update({'d_center':(du+dd)/2,'d_half':dh,'d_iv_1cal':center*iv*np.sqrt(1/365),'d_iv_1trd':center*iv*np.sqrt(1/252),'d_iv_3cal':center*iv*np.sqrt(3/365)})
    rows.append(r)
df = pd.DataFrame(rows).set_index('sym')
pd.set_option('display.width',250); pd.set_option('display.precision',3)
print(df[['spot','center','center_minus_spot','half','em_iv_7cal','em_iv_5trd','straddle','straddle85']])
for col in ['em_iv_7cal','em_iv_5trd','straddle','straddle85']:
    ratio = df['half']/df[col]; print(f'weekly half / {col}: median {ratio.median():.3f}  IQR {ratio.quantile(.25):.3f}-{ratio.quantile(.75):.3f}')
d = df.dropna(subset=['d_half'])
print(d[['d_center','center','d_half','d_iv_1cal','d_iv_1trd','d_iv_3cal']])
for col in ['d_iv_1cal','d_iv_1trd','d_iv_3cal']:
    ratio = d['d_half']/d[col]; print(f'daily half / {col}: median {ratio.median():.3f}  IQR {ratio.quantile(.25):.3f}-{ratio.quantile(.75):.3f}')
print('captured_at', c.captured_at.min(), c.quote_time_utc.min() if 'quote_time_utc' in c else '')
