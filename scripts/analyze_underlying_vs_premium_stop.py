"""Counterfactual: premium stop vs underlying stop on live option exits.

Reads every Data/inference/*/closed_trades.jsonl option stop that carries an
entry_bar, resolves the UNDERLYING's 4H bars around it, and reports how far the
underlying had actually moved at the moment the premium stop fired.

  PYTHONPATH=. .venv/bin/python scripts/analyze_underlying_vs_premium_stop.py

Writes the per-trade table to research/daily_live_reports/underlying_vs_premium_stop.csv.
Findings are written up in underlying_vs_premium_stop.md next to it.

Scope/limits: this measures WHEN each stop fired against the underlying. It does
NOT price the counterfactual exit, because Data/options_history bars are trade
prints, not marks, and were invalidated for P&L by
research/options_experiment/10_RETRACTION_option_pnl_invalid.md. Treat the
dollar figures here as "what the premium stop cost", not "what the underlying
stop would have made".
"""
import json, glob, pandas as pd, numpy as np
from strategies.momentum_expansion.data.load_bars import load_4h

rows=[]
for f in sorted(glob.glob('Data/inference/*/closed_trades.jsonl')):
    for line in open(f):
        if line.strip(): rows.append(json.loads(line))
d=pd.DataFrame(rows)
d=d[d['ts'].notna() & d['realized_pnl'].notna()].copy()
d['ts']=pd.to_datetime(d['ts'],utc=True,format='mixed')
d['entry_dt']=pd.to_datetime(d['entry_bar'],utc=True,format='mixed',errors='coerce')
d['tkr']=d['ticker'].str.replace(r'^([A-Z]+)\d{6}[CP]\d{8}$',r'\1',regex=True)
# option full-exit stops only, with a recorded entry bar
s=d[(d['route']=='option') & d['exit_reason'].str.startswith('stop') & d['entry_dt'].notna()].copy()
print("option stop exits with entry_bar: %d of %d option stops" %
      (len(s), ((d['route']=='option')&d['exit_reason'].str.startswith('stop')).sum()))
# direction: OCC 'P' = put. order_symbol carries the OCC.
s['is_put']=s['order_symbol'].str.contains(r'\d{6}P\d{8}$',regex=True)
print("puts among them:", int(s['is_put'].sum()))

ATR_MULT=1.5
out=[]
for _,r in s.iterrows():
    try: b=load_4h(r['tkr'])
    except FileNotFoundError: continue
    if b.empty: continue
    b=b[~b.index.duplicated(keep='last')]
    tr=(b['high']-b['low']).rolling(14).mean()
    ent_i=b.index.searchsorted(r['entry_dt'])
    xit_i=b.index.searchsorted(r['ts'])
    if ent_i>=len(b) or xit_i>=len(b) or xit_i<=ent_i: continue
    ent_px=float(b['close'].iloc[ent_i]); atr=float(tr.iloc[ent_i])
    if not np.isfinite(atr) or atr<=0 or ent_px<=0: continue
    seg=b.iloc[ent_i+1:xit_i+1]                    # entry -> premium-stop bar
    fwd=b.iloc[xit_i+1:xit_i+1+40]                 # ~40 4H bars (~16 sessions) after
    stop_lvl=ent_px-ATR_MULT*atr
    hit=seg.index[seg['low']<=stop_lvl]
    out.append(dict(
        module=r['module'], tkr=r['tkr'], route=r['route'], reason=r['exit_reason'],
        entry=r['entry_dt'].tz_convert('America/New_York').date(),
        exit=r['ts'].tz_convert('America/New_York').date(),
        prem_ret=(r['exit_fill_price']-r['entry_avg_price'])/r['entry_avg_price']*100,
        pnl=r['realized_pnl'],
        u_ent=ent_px, u_xit=float(b['close'].iloc[xit_i]),
        u_ret=(float(b['close'].iloc[xit_i])/ent_px-1)*100,          # underlying move at premium-stop
        u_mae=(float(seg['low'].min())/ent_px-1)*100 if len(seg) else 0.0,
        atr_pct=atr/ent_px*100,
        u_stop_pct=(stop_lvl/ent_px-1)*100,
        u_stop_hit=bool(len(hit)),                                    # would ATR stop have fired by then?
        u_fwd_max=(float(fwd['high'].max())/ent_px-1)*100 if len(fwd) else np.nan,  # recovery after stop
        u_fwd_last=(float(fwd['close'].iloc[-1])/ent_px-1)*100 if len(fwd) else np.nan,
        bars_fwd=len(fwd)))
c=pd.DataFrame(out)
pd.set_option('display.width',250); pd.set_option('display.max_rows',80)
print("\nresolved %d/%d stops against 4H underlying bars\n" % (len(c),len(s)))
print("=== UNDERLYING MOVE AT THE MOMENT THE PREMIUM STOP FIRED ===")
print(c['u_ret'].describe().round(2).to_string())
for th in (2,3,5,8):
    m=c['u_ret']>-th
    print("  underlying down less than %d%%: %2d/%d (%.0f%%)  premium avg %.1f%%  pnl $%.0f"
          %(th,m.sum(),len(c),m.mean()*100,c.loc[m,'prem_ret'].mean(),c.loc[m,'pnl'].sum()))
print("\n=== WOULD A %.1fx-ATR UNDERLYING STOP HAVE FIRED BY THEN? ===" % ATR_MULT)
print(c['u_stop_hit'].value_counts().to_string())
nh=c[~c['u_stop_hit']]
print("NOT hit (premium stop was pure premium decay/noise): %d trades, realized $%.0f" % (len(nh),nh['pnl'].sum()))
print("  of those, underlying recovered above entry within 40 bars: %d (%.0f%%)"
      %((nh['u_fwd_max']>0).sum(),(nh['u_fwd_max']>0).mean()*100))
print("  their mean fwd max excursion: %+.1f%%   mean fwd close: %+.1f%%"
      %(nh['u_fwd_max'].mean(),nh['u_fwd_last'].mean()))
print("\n=== ALL STOPS ===")
print(c.sort_values('u_ret',ascending=False).round(2).to_string(index=False))
c.to_csv('research/daily_live_reports/underlying_vs_premium_stop.csv',index=False)
