"""Sensitivity checks, signal controls, and execution/accounting diagnostics."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.profit_investigation.build_study import DATA,read,ts
from scripts.profit_investigation.analyze import stats,periods

OUT=DATA.parent/'analysis'

def main():
    d=pd.read_csv(OUT/'trades.csv');s=pd.read_csv(OUT/'signal_outcomes.csv');q=pd.read_csv(OUT/'exit_quote_costs.csv')
    s=s[s.available_at>='2026-07-13']
    closed=d[d.closed & ~d.exercise_transfer];rows=[]
    for period,g in periods(closed,'exit_session').items():
        for (mod,route),z in g.groupby(['module','route']):
            z=z.dropna(subset=['underlying_return']);pos=z[z.underlying_return>0]
            rows.append({'period':period,'module':mod,'route':route,'endpoints':len(z),
                'underlying_positive':len(pos),'underlying_positive_option_loss':int((pos.realized_pnl< -1e-7).sum()),
                'positive_move_precision':float((pos.realized_pnl>1e-7).mean()) if len(pos) else None})
    pd.DataFrame(rows).to_csv(OUT/'underlying_translation.csv',index=False)
    rows=[]
    for period,g in periods(q,'session').items():
        for cutoff in [60,300]:
            fresh=g[g.quote_age_s.between(0,cutoff)]
            for winner,z in [('all',fresh),('winner',fresh[fresh.winner==True]),('loser',fresh[fresh.winner==False])]:
                rows.append({'period':period,'max_quote_age_seconds':cutoff,'cohort':winner,'n':len(z),
                    'median_exit_cost':z.exit_cost_vs_mid.median(),'median_spread':z.spread.median(),
                    'mid_minus_fill_dollars':z.cost_dollars.sum()})
    pd.DataFrame(rows).to_csv(OUT/'fresh_exit_cost_summary.csv',index=False)
    rows=[]
    for module in ['momentum_expansion','multi_ticker_swing_htf']:
        z=closed[(closed.module==module)&(closed.route=='equity')].dropna(subset=['prior_return_20d'])
        for threshold in [-.1,0,.1,.2]:
            for passed,g in z.groupby(z.prior_return_20d>threshold):
                rows.append({'module':module,'threshold':threshold,'passed':passed,**stats(g,'realized_pnl'),
                    'mean_return':g['return'].mean(),'entry_min':g.entry_date.min(),'entry_max':g.entry_date.max()})
    pd.DataFrame(rows).to_csv(OUT/'trend_sensitivity.csv',index=False)
    # Chronological maturity: closed-trade feature separation cannot be
    # declared validated when it reverses on newer still-open observations.
    positions=pd.read_csv(OUT/'open_positions.csv')
    m=positions.merge(d.loc[~d.closed,['symbol','prior_return_20d','realized_pnl','basis']],on='symbol',validate='one_to_one')
    m.to_csv(OUT/'open_positions_with_prior_trend.csv',index=False)
    rows=[]
    for module in ['momentum_expansion','multi_ticker_swing_htf']:
        z=m[m.module==module].dropna(subset=['prior_return_20d'])
        for passed,g in z.groupby(z.prior_return_20d>0):
            rows.append({'module':module,'positive_prior_trend':passed,'n':len(g),
                'positive_marks':int((g.unrealized_pl>0).sum()),'net_open_mark':g.unrealized_pl.sum(),
                'median_unrealized_return':g.unrealized_plpc.median()})
    pd.DataFrame(rows).to_csv(OUT/'open_trend_control.csv',index=False)
    # Same-bar top-three vs ranks 4+ eliminates common decision-day drift.
    rows=[]
    for period,g in periods(s,'available_at').items():
        for module,z in g.groupby('module'):
            for h in [1,5,10]:
                col=f'forward_{h}d';diff=[];rhos=[]
                for _,bar in z.dropna(subset=[col,'rank','score']).groupby('signal_bar'):
                    a=bar[bar['rank']<=3];b=bar[bar['rank']>3]
                    if len(a)>=2 and len(b)>=2:
                        diff.append(a[col].mean()-b[col].mean())
                        if bar.score.nunique()>1:rhos.append(spearmanr(bar.score,bar[col]).statistic)
                rows.append({'period':period,'module':module,'horizon_sessions':h,'decisions':len(diff),
                    'mean_top3_minus_rest_return':np.mean(diff) if diff else None,
                    'median_within_decision_score_rho':np.nanmedian(rhos) if rhos else None})
    pd.DataFrame(rows).to_csv(OUT/'within_decision_rank_control.csv',index=False)
    # Attempt-level fill rate is distinct from predictive precision. Resting
    # ladder rungs are separate attempts and can legitimately be cancelled.
    orders=read(DATA/'orders.jsonl');attempts=pd.DataFrame(orders)
    attempts['session']=pd.to_datetime(attempts.submitted_at,utc=True).dt.tz_convert('America/New_York').dt.date.astype(str)
    rows=[]
    for period,g in periods(attempts,'session').items():
        for side,z in g.groupby('side'):
            rows.append({'period':period,'side':side,'attempts':len(z),'filled_orders':int((z.status=='filled').sum()),
                'cancelled':int((z.status=='canceled').sum()),'expired':int((z.status=='expired').sum()),
                'fill_rate':(z.status=='filled').mean()})
    pd.DataFrame(rows).to_csv(OUT/'order_attempts.csv',index=False)
    # More complete entry latency includes cancelled rungs in the previous
    # 60 minutes, bounded after the prior lifecycle of the same symbol.
    campaign=[]
    for _,r in closed.iterrows():
        if pd.isna(r.submit_time):continue
        t=ts(r.submit_time);a=attempts[(attempts.symbol==r.symbol)&(attempts.side=='buy')].copy()
        at=pd.to_datetime(a.submitted_at,utc=True)
        a=a[(at<=t)&(at>=t-pd.Timedelta(minutes=60))]
        if a.empty:continue
        earliest=pd.to_datetime(a.submitted_at,utc=True).min()
        campaign.append({'cycle_id':r.cycle_id,'module':r.module,'winner':r.realized_pnl>0,'entry_session':r.entry_date,
            'attempts_previous_hour':len(a),'campaign_to_fill_seconds':(ts(r.entry_time)-earliest).total_seconds()})
    pd.DataFrame(campaign).to_csv(OUT/'entry_campaigns.csv',index=False)
    print('Supplement complete')

if __name__=='__main__':main()
