"""Descriptive cohort comparisons; no model selection or policy promotion."""
from __future__ import annotations
from collections import defaultdict
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.profit_investigation.build_study import DATA,read,ts,occ_meta

OUT=DATA.parent/'analysis'

def stats(d,key='pnl'):
    p=d[key].dropna();w=p[p>1e-7];l=p[p< -1e-7]
    return {'n':len(p),'wins':len(w),'losses':len(l),'flat':int((abs(p)<=1e-7).sum()),
        'win_rate':len(w)/len(p) if len(p) else None,'gross_profit':w.sum(),'gross_loss':l.sum(),
        'net':p.sum(),'profit_factor':w.sum()/(-l.sum()) if l.sum()<0 else None,
        'mean_pnl':p.mean(),'median_pnl':p.median(),
        'avg_winner':w.mean(),'avg_loser':l.mean()}

def periods(d,col):
    day=d[col].astype(str).str[:10]
    return {'overall':d,'last_month':d[(day>='2026-08-19')&(day<='2026-09-18')],
        'august':d[(day>='2026-08-01')&(day<='2026-08-31')]}

def bh(p):
    p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    return np.minimum(1,q)

def score_ci(d,seed=1909):
    # Resampling the single ticker SPY would yield a false zero-width CI.
    # Cluster that module by entry session; refuse CIs with <5 minority cases.
    group=d.ticker if d.ticker.nunique()>1 else d.entry_date
    if min(d.win.sum(),(~d.win).sum())<5 or group.nunique()<5:return [None,None]
    rng=np.random.default_rng(seed)
    groups=[g[['win','score']].to_numpy(dtype=float) for _,g in d.groupby(group)]
    v=[]
    for _ in range(1000):
        b=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        if np.unique(b[:,0]).size==2:v.append(roc_auc_score(b[:,0],b[:,1]))
    return np.quantile(v,[.025,.975]).tolist() if v else [None,None]

def main():
    d=pd.read_csv(OUT/'trades.csv');e=pd.read_csv(DATA.parent/'run3/realized_exits.csv')
    # An order can have many partial execution events: aggregate those into
    # one realized exit per lifecycle/order before calculating win rates.
    e=e[e.kind!='BASIS_TRANSFER']
    e=e.groupby(['cycle_id','exit_id','symbol','ticker','route','module','session','kind'],dropna=False).agg(
        pnl=('pnl','sum'),cost=('cost','sum'),proceeds=('proceeds','sum'),qty=('qty','sum')).reset_index()
    e['return']=e.proceeds/e.cost-1
    activities=read(DATA/'activities.jsonl');orders=read(DATA/'orders.jsonl');oid={o['id']:o for o in orders}
    b=lambda x:bool(x) if pd.notna(x) else False
    d['win']=d.realized_pnl>1e-7
    closed=d[d.closed & ~d.exercise_transfer].copy()
    summaries=[]
    for period,g in periods(e,'session').items():
        for grouping in [[],['module'],['route']]:
            groups=[(('ALL',),g)] if not grouping else g.groupby(grouping)
            for keys,z in groups:
                keys=keys if isinstance(keys,tuple) else (keys,)
                summaries.append({'period':period,'grouping':'+'.join(grouping) or 'all','group':'/'.join(map(str,keys)),**stats(z)})
    pd.DataFrame(summaries).to_csv(OUT/'realized_summary.csv',index=False)
    cs=[];concentration=[]
    for period,g in periods(closed,'exit_session').items():
        for (mod,route),z in g.groupby(['module','route']):
            cs.append({'period':period,'module':mod,'route':route,**stats(z,'realized_pnl'),
                'median_return':z['return'].median(),'median_hold_days':z.holding_days.median()})
        wins=g[g.win].sort_values('realized_pnl',ascending=False)
        concentration.append({'period':period,'closed_cycles':len(g),'winners':len(wins),
            'top5_profit_share':wins.head(5).realized_pnl.sum()/wins.realized_pnl.sum(),
            'top10_profit_share':wins.head(10).realized_pnl.sum()/wins.realized_pnl.sum(),
            'net_without_top5':g.realized_pnl.sum()-wins.head(5).realized_pnl.sum(),
            'net_without_worst5':g.realized_pnl.sum()-g.nsmallest(5,'realized_pnl').realized_pnl.sum()})
    pd.DataFrame(cs).to_csv(OUT/'closed_cohort_summary.csv',index=False)
    pd.DataFrame(concentration).to_csv(OUT/'concentration.csv',index=False)
    score_rows=[];bins=[]
    for period,g in periods(closed,'exit_session').items():
        for (mod,route),z in g.groupby(['module','route']):
            z=z.dropna(subset=['score']);z=z[z.realized_pnl.abs()>1e-7]
            if mod in ['mixed','unattributed'] or len(z)<15 or z.win.nunique()<2:continue
            lo,hi=score_ci(z)
            score_rows.append({'period':period,'module':mod,'route':route,'n':len(z),
                'winner_score':z.loc[z.win,'score'].median(),'loser_score':z.loc[~z.win,'score'].median(),
                'auc':roc_auc_score(z.win,z.score),'auc_cluster_low':lo,'auc_cluster_high':hi,
                'winners':int(z.win.sum()),'ci_cluster':'ticker' if z.ticker.nunique()>1 else 'entry_session',
                'rho_score_return':spearmanr(z.score,z['return']).statistic})
            z=z.copy();z['bucket']=pd.qcut(z.score,3,labels=['low','middle','high'],duplicates='raise')
            for bucket,zz in z.groupby('bucket',observed=True):
                bins.append({'period':period,'module':mod,'route':route,'bucket':bucket,
                    'min_score':zz.score.min(),'max_score':zz.score.max(),**stats(zz,'realized_pnl'),
                    'median_return':zz['return'].median()})
    pd.DataFrame(score_rows).to_csv(OUT/'score_discrimination.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'score_buckets.csv',index=False)
    feature_cols=[c for c in d if c.startswith(('prior_','x_','confirm_')) and pd.api.types.is_numeric_dtype(d[c])]
    feature_cols=[c for c in feature_cols if c not in ['x_p_dir','prior_gap_flag','x_htf_score','x_setup_peak_prob']]
    feature_cols+=['entry_premium','entry_delta','entry_spread','dte','entry_hour_et','rank','moneyness_pct']
    feature_rows=[];pairs=[];pair_stats=[]
    for period,g in periods(closed,'exit_session').items():
        for (mod,route),z in g.groupby(['module','route']):
            if mod in ['mixed','unattributed']:continue
            for col in feature_cols:
                zz=z.dropna(subset=[col]);w=zz.loc[zz.win,col];l=zz.loc[~zz.win & (zz.realized_pnl< -1e-7),col]
                if min(len(w),len(l))<8 or zz[col].nunique()<3:continue
                test=mannwhitneyu(w,l,alternative='two-sided')
                feature_rows.append({'period':period,'module':mod,'route':route,'feature':col,
                    'winners':len(w),'losers':len(l),'winner_median':w.median(),'loser_median':l.median(),
                    'auc':test.statistic/len(w)/len(l),'p_unclustered':test.pvalue})
            # Greedy without-replacement matching within module, route, 14
            # calendar days, and 0.25 within-group score SD. Descriptive only.
            zz=z.dropna(subset=['score']).copy();caliper=zz.score.std()*.25
            losers=zz[zz.realized_pnl< -1e-7].copy();used=set();matched=[]
            for _,w in zz[zz.win].sort_values('entry_time').iterrows():
                cand=losers[~losers.cycle_id.isin(used)].copy()
                cand=cand[(cand.score-w.score).abs()<=caliper]
                cand=cand[(pd.to_datetime(cand.entry_time,utc=True)-ts(w.entry_time)).abs()<=pd.Timedelta(days=14)]
                if cand.empty:continue
                l=cand.loc[(cand.score-w.score).abs().idxmin()];used.add(l.cycle_id);matched.append((w,l))
                pairs.append({'period':period,'module':mod,'route':route,'winner':w.symbol,'loser':l.symbol,
                    'winner_id':w.cycle_id,'loser_id':l.cycle_id,'winner_score':w.score,'loser_score':l.score,
                    'score_gap':abs(w.score-l.score),'score_caliper':caliper,'winner_pnl':w.realized_pnl,'loser_pnl':l.realized_pnl,
                    'winner_return':w['return'],'loser_return':l['return'],'entry_day_gap':abs((ts(w.entry_time)-ts(l.entry_time)).total_seconds()/86400)})
            for col in feature_cols+['entry_vs_mid','signal_to_submit_min','holding_days','underlying_entry_drift']:
                delta=[float(w[col])-float(l[col]) for w,l in matched if pd.notna(w.get(col)) and pd.notna(l.get(col))]
                if len(delta)<8 or not any(x!=0 for x in delta):continue
                pair_stats.append({'period':period,'module':mod,'route':route,'feature':col,'pairs':len(delta),
                    'median_winner_minus_loser':np.median(delta),'p_unclustered':wilcoxon(delta).pvalue})
    for name,rows in [('feature_separation',feature_rows),('matched_feature_separation',pair_stats)]:
        f=pd.DataFrame(rows)
        if len(f):f['q_bh_all_tests']=bh(f.p_unclustered)
        f.to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'similar_score_pairs.csv',index=False)
    execution=[]
    excols=['submit_to_first_fill_s','signal_to_submit_min','entry_vs_mid','entry_vs_reference',
        'entry_spread','holding_days','underlying_return','underlying_entry_drift',
        'underlying_mfe','underlying_mae','underlying_giveback','peak_fraction_of_hold','postexit_60m_return']
    for period,g in periods(closed,'exit_session').items():
        for (mod,route),z in g.groupby(['module','route']):
            for col in excols:
                for winner,zz in z.groupby('win'):
                    v=zz[col].dropna()
                    if len(v):execution.append({'period':period,'module':mod,'route':route,'winner':winner,
                        'metric':col,'n':len(v),'median':v.median(),'p90':v.quantile(.9)})
    pd.DataFrame(execution).to_csv(OUT/'execution_comparison.csv',index=False)
    # Exit quote comparisons are same-contract and order-ID joins; record
    # quote age and exclude stale observations in the primary comparison.
    rawex=pd.read_csv(DATA.parent/'run3/realized_exits.csv').drop_duplicates(['cycle_id','exit_id'])
    quote=[]
    for _,r in rawex.iterrows():
        o=oid.get(r.exit_id)
        if not o or not pd.notna(r.get('exit_mid')) or r.exit_mid<=0:continue
        qts=ts(r.exit_quote_ts) if pd.notna(r.exit_quote_ts) else None
        age=(ts(o['filled_at'])-qts).total_seconds() if qts is not None else None
        cr=closed[closed.cycle_id==r.cycle_id]
        quote.append({'cycle_id':r.cycle_id,'symbol':r.symbol,'module':r.module,'session':r.session,
            'winner':bool(cr.iloc[0].win) if len(cr) else None,
            'exit_cost_vs_mid':1-float(o['filled_avg_price'])/r.exit_mid,
            'spread':r.exit_spread,'quote_age_s':age,'qty':float(o['filled_qty']),
            'cost_dollars':(r.exit_mid-float(o['filled_avg_price']))*float(o['filled_qty'])*100})
    pd.DataFrame(quote).to_csv(OUT/'exit_quote_costs.csv',index=False)
    recall=[];s=pd.read_csv(OUT/'signal_outcomes.csv')
    s=s[s.available_at>='2026-07-13']
    for period,g in periods(s,'available_at').items():
        for mod,z in g.groupby('module'):
            for h in [1,5,10]:
                col=f'forward_{h}d'
                for hurdle in [0.,.05]:
                    zz=z.dropna(subset=[col]);positive=zz[col]>hurdle;selected=zz.was_traded
                    tp=int((positive&selected).sum());fp=int((~positive&selected).sum());fn=int((positive&~selected).sum());tn=int((~positive&~selected).sum())
                    recall.append({'period':period,'module':mod,'horizon_sessions':h,'return_hurdle':hurdle,
                        'n':len(zz),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                        'candidate_positive_rate':positive.mean(),'traded_mean_return':zz.loc[selected,col].mean(),
                        'untraded_mean_return':zz.loc[~selected,col].mean()})
    pd.DataFrame(recall).to_csv(OUT/'candidate_precision_recall.csv',index=False)
    # Export every realized winning/losing exit and every closed winner,
    # preserving partial-exit vs completed-lifecycle distinction.
    e[e.pnl>1e-7].sort_values('pnl',ascending=False).to_csv(OUT/'all_profitable_exits.csv',index=False)
    e[e.pnl< -1e-7].sort_values('pnl').to_csv(OUT/'all_losing_exits.csv',index=False)
    closed[closed.win].sort_values('realized_pnl',ascending=False).to_csv(OUT/'all_profitable_lifecycles.csv',index=False)
    closed[~closed.win].sort_values('realized_pnl').to_csv(OUT/'all_nonprofitable_lifecycles.csv',index=False)
    extra={'concentration':concentration,'fees_by_month':{},'order_status':pd.Series([o['status'] for o in orders]).value_counts().to_dict()}
    for period,g in periods(pd.DataFrame([a for a in activities if a['activity_type']=='FEE']),'date').items():
        extra['fees_by_month'][period]=pd.to_numeric(g.net_amount).sum()
    (OUT/'additional_summary.json').write_text(json.dumps(extra,indent=2,default=str))
    print('Analysis complete',len(feature_rows),'feature tests',len(pairs),'matched pairs',flush=True)

if __name__=='__main__':main()
