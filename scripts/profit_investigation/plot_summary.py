"""Standalone research figures; shared candlestick helpers are inapplicable."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/profit_investigation_2026-09-19/analysis'

def main():
    d=pd.read_csv(OUT/'trades.csv');summary=pd.read_csv(OUT/'realized_summary.csv')
    auc=pd.read_csv(OUT/'score_discrimination.csv');q=pd.read_csv(OUT/'fresh_exit_cost_summary.csv')
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('Paper trading: July 13–September 18, 2026\nExploratory analysis of broker fills; last month = August 19–September 18',fontsize=15)
    s=summary[(summary.grouping=='all')&summary.period.isin(['overall','last_month'])]
    x=np.arange(len(s));ax[0,0].bar(x-.16,s.gross_profit/1000,.32,label='Realized profits',color='#24876b')
    ax[0,0].bar(x+.16,s.gross_loss/1000,.32,label='Realized losses',color='#c75146')
    ax[0,0].set_xticks(x,['Overall','Last month']);ax[0,0].set_ylabel('USD thousands, before fees')
    ax[0,0].axhline(0,color='black',lw=.6);ax[0,0].legend(frameon=False);ax[0,0].set_title('Profits have not covered losses')
    a=auc[(auc.period=='overall')&auc.auc_cluster_low.notna()].copy()
    names={'dealer_ranker':'Dealer options','meta_ranker':'Meta shares','momentum_expansion':'Momentum shares',
        'multi_ticker_swing':'Swing 30m options','multi_ticker_swing_htf':'HTF shares','spy_daytrader':'SPY options'}
    y=np.arange(len(a));ax[0,1].errorbar(a.auc,y,xerr=[a.auc-a.auc_cluster_low,a.auc_cluster_high-a.auc],fmt='o',color='#315c91',capsize=3)
    ax[0,1].set_yticks(y,[names[v] for v in a.module]);ax[0,1].axvline(.5,ls='--',color='gray')
    ax[0,1].set_xlim(0,1);ax[0,1].set_xlabel('Winner discrimination AUC, clustered 95% interval')
    ax[0,1].set_title('Entry scores do not reliably distinguish winners')
    m=d[(d.module=='momentum_expansion')&(d.route=='equity')&d.closed].dropna(subset=['prior_return_20d','score'])
    for winner,label,color in [(False,'Losing lifecycle','#c75146'),(True,'Winning lifecycle','#24876b')]:
        z=m[(m.realized_pnl>1e-7)==winner]
        ax[1,0].scatter(z.score,z.prior_return_20d,c=color,label=label,s=38,alpha=.8)
    ax[1,0].axhline(0,color='gray',ls='--');ax[1,0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1,0].set_xlabel('Momentum score at entry');ax[1,0].set_ylabel('Prior 20-session underlying return')
    ax[1,0].set_title('A promising historical split, not yet validated (n=41)');ax[1,0].legend(frameon=False,fontsize=8)
    a=q[(q.period=='overall')&(q.max_quote_age_seconds==60)&q.cohort.isin(['winner','loser'])]
    x=np.arange(len(a));ax[1,1].bar(x-.16,a.median_spread,.32,label='Quoted spread / midpoint',color='#869eb9')
    ax[1,1].bar(x+.16,a.median_exit_cost,.32,label='Exit fill below midpoint',color='#ce9551')
    ax[1,1].set_xticks(x,[f'{c.title()} exits (n={n})' for c,n in zip(a.cohort,a.n)])
    ax[1,1].yaxis.set_major_formatter(PercentFormatter(1));ax[1,1].set_title('Swing 30m: losing exits face larger spread costs')
    ax[1,1].legend(frameon=False,fontsize=8);ax[1,1].set_ylabel('Median fraction of option midpoint')
    for a in ax.flat:a.spines[['right','top']].set_visible(False)
    fig.savefig(OUT/'findings.png',dpi=170)
    fig.savefig(OUT/'findings.pdf')

if __name__=='__main__':main()
