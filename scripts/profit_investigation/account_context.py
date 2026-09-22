"""Preserve the account mark context separately from realized trading P&L."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/profit_investigation_2026-09-19/analysis'

def main():
    d=pd.read_csv(OUT/'trades.csv');rows=[];manifest=[]
    for p in sorted((ROOT/'Data/inference/account_snapshots').glob('broker_equity_*_paper.jsonl')):
        if p.name>'broker_equity_20260918_paper.jsonl':continue
        raw=p.read_bytes();r=json.loads(raw.splitlines()[-1]);account=r.get('account') or {}
        if not account.get('equity'):continue
        rows.append({'session':r.get('session_date_et'),'captured_at':r.get('captured_at_et'),
                     'equity':float(account['equity'])})
        manifest.append({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()})
    f=pd.DataFrame(rows);f['drawdown']=f.equity/f.equity.cummax()-1
    f.to_csv(OUT/'account_equity.csv',index=False)
    (OUT/'account_source_hashes.json').write_text(json.dumps(manifest,indent=2))
    p=OUT.parent/'data/local/broker_equity_20260918_paper.jsonl'
    snapshot=json.loads(p.read_text().splitlines()[-1])
    cols=['symbol','qty','avg_entry_price','cost_basis','market_value','unrealized_pl','unrealized_plpc']
    positions=pd.DataFrame(snapshot['positions'])[cols].copy()
    for c in cols[1:]:positions[c]=pd.to_numeric(positions[c],errors='coerce')
    positions=positions.merge(d.loc[~d.closed,['symbol','module','entry_time','score']],on='symbol',how='left',validate='one_to_one')
    positions.sort_values('unrealized_pl',ascending=False).to_csv(OUT/'open_positions.csv',index=False)

if __name__=='__main__':main()
