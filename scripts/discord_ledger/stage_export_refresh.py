"""Snapshot, normalize and version-merge an updated Discord HTML export.

Original exports and prior study outputs are read-only. Every input HTML is
copied and hashed into a new study. Repeated IDs are compared semantically,
including authors, timestamps, edits, replies and attachment identity; rotating
CDN signatures alone do not fabricate a content edit. Changed variants remain
in the version audit and the newest conservatively available version supplies
the refreshed source view. Ambiguous same-availability conflicts are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

from scripts.discord_ledger.build import available_at,digest,read_jsonl,write_jsonl
from scripts.discord_ledger.normalize import build_manifest,normalize_file


def semantic_key(r:dict)->tuple:
    return (r.get('author_id'),r.get('channel_id'),r.get('timestamp_utc'),r.get('text'),
            r.get('is_edited'),r.get('edited_timestamp_utc'),r.get('reply_to_message_id'),r.get('reply_text'),
            tuple((a.get('filename'),urlparse(a.get('url') or a.get('media_url') or '').path) for a in r.get('attachments') or []),
            json.dumps(r.get('embeds') or [],sort_keys=True))


def merge_versions(base:list[dict],current:list[dict])->tuple[list[dict],list[dict],dict]:
    variants=defaultdict(list)
    for r in base:variants[r['message_id']].append(('prior',r))
    for r in current:variants[r['message_id']].append(('current',r))
    merged,audit=[],[]
    old_ids={r['message_id'] for r in base}
    current_ids={r['message_id'] for r in current}
    for mid,vs in variants.items():
        def rank(v):
            kind,r=v;avail,_=available_at(r)
            return (avail or r.get('timestamp_utc') or '',int(kind=='current'),r.get('source_file') or '')
        ordered=sorted(vs,key=rank)
        selected=ordered[-1][1]
        fingerprints={semantic_key(r) for _,r in vs}
        if len(fingerprints)>1:
            audit.append({'message_id':mid,'status':'different_versions_preserved',
                          'selection_rule':'latest_conservative_availability_then_current_export_then_source_path',
                          'selected_source_file':selected['source_file'],
                          'variants':[{'scope':kind,'record':r,'available_at_utc':available_at(r)[0]} for kind,r in vs]})
        row=dict(selected)
        row['source_observations']=[{'scope':kind,'source_file':r['source_file'],'raw_html_sha256':r.get('raw_html_sha256'),
                                    'source_file_sha256':r['original_html_metadata'].get('source_file_sha256')}
                                   for kind,r in vs]
        merged.append(row)
    merged.sort(key=lambda r:(r['timestamp_utc'],r['message_id']))
    summary={'prior_unique_messages':len(old_ids),'current_export_observations':len(current),
             'current_unique_messages':len(current_ids),'new_message_ids':len(current_ids-old_ids),
             'prior_ids_absent_from_current_preserved':len(old_ids-current_ids),
             'same_id_version_changes':len(audit),'merged_unique_messages':len(merged)}
    return merged,audit,summary


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('VaultOfAceDiscordLogs'))
    parser.add_argument('--base',type=Path,default=Path('research/discord_ledger_2026-09-21/source_post_profits_delta_20260922_01/messages.jsonl'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    raw_dir=args.out/'raw_exports';raw_dir.mkdir()
    contexts,rows,snapshots=[],[],[]
    for src in sorted(args.input_dir.glob('*.html')):
        dest=raw_dir/src.name;shutil.copy2(src,dest)
        if digest(src)!=digest(dest):raise ValueError(f'input changed while snapshotting {src}')
        context,rs=normalize_file(dest,raw_dir)
        for r in rs:r['source_file']=str(dest.resolve())
        contexts.append(context);rows+=rs
        snapshots.append({'original_path':str(src),'snapshot_path':str(dest),'sha256':digest(dest)})
    manifest=build_manifest(contexts,rows)
    merged,changes,summary=merge_versions(read_jsonl(args.base),rows)
    source_dir=args.out/'source';source_dir.mkdir()
    write_jsonl(source_dir/'messages.jsonl',merged)
    write_jsonl(source_dir/'export_observations.jsonl',rows)
    write_jsonl(source_dir/'changed_message_versions.jsonl',changes)
    old_ids={r['message_id'] for r in read_jsonl(args.base)}
    write_jsonl(source_dir/'new_messages.jsonl',[r for r in merged if r['message_id'] not in old_ids])
    manifest.update({'snapshot_inputs':snapshots,'base_path':str(args.base),'base_sha256':digest(args.base),'refresh_summary':summary,
                     'user_confirmed_display_timezone':'America/New_York; creation UTC remains snowflake-authoritative'})
    (source_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
