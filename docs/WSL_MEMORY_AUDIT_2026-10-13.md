# WSL and Windows memory audit — 2026-10-13

## Attribution

The initial Windows sample had 31.76 GiB usable RAM and only 1.27 GiB free.
Ubuntu's process list did not include Docker Desktop's separate process namespace,
although both use the WSL VM's memory. Adding the process RSS values to the
filesystem cache double-counts mapped file pages; Windows working sets are also
not the same measurement as guest RSS or Docker cgroup usage.

| Component | Initial sample |
| --- | ---: |
| Cynolycus combined server | about 2.7–2.8 GiB RSS |
| Intraday catalyst worker | about 1.8 GiB RSS |
| ACE recorder | about 0.12 GiB RSS |
| All processes visible in Ubuntu | about 6.76 GiB summed RSS |
| Odysseus main Docker container | 7.41 GiB reported by Docker |
| Windows local model runner | about 1.96 GiB working set |
| Chrome | about 3.14 GiB summed working sets |
| Windows VS Code | about 1.25 GiB summed working sets |
| Windows memory compression | about 9.94 GiB working set |

The compressed store reflects memory pressure; it is not an independently
installed application to terminate. Other Odysseus containers were small
(SearXNG ~62 MiB, Chroma ~9 MiB, ntfy ~42 MiB). The Windows model automatically
unloaded during the investigation.

The Odysseus model had a CPU-only llama.cpp build despite its GPU-offload command
flag. It served a 9B Q4_K_M GGUF, with four 8,192-token slots. All slots were idle.
The trading server's memory-keeper log stayed near 2.7 GiB over the observed hour;
this does not demonstrate a growing server leak. Linux showed effectively no
current memory pressure or sustained swap traffic, while Windows was low on RAM.

## Changes applied

- Reclaimed clean guest filesystem cache using `sync` then `vm.drop_caches=1`.
  This deletes no files and stops no services; cache may grow again.
- Preserved Odysseus and its model, with one 8,192-token serving slot,
  `--cache-ram 256`, and `--sleep-idle-seconds 120`. Idle sleep unloads the model
  and context; a new generation request automatically reloads it. First-request
  latency after sleeping is the tradeoff; simultaneous requests share one slot.
- Updated matching Odysseus saved commands and fields in
  `/home/luket/repos/odysseus/data/cookbook_state.json` and its running tmux
  launch script `/tmp/odysseus-tmux/serve-90e5a5b2_run.sh` inside the container.
  The original files have `.bak.memory-20261013` backups beside them; intermediate
  batch and sleep backups are also retained. Reload an already-open Cookbook UI
  before editing its settings so stale browser state does not overwrite changes.
- Backed up `C:\Users\luket\.wslconfig` to
  `.wslconfig.bak.memory-20261013`, and changed `autoMemoryReclaim=gradual` to
  `autoMemoryReclaim=dropCache`. The 20 GB guest cap and 24 GB swap limit remain.
  **This setting takes effect only after the next WSL restart.** Trading was not
  interrupted to apply it. Idle CPU detection still governs automatic reclaim.

A smaller inference batch (512 logical / 128 physical tokens) saved no measurable
anonymous RAM in a one-token test, so that experiment was reverted. One slot
alone only modestly reduced container usage; idle unloading is the primary model
memory reduction. Reducing the WSL cap is unsuitable without separately handling
the documented nightly news peak (~18.4 GB) and previous 16 GB OOM failures.

## Verification

Single-slot health and one-token CPU inference passed. Saved commands retained
one slot and 8,192-token context. The combined server and catalyst worker kept
their existing PIDs throughout. Final idle-release and wake-up measurements are
recorded below once completed. No trading code, data, or broker orders changed.

Sources: [WSL settings](https://learn.microsoft.com/windows/wsl/wsl-config),
[llama.cpp idle sleep](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#sleeping-on-idle).

### Measured results

- The first idle cycle reduced model RSS from approximately 9.75 GiB (including
  mapped weights) to 108.5 MiB; anonymous RSS fell to about 92 MiB.
- After reclaiming unused clean cache, the main Odysseus container reported
  275.8 MiB, versus 7.41 GiB initially. This is an idle-state comparison; active
  inference reloads the same model and again consumes multiple GiB.
- Windows free memory rose from about 1.27 GiB initially to 4.55 GiB in the
  first idle sample, with overall physical usage 85.7%. Guest memory release
  does not translate one-for-one or instantly into Windows working-set changes.
- A request to the sleeping server automatically reloaded the model and produced
  one token successfully in 18.36 seconds. The server reported awake afterward.
- Source inspection confirms that idle sleep destroys the model/context and
  automatically reloads them when a generation request arrives. Metadata polling
  via `/props` does not wake the model.

The next project-specific investigation, if further reduction is needed, is the
catalyst scorer's full-universe FINRA lookup: the source contains 9,695,519 rows,
while the worker trades a curated universe. A ticker-filtered loader could reduce
its startup/retained memory, but requires score-equivalence and rolling-window
validation. No such trading-path change was made in this audit.

### Final host reclaim

A second automatic sleep after the wake-up test passed; the main container was
295.6 MiB. Before compaction, Windows still charged substantial guest-free memory
to VmmemWSL (overall usage 87.1%). Free page reporting was enabled, but automatic
compaction depends on idle CPU. Ran the Microsoft-documented
`echo 1 > /proc/sys/vm/compact_memory` once as guest root; this compacts RAM,
not the disk/VHDX. Windows then measured **7.57 GiB free / 76.2% physical RAM
used**; VmmemWSL working set was 12.2 GiB and memory compression 2.49 GiB.
Ubuntu had about 12.3 GiB available. All three Cynolycus process IDs remained
unchanged. Measurements are snapshots, not a guarantee under nightly jobs or
active model inference.

For manual recurrence, this one-line Windows PowerShell command reclaims clean
cache and compacts guest RAM without shutting down WSL:

```powershell
wsl.exe -d Ubuntu -u root --exec sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
```

It can cause brief disk/CPU activity and should be used when idle, not as a
frequent polling task. Existing files and running processes are preserved.
[Microsoft's memory reclaim explanation](https://devblogs.microsoft.com/commandline/memory-reclaim-in-the-windows-subsystem-for-linux-2/)
documents both commands and why compaction helps free-page reporting.
