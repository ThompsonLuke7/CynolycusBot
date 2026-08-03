# CynolycusBot → Google Cloud Platform: Console-First Migration Tutorial

**Created:** 2026-07-26 · **Rewritten Console-first:** 2026-07-26 · **Phase 2 upload set re-verified:** 2026-07-29
**Status:** Phase 0 ✅ complete — Phase 1 next
**Audience:** You, learning GCP by porting this repo, one small verified step at a time.

---

## 0. Read this first

### How much of this is clicking vs. typing

**Almost all of it is clicking.** Here is the honest breakdown:

| Phase | Console (web UI) | CLI required? |
|---|---|---|
| 1 · Project, billing, budget | ✅ fully | no |
| 2 · Cloud Storage setup | ✅ bucket, lifecycle, versioning | **yes — bulk upload only** |
| 3 · Secrets & service account | ✅ fully | no |
| 4 · First deploy | ✅ fully (connect GitHub) | no |
| 5 · Jobs & scheduling | ✅ fully, incl. volume mounts | no |
| 6 · Shared dashboard | ✅ fully | no |
| 7 · Logs, alerts, cost | ✅ fully — Console is *better* here | no |
| 8 · Live loop | ✅ fully | no |

**The one unavoidable CLI step** is Phase 2.4: uploading ~72 GB across **~108,000 files** (re-measured 2026-07-29 — see §2.3b). The Console's uploader is a browser drag-and-drop — it will stall, time out, or silently drop files at that scale, and it can't resume. `gcloud storage rsync` is resumable, parallel, and verifies checksums. That's a 20-minute exception in an otherwise click-driven migration.

> **Trap: Cloud Shell won't save you here.** The Console has a built-in terminal (Cloud Shell) with `gcloud` preinstalled, and it's great for one-off commands. But Cloud Shell runs on a Google VM — **it cannot see your laptop's filesystem.** Uploading local data must run from your own WSL terminal.

Since you've already completed Phase 0 and have a working `gcloud`, that step is ready to go when you reach it.

### Why the Console is the right choice for learning

Clicking through forms shows you **every option a resource has** — memory, concurrency, timeouts, volumes, secrets — laid out with inline help. A `gcloud` one-liner hides all of that behind flags you didn't know existed. Build the mental model in the UI; automate later, once you know what you're automating.

Every Console screen has a **"Equivalent command line"** or **"Equivalent REST"** link, usually near the Create button. Click it every time. That's how the CLI stops being intimidating — you'll have already built the resource and can see exactly which flag maps to which field.

> **UI wording drifts.** Google renames buttons and reshuffles panels a few times a year. This guide gives navigation paths and field names, not pixel positions. If a label doesn't match exactly, look for the nearest equivalent — the underlying concepts in Section 1 are stable even when the chrome isn't.

### The three goals you actually stated

| Goal | Which phase solves it |
|---|---|
| Stop running out of disk space | **Phase 2** (Cloud Storage) — biggest, safest, earliest win |
| Other people can access it | **Phase 6** (Cloud Run service + IAM) |
| Runs automatically and continuously | **Phase 5** (Cloud Run Jobs + Scheduler), then **Phase 8** (the live loop) |

They're in that order deliberately. Storage is reversible and nearly free. The live trading loop is last because it's the only piece where a mistake costs money in the market, not just on the bill.

### Ground rules

1. **Cost safety first.** Phase 1 sets a budget before you create a single resource.
2. **Paper mode only in the cloud** until Phase 8, decided deliberately and recorded in §12. Per `AGENTS.md`, live order paths stay separated by configuration and credentials.
3. **Local stays authoritative** until a phase is verified. Upload → verify → *then* delete, in a separate session.
4. **One phase per sitting.** Each has a ✅ Checkpoint you must pass.
5. **Write notes in §12** as you go. That's what this file is for long-term.

### Your starting state (re-measured 2026-07-29)

```
Repo total:                86 GB
  strategies/              51 GB   ← .parquet feature/training matrices      9,076 files
  signals/                 11 GB   ← news embeddings, catalyst matrices      8,340 files
  Data/                    11 GB   ← models, bars cache, options history     89,165 files
  .venv/                  8.6 GB   ← regenerable, never upload
  .git/                   5.1 GB   ← already on GitHub, never upload
  theme_expansion/        408 MB   ← 100% untracked theme outputs            1,869 files
  UI/swing_audit/         332 MB   ← paper/live session audit logs             298 files
  research/               169 MB   ← study datasets (confluence, options)      273 files
  themes/                 120 MB   ← dynamic_theme outputs (live path)         230 files
  root *.tgz               82 MB   ← meta_ranker model bundles                   2 files
  logs/ backtests/ etc.    78 MB

GitHub remote:          ThompsonLuke7/CynolycusBot (branch main)
```

**Correction to the 2026-07-26 version of this section.** It claimed "`.gitignore` already excludes `Data/**` … nothing in Phase 2 touches tracked source code." That is **false** — `.gitignore` rules do not untrack files already committed. Actual tracked-file counts inside the upload trees:

| Tree | Tracked files | Tracked bytes | What they are |
|---|---|---|---|
| `Data/` | 4,591 | 82 MB | 4,000 `ga_xgboost/**/tree_dot` files, `inference/live_runs/*.jsonl` |
| `strategies/` | 374 | 54 MB | live `*.json` models, universe CSVs, frozen-test summaries |
| `signals/` | 159 | 4.5 MB | meta-ranker native boosters, blacklist, kmeans pickle |

Two consequences, both handled below: the bucket becomes a **second copy** of git-tracked artifacts (divergence risk — git stays authoritative for those), and Phase 2.7's "delete the local directory" would **destroy tracked working-tree files** unless you check first.

---

## 1. The mental model, compressed

Six categories cover nearly everything:

```
COMPUTE      run code            → Cloud Run (start here), Compute Engine, GKE (don't)
STORAGE      files & objects     → Cloud Storage (buckets)
DATABASES    structured state    → Cloud SQL, Firestore, BigQuery
NETWORKING   connect things      → mostly automatic on Cloud Run
IAM          who may do what     → principals + roles + resources
OPERATIONS   observe & pay       → Cloud Logging, Monitoring, Budgets
```

The distinction that will bite you:

| Thing | Use for | Do **not** use for |
|---|---|---|
| **Cloud Storage** | parquet, models, snapshots, backtest outputs | anything needing `UPDATE ... WHERE` |
| **Persistent Disk** | a VM's actual filesystem | sharing data between services |
| **Cloud SQL / Firestore** | live app state, positions, orders | 8-million-row feature matrices |
| **BigQuery** | analytical SQL over huge history | the query behind a dashboard click |

And the identity distinction:

```
Human account    = you (lthompson7835@gmail.com) — used to administer
Service account  = your code — used at runtime, never your personal creds
```

### Target architecture

```
                     ┌──────────────── Secret Manager (Alpaca/Tiingo/Finnhub/Anthropic keys)
                     │
GitHub ──► Cloud Build ──► Artifact Registry ──► Cloud Run
  (push)   (auto on push)                          ├── Job:     nightly pipeline    ◄── Cloud Scheduler
                                                   ├── Job:     module scoring       ◄── Cloud Scheduler
                                                   └── Service: dashboards (IAM)     ◄── other people
                                                        │
                                                        ▼
                                              Cloud Storage  gs://cynolycusbot-data/
                                                   ├── Data/     ├── signals/   ├── strategies/
                                                   ├── themes/   ├── research/  ├── backtests/
                                                   ├── theme_expansion/  ├── UI/swing_audit/
                                                   └── model_bundles/
                                                        │
                                              Cloud Logging + Monitoring + Budget alerts
```

---

## Phase 0 — Prerequisites ✅ COMPLETE

Done 2026-07-26. Linux-native `gcloud` **577.0.0** installed via apt; `which gcloud` → `/usr/bin/gcloud`. The stale Windows SDK (513.0.0) is still on `PATH` but ranks below `/usr/bin`, so it never wins — nothing to uninstall.

You need `gcloud` for exactly one step (Phase 2.4). Everything else is the browser.

**Docker: not needed.** Cloud Build builds your image in the cloud from your GitHub repo.

### 0.3 Know what must never leave your machine

- `.env` — holds `APCA_API_KEY_ID_LIVE`, `APCA_API_SECRET_KEY_LIVE`, `TIINGO_API_KEY`, `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY`
- `.venv/` — regenerable, huge, platform-specific
- The `THEME_EXPLORER_DEPLOY_KEY_PATH` target — an SSH private key

These go to **Secret Manager** (Phase 3) — never a bucket, never a container image, never an env-var field.

### 0.4 One CLI step you should do now (30 seconds)

So local Python scripts can authenticate to GCP later:

```bash
gcloud auth login
gcloud auth application-default login
```

The second is separate and matters: it creates the credentials that `from google.cloud import storage` picks up automatically.

---

## Phase 1 — Project, billing, and the budget that protects you (30 min · 100% Console)

**Complete this phase before creating any resource.** The budget is your seatbelt.

### 1.1 Sign up / confirm billing

Go to <https://cloud.google.com/free>. New accounts get **$300 in credits valid for 90 days**, plus the **Always Free tier**, which does not expire.

> ⚠️ A budget **alerts** you. It does **not** shut anything down. Treat every alert as urgent.

### 1.2 Create your projects

**Console → the project dropdown in the top blue bar → "New Project"**

Create two. You don't need staging yet.

- Project name: `Cynolycus Dev` → note the auto-generated **Project ID**
- Repeat for `Cynolycus Prod`

> **Project IDs are globally unique across all of GCP and permanently immutable.** The Console auto-suggests one and appends digits if taken. Write the exact ID it gives you into §12 — you'll type it into a dozen forms later. `cynolycus-dev` may already be claimed by a stranger; whatever you get, use it consistently.

Switch to the dev project using the same dropdown. **Confirm the top bar shows `Cynolycus Dev` before every subsequent phase** — deploying into the wrong project is the most common Console mistake.

### 1.3 Link billing

**Console → Billing.** If the project isn't linked, you'll see a prompt — link it to your billing account. Repeat for prod.

### 1.4 Set the budget — do not skip

**Console → Billing → Budgets & alerts → "Create budget"**

- **Scope:** your billing account (covers both projects)
- **Amount:** `$25`
- **Threshold rules:** 50%, 90%, 100%
- **Actions:** email alerts to yourself

$25/month is intentionally low. Everything through Phase 7 should cost **under $10/month**. A 50% alert in week one means something is misconfigured — investigate before continuing.

### 1.5 Enable APIs

**Console → APIs & Services → "Enable APIs and Services"**, then search and enable each:

Cloud Run · Cloud Build · Artifact Registry · Secret Manager · Cloud Scheduler · Cloud Storage · Cloud Logging · Cloud Monitoring

You can also skip this — the Console prompts you to enable an API the first time you use a service. Enabling upfront just avoids interruptions.

### 1.6 Pick your region and stick to it

Use **`us-central1`** everywhere. It's one of only three regions qualifying for the Cloud Storage Always Free tier (with `us-east1`, `us-west1`) and is the standard-price reference region. Choosing it consistently means never thinking about regions again.

✅ **Checkpoint 1**

- Top bar shows your dev project
- **Billing → Overview** shows the project linked, spend **$0.00**
- **Billing → Budgets & alerts** lists your $25 budget

📝 Record your Project IDs in §12 now.

---

## Phase 2 — Cloud Storage: reclaim your disk (the big win)

Solves the space problem. Touches **no** application code.

### 2.0 Why buckets and not a database (answered 2026-07-26)

Reasonable question, since the big files *are* structured data. Answer: **object storage is right; a relational database would cost ~100× more and would not make research easier.**

Cost for your ~72 GB:

| Option | Storage/mo | Compute/mo | **Total** |
|---|---|---|---|
| **GCS parquet** (this plan) | $1.28 | $0 — pay per job | **~$1.28** |
| GCS + lifecycle → Coldline | $0.31 | $0 | **~$0.31** |
| BigQuery native tables | $0.64–$1.28 | $6.25/TiB scanned (1 TiB/mo free) | **~$1–2** |
| **Cloud SQL Postgres** | ~$28.60 | ~$100.30 (`db-n1-standard-2`, 24/7) | **~$129** |

Cloud SQL storage is **$0.22/GB-month vs $0.020 for GCS** — 11× before the always-on instance. And the data grows: `features_4h.parquet` (8.46M rows × 134 float64 cols) is ~9.3 GB as Postgres rows vs 4.37 GB as parquet, before indexes and WAL.

**Parquet already is the database part.** It's a columnar storage format — columnar layout, compression, predicate pushdown, per-row-group min/max statistics. (Your chunked column-pruned reader works *because* of those statistics.) What a database adds on top is transactions, point-lookup indexes, concurrent mutation, and a query planner, served by a 24/7 process. This workload — "read 40 columns across 8.4M rows, hand a matrix to XGBoost" — needs none of it. A row store would read every byte of every row to answer that.

**Immutability is the stronger argument.** `AGENTS.md` requires immutable raw data and versioned transformations. A parquet file is an artifact you can hash and pin to an experiment; a table row can be silently `UPDATE`d, and the backtest stops being reproducible. Mutability is a liability for research infrastructure.

**Training settles it:** XGBoost/sklearn/torch need materialized in-memory arrays. Via a database you'd `SELECT` into pandas anyway — a network round-trip and a serialization step to arrive at what `read_parquet` already gave you.

**Where BigQuery does earn a place — later, without migrating:** *external tables* query the parquet already in your GCS bucket. No second copy, no extra storage cost, no migration. Pay per query scanned, **1 TiB/month free**. Good for ad-hoc analytics ("mean forward excursion by regime across 8.4M rows" as a 10-second query instead of a chunked-reader script). Useless for training loops and bar-by-bar backtests.

**Decision:** GCS + parquet is the source of truth. Add BigQuery external tables *only if* ad-hoc SQL becomes painful. Phase 2 is unaffected either way.

### 2.0b The real storage lever: fix the encoding, not the engine

Measured 2026-07-26 on one row group of `strategies/momentum_expansion/data/processed/features_4h.parquet` — the file compresses only **1.04×** because 131 of its 134 columns are `float64`:

```
float64 + snappy :  458.7 MB   ← current
float64 + zstd-3 :  401.4 MB   12.5% smaller
float32 + snappy :  271.9 MB   40.7% smaller
float32 + zstd-3 :  244.8 MB   46.6% smaller
```

That file would go **4.37 GB → 2.33 GB**. It also compounds — smaller files mean less RAM pressure, the thing that crashed WSL on 2026-07-21.

**Repo-wide audit (2026-07-28) — the prize is smaller than that one file suggests.** Footer scan of all 142 parquet files >10 MB (41.0 GB of the ~72 GB):

| arrow dtype | GB | share |
|---|---|---|
| `float` (float32) | 19.62 | **47.9%** ← already downcast |
| `double` (float64) | 15.50 | 37.8% |
| `string` | 4.96 | 12.1% |
| `timestamp[ns, tz=UTC]` | 0.81 | 2.0% |
| everything else | 0.11 | 0.2% |

Codec is **100% SNAPPY** — so zstd applies everywhere, but the float32 downcast only reaches 37.8% of bytes, because nearly half your data is already float32.

float64 share by tree: `strategies/` 41.8% (14.44 GB) · `Data/` 21.4% · `signals/` 7.6% · `theme_expansion/` 97.6% · `research/` 98.6%.

Measured zstd-3 on string-heavy files (`finra_short_volume`, `cboe_unusual_strikes`): **15–19% smaller**, better than the 12.5% seen on float-only data.

**Realistic projection on the 41 GB analyzed:**

| Change | Saved |
|---|---|
| zstd-3 everywhere (~15%) | ~3.8 GB (non-float64 portion) |
| float32 on float64 columns (~40.7%) | ~6.3 GB |
| **both** | **~11 GB (≈27%)** |

Not the 46.6% that single file implied — **that file was unusually float64-heavy and is not representative.** ~11 GB is still worth having, but it's a tidy-up, not a rescue.

> The remaining **38,673 parquet files under 10 MB (23.7 GB)** weren't footer-scanned. Their file *count* matters more than their dtypes: ~108,000 objects makes `rsync` slow and burns Class A operations (free tier is 5,000/month; the overage is cents, but expect to exceed it on first upload).

⚠️ **Validate before adopting — do not bulk-convert.**
- Technical features (ATR, returns, cross-sectional ranks) don't need float64's 15 significant digits; float32 gives ~7. But dtype changes can shift model outputs.
- **XGBoost's `DMatrix` is float32 internally**, so for XGBoost models the downcast already happens and costs nothing. sklearn and torch paths need checking.
- `strategies/multi_ticker_swing/data/processed/features_30m.parquet` is *already* float32 — there the lever is zstd alone (~12%), not the downcast. The audit shows this is true of **47.9% of all your parquet bytes**, so target the float64 files specifically rather than converting broadly.
- Compare against a frozen-test baseline before changing any pipeline. **Treat this as a separate change from the migration** — don't bundle it into Phase 2.

### 2.1 The cost math

Verified `us-central1` regional prices (2026-07-26):

| Class | $/GB-month | Min. duration | Retrieval fee | **72 GB uploaded costs** |
|---|---|---|---|---|
| Standard | $0.020 | none | none | **$1.44/mo** |
| Nearline | $0.010 | 30 days | $0.01/GB | **$0.72/mo** |
| Coldline | $0.004 | 90 days | $0.02/GB | **$0.29/mo** |
| Archive | $0.0012 | 365 days | $0.05/GB | **$0.09/mo** |

Free tier: **5 GB-months Standard** (US regions), 5,000 Class A ops, 50,000 Class B ops, 100 GB North America egress/month.

Your entire 72 GB problem costs **about a coffee per year** in Coldline, or **$19/year** in Standard. Highest-leverage item in this document.

> **Minimum storage duration is a real trap.** An object written to Nearline and deleted after 3 days is billed for the full 30. Keep actively-rewritten data (live state, daily caches) in Standard; let lifecycle rules age *cold* things down.

> **Egress is the other trap.** Reading a bucket *into Cloud Run in the same region* is free. Downloading to your laptop costs money past 100 GB/month. Send the compute to the data, not the data to the compute.

### 2.2 Create the bucket — Console

**Console → Cloud Storage → Buckets → "Create"**

| Field | Value | Why |
|---|---|---|
| Name | `cynolycusbot-data` (add a suffix if taken) | **Globally unique across all of GCP** — same namespace as every other customer |
| Location type | **Region** | Cheaper than multi-region; you don't need geo-redundancy |
| Location | `us-central1` | Free-tier eligible |
| Storage class | **Standard** | Lifecycle rules will age things down in 2.5 |
| Access control | **Uniform** | ⚠️ Important — see below |
| Public access prevention | **Enabled** | Blocks accidental internet exposure |
| Protection | leave default for now | Versioning comes in 2.6 |

> **Choose "Uniform", not "Fine-grained".** Uniform disables the legacy per-object ACL system so **IAM is the only permission model**. It eliminates an entire category of accidental-public-exposure bugs. Fine-grained exists only for S3-migration compatibility.

### 2.3 Learn the tool on ONE small directory (CLI)

Don't start with 51 GB. Start with 3.6 MB. From your WSL terminal:

```bash
cd /home/luket/repos/CynolycusBot
gcloud storage rsync Data/outputs gs://cynolycusbot-data/Data/outputs --recursive --dry-run
```

`--dry-run` shows exactly what *would* happen without transferring anything. Read it, then drop the flag to run for real.

> **Corrected 2026-07-29.** This step used to practise on `Data/options_history` described as "17 MB" — that directory is now **4.4 GB** (3.6 GB of it `trades/`), which is no longer a cheap first lesson. It also wrote to `gs://cynolycusbot-data/options_history` while 2.4 writes the same files to `gs://cynolycusbot-data/Data/options_history` — running both would have stored **4.4 GB twice** under two prefixes. Every destination in Phase 2 now mirrors the local path exactly, so `rsync` recognises what it already uploaded.

Verify in the **Console → Cloud Storage → Buckets → cynolycusbot-data**, and compare against `du -sh Data/outputs`.

✅ **Checkpoint 2.3** — bucket byte total matches local. If not, stop and find out why before scaling up.

### 2.3b What actually gets uploaded — verified 2026-07-29

Re-derived from the filesystem, not from the earlier estimate. Nine sources, ~72 GB, ~108k objects:

| # | Local path | Size | Files | Why it goes | Tracked in git? |
|---|---|---|---|---|---|
| 1 | `Data/` | 11 GB | 89,165 | models, bars cache, options history, `inference/live_runs` logs | 4,591 files |
| 2 | `signals/` | 11 GB | 8,340 | news embeddings, catalyst + meta-ranker matrices | 159 |
| 3 | `strategies/` | 51 GB | 9,076 | the feature/training matrices — the whole disk problem | 374 |
| 4 | `theme_expansion/` | 408 MB | 1,869 | **was missing.** 100% untracked; `theme_daily.parquet`, theme training matrix | 0 |
| 5 | `themes/` | 120 MB | 230 | **was missing.** `dynamic_theme/outputs/` is gitignored and read by the live theme path | 157 |
| 6 | `UI/swing_audit/` | 332 MB | 298 | **was missing.** Paper/live session audit logs — gitignored, single copy, no backup | 32 |
| 7 | `research/` | 169 MB | 273 | **was missing.** Study datasets behind the capstone and the options retraction | 94 |
| 8 | `backtests/` | 29 MB | 104 | **was missing.** Gitignored per-run outputs | 9 |
| 9 | `meta_ranker_model_bundle_{quality,upside}.tgz` | 82 MB | 2 | **was missing.** Trained bundles, gitignored (`*.tgz`), only copy on earth | 0 |

Items 4–9 total **~1.1 GB / ~2,800 files** — 1.5% of the bytes, but they include the only copies of the live theme features, every paper-trade audit trail, and two trained model bundles. Leaving them out is the real risk in this phase; the 51 GB was never in danger of being forgotten.

**Deliberately excluded:**

| Excluded | Size | Why |
|---|---|---|
| `.venv/` | 8.6 GB | regenerable, platform-specific |
| `.git/` | 5.1 GB | already on GitHub |
| `__pycache__/`, `*.py`, `*.pyc` | ~315 files | source belongs in git, not a bucket — **but see §2.7b: 4 gitignored Colab scripts break this assumption and exist nowhere else** |
| `strategies/momentum_expansion/data/training_export/training_matrix_4h.parquet` | 1.15 GB | **verified byte-identical** to `data/processed/training_matrix_4h.parquet` (same size, same md5 over first 200 MB) |
| `.../training_export_ablation/ablation_colab_bundle.tgz` | 1.04 GB | it is a tar of `training_matrix_4h_with_regime.parquet`, which uploads uncompressed in the same directory |
| `Data/runtime/live_data_jobs.lock`, `startup_queue.json` | ~1 KB | live process state; meaningless and misleading in a bucket |
| `logs/` | 39 MB | rotating local churn; not a research artifact |
| `.env`, `core/API/Schwab_API/schwab_token*.json` | — | secrets → Phase 3 (verified: no `.env`/key/token/pem file exists anywhere in trees 1–9) |

Skipping the two verified duplicates saves **2.2 GB** — more than every missing directory added together.

> `Data/runtime/meta_ranker_matrix_before_{news,treasury}_repair_20260707*.parquet` (128 MB) are pre-repair forensic snapshots. Kept deliberately — `AGENTS.md` treats experiment outputs as non-overwritable, and 128 MB in Coldline is $0.0005/month.

> **`gsutil` is dead — don't learn it.** Google's docs state plainly that "gsutil is not the recommended CLI for Cloud Storage." `gcloud storage` is 79–94% faster on downloads, parallelizes automatically with no `-m` flag, and **gsutil is removed from the CLI package after March 2027.** Any tutorial using `gsutil` is teaching a soon-to-be-deleted tool. (Your install lists `gsutil 5.37` — ignore it.)

### 2.4 The real upload — the one genuinely CLI-only step

~108,000 files, ~72 GB, in the order of §2.3b. Work up in size, verifying between each.

```bash
cd /home/luket/repos/CynolycusBot

# The same code/cache exclusion applies to every tree below.
CODE_EX='(.*/)?__pycache__/.*|.*\.pyc?$'

# --- small ones first: these are the ones that were missing, do them while you're paying attention ---

# 82 MB — trained meta-ranker bundles; gitignored, no other copy exists.
# Named files via `cp`, NOT rsync of the repo root — the root also holds .env.
gcloud storage cp meta_ranker_model_bundle_quality.tgz meta_ranker_model_bundle_upside.tgz \
  gs://cynolycusbot-data/model_bundles/

# 332 MB — paper/live audit logs; AGENTS.md forbids losing these
gcloud storage rsync UI/swing_audit gs://cynolycusbot-data/UI/swing_audit --recursive --exclude="$CODE_EX"

# 408 MB — theme outputs, 100% untracked
gcloud storage rsync theme_expansion gs://cynolycusbot-data/theme_expansion --recursive --exclude="$CODE_EX"

# 120 MB — dynamic_theme outputs are read by the live theme path
gcloud storage rsync themes gs://cynolycusbot-data/themes --recursive --exclude="$CODE_EX"

# 169 MB + 29 MB — study datasets and per-run backtest outputs
gcloud storage rsync research  gs://cynolycusbot-data/research  --recursive --exclude="$CODE_EX"
gcloud storage rsync backtests gs://cynolycusbot-data/backtests --recursive --exclude="$CODE_EX"

# --- then the bulk ---

# ~11 GB / 89k files — models, bars cache, options history, live_runs logs
gcloud storage rsync Data gs://cynolycusbot-data/Data --recursive \
  --exclude="$CODE_EX|runtime/live_data_jobs\.lock$|runtime/startup_queue\.json$"

# ~11 GB — news embeddings, catalyst + meta-ranker matrices
gcloud storage rsync signals gs://cynolycusbot-data/signals --recursive --exclude="$CODE_EX"

# ~51 GB — the big one, minus 2.2 GB of verified duplicates
gcloud storage rsync strategies gs://cynolycusbot-data/strategies --recursive \
  --exclude="$CODE_EX|momentum_expansion/data/training_export/training_matrix_4h\.parquet$|momentum_expansion/data/training_export_ablation/ablation_colab_bundle\.tgz$"
```

Add `--dry-run` to any of these first. On the 51 GB line, do it — it costs nothing and shows you the object count before you commit.

> **`--exclude` takes a Python regex, not a glob**, matched against the path **relative to the source directory**. Two traps, both of which the 2026-07-26 version of this section fell into:
> - `*.py` silently won't work; `.*\.pyc?$` will.
> - `.*/__pycache__/.*` does **not** match a top-level `__pycache__/` — the pattern requires a literal `/` before the directory name, and `signals/__pycache__/x.pyc` has relative path `__pycache__/x.pyc` with nothing before it. Both `signals/` and `strategies/` do have a top-level `__pycache__`. Verified by local dry-run: the old pattern leaks any **non-`.pyc`** file there (a stale `.json` came through), and is saved only by the separate `.*\.pyc$` alternative. Today that leak is harmless — 0 of the 315 `__pycache__` files in these trees are non-`.pyc` — but `(.*/)?__pycache__/.*` is the form that is actually correct.

> **Before Phase 2.7 deletes anything**, remember trees 1, 2, 3, 5, 6, 7, 8 contain git-tracked files (§ Your starting state). `git ls-files <dir> | wc -l` must be `0` before you `rm -rf` a directory, or use `git clean -nXd <dir>` to delete only ignored files.

Run the 51 GB transfer in a terminal you can leave open (or under `tmux`) so a disconnect doesn't kill it. **`rsync` is resumable** — re-running transfers only what's missing, so an interruption costs you nothing.

Watch progress live in the Console bucket view; it updates as objects land.

### 2.5 Lifecycle rules — Console

**Bucket → "Lifecycle" tab → "Add a rule"**

Rule 1:
- Action: **Set storage class to Nearline**
- Condition: **Age = 45 days**, **Prefix matches** `strategies/`, `research/`, `backtests/`, `theme_expansion/`

Rule 2:
- Action: **Set storage class to Coldline**
- Condition: **Age = 180 days**, same prefixes

This ages *only* frozen research artifacts. Never `Data/`, which the live modules read daily and must stay Standard.

> **Corrected 2026-07-29 — do not age `signals/` wholesale.** The earlier rule listed `signals/` as a frozen-research prefix. It isn't: a grep of live code shows daily reads and nightly **rewrites** of `signals/news/data/processed/news_records.parquet`, `live_catalyst_records.parquet`, `signals/catalysts/data/processed/catalyst_records.parquet`, and `signals/meta_context/meta_ranker/meta_ranker_matrix.parquet`. Aging those hits both traps named in 2.1 at once — Nearline's **30-day minimum duration** billed against a file rewritten nightly, plus a retrieval fee every time a module reads it. If you want to age part of `signals/`, use narrow prefixes for the frozen subtrees only (e.g. `signals/events/`) and leave `news/`, `catalysts/`, and `meta_context/meta_ranker/` in Standard.

After 180 days the ~52 GB of `strategies/` + the small research prefixes costs about **$0.21/month**; the ~11 GB of `signals/` staying in Standard costs $0.22/month. Total still under $1.

### 2.6 Versioning — Console

**Bucket → "Protection" tab → Object versioning → Enable**

Per `AGENTS.md`: *"Never overwrite raw market data, historical labels, experiment outputs, or live trading logs without explicit approval."* Versioning enforces that mechanically — an overwrite or delete keeps the prior version recoverable.

Later, add a lifecycle rule to delete **noncurrent** versions after ~90 days so they don't accumulate cost.

### 2.7 Reclaim the space — separate session, carefully

**Do not do this the same day as the upload.**

Read-back test first — download one file through the Console (click any object → Download) or:

```bash
gcloud storage cp gs://cynolycusbot-data/Data/shared/bars/SOME_FILE.parquet /tmp/readback.parquet
python3 -c "import pandas as pd; d=pd.read_parquet('/tmp/readback.parquet'); print(d.shape); print(d.head())"
```

Read-back passed 2026-07-30 on `Data/shared/bars/1d/A.parquet` (1,505 × 9, sane OHLCV).

#### 2.7a Full-bucket audit — measured 2026-07-30

A read-back proves *one* object is retrievable. It does not prove the upload was complete, so a per-file comparison was run (`gcloud storage ls -r --long` over the whole bucket vs. a local walk applying the same exclusion regexes):

```
LOCAL   108,369 objects   74,252,571,283 bytes
BUCKET  141,340 objects   78,776,932,997 bytes   (gcloud storage du -s)
```

| Finding | Count | Verdict |
|---|---|---|
| Present, byte-size identical | 95,372 | ✅ |
| **Size differs** (local grew after upload) | 12,966 | ⚠️ stale in bucket — re-sync |
| **Missing** from bucket | 31 | ⚠️ created after upload — re-sync |
| **Extra** in bucket | 33,002 / 4.54 GB | ⚠️ duplicate prefix — delete |

None of this is upload failure. Every one of the 141,340 objects carries a write timestamp of either `2026-07-28` (33,002 — the §2.3 practice run) or `2026-07-30T04:07–04:08Z` (108,338 — the real upload). The bucket is an accurate snapshot of **00:07 ET on 2026-07-30**; the live system has been writing ever since. The 12,966 drifted files are the nightly bar refresh (`Data/shared/bars` 9,172), the momentum feature rebuild (2,850), and the swing rebuild (918); local is larger in 12,961 of them. The 31 missing are files created after 04:07Z (today's swing session audits, today's daily report).

**⚠️ The 4.54 GB "extra" is the prefix collision.** The §2.3 practice upload put `Data/options_history` at the bucket root as `options_history/`, so it is stored twice. Harmless but pointless — delete the top-level copy, **not** the one under `Data/`:

```bash
gcloud storage ls gs://cynolycusbot-data/options_history/ | head   # confirm what you're about to remove
gcloud storage rm --recursive gs://cynolycusbot-data/options_history/
```

**The "Observability" tab lies about size.** It plots `storage.googleapis.com/storage/total_bytes`, which GCS samples **once per day**. On 2026-07-30 it read 4.33 GB — the practice upload alone, sampled before the bulk transfer landed. Trust `gcloud storage du -s` or the bucket details header, never the Observability chart, for "did my upload finish."

#### 2.7b Re-sync, then verify, then delete

Because of the drift above, **re-run the §2.4 rsync commands immediately before deleting anything.** rsync is incremental — it will move only the ~13k changed/missing files, not 72 GB. Then gate each deletion on:

```bash
python3 scripts/verify_gcs_backup.py strategies/multi_ticker_swing/data/training_export
```

It walks the directory, applies the same exclusion rules the upload used, compares every file to the bucket by name and byte size, and exits non-zero if anything is missing, differs, or falls into the trap below.

⚠️ **The `*.py` exclusion is only safe if the source is actually in git.** §2.3b excluded `*.py` on the reasoning that "source belongs in git, not a bucket." That is false for **4 unique files** — Colab training scripts sitting in gitignored export directories, so they are in neither git nor the bucket and the local disk is their only copy on earth:

| File | Also gitignored via |
|---|---|
| `strategies/multi_ticker_swing/data/training_export/{colab_competition,swing_train_colab,oof_ranker_colab}.py` | `.gitignore:238` |
| `Data/processed/spy/training_export/{colab_competition,spy_daytrader_train_colab}.py` | `.gitignore:227` |

(The two `colab_competition.py` copies are identical to each other — `eea9fbb` — but **differ** from the tracked momentum-expansion copy, so that one is not a substitute.) The first of these directories is delete candidate #1.

**Rescue (done 2026-07-30): put them in git, not the bucket.** The equivalent momentum scripts at `strategies/momentum_expansion/data/training_export/*.py` are tracked normally — that directory has no ignore rule — so git is the established home for Colab trainers. Re-inclusion via `!` negation does **not** work here (git cannot re-include a file whose parent directory is excluded, and both `Data/**` and `.../training_export/` exclude the parents), so force-add is the correct mechanism. Once tracked, `.gitignore` no longer applies to them:

```bash
git add -f strategies/multi_ticker_swing/data/training_export/{colab_competition,oof_ranker_colab,swing_train_colab}.py \
           Data/processed/spy/training_export/{colab_competition,spy_daytrader_train_colab}.py
git commit -m "Track Colab training scripts that existed only on local disk"
git push          # not backed up until this succeeds
```

⚠️ **Staged is not backed up.** A `git add`-ed file still lives only on this disk. `verify_gcs_backup.py` checks HEAD (via `git ls-tree`), not the index, and reports `STAGED BUT NOT COMMITTED` as a hard failure for exactly this reason.

⚠️ **Never `rm -rf` a directory without checking for tracked files first.** `Data/` alone holds 4,591 git-tracked files (§ Your starting state); deleting it would gut the working tree and show 4,591 deletions in `git status`. For each candidate:

```bash
git ls-files <dir> | wc -l      # must be 0 before a plain rm -rf
git clean -nXd <dir>            # otherwise: dry-run deleting ONLY gitignored files
git clean -fXd <dir>            # ...then for real
```

Delete **one directory at a time**, checking the relevant module still starts between each. Regenerable derived matrices go first; raw sources and trained models go last, or never.

Safest first deletions — re-checked 2026-07-30 against the bucket:

| Candidate | Size | Verifier | Live readers? | Method |
|---|---|---|---|---|
| `Data/options_history/trades/` | 3.6 GB | ✅ 10,192/10,192 match | none — only `research/options_lab/chain_cache.py` | plain `rm -rf` (and per the 2026-07 retraction, unfit for the study it was collected for) |
| `theme_expansion/outputs/` | 300 MB | ✅ 17/17 match | none — only `scripts/sweep_multiticker_swing_risk_profiles.py` + `scripts/capstone/`; absent from both nightly scripts and `combined_server` | plain `rm -rf` |
| `strategies/multi_ticker_swing/data/training_export/` | 2.0 GB | ✅ after the 2026-07-30 rescue commit | ⚠️ **YES** — see below | **delete one file, not the directory** |
| `strategies/momentum_expansion/data/training_export/` | 1.2 GB | ✅ (1.15 GB of it is the excluded duplicate) | none | `git clean -fXd` only — 2 tracked `.py` here |

🛑 **Corrected 2026-07-30 — do NOT `git clean -fXd` the multi_ticker_swing export directory.** The earlier instruction would have broken live trading. That directory holds two very differently-sized files:

| File | Size | Read by |
|---|---|---|
| `bar_location_context_features.parquet` | **2.09 GB** | offline only — `backtest/build_competition_test_matrix.py`, written by `models/export_for_colab.py` |
| `daily_context_features.parquet` | 25.5 MB | **LIVE** — `live/feature_builder.py:555` builds `DailyContextLookup` from it for `RankerSwingScanner.compute_latest_full()` |

The whole directory is gitignored, so `git clean -fXd` would take *both* — and the live swing ranker would lose its as-of daily context. All the space is in the first file anyway, so delete only that:

```bash
python3 scripts/verify_gcs_backup.py strategies/multi_ticker_swing/data/training_export   # must exit 0
rm -f strategies/multi_ticker_swing/data/training_export/bar_location_context_features.parquet
```

This is precisely the gap `verify_gcs_backup.py` disclaims in its all-clear message: it proves a copy exists in the bucket, **not** that nothing local still reads the path. Always grep for readers before deleting.

> The 1.15 GB `training_export/training_matrix_4h.parquet` is excluded from upload as a duplicate of `data/processed/training_matrix_4h.parquet`. The 2026-07-29 note verified only the first 200 MB; **full-file md5 now confirms identity** (`3ed708deceb55eadadca81dce54e5c47` both). Deleting the export copy is safe as long as the `processed/` copy stays.

> **There is no space emergency.** `df -h /` on 2026-07-30: 1007 GB total, 96 GB used, **860 GB available (11%)**. Phase 2's framing as "reclaim your disk" was written against a smaller volume. Deleting all 74 GB buys 8% of a disk that is 89% empty, so prefer the conservative subset above over aggressive deletion.

#### 2.7c Where deletion stops — measured 2026-07-30

Executed: `Data/options_history/trades` (3.6 GB), `theme_expansion/outputs` (300 MB), `bar_location_context_features.parquet` (2.09 GB). Disk went 96 GB → 91 GB used, **866 GB free**. Remaining trees: `strategies` 49 GB, `signals` 11 GB, `Data` 6.9 GB.

**Only 2.2 GB of the remaining 67 GB is safely deletable today**, and both are verified-redundant files the upload already skipped:

| File | Size | Why redundant |
|---|---|---|
| `strategies/momentum_expansion/data/training_export/training_matrix_4h.parquet` | 1.15 GB | full-file md5 identical to the `processed/` copy, which is in the bucket |
| `.../training_export_ablation/ablation_colab_bundle.tgz` | 1.04 GB | `tar -tzvf` confirms its 4 members all exist uncompressed in the same directory (parquet + manifest in the bucket, both `.py` in git) |

**Everything else must stay, for two distinct reasons:**

1. **Nightly regeneration.** `strategies/momentum_expansion/data/processed/features_4h.parquet` (4.07 GB) is rebuilt every night by `scripts/nightly_data_readiness.sh` step 4/5 (`--build-features --refresh-stale`, 7200 s timeout). Deleting it does not save space — it returns within a day — and it forces a cold rebuild of the exact step that has been getting killed at ~99% (see LIVING_SUMMARY 2026-07-30). Same for `signals/news/data/processed/news_embeddings.parquet` (1.87 GB, nightly `--stage incremental`).
2. **Egress makes re-download cost real money.** `features_30m.parquet` (9.31 GB) and `training_matrix.parquet` (7.2 GB) are genuinely offline-only — `strategies/multi_ticker_swing/main.py:168` lists them as pipeline artifacts, and the live path computes features through `live/feature_builder.py` instead. But pulling 16.5 GB back out of GCS costs ~$0.12/GB ≈ **$2 plus the download time**, to free 1.6% of a disk that is 90% empty. Bad trade. Delete these only if you actually need the space.

> **The bucket is a point-in-time backup, not a live mirror.** Files the nightly pipeline rewrites (`features_4h`, `news_records`, `news_embeddings`, `Data/shared/bars`) drift out of sync within 24 h — that is exactly the 12,966-file drift §2.7a measured. Re-run the §2.4 rsync on a schedule if you want the bucket current; automating it is a natural first Cloud Run Job in Phase 5.

**The real blocker on deleting the bulk is that `combined_server.py` still runs locally and reads these paths directly.** Local deletion beyond the list above is not a Phase 2 task — it becomes possible only once the code reads from GCS, which is Phase 4/8. Do not force it here.

✅ **Checkpoint 2**

- `gcloud storage du -s gs://cynolycusbot-data` shows ~73 GiB (the Observability tab will lag a day)
- `scripts/verify_gcs_backup.py` exits 0 for every directory you intend to delete
- Lifecycle and versioning both visible on their tabs
- Billing still near $0

**What you learned:** buckets, storage classes, lifecycle rules, uniform access, versioning, and the egress/minimum-duration traps. That's most of GCP storage.

📝 **Notes:**

---

## Phase 3 — Secret Manager (30 min · 100% Console)

Your `.env` holds 5 API credentials including **live-trading keys**. Those can never enter a container image or an env-var field.

Free tier: **6 active secret versions**, **10,000 access operations**/month. You have slightly more than 6 keys — expect a few cents. Fine.

### 3.1 Create secrets — Console

**Console → Security → Secret Manager → "Create secret"**

For each, paste the value from `.env` into the **Secret value** box:

| Secret name | From `.env` |
|---|---|
| `alpaca-paper-key-id` | `APCA_API_KEY_ID` |
| `alpaca-paper-secret` | `APCA_API_SECRET_KEY` |
| `tiingo-key` | `TIINGO_API_KEY` |
| `finnhub-key` | `FINNHUB_API_KEY` |

Leave replication as **Automatic**.

> **Watch for trailing whitespace.** Pasting a value that picks up a newline or space produces baffling 401s later. Select precisely, and use the eye icon to re-check after saving.

> **Deliberately excluded:** `APCA_API_KEY_ID_LIVE` and `APCA_API_SECRET_KEY_LIVE`. These do not go to the cloud until Phase 8, and only with a decision recorded in §12.

### 3.2 Create the runtime service account — Console

**Console → IAM & Admin → Service Accounts → "Create service account"**

- Name: `cynolycus-runtime`
- Description: `CynolycusBot runtime identity`
- **Skip the "Grant this service account access to project" step entirely** — click Continue, then Done.

> That skip is the whole point. The Console *invites* you to grant a project-wide role right there. Don't. Project-wide grants are how service accounts end up with far more power than they need. You'll grant narrow, per-resource access in 3.3 instead.

Copy the full email — it looks like `cynolycus-runtime@YOUR-PROJECT-ID.iam.gserviceaccount.com`. Record it in §12.

### 3.3 Grant least privilege — per resource, not per project

**Secrets:** For each of the 4 secrets — **Secret Manager → click the secret → "Permissions" tab → "Grant access"**
- Principal: your `cynolycus-runtime@...` email
- Role: **Secret Manager Secret Accessor**

**Bucket:** **Cloud Storage → cynolycusbot-data → "Permissions" tab → "Grant access"**
- Principal: same email
- Role: **Storage Object User** (read + write objects)

> Use **Storage Object Viewer** for anything that should only read. The Phase 6 dashboard gets Viewer, never User.

**Never** grant `Owner` or `Editor` to a service account. That's the single most common real-world GCP security failure.

### 3.4 Never download a service account key

You'll find many tutorials saying to create and download a JSON key. **Don't.** A downloaded key is a permanent credential that leaks into git, logs, and backups.

Cloud Run *attaches* the service account to the workload — your code gets credentials automatically from the metadata server, with no file anywhere. You'll select it from a dropdown in Phase 5.

✅ **Checkpoint 3** — 4 secrets listed; service account exists; each secret's Permissions tab and the bucket's Permissions tab show `cynolycus-runtime`.

📝 **Notes:**

---

## Phase 4 — Connect GitHub and deploy (1 hour · 100% Console)

Because your repo is already on GitHub, the Console gives you something better than the CLI flow: **push to `main` → automatic rebuild and redeploy.**

### 4.1 What happens on deploy

```
git push origin main
   ▼
Cloud Build          builds the container in the cloud (no local Docker)
   ▼
Artifact Registry    us-central1-docker.pkg.dev/PROJECT/...
   ▼
Cloud Run revision   immutable, versioned, rollback-able
```

> **Container Registry (`gcr.io`) is fully shut down** — writes ended March 2025, reads June 2025. **Artifact Registry is the only option.** Ignore any tutorial mentioning `gcr.io`.

> **Behavioral difference from the CLI flow:** Cloud Build builds **what you pushed to GitHub**, not what's on your disk. Uncommitted local changes won't deploy. This is a feature — deploys become reproducible — but it will confuse you once.

### 4.2 Commit the Dockerfile first

Nothing can build until these two files are pushed. Create `Dockerfile` at the repo root:

```dockerfile
FROM python:3.12-slim

# TA-Lib needs the C library built from source; numba/torch need build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential wget ca-certificates \
 && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
 && tar -xzf ta-lib-0.6.4-src.tar.gz \
 && cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && make install \
 && cd .. && rm -rf ta-lib-0.6.4* \
 && apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so code edits don't retrigger a 10-minute rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
```

And `.dockerignore` — **not optional**, it's what stops you shipping 86 GB to Cloud Build:

```
.venv/
.git/
Data/
logs/
local_artifacts/
backtests/
reports/
**/__pycache__/
**/*.parquet
**/*.pyc
**/*.joblib
*.tgz
.env
.pytest_cache/
.worktrees/
```

> **`torch` pulls ~2.5 GB of CUDA libraries you'll never use on Cloud Run (CPU-only).** Switching to the CPU wheel cuts image size by more than half and saves minutes per build:
> ```
> --extra-index-url https://download.pytorch.org/whl/cpu
> torch
> ```
> Verify nothing in the live path calls `.cuda()` before doing this.

Commit and push both.

### 4.3 Give yourself the required roles

**Console → IAM & Admin → IAM**, confirm your own account has: **Cloud Build Editor**, **Cloud Run Developer**, **Artifact Registry Administrator**. As project creator you're likely Owner, which covers all three.

### 4.4 Connect the repo

**Console → Cloud Run → "Deploy container" → "Continuously deploy from a repository"** (wording varies: "Connect repository" / "Set up with Cloud Build")

1. Provider: **GitHub**
2. Click **Authenticate** → authorize the **Cloud Build GitHub App**
3. Select `ThompsonLuke7/CynolycusBot`. Not listed? → **"Manage connected repositories"** and grant access.
4. Branch: `^main$` (it's a regex)
5. Build type: **Dockerfile**, location `/Dockerfile`
6. **Save**

### 4.5 First deploy

Complete the service form: name `hello-test`, region `us-central1`, **Require authentication** (never "Allow unauthenticated" yet), then **Create**.

Watch the build stream in **Cloud Build → History**. The first build takes 10–20 minutes — TA-Lib compiles from source and torch is large. Subsequent builds reuse cached layers and are much faster.

If it fails, the log tells you which line of the Dockerfile broke. Read it top-down; the first error is the real one.

### 4.6 Understand this, then delete it

**A Cloud Run *service* must listen on `$PORT` (default 8080) and answer HTTP.** Cloud Run decides your container is healthy by connecting to that port. This is precisely why `UI/combined_server.py` can't be lifted into a service as-is — see Phase 8.

Then **Cloud Run → hello-test → Delete**.

✅ **Checkpoint 4** — a build succeeded, a revision deployed, and you understand the pipeline. Also click **"Equivalent command line"** on a Cloud Run form once, just to see the mapping.

📝 **Notes:**

---

## Phase 5 — Cloud Run Jobs: automated nightly pipeline (2–3 hrs · 100% Console)

Where "runs automatically and continuously" becomes real.

### 5.1 Services vs. Jobs

| | Cloud Run **Service** | Cloud Run **Job** |
|---|---|---|
| Trigger | an HTTP request | you run it (or a scheduler does) |
| Must serve a port? | **yes** | **no** |
| Ends? | stays warm, scales to zero | **runs to completion and exits** |
| Max duration | 60 min request timeout | **168 hours (7 days)** |
| Your... | dashboards | nightly pipeline, module scoring |

Your nightly work is batch that exits. **It's a Job.** This maps onto `_run_nightly_jobs` in [UI/combined_server.py](UI/combined_server.py) — today driven by an in-process `NightlyScheduler` thread that only fires if your laptop is awake and un-crashed.

### 5.2 Limits to design around

Verified Cloud Run Job limits:

- **Max memory 32 GiB**, max **8 vCPU**
- CPU↔memory pairing is constrained: 1 vCPU → ≤4 GiB · 2 → ≤8 GiB · 4 → ≤16 GiB · 6 → ≤24 GiB · **8 → ≤32 GiB**
- **Default task timeout is 10 minutes** — far too short, you must raise it

> **This directly addresses your WSL OOM history.** The 2026-07-21 crash came from a Python process ballooning to 2.2 GB reading a 4.0 GB `features_4h.parquet` inside a 16 GB WSL cap. A job at 8 vCPU / 32 GiB doubles that ceiling, isolated per-run, and dies cleanly instead of taking the whole machine down. Your chunked-parquet reader stays valuable — it just has more headroom now.

### 5.3 Write a cloud entrypoint

Don't cloud-ify `combined_server.py` yet. Add a thin single-purpose entrypoint — `scripts/cloud/run_nightly.py`:

```python
"""Cloud Run Job entrypoint: nightly data collection/enrichment.

Runs once and exits. Reads/writes datasets under CYNOLYCUS_DATA_ROOT.
Exits non-zero on failure so Cloud Run marks the execution failed and
Monitoring can alert on it.
"""
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nightly")


def main() -> int:
    data_root = os.environ["CYNOLYCUS_DATA_ROOT"]   # fail fast if unset
    log.info("nightly start | data_root=%s", data_root)

    # TODO: call the same functions UI/combined_server.py:_run_nightly_jobs calls.
    # Import them directly — do NOT duplicate the logic (AGENTS.md: no competing
    # implementations, research/live parity).

    log.info("nightly complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Commit and push.

### 5.4 Create the job — Console

**Console → Cloud Run → "Jobs" tab → "Deploy container" → "Continuously deploy from a repository"** (same GitHub connection as Phase 4)

**General:**
- Job name: `nightly-pipeline`
- Region: `us-central1`
- Tasks: `1`

Expand **"Containers, Connections, Security"**:

**Container tab:**
- Container command: `python`
- Container arguments: `scripts/cloud/run_nightly.py`
- Memory: `32 GiB` · CPU: `8`
- Task timeout: `7200` seconds
- Max retries: `1`

**Variables & Secrets tab:**
- Environment variable: `CYNOLYCUS_DATA_ROOT` = `/mnt/data`
- **"Reference a secret"** for each — pick the secret, version `latest`, expose **as environment variable**:

| Env var | Secret |
|---|---|
| `APCA_API_KEY_ID` | `alpaca-paper-key-id` |
| `APCA_API_SECRET_KEY` | `alpaca-paper-secret` |
| `TIINGO_API_KEY` | `tiingo-key` |
| `FINNHUB_API_KEY` | `finnhub-key` |

Secrets arrive as ordinary env vars — your existing `os.environ` reads work unchanged.

**Security tab:**
- Service account: **`cynolycus-runtime`** ← this is what replaces downloaded key files

**Volumes tab** — this is the big shortcut:
1. **"Mount volume"** (or Add volume)
2. Volume type: **Cloud Storage bucket**
3. Mount path: `/mnt/data`
4. Browse and select `cynolycusbot-data`
5. Leave **Read-only unchecked** (the nightly job writes)
6. **Save**

Your bucket now appears as an ordinary directory at `/mnt/data`. **No code changes to any path-handling logic.**

Click **Create**.

> ⚠️ **FUSE caveat that matters for you:** Cloud Storage FUSE has **no concurrency control for concurrent writes — last write wins, earlier writes are silently lost.** It's also not fully POSIX-compliant. Never run two jobs writing the same parquet at once, and don't expect file locking, atomic rename, or in-place partial updates to work. For anything needing atomicity, write a unique temp key and use the Python client's object operations.

### 5.5 Run it manually before scheduling

**Cloud Run → Jobs → nightly-pipeline → "Execute"**

Watch the **Executions** tab, then click into the execution → **Logs**.

✅ **Checkpoint 5.5** — execution shows **Succeeded** and your own log lines are visible. Do not proceed until a manual run is green.

### 5.6 Schedule it — Console

**Cloud Run → Jobs → nightly-pipeline → "Triggers" tab → "Add Scheduler Trigger"**

- Frequency: `30 20 * * 1-5` (20:30, Mon–Fri, unix-cron)
- **Timezone: `America/New_York`** ← essential
- Service account: accept the suggested compute service account

> **The timezone field is not cosmetic.** It makes the schedule follow EDT/EST automatically — exactly the market-calendar correctness `AGENTS.md` demands. Leave it UTC and you silently drift an hour twice a year, firing your nightly job during market hours half the year.

Test-fire immediately via **Cloud Scheduler → your job → "Force run"** rather than waiting for 20:30.

Pricing: **$0.10 per job per 31 days, 3 free per billing account.** CynolycusBot has many scheduled times (Meta 14:35/16:35, HTF, momentum, dealer ranker, flushes) so you'll pass 3 quickly. 10 jobs ≈ $0.70/month.

✅ **Checkpoint 5** — a scheduler-triggered execution appears, distinct from your manual one.

📝 **Notes:**

---

## Phase 6 — Dashboards other people can open (2 hrs · 100% Console)

### 6.1 Design decision: read-only

Your local dashboards both **display** state and **drive trading sessions** (the "Run" button). Don't port that. The cloud dashboard is a **read-only view** of state written to GCS by the jobs.

Right call for three independent reasons: it's the `AGENTS.md` separation of concerns; a viewer can never trigger an order; and a stateless read-only service scales to zero and costs nothing.

### 6.2 Deploy — Console

**Cloud Run → "Services" tab → "Deploy container" → continuous deploy from your repo**

- Name: `cynolycus-dashboard` · Region `us-central1`
- **Authentication: Require authentication**
- CPU `1` · Memory `2 GiB`
- **Min instances `0`** · **Max instances `3`**
- Container command `python`, arguments `-m,UI.hub_dashboard`
- Env var `CYNOLYCUS_DATA_ROOT` = `/mnt/data`
- Security → service account: `cynolycus-runtime`
- Volumes → Cloud Storage bucket `cynolycusbot-data`, mount `/mnt/data`, **✅ Read-only checked**

Three settings doing real work:

- **Max instances 3** — your runaway-cost circuit breaker. Always set it.
- **Min instances 0** — scale to zero; you pay nothing when nobody's looking.
- **Read-only volume** — the dashboard *cannot* corrupt your data even if compromised.

### 6.3 Share it — Console

**Cloud Run → cynolycus-dashboard → "Security" tab (or Permissions) → "Add principal"**

- New principals: `colleague@example.com`
- Role: **Cloud Run Invoker**

They sign in with their Google account. You revoke by removing the principal. No passwords, no shared secrets, full audit trail in Cloud Logging.

> **The alternative — "Allow unauthenticated invocations" — publishes your trading system's positions and signals to the entire internet.** Given this dashboard shows live positions, the answer is no. For viewers without Google accounts, put **Identity-Aware Proxy (IAP)** in front of it later. Don't build custom auth.

### 6.4 Cost

Free tier: **2M requests, 180,000 vCPU-seconds, 360,000 GiB-seconds/month.** A few people checking a dashboard daily won't come close. Expect **$0.00**.

Beyond free tier, roughly **$0.000024/vCPU-second and $0.0000025/GiB-second** in us-central1 **[verify]**.

✅ **Checkpoint 6** — you open the URL signed in and see real data; a signed-out/incognito browser gets 403; one other person can open it.

📝 **Notes:**

---

## Phase 7 — Operations (1 hr · 100% Console — and the Console is better here)

A deployed system with no monitoring is a car with no dashboard.

### 7.1 Logs

**Console → Logging → Logs Explorer.** Use the query box:

```
resource.type="cloud_run_job" AND severity>=ERROR
```

The Console beats the CLI here: clickable severity filters, a histogram showing *when* errors clustered, and one-click jumps from an error to surrounding context.

Free tier: **first 50 GiB of logs per project per month.** Your `logs/` dir is 32 MB, so you're fine — unless something retry-loops and floods. Which the next step catches.

### 7.2 Two alerts

**Console → Monitoring → Alerting → "Create policy"**

1. **Job failed** — log-based condition on `resource.type="cloud_run_job" AND severity=ERROR`, notify by email.
2. **Job didn't run at all** — the one people forget. A job that silently stops firing looks exactly like a healthy quiet night. Alert on the *absence* of a success log within a window.

That second alarm is what would have told you your live server was dead during the WSL crashes, instead of you reconstructing crash times from per-module audit-write timestamps afterward.

### 7.3 Check the bill weekly by hand, first month

**Console → Billing → Reports**, grouped by service. Look for anything unfamiliar. Budget alerts are a backstop, not a substitute for looking.

### 7.4 Standing cost hygiene

- **Max instances** on every service, always
- Never leave a Compute Engine VM running "just to test"
- Never attach a GPU without a reason and a hard stop time
- Delete unattached persistent disks — they bill with or without a VM
- Delete old Artifact Registry images (4 GB × 20 revisions = 80 GB)

Sweep periodically: **Cloud Run → Services / Jobs · Compute Engine → VM instances / Disks · Artifact Registry → Repositories**

**Artifact Registry → your repo → "Cleanup policies"** can auto-delete old images. Set it once: keep the 5 most recent, delete the rest.

📝 **Notes:**

---

## Phase 8 — The live trading loop (DO NOT START UNTIL 0–7 ARE DONE)

The genuinely hard part, and the only one where mistakes cost real money.

### 8.1 Why `combined_server.py` doesn't lift-and-shift

It's a single always-on process that:

- holds **one persistent Alpaca WebSocket** (IEX free tier allows exactly one concurrent stream)
- runs ~10 in-process `NightlyScheduler` threads at fixed ET times
- serves 4+ HTTP dashboards on ports 8765/8766/8768/8773
- fights RSS growth with `malloc_trim` across a ~900-ticker all-day churn
- keeps mutable `live_state.json` per module

A Cloud Run **service** is request-driven and can be evicted between requests — bad fit for a persistent WebSocket consumer. A Cloud Run **job** exits — wrong shape for a continuous stream.

### 8.2 The three real options

| Option | Fit | Cost/mo | Verdict |
|---|---|---|---|
| **Compute Engine VM** (`e2-standard-2`, 2 vCPU/8 GB) | Lift-and-shift. It's a Linux box — run what you run now, under `systemd` so it restarts on crash. | ~$49 **[verify]** | **Recommended first.** Zero architectural risk; solves the WSL-crash problem outright. |
| **Cloud Run worker pools** | **GA April 2026.** Purpose-built for always-on non-request workloads; ~40% cheaper than services/jobs for long-running background work; supports GCS volume mounts. | lower | **Right long-term target.** Needs the stream consumer restructured around pull-based work. |
| **Decompose into Jobs** | Each module's scoring loop becomes a scheduled Job; the WebSocket becomes a separate always-on component. | lowest | Cleanest architecture, most work, most risk of research/live parity drift. |

Recommendation: **VM first** — it makes the system reliable immediately, which is the actual pain — then migrate to worker pools once the data layer is proven.

Console path: **Compute Engine → VM instances → Create instance**, `e2-standard-2`, `us-central1`, Debian 12. Then SSH **from the browser** (the SSH button in the Console — no key management needed).

### 8.3 Non-negotiable gates before any live cloud trading

1. **Paper mode only**, minimum two full weeks, fills compared against the local paper record.
2. **Live Alpaca keys stay out of the cloud** until 1 is complete and the result is written in §12.
3. **Exactly one instance may hold the Alpaca stream.** Two processes = exceeded stream limit = both degrade. This is a correctness constraint, not a cost one.
4. **The cloud must not duplicate orders your local server is placing.** Decide explicitly which is authoritative and shut the other's order path off. Two live servers submitting the same signal is the worst outcome available here.
5. Verify ET-timezone behavior of every scheduled fire across a DST boundary before trusting it.

**Do not begin Phase 8 in the same session you finish Phase 7.**

📝 **Notes:**

---

## 9. Expected total cost

| Item | Monthly |
|---|---|
| Cloud Storage — 72 GB aging to Nearline/Coldline | $0.30 – $1.45 |
| Cloud Run jobs (nightly ~1 hr/day, 8 vCPU/32 GiB) | $3 – $8 |
| Cloud Run dashboard (scale-to-zero) | ~$0.00 (free tier) |
| Cloud Scheduler (~10 jobs, 3 free) | ~$0.70 |
| Secret Manager (~6 secrets) | ~$0.06 |
| Artifact Registry (~4 GB images) | ~$0.40 **[verify]** |
| Cloud Logging | ~$0.00 (50 GiB free) |
| **Phases 0–7 subtotal** | **≈ $5 – $10/month** |
| *Phase 8: always-on `e2-standard-2`* | *+$49 **[verify]*** |

Your $300 trial credit covers roughly **3–5 months of the full stack including the VM**, or over a year of Phases 0–7.

---

## 10. Console navigation cheat sheet

| I want to… | Console path |
|---|---|
| Switch project | Top blue bar → project dropdown |
| See spend | Billing → Reports |
| Check/edit budget | Billing → Budgets & alerts |
| Browse my data | Cloud Storage → Buckets → cynolycusbot-data |
| Change storage aging | Bucket → Lifecycle |
| Add/rotate an API key | Secret Manager → secret → New version |
| See who can access what | IAM & Admin → IAM (or a resource's Permissions tab) |
| Run a job now | Cloud Run → Jobs → job → Execute |
| See why a job failed | Cloud Run → job → Executions → click → Logs |
| Change a schedule | Cloud Scheduler → job → Edit |
| Roll back a bad deploy | Cloud Run → service → Revisions → Manage traffic → point 100% at the previous revision |
| See a build's output | Cloud Build → History |
| Search all logs | Logging → Logs Explorer |
| Delete unused stuff | Cloud Run → Services/Jobs · Compute Engine → VM instances/Disks |

**Rollback is worth a dry run once.** Cloud Run revisions are immutable, so shifting 100% of traffic to the previous revision is instant and safe — but you don't want the first time you try it to be during an incident.

---

## 11. Appendix — CLI equivalents (optional)

Not needed for the Console path. Useful when you want to script something, or when a doc/StackOverflow answer is CLI-only.

```bash
# Context
gcloud config list
gcloud config set project YOUR-PROJECT-ID

# Storage
gcloud storage ls gs://cynolycusbot-data
gcloud storage du --summarize gs://cynolycusbot-data
gcloud storage rsync LOCAL gs://cynolycusbot-data/PATH --recursive --dry-run
gcloud storage cp gs://cynolycusbot-data/PATH/file.parquet .

# Run
gcloud run jobs execute JOB --region=us-central1 --wait
gcloud run services list --region=us-central1
gcloud run revisions list --service=SVC --region=us-central1
gcloud run services update-traffic SVC --to-revisions=REV=100 --region=us-central1

# Secrets
gcloud secrets versions access latest --secret=NAME
gcloud secrets versions add NAME --data-file=-        # rotate

# Scheduler
gcloud scheduler jobs run NAME --location=us-central1
gcloud scheduler jobs pause NAME --location=us-central1

# Logs
gcloud logging read 'severity>=ERROR' --freshness=1d --limit=50

# IAM introspection
gcloud storage buckets get-iam-policy gs://cynolycusbot-data
```

**Deliberately out of scope** until something forces it: Kubernetes/GKE, Terraform, custom VPC, load balancers, multi-region, Anthos, service meshes, org policy.

**Reconsider Terraform after Phase 7** — once you can build these by hand and know what each does, encoding them as code is genuinely valuable. Not before. **Reconsider BigQuery** if you start running analytical SQL across full multi-year feature matrices (1 TiB/month free) — but parquet-on-GCS read by pandas is simpler and probably sufficient.

---

## 12. Running notes & decision log

> Append as you go. Date every entry. Record what worked, what broke, exact error text, decisions and why.

### Decisions made

| Date | Decision | Reason |
|---|---|---|
| 2026-07-26 | Region `us-central1` everywhere | Only us-central1/east1/west1 qualify for the GCS Always Free tier; standard-price reference region |
| 2026-07-26 | Storage before compute | Reversible, ~$1/mo, solves the stated disk pain, touches no code |
| 2026-07-26 | Live Alpaca keys excluded until Phase 8 | `AGENTS.md`: default to paper; live paths separated by credentials |
| 2026-07-26 | Cloud dashboard read-only, no "Run" button | Prevents a viewer triggering orders; enables scale-to-zero |
| 2026-07-26 | **Console-first; CLI only for bulk upload** | ~108,000 files defeats browser upload; everything else is clearer in the UI while learning |
| 2026-07-26 | **GitHub continuous deployment over `--source .`** | Repo already on GitHub; deploys become reproducible from pushed commits |
| 2026-07-26 | **Object storage, not a database** (§2.0) | Cloud SQL ≈ $129/mo vs ~$1.28 for GCS; parquet already is a columnar store; immutable files beat mutable rows for reproducibility |
| 2026-07-26 | **BigQuery deferred, external tables if ever** (§2.0) | Queries parquet in place — no copy, no migration, 1 TiB/mo free. Revisit only if ad-hoc SQL becomes painful |
| 2026-07-26 | **float32 + zstd re-encoding = separate change** (§2.0b) | Size cut measured, but dtype changes can shift model outputs; needs frozen-test validation, must not ride along with the migration |
| 2026-07-29 | **Phase 2.4 upload set re-verified from the filesystem** (§2.3b) | Added 6 missing sources (`theme_expansion/`, `themes/`, `UI/swing_audit/`, `research/`, `backtests/`, root model bundles — ~1.1 GB incl. the only copies of live theme features, all paper-trade audit logs, 2 trained bundles); dropped 2.2 GB of verified duplicates; fixed a 4.4 GB double-upload from a 2.3/2.4 prefix mismatch; corrected the false "no tracked files" claim (5,124 tracked files sit inside the upload trees) |
| 2026-07-29 | **`signals/` removed from the lifecycle-aging rule** (§2.5) | Live code reads and nightly-rewrites `signals/news`, `signals/catalysts`, `signals/meta_context/meta_ranker` — Nearline's 30-day minimum duration plus retrieval fees would apply to daily-churn files |
| 2026-07-28 | **Re-encoding downgraded to low priority** (§2.0b) | Repo-wide audit: only 37.8% of parquet bytes are float64 (47.9% already float32), so realistic saving is ~11 GB / 27%, not the 46.6% one unrepresentative file implied. Worth doing eventually; not worth blocking the migration |

### Phase log

**Phase 0 — Prerequisites** · ✅ **complete 2026-07-26**
```
gcloud 577.0.0 installed via apt (google-cloud-cli), which gcloud -> /usr/bin/gcloud.
Windows SDK 513.0.0 still on PATH below /usr/bin; harmless, left in place.
Docker deliberately not installed — Cloud Build handles image builds.
TODO: gcloud auth login + gcloud auth application-default login
```

**Phase 1 — Project & budget** · ✅ **complete** (verified 2026-07-30)
```
Dev project ID (exact, from Console):  cynolycusbot-dev   (473199336957) — gcloud active project
Prod project ID:                       cynolycusbot       (806989645664)
Billing account:                       (linked — bucket + APIs are live)
Budget amount / thresholds:            CONFIRM IN CONSOLE — §1.4 says do not skip
```

**Phase 2 — Cloud Storage** · ✅ **complete 2026-07-30**
```
Bucket name (exact):   cynolycusbot-data   (US-EAST5, uniform access, public access prevented)
Protection:            object versioning ENABLED; soft-delete 7 days;
                       lifecycle = Nearline@45d / Coldline@180d on
                       strategies|research|backtests|theme_expansion,
                       + delete noncurrent versions @90d / >3 newer
Uploaded:              108,338 objects @ 2026-07-30T04:07-04:08Z (the §2.4 run)
                     +  33,002 objects @ 2026-07-28 (the §2.3 practice run, DUPLICATE)
                     = 141,340 objects / 78,776,932,997 bytes (73.4 GiB)
Read-back verified:    y — Data/shared/bars/1d/A.parquet, 1505x9, sane OHLCV
Full audit:            y — see §2.7a. 31 missing + 12,966 stale (both = post-upload
                       drift, fixed by re-running rsync); 4.54 GB duplicate prefix
                       `options_history/` to delete.
Duplicate prefix:      deleted — `options_history/` (33,002 objs / 4.54 GB) removed 2026-07-30
Re-sync after drift:   done — Data/shared/bars back to 9,885/9,885 byte-matched
Local GB reclaimed:    5.9 GB (96G -> 91G used, 866G free). Stops here by design — see
                       §2.7c: only 2.2 GB more is safely deletable, the rest is either
                       rebuilt nightly or costs egress to retrieve, and combined_server
                       still reads these paths locally until Phase 4/8.
```

**Phase 3 — Secret Manager** · ☐ not started
```
Secrets created:
alpaca-paper-key-id		
alpaca-paper-secret
finnhub-key
tiingo-key
Service account email: cynolycus-runtime@cynolycusbot-dev.iam.gserviceaccount.com
```

**Phase 4 — GitHub connect & first deploy** · ☐ not started
```
Dockerfile committed (y/n):
First build duration:
Image size:
```

**Phase 5 — Jobs & Scheduler** · ☐ not started
```
Job memory/CPU:
First green execution:
Schedule + timezone:
```

**Phase 6 — Dashboard** · ☐ not started
```
Service URL:
People granted Invoker:
```

**Phase 7 — Operations** · ☐ not started
```
Alerts configured:
First month actual cost:
```

**Phase 8 — Live loop** · ☐ BLOCKED until 0–7 complete
```
Paper-mode start date:
Two-week comparison result:
Live authorization decision:
```

### Gotchas hit

| Date | Symptom | Cause | Fix |
|---|---|---|---|
| 2026-07-30 | Observability tab showed **4.33 GB** after a 72 GB upload | `storage/total_bytes` is a **once-daily** Cloud Monitoring sample; it still reflected the §2.3 practice upload | Use `gcloud storage du -s` (returned 73.4 GiB) or the bucket details header; ignore the Observability chart for completeness checks |
| 2026-07-30 | Bucket held 33,002 objects / 4.54 GB more than the manifest | §2.3's practice run wrote `Data/options_history` to the bucket **root** as `options_history/`, storing it twice | `gcloud storage rm --recursive gs://cynolycusbot-data/options_history/` (keep `Data/options_history/`) |
| 2026-07-30 | 12,966 objects size-mismatched, 31 absent | Not upload failure — the bucket is a 00:07 ET snapshot and the live system kept writing (nightly bars, feature rebuilds, session audits) | Re-run the §2.4 rsync right before any deletion; it is incremental |
| 2026-07-30 | 4 Colab training scripts were in neither git nor the bucket | §2.3b excluded `*.py` assuming "source belongs in git", but these live in **gitignored** export dirs | Rescue before deleting; `scripts/verify_gcs_backup.py` now hard-fails on this class |

---

## 13. Sources

Verified 2026-07-26:

- [Google Cloud Free Program](https://docs.cloud.google.com/free/docs/free-cloud-features) — Always Free limits
- [Google Cloud Free Trial](https://cloud.google.com/free) — $300 / 90 days
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
- [Cloud Run job memory limits](https://docs.cloud.google.com/run/docs/configuring/jobs/memory-limits) — 32 GiB max, CPU/memory ratios
- [Execute Cloud Run jobs on a schedule](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule) — 168 h max; Triggers tab → Add Scheduler Trigger
- [Cloud Storage volume mounts for jobs](https://docs.cloud.google.com/run/docs/configuring/jobs/cloud-storage-volume-mounts) — Console steps & FUSE concurrency caveat
- [Continuous deployment from GitHub](https://docs.cloud.google.com/run/docs/continuous-deployment-with-cloud-build) — Console flow, required roles
- [Cloud Scheduler pricing](https://cloud.google.com/scheduler/pricing) — $0.10/job/31 days, 3 free
- [Cloud Storage pricing](https://cloud.google.com/storage/pricing)
- [Transition from gsutil to gcloud storage](https://docs.cloud.google.com/storage/docs/gsutil-transition-to-gcloud) — removal after March 2027
- [Container Registry deprecation](https://docs.cloud.google.com/artifact-registry/docs/transition/prepare-gcr-shutdown)
- [Install gcloud CLI](https://docs.cloud.google.com/sdk/docs/install) — current 577.0.0
- [Deploy worker pools to Cloud Run](https://docs.cloud.google.com/run/docs/deploy-worker-pools) — GA April 2026

Items marked **[verify]** (Compute Engine VM pricing, Artifact Registry per-GB, Cloud Run per-second rates) come from third-party aggregators — confirm against the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) before relying on them.
