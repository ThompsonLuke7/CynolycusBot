# CynolycusBot → Google Cloud Platform: Console-First Migration Tutorial

**Created:** 2026-07-26 · **Rewritten Console-first:** 2026-07-26
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

**The one unavoidable CLI step** is Phase 2.4: uploading ~64 GB across **93,460 files**. The Console's uploader is a browser drag-and-drop — it will stall, time out, or silently drop files at that scale, and it can't resume. `gcloud storage rsync` is resumable, parallel, and verifies checksums. That's a 20-minute exception in an otherwise click-driven migration.

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

### Your starting state (measured 2026-07-26)

```
Repo total:                78 GB
  strategies/              48 GB   ← gitignored .parquet feature/training matrices
  signals/                9.9 GB   ← news embeddings, catalyst matrices
  Data/                   6.0 GB   ← models, shared bars cache, dealer snapshots
  everything else         ~14 GB   ← .venv, .git, UI, research, themes

Files to upload:        93,460
GitHub remote:          ThompsonLuke7/CynolycusBot (branch main)
```

`.gitignore` already excludes `Data/**` and `*.parquet`, so **~64 GB is untracked generated artifacts** — exactly what Cloud Storage is for. Nothing in Phase 2 touches tracked source code.

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
                                              Cloud Storage  gs://cynolycus-data/
                                                   ├── Data/  ├── signals/  └── strategies/
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

Cost for your 64 GB:

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

**Repo-wide audit (2026-07-28) — the prize is smaller than that one file suggests.** Footer scan of all 142 parquet files >10 MB (41.0 GB of the ~64 GB):

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

> The remaining **38,673 parquet files under 10 MB (23.7 GB)** weren't footer-scanned. Their file *count* matters more than their dtypes: 93,460 objects makes `rsync` slow and burns Class A operations (free tier is 5,000/month; the overage is cents, but expect to exceed it on first upload).

⚠️ **Validate before adopting — do not bulk-convert.**
- Technical features (ATR, returns, cross-sectional ranks) don't need float64's 15 significant digits; float32 gives ~7. But dtype changes can shift model outputs.
- **XGBoost's `DMatrix` is float32 internally**, so for XGBoost models the downcast already happens and costs nothing. sklearn and torch paths need checking.
- `strategies/multi_ticker_swing/data/processed/features_30m.parquet` is *already* float32 — there the lever is zstd alone (~12%), not the downcast. The audit shows this is true of **47.9% of all your parquet bytes**, so target the float64 files specifically rather than converting broadly.
- Compare against a frozen-test baseline before changing any pipeline. **Treat this as a separate change from the migration** — don't bundle it into Phase 2.

### 2.1 The cost math

Verified `us-central1` regional prices (2026-07-26):

| Class | $/GB-month | Min. duration | Retrieval fee | **78 GB costs** |
|---|---|---|---|---|
| Standard | $0.020 | none | none | **$1.56/mo** |
| Nearline | $0.010 | 30 days | $0.01/GB | **$0.78/mo** |
| Coldline | $0.004 | 90 days | $0.02/GB | **$0.31/mo** |
| Archive | $0.0012 | 365 days | $0.05/GB | **$0.09/mo** |

Free tier: **5 GB-months Standard** (US regions), 5,000 Class A ops, 50,000 Class B ops, 100 GB North America egress/month.

Your entire 78 GB problem costs **about a coffee per year** in Coldline, or **$19/year** in Standard. Highest-leverage item in this document.

> **Minimum storage duration is a real trap.** An object written to Nearline and deleted after 3 days is billed for the full 30. Keep actively-rewritten data (live state, daily caches) in Standard; let lifecycle rules age *cold* things down.

> **Egress is the other trap.** Reading a bucket *into Cloud Run in the same region* is free. Downloading to your laptop costs money past 100 GB/month. Send the compute to the data, not the data to the compute.

### 2.2 Create the bucket — Console

**Console → Cloud Storage → Buckets → "Create"**

| Field | Value | Why |
|---|---|---|
| Name | `cynolycus-data` (add a suffix if taken) | **Globally unique across all of GCP** — same namespace as every other customer |
| Location type | **Region** | Cheaper than multi-region; you don't need geo-redundancy |
| Location | `us-central1` | Free-tier eligible |
| Storage class | **Standard** | Lifecycle rules will age things down in 2.5 |
| Access control | **Uniform** | ⚠️ Important — see below |
| Public access prevention | **Enabled** | Blocks accidental internet exposure |
| Protection | leave default for now | Versioning comes in 2.6 |

> **Choose "Uniform", not "Fine-grained".** Uniform disables the legacy per-object ACL system so **IAM is the only permission model**. It eliminates an entire category of accidental-public-exposure bugs. Fine-grained exists only for S3-migration compatibility.

### 2.3 Learn the tool on ONE small directory (CLI)

Don't start with 48 GB. Start with 17 MB. From your WSL terminal:

```bash
cd /home/luket/repos/CynolycusBot
gcloud storage rsync Data/options_history gs://cynolycus-data/options_history --recursive --dry-run
```

`--dry-run` shows exactly what *would* happen without transferring anything. Read it, then drop the flag to run for real.

Verify in the **Console → Cloud Storage → Buckets → cynolycus-data**, and compare against `du -sh Data/options_history`.

✅ **Checkpoint 2.3** — bucket byte total matches local. If not, stop and find out why before scaling up.

> **`gsutil` is dead — don't learn it.** Google's docs state plainly that "gsutil is not the recommended CLI for Cloud Storage." `gcloud storage` is 79–94% faster on downloads, parallelizes automatically with no `-m` flag, and **gsutil is removed from the CLI package after March 2027.** Any tutorial using `gsutil` is teaching a soon-to-be-deleted tool. (Your install lists `gsutil 5.37` — ignore it.)

### 2.4 The real upload — the one genuinely CLI-only step

93,460 files. Work up in size, verifying between each.

```bash
# ~6 GB — shared caches and models the live modules read
gcloud storage rsync Data gs://cynolycus-data/Data --recursive

# ~9.9 GB — news embeddings, catalyst matrices
gcloud storage rsync signals gs://cynolycus-data/signals --recursive \
  --exclude=".*/__pycache__/.*|.*\.py$|.*\.pyc$"

# ~48 GB — the big one
gcloud storage rsync strategies gs://cynolycus-data/strategies --recursive \
  --exclude=".*/__pycache__/.*|.*\.py$|.*\.pyc$"
```

> `--exclude` takes a **Python regex, not a glob.** `*.py` silently won't work; `.*\.py$` will. Everyone trips on this once.

Run the 48 GB transfer in a terminal you can leave open (or under `tmux`) so a disconnect doesn't kill it. **`rsync` is resumable** — re-running transfers only what's missing, so an interruption costs you nothing.

Watch progress live in the Console bucket view; it updates as objects land.

### 2.5 Lifecycle rules — Console

**Bucket → "Lifecycle" tab → "Add a rule"**

Rule 1:
- Action: **Set storage class to Nearline**
- Condition: **Age = 45 days**, **Prefix matches** `strategies/`, `signals/`

Rule 2:
- Action: **Set storage class to Coldline**
- Condition: **Age = 180 days**, same prefixes

This ages *only* frozen research artifacts — never `Data/`, which the live modules read daily and must stay Standard. After 180 days the 58 GB of `strategies/` + `signals/` costs about **$0.23/month.**

### 2.6 Versioning — Console

**Bucket → "Protection" tab → Object versioning → Enable**

Per `AGENTS.md`: *"Never overwrite raw market data, historical labels, experiment outputs, or live trading logs without explicit approval."* Versioning enforces that mechanically — an overwrite or delete keeps the prior version recoverable.

Later, add a lifecycle rule to delete **noncurrent** versions after ~90 days so they don't accumulate cost.

### 2.7 Reclaim the space — separate session, carefully

**Do not do this the same day as the upload.**

Read-back test first — download one file through the Console (click any object → Download) or:

```bash
gcloud storage cp gs://cynolycus-data/Data/shared/bars/SOME_FILE.parquet /tmp/readback.parquet
python3 -c "import pandas as pd; d=pd.read_parquet('/tmp/readback.parquet'); print(d.shape); print(d.head())"
```

Only after that succeeds, delete **one local directory at a time**, checking the relevant module still starts between each. Regenerable derived matrices go first; raw sources and trained models go last, or never.

✅ **Checkpoint 2**

- Console bucket shows ~64 GB
- `df -h /` shows meaningfully more free space
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

**Bucket:** **Cloud Storage → cynolycus-data → "Permissions" tab → "Grant access"**
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

And `.dockerignore` — **not optional**, it's what stops you shipping 78 GB to Cloud Build:

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
4. Browse and select `cynolycus-data`
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
- Volumes → Cloud Storage bucket `cynolycus-data`, mount `/mnt/data`, **✅ Read-only checked**

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
| Cloud Storage — 64 GB aging to Nearline/Coldline | $0.30 – $1.30 |
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
| Browse my data | Cloud Storage → Buckets → cynolycus-data |
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
gcloud storage ls gs://cynolycus-data
gcloud storage du --summarize gs://cynolycus-data
gcloud storage rsync LOCAL gs://cynolycus-data/PATH --recursive --dry-run
gcloud storage cp gs://cynolycus-data/PATH/file.parquet .

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
gcloud storage buckets get-iam-policy gs://cynolycus-data
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
| 2026-07-26 | **Console-first; CLI only for bulk upload** | 93,460 files defeats browser upload; everything else is clearer in the UI while learning |
| 2026-07-26 | **GitHub continuous deployment over `--source .`** | Repo already on GitHub; deploys become reproducible from pushed commits |
| 2026-07-26 | **Object storage, not a database** (§2.0) | Cloud SQL ≈ $129/mo vs ~$1.28 for GCS; parquet already is a columnar store; immutable files beat mutable rows for reproducibility |
| 2026-07-26 | **BigQuery deferred, external tables if ever** (§2.0) | Queries parquet in place — no copy, no migration, 1 TiB/mo free. Revisit only if ad-hoc SQL becomes painful |
| 2026-07-26 | **float32 + zstd re-encoding = separate change** (§2.0b) | Size cut measured, but dtype changes can shift model outputs; needs frozen-test validation, must not ride along with the migration |
| 2026-07-28 | **Re-encoding downgraded to low priority** (§2.0b) | Repo-wide audit: only 37.8% of parquet bytes are float64 (47.9% already float32), so realistic saving is ~11 GB / 27%, not the 46.6% one unrepresentative file implied. Worth doing eventually; not worth blocking the migration |

### Phase log

**Phase 0 — Prerequisites** · ✅ **complete 2026-07-26**
```
gcloud 577.0.0 installed via apt (google-cloud-cli), which gcloud -> /usr/bin/gcloud.
Windows SDK 513.0.0 still on PATH below /usr/bin; harmless, left in place.
Docker deliberately not installed — Cloud Build handles image builds.
TODO: gcloud auth login + gcloud auth application-default login
```

**Phase 1 — Project & budget** · ☐ not started
```
Dev project ID (exact, from Console):
Prod project ID:
Billing account:
Budget amount / thresholds:
```

**Phase 2 — Cloud Storage** · ☐ not started
```
Bucket name (exact):
GB uploaded:
Local GB reclaimed:
Read-back verified (y/n):
```

**Phase 3 — Secret Manager** · ☐ not started
```
Secrets created:
Service account email:
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
| | | | |

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
