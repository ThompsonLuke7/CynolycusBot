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
# CPU-only torch: the default wheel pulls ~2.5 GB of CUDA libraries Cloud Run cannot use.
# Installed before requirements.txt so the unpinned `torch` line resolves as already satisfied,
# which keeps requirements.txt GPU-capable for local development.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1