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