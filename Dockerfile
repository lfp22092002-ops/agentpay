# ── Build stage ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime-only system deps (libpq for psycopg, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages (no gcc in final image)
COPY --from=builder /install /usr/local

# Copy source
COPY . .

# Create non-root user
RUN useradd -r -s /bin/false agentpay && \
    mkdir -p /app/data/wallets /app/data/cards && \
    chown -R agentpay:agentpay /app/data

USER agentpay

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
