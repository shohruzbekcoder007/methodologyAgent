# =============================================================================
# Hermes host agent - starter
# =============================================================================

FROM python:3.12-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG HERMES_REPO=https://github.com/NousResearch/hermes-agent.git
ARG HERMES_REF=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Hermes framework. Its build backend refuses pip wheel builds by design, so the
# source tree is used directly via PYTHONPATH instead of being installed.
RUN pip install --upgrade pip setuptools wheel \
    && git clone --depth 1 --branch "${HERMES_REF}" "${HERMES_REPO}" /opt/hermes-agent

COPY requirements.txt pyproject.toml README.md ./
COPY agents ./agents
COPY app ./app

RUN pip install -r requirements.txt \
    && pip install .

# Hermes' own runtime dependencies, read from its pyproject so a new upstream
# requirement (e.g. psutil) does not break the import check below.
RUN python -c "import tomllib; d=tomllib.load(open('/opt/hermes-agent/pyproject.toml','rb')); print('\n'.join(d['project'].get('dependencies', [])))" > /tmp/hermes-reqs.txt \
    && cat /tmp/hermes-reqs.txt \
    && pip install -r /tmp/hermes-reqs.txt

# Hermes is a hard requirement: fail the build rather than ship an image that
# silently falls back to hermes_lite.
ENV PYTHONPATH=/opt/hermes-agent
RUN python -c "import run_agent, hermes_cli.plugins; print('hermes-agent importable')"

# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    HERMES_HOME=/home/appuser/.hermes \
    HERMES_ENABLE_PROJECT_PLUGINS=true \
    PYTHONPATH=/opt/hermes-agent \
\
    HERMES_ENABLED_TOOLSETS=memory,session_search,skills,todo \
    HERMES_SYSTEM_PROMPT_PATH=/app/prompts/hermes_coordinator.md \
    LOG_DIR=/app/logs \
    APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    TZ=UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
        tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
# May be missing if hermes pip install failed — hermes_lite still works
COPY --from=builder /opt/hermes-agent /opt/hermes-agent

WORKDIR /app

COPY --chown=appuser:appuser agents ./agents
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser prompts ./prompts
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser requirements.txt pyproject.toml README.md ./

# Strip Windows CRLF from shell scripts (avoids: /usr/bin/env: 'bash\r')
RUN sed -i 's/\r$//' /app/scripts/*.sh \
    && chmod +x /app/scripts/*.sh \
    && mkdir -p /app/logs /app/data \
        /home/appuser/.hermes/plugins /home/appuser/.hermes/logs \
    && if [ -f /app/config/hermes_config.yaml ]; then cp /app/config/hermes_config.yaml /home/appuser/.hermes/config.yaml; fi \
    && chown -R appuser:appuser /app/logs /app/data /home/appuser

# start.sh runs as root to chown named volumes, then drops to appuser
USER root

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD ["/app/scripts/healthcheck.sh"]

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/start.sh"]
CMD []
