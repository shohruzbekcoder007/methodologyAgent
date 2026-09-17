#!/usr/bin/env bash
# Container entrypoint - Hermes host agent
set -euo pipefail

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [start] $*"
}

APP_HOME="${APP_HOME:-/app}"
HERMES_HOME="${HERMES_HOME:-/home/appuser/.hermes}"
HERMES_USERS_HOME="${HERMES_USERS_HOME:-/home/appuser/.hermes-users}"
export APP_HOME HERMES_HOME HERMES_USERS_HOME
export LOG_DIR="${LOG_DIR:-$APP_HOME/logs}"
export PYTHONUNBUFFERED=1
export HERMES_ENABLE_PROJECT_PLUGINS="${HERMES_ENABLE_PROJECT_PLUGINS:-true}"
export HERMES_SYSTEM_PROMPT_PATH="${HERMES_SYSTEM_PROMPT_PATH:-$APP_HOME/prompts/hermes_coordinator.md}"

mkdir -p "$LOG_DIR" "$HERMES_HOME/plugins" "$HERMES_HOME/logs" "$HERMES_USERS_HOME"

# Named volumes often mount as root — fix ownership when we can (root entrypoint)
if [[ "$(id -u)" -eq 0 ]]; then
  chown -R appuser:appuser "$LOG_DIR" "$HERMES_HOME" 2>/dev/null || true
  # Per-user profiles: own named volume, so it mounts as root too. Only the
  # top level is chowned -- the tree below it holds a directory per user and
  # a recursive pass would grow into a slow startup.
  chown appuser:appuser "$HERMES_USERS_HOME" 2>/dev/null || true
  chown -R appuser:appuser "$APP_HOME/data" 2>/dev/null || true
fi

# Hermes config
if [[ ! -f "$HERMES_HOME/config.yaml" ]]; then
  if [[ -f "$APP_HOME/config/hermes_config.yaml" ]]; then
    cp "$APP_HOME/config/hermes_config.yaml" "$HERMES_HOME/config.yaml"
    log "Installed hermes config.yaml"
  fi
fi

# The agent's SOUL.md. Per-user profiles get theirs from
# agents/user_profiles.py; this covers the shared home, which the framework
# also seeds with a stock file naming itself and its vendor. Only a stock file
# is replaced -- one the agent wrote about itself is left alone.
SOUL_TEMPLATE="${AGENT_SOUL_TEMPLATE:-$APP_HOME/prompts/agent_soul.md}"
if [[ -f "$SOUL_TEMPLATE" ]]; then
  if [[ ! -f "$HERMES_HOME/SOUL.md" ]] \
     || grep -qE 'Hermes Agent|Nous Research' "$HERMES_HOME/SOUL.md"; then
    cp "$SOUL_TEMPLATE" "$HERMES_HOME/SOUL.md"
    chown appuser:appuser "$HERMES_HOME/SOUL.md" 2>/dev/null || true
    log "Installed SOUL.md"
  fi
fi

log "Starting Hermes host service"

cd "$APP_HOME"
if [[ "$(id -u)" -eq 0 ]]; then
  exec runuser -u appuser -- python -m app.main
fi
exec python -m app.main
