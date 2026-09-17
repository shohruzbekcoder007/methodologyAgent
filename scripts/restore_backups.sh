#!/usr/bin/env bash
# =============================================================================
# Restore MongoDB and Neo4j from data/backups/
#
#   ./scripts/restore_backups.sh                 # newest of each, with a prompt
#   ./scripts/restore_backups.sh --yes           # no prompt (CI, re-runs)
#   ./scripts/restore_backups.sh --only mongo
#   ./scripts/restore_backups.sh --neo4j-file nmc-neo4j-5.26-20260915-2111.dump
#   ./scripts/restore_backups.sh --list          # what is available, restore nothing
#
# Re-runnable by design: Mongo restores with --drop and Neo4j with
# --overwrite-destination, so a second run lands on the same state as the
# first. Both stores are wiped of whatever they held.
#
# The two backups are taken as a pair -- Mongo documents point at Neo4j nodes
# through `mongo_id` -- so restoring only one side, or two files with different
# timestamps, can leave dangling references. The script says so when it spots
# it; it does not refuse, because re-restoring one side alone is a normal
# thing to do while iterating.
# =============================================================================
set -euo pipefail

# Git Bash rewrites anything that looks like a path in a docker argument, so
# a container-side `/data` arrives as `C:/Program Files/Git/data`. Turning the
# rewriting off protects those, but then host paths have to be spelled in a
# form Windows docker understands -- hence `host_path` below. On Linux and
# macOS both are no-ops.
export MSYS_NO_PATHCONV=1

host_path() {
  if command -v cygpath >/dev/null 2>&1; then
    # -m: D:/GROK/... — forward slashes, which docker parses unambiguously
    # next to the `:/container:ro` half of a volume argument.
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/data/backups}"

# Mount point of BACKUP_DIR inside the mongo/neo4j containers -- see the
# `./data/backups:/backups:ro` volumes in docker-compose.yml.
BACKUP_MOUNT=/backups

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
[[ -t 1 ]] || { RED=""; GRN=""; YLW=""; DIM=""; OFF=""; }

log()  { printf '%s[restore]%s %s\n' "$DIM" "$OFF" "$*"; }
ok()   { printf '%s[restore]%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s[restore]%s %s\n' "$YLW" "$OFF" "$*" >&2; }
die()  { printf '%s[restore]%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ONLY=all
MONGO_FILE=""
NEO4J_FILE=""
ASSUME_YES=false
LIST_ONLY=false

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)        ONLY="${2:-}"; shift 2 ;;
    --only=*)      ONLY="${1#*=}"; shift ;;
    --mongo-file)  MONGO_FILE="${2:-}"; shift 2 ;;
    --mongo-file=*) MONGO_FILE="${1#*=}"; shift ;;
    --neo4j-file)  NEO4J_FILE="${2:-}"; shift 2 ;;
    --neo4j-file=*) NEO4J_FILE="${1#*=}"; shift ;;
    -y|--yes)      ASSUME_YES=true; shift ;;
    -l|--list)     LIST_ONLY=true; shift ;;
    -h|--help)     usage 0 ;;
    *)             die "unknown argument: $1 (--help for usage)" ;;
  esac
done

case "$ONLY" in
  all|mongo|neo4j) ;;
  *) die "--only takes: all | mongo | neo4j (got: $ONLY)" ;;
esac

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Credentials come from .env, the same file compose reads, so the script and
# the containers can never disagree about the password.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(sed -e 's/\r$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$ROOT/.env")
  set +a
else
  warn ".env not found — relying on the current environment"
fi

MONGO_ROOT_USER="${MONGO_ROOT_USER:-nmc}"
MONGO_DB="${MONGO_DB:-nmc}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

# Compose project name, for the named volume the Neo4j load writes into.
PROJECT="$(sed -n 's/^name:[[:space:]]*\([A-Za-z0-9_-]\{1,\}\).*/\1/p' "$ROOT/docker-compose.yml" | head -1)"
PROJECT="${PROJECT:-methodologyagent}"
NEO4J_DATA_VOLUME="${NEO4J_DATA_VOLUME:-${PROJECT}-neo4j-data}"

compose() {
  docker compose \
    --project-directory "$(host_path "$ROOT")" \
    -f "$(host_path "$ROOT/docker-compose.yml")" "$@"
}

# The image the dump is loaded with must be the one that serves it afterwards;
# a store written by a newer Neo4j will not open in an older one.
neo4j_image() {
  local img
  img="$(compose config --format json 2>/dev/null \
    | tr ',' '\n' | sed -n 's/.*"image":[[:space:]]*"\(neo4j:[^"]*\)".*/\1/p' | head -1)"
  printf '%s' "${img:-neo4j:5.26-community}"
}

# ---------------------------------------------------------------------------
# Backup selection
# ---------------------------------------------------------------------------
[[ -d "$BACKUP_DIR" ]] || die "backup directory not found: $BACKUP_DIR"

# Newest by filename, not mtime: the timestamp in the name is when the dump was
# taken, while mtime is when the file last moved between machines.
newest() {
  local pattern="$1" f
  f="$(ls -1 "$BACKUP_DIR"/$pattern 2>/dev/null | sort | tail -1 || true)"
  [[ -n "$f" ]] && basename "$f"
}

list_backups() {
  log "backups in $BACKUP_DIR:"
  local found=false f
  for f in "$BACKUP_DIR"/*.archive.gz "$BACKUP_DIR"/*.dump; do
    [[ -e "$f" ]] || continue
    found=true
    printf '  %-46s %6s MB\n' "$(basename "$f")" "$(( $(stat -c %s "$f") / 1048576 ))"
  done
  $found || printf '  (none)\n'
}

if $LIST_ONLY; then
  list_backups
  exit 0
fi

if [[ "$ONLY" == all || "$ONLY" == mongo ]]; then
  MONGO_FILE="${MONGO_FILE:-$(newest '*mongo*.archive.gz')}"
  [[ -n "$MONGO_FILE" ]] || die "no Mongo archive (*mongo*.archive.gz) in $BACKUP_DIR"
  [[ -f "$BACKUP_DIR/$MONGO_FILE" ]] || die "no such file: $BACKUP_DIR/$MONGO_FILE"
fi

if [[ "$ONLY" == all || "$ONLY" == neo4j ]]; then
  NEO4J_FILE="${NEO4J_FILE:-$(newest '*neo4j*.dump')}"
  [[ -n "$NEO4J_FILE" ]] || die "no Neo4j dump (*neo4j*.dump) in $BACKUP_DIR"
  [[ -f "$BACKUP_DIR/$NEO4J_FILE" ]] || die "no such file: $BACKUP_DIR/$NEO4J_FILE"
fi

# The date-time stamp both file names carry, e.g. 20260917-1436.
stamp_of() { printf '%s' "$1" | sed -n 's/.*-\([0-9]\{8\}-[0-9]\{4\}\).*/\1/p'; }

if [[ -n "$MONGO_FILE" && -n "$NEO4J_FILE" ]]; then
  ms="$(stamp_of "$MONGO_FILE")"; ns="$(stamp_of "$NEO4J_FILE")"
  if [[ -n "$ms" && -n "$ns" && "${ms%-*}" != "${ns%-*}" ]]; then
    warn "the two backups are from different days ($ns vs $ms);"
    warn "cross-store references (mongo_id, edges) may not line up."
  fi
fi

# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------
log "project      : $PROJECT"
[[ -n "$MONGO_FILE" ]] && log "mongo archive: $MONGO_FILE  → database '$MONGO_DB' (--drop)"
[[ -n "$NEO4J_FILE" ]] && log "neo4j dump   : $NEO4J_FILE  → database '$NEO4J_DATABASE' (overwrite)"

if ! $ASSUME_YES; then
  if [[ -t 0 ]]; then
    read -r -p "$(printf '%s[restore]%s this replaces the current contents. continue? [y/N] ' "$YLW" "$OFF")" reply
    [[ "$reply" == [yY]* ]] || die "aborted"
  else
    die "not a terminal and --yes was not given; refusing to overwrite"
  fi
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_healthy() {
  local service="$1" timeout="${2:-180}" cid status waited=0
  cid="$(compose ps -q "$service")"
  [[ -n "$cid" ]] || die "service '$service' is not running"
  while (( waited < timeout )); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid")"
    case "$status" in
      healthy|none) return 0 ;;
      unhealthy)    warn "$service reports unhealthy; still waiting" ;;
    esac
    sleep 3
    waited=$(( waited + 3 ))
  done
  die "$service did not become healthy within ${timeout}s"
}

# ---------------------------------------------------------------------------
# MongoDB — restores while the server runs; no downtime needed.
# ---------------------------------------------------------------------------
restore_mongo() {
  log "mongo: starting the service"
  compose up -d mongo >/dev/null
  wait_healthy mongo 120

  log "mongo: restoring $MONGO_FILE (this takes a few minutes)"
  # Read from the read-only /backups mount rather than piping ~90 MB through
  # docker's stdin, which is markedly slower on Windows.
  compose exec -T \
    -e RESTORE_PW="${MONGO_ROOT_PASSWORD:?MONGO_ROOT_PASSWORD is not set}" \
    mongo sh -c "mongorestore \
      --username '$MONGO_ROOT_USER' --password \"\$RESTORE_PW\" \
      --authenticationDatabase admin \
      --gzip --archive='$BACKUP_MOUNT/$MONGO_FILE' \
      --drop --numParallelCollections 4"

  local summary
  summary="$(compose exec -T \
    -e RESTORE_PW="$MONGO_ROOT_PASSWORD" \
    mongo sh -c "mongosh --quiet \
      --username '$MONGO_ROOT_USER' --password \"\$RESTORE_PW\" \
      --authenticationDatabase admin \
      --eval 'const d = db.getSiblingDB(\"$MONGO_DB\"); \
               print(d.getCollectionNames().length + \" collections, \" + \
                     d.stats().objects + \" documents\")'" | tr -d '\r')"
  ok "mongo: $summary"
}

# ---------------------------------------------------------------------------
# Neo4j — Community Edition loads a dump only into a stopped store, so the
# load runs in a throwaway container attached to the data volume instead of
# through the service container.
# ---------------------------------------------------------------------------
restore_neo4j() {
  local image
  image="$(neo4j_image)"

  log "neo4j: stopping the service"
  compose stop neo4j >/dev/null 2>&1 || true

  log "neo4j: loading $NEO4J_FILE with $image (this takes a few minutes)"
  # `database load` looks for <database>.dump in --from-path, so the backup is
  # staged under that name. /backups is mounted read-only, hence the copy.
  docker run --rm \
    -v "${NEO4J_DATA_VOLUME}:/data" \
    -v "$(host_path "$BACKUP_DIR"):${BACKUP_MOUNT}:ro" \
    --entrypoint sh \
    "$image" -c "set -e
      cp '$BACKUP_MOUNT/$NEO4J_FILE' '/tmp/${NEO4J_DATABASE}.dump'
      neo4j-admin database load '$NEO4J_DATABASE' \
        --from-path=/tmp --overwrite-destination=true
      rm -f '/tmp/${NEO4J_DATABASE}.dump'"

  log "neo4j: starting the service"
  compose up -d neo4j >/dev/null
  wait_healthy neo4j 240

  local counts
  counts="$(compose exec -T \
    -e RESTORE_PW="${NEO4J_PASSWORD:?NEO4J_PASSWORD is not set}" \
    neo4j sh -c "cypher-shell -u '$NEO4J_USER' -p \"\$RESTORE_PW\" \
      -d '$NEO4J_DATABASE' --format plain \
      'MATCH (n) RETURN count(n) AS nodes'" | tail -1 | tr -d '\r')"
  ok "neo4j: $counts nodes"
}

# ---------------------------------------------------------------------------
[[ "$ONLY" == all || "$ONLY" == mongo ]] && restore_mongo
[[ "$ONLY" == all || "$ONLY" == neo4j ]] && restore_neo4j

ok "done. mongo → 127.0.0.1:${MONGO_PORT:-27118}, neo4j browser → http://127.0.0.1:${NEO4J_HTTP_PORT:-9096}"
