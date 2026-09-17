#!/usr/bin/env bash
# =============================================================================
# Embed the corpus and write the vectors onto the Neo4j nodes.
#
#   ./scripts/embed_chunks.sh                    # everything still pending
#   ./scripts/embed_chunks.sh --only terms
#   ./scripts/embed_chunks.sh --limit 200        # a small slice, to try it
#   ./scripts/embed_chunks.sh --force            # re-embed from scratch
#   ./scripts/embed_chunks.sh --check "narx indeksi qanday hisoblanadi"
#   ./scripts/embed_chunks.sh --status           # what is embedded right now
#
# Run it as often as you like. A node counts as pending when it has no vector
# or one from a different model, so an interrupted run resumes and a finished
# one is a no-op. Change the embedding model and every vector is rebuilt --
# vectors from two models are not comparable, and mixing them would quietly
# spoil every search afterwards.
#
# The work happens inside the app container: that is where the Neo4j and
# MongoDB drivers and the credentials already live, and where `mongo` and
# `neo4j` resolve as service names. The embedding service itself is external
# (EMBEDDING_BASE_URL) and is reached over host.docker.internal.
# =============================================================================
set -euo pipefail

export MSYS_NO_PATHCONV=1

host_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
[[ -t 1 ]] || { RED=""; GRN=""; YLW=""; DIM=""; OFF=""; }

log()  { printf '%s[embed]%s %s\n' "$DIM" "$OFF" "$*"; }
ok()   { printf '%s[embed]%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s[embed]%s %s\n' "$YLW" "$OFF" "$*" >&2; }
die()  { printf '%s[embed]%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

STATUS_ONLY=false
PASS_THROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)  STATUS_ONLY=true; shift ;;
    -h|--help) usage 0 ;;
    *)         PASS_THROUGH+=("$1"); shift ;;
  esac
done

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(sed -e 's/\r$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$ROOT/.env")
  set +a
else
  warn ".env not found — relying on the current environment"
fi

compose() {
  docker compose \
    --project-directory "$(host_path "$ROOT")" \
    -f "$(host_path "$ROOT/docker-compose.yml")" "$@"
}

# ---------------------------------------------------------------------------
# The embedding service is external and not managed here, so it is checked
# rather than started. From the host it answers on localhost; the container
# reaches the same service through host.docker.internal, so the two URLs
# differ by hostname alone.
# ---------------------------------------------------------------------------
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://host.docker.internal:8090}"
PROBE_URL="${EMBEDDING_BASE_URL/host.docker.internal/localhost}"

log "embedding service: $EMBEDDING_BASE_URL"
if ! curl -fsS --max-time 10 "$PROBE_URL/health" >/dev/null 2>&1; then
  die "embedding service is not answering at $PROBE_URL/health.
     Start it, or point EMBEDDING_BASE_URL at one that is running."
fi
curl -fsS --max-time 20 "$PROBE_URL/v1/info" 2>/dev/null \
  | python -c 'import json,sys
i = json.load(sys.stdin)
d = i.get("identity", {})
print("  model  : %s  (dim %s, %s)" % (d.get("model"), d.get("dim"), d.get("device")))
print("  gpu    : %s" % (d.get("runtime", {}).get("gpu") or "-"))
print("  batch  : %s, max_chars %s" % (i.get("batch_max"), i.get("max_chars")))' \
  2>/dev/null || warn "could not read /v1/info (continuing)"

# ---------------------------------------------------------------------------
wait_healthy() {
  local service="$1" timeout="${2:-180}" cid status waited=0
  cid="$(compose ps -q "$service")"
  [[ -n "$cid" ]] || die "service '$service' is not running"
  while (( waited < timeout )); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid")"
    case "$status" in healthy|none) return 0 ;; esac
    sleep 3; waited=$(( waited + 3 ))
  done
  die "$service did not become healthy within ${timeout}s"
}

log "making sure the stores are up"
compose up -d mongo neo4j >/dev/null
wait_healthy mongo 120
wait_healthy neo4j 240

# ---------------------------------------------------------------------------
if $STATUS_ONLY; then
  compose exec -T -e P="${NEO4J_PASSWORD:?NEO4J_PASSWORD is not set}" neo4j \
    sh -c "cypher-shell -u '${NEO4J_USER:-neo4j}' -p \"\$P\" --format plain \
      'MATCH (n) WHERE n.embedding_model IS NOT NULL
       RETURN labels(n)[0] AS label, n.embedding_model AS model, count(*) AS embedded
       ORDER BY embedded DESC'"
  compose exec -T -e P="$NEO4J_PASSWORD" neo4j \
    sh -c "cypher-shell -u '${NEO4J_USER:-neo4j}' -p \"\$P\" --format plain \
      'SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties
       WHERE type = \"VECTOR\" RETURN name, state, labelsOrTypes, properties'"
  exit 0
fi

# `run --rm`, not `exec`: a one-off job should not share the serving
# container's lifetime, and compose still waits for mongo and neo4j to be
# healthy because the app service declares them as dependencies.
log "running the embedding worker"
compose run --rm --no-deps \
  -e EMBEDDING_BASE_URL="$EMBEDDING_BASE_URL" \
  --entrypoint python app /app/scripts/embed_chunks.py "${PASS_THROUGH[@]}"

ok "done. check it with:  ./scripts/embed_chunks.sh --status"
