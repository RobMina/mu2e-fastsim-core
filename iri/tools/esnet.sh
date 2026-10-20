#!/bin/bash
# ESnet East IRI v2 (AWS EKS) driver: curl + AmSC PAT. See iri/README.md "ESnet East".
# usage: esnet.sh whoami | alloc | submit <jobspec.json> | state <jobid> | wait <jobid> [maxsec] | cancel <jobid>
#        | mkdir <path> | ls <path> | cat <path> | upload <local> <remote path> | download <remote> <local>
# Token: $AMSC_TOKEN, else the file $AMSC_TOKEN_FILE (default ~/.amsc/pat, mode 600).
set -u
BASE=${ESNET_IRI:-https://iri-dev.ppg.es.net/api/v2}
COMPUTE=fb0aafe1-c780-55c0-b635-a7121f1b0ce5   # Low Priority compute (EKS, Kueue)
STORAGE=b6d59ba3-5c2d-57dd-b3b9-108832da0578   # EFS storage (home, project)
PROJECT=930784b9-161d-5750-ba65-5d2b9b3b0ea1   # Hackathon 2610 project
TOKEN=${AMSC_TOKEN:-$(cat "${AMSC_TOKEN_FILE:-$HOME/.amsc/pat}" 2>/dev/null)}
[ -n "$TOKEN" ] || { echo "no AmSC PAT: set AMSC_TOKEN or AMSC_TOKEN_FILE" >&2; exit 2; }
api() { local m=$1 p=$2; shift 2; curl -sS -m 60 -X "$m" -H "Authorization: Bearer $TOKEN" "$@" "$BASE$p"; }
jpost() { api POST "$1" -H "Content-Type: application/json" --data "$2"; }
task() {  # poll a filesystem task until it ends; print the task JSON
  local tid st out; tid=$(jq -r '.task_id // empty' <<<"$1")
  [ -n "$tid" ] || { echo "$1" >&2; return 1; }
  for _ in $(seq 1 60); do
    out=$(api GET "/task/$tid"); st=$(jq -r .status <<<"$out")
    case "$st" in pending|active) sleep 2 ;; *) echo "$out"; return 0 ;; esac
  done
  echo "task $tid still $st" >&2; return 1
}
case "${1:-}" in
  whoami) api GET /account/whoami; echo ;;
  alloc) api GET "/account/projects/$PROJECT/project_allocations" \
         | jq -r '.[].entries[] | "\(.usage|floor) / \(.allocation) \(.unit|split(":")|last)"' ;;
  submit) jpost "/compute/job/$COMPUTE" "@$2"; echo ;;
  state) api GET "/compute/status/$COMPUTE/$2" | jq -c .status ;;
  wait)
    end=$(( $(date +%s) + ${3:-600} ))
    while :; do
      st=$(api GET "/compute/status/$COMPUTE/$2" | jq -r '"\(.status.state) \(.status.exit_code) \(.status.message)"')
      echo "$(date +%T) $st"
      case "$st" in completed*|failed*|canceled*) break ;; esac
      [ "$(date +%s)" -ge "$end" ] && { echo "still running; call wait again"; break; }
      sleep 15
    done ;;
  cancel) api DELETE "/compute/cancel/$COMPUTE/$2" -o /dev/null -w "HTTP %{http_code}\n" ;;
  mkdir) task "$(jpost "/filesystem/mkdir/$STORAGE" "$(jq -nc --arg p "$2" '{path: $p, parent: true}')")" | jq -c '{status, result}' ;;
  ls) task "$(jpost "/filesystem/ls/$STORAGE" "$(jq -nc --arg p "$2" '{path: $p}')")" \
      | jq -r 'if (.result.output|type)=="array" then .result.output[] | "\(.size) \(.last_modified) \(.name)" else .result end' ;;
  cat) task "$(jpost "/filesystem/view/$STORAGE" "$(jq -nc --arg p "$2" --argjson n "${VIEW_SIZE:-200000}" '{path: $p, size: $n, offset: 0}')")" \
       | jq -r '.result.output.content // .result' ;;
  upload) task "$(api POST "/filesystem/upload/$STORAGE?path=$3" -F "file=@$2")" | jq -c '{status, result}' ;;
  download) task "$(jpost "/filesystem/download/$STORAGE" "$(jq -nc --arg p "$2" '{path: $p}')")" \
            | jq -r '.result.output | if type=="object" then (.content // .data) else . end' \
            | base64 -d > "$3" && ls -l "$3" ;;
  *) sed -n 3,5p "$0"; exit 2 ;;
esac
