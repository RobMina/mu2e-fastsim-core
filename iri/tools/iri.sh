#!/bin/bash
# IRI driver for the fastsim endpoints: submit, poll, list and read files. Facility from IRI_FACILITY (default alcf).
# usage: [IRI_FACILITY=alcf|esnet] iri.sh submit <jobspec.json> | wait <jobid> [maxsec] | state <jobid> | cancel <jobid>
#        | cat <path> | ls <path> | mkdir <path> | whoami | alloc | upload <local> <remote> | download <remote> <local>
# alcf:  genesis iri-api scripts + Globus token (site.env: IRI, IRI_PY); Polaris + Eagle. Only submit/wait/state/cat/ls.
# esnet: curl + jq + AmSC PAT ($AMSC_TOKEN or the mode-600 file $AMSC_TOKEN_FILE, default ~/.amsc/pat); EKS + EFS.
set -u
FAC=${IRI_FACILITY:-alcf}
cmd=${1:-}; shift || true

wait_loop() {  # wait_loop <jobid> <maxsec> <state-command...>: poll every 15 s until a final state
  local id=$1 max=$2; shift 2
  local end=$(( $(date +%s) + max )) st
  while :; do
    st=$("$@" "$id")
    echo "$(date +%T) $st"
    case "$st" in completed*|failed*|canceled*) break ;; esac
    [ "$(date +%s)" -ge "$end" ] && { echo "still running; call wait again"; break; }
    sleep 15
  done
}

case "$FAC" in
alcf)
  IRI=${IRI:?source site.env}
  PY=${IRI_PY:-python3}
  POLARIS=55c1c993-1124-47f9-b823-514ba3849a9a
  EAGLE=1c3ad9d4-2e91-42bc-becb-72b1fde1235c
  OPENAPI=${IRI_OPENAPI_URL:-https://api.alcf.anl.gov/api/v2/openapi.json}  # ALCF moved the API to /api/v2 by 2026-10-19
  call() { "$PY" "$IRI/scripts/iri_api_call.py" --facility alcf --openapi-url "$OPENAPI" call "$@" --ensure-token; }
  task() {  # run a filesystem op and print its result once the task completes
    local tid out st; tid=$(call "$@" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['task_id'])") || return 1
    for _ in $(seq 1 30); do
      out=$(call --operation-id getTask --path-param task_id="$tid")
      st=$(echo "$out" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['status'])")
      [ "$st" = completed ] || [ "$st" = failed ] && { echo "$out"; return 0; }
      sleep 2
    done
    echo "task $tid still $st" >&2; return 1
  }
  state() { call --operation-id getJob --path-param resource_id=$POLARIS --path-param job_id="$1" \
            | "$PY" -c "import json,sys;d=json.load(sys.stdin)['status'];print(d['state'],d.get('exit_code'))"; }
  case "$cmd" in
    submit) call --operation-id launchJob --path-param resource_id=$POLARIS --json-file "$1" ;;
    state) state "$1" ;;
    wait) wait_loop "$1" "${2:-280}" state ;;
    cat) task --operation-id view --path-param resource_id=$EAGLE --json-body "{\"path\": \"$1\"}" \
         | "$PY" -c "import json,sys;r=json.load(sys.stdin)['result'];print(r['output']['content'] if 'output' in r else 'NOFILE '+json.dumps(r)[:300])" ;;
    ls) task --operation-id ls --path-param resource_id=$EAGLE --json-body "{\"path\": \"$1\"}" \
        | "$PY" -c "import json,sys;[print(f\"{e['size']:>10} {e['last_modified']} {e['name']}\") for e in json.load(sys.stdin)['result']['output']]" ;;
    *) echo "alcf: submit | wait | state | cat | ls (see the header of $0)"; exit 2 ;;
  esac ;;
esnet)
  BASE=${ESNET_IRI:-https://iri-dev.ppg.es.net/api/v2}
  COMPUTE=fb0aafe1-c780-55c0-b635-a7121f1b0ce5   # Low Priority compute (EKS, Kueue)
  STORAGE=b6d59ba3-5c2d-57dd-b3b9-108832da0578   # EFS storage (home, project)
  PROJECT=${ESNET_PROJECT:-930784b9-161d-5750-ba65-5d2b9b3b0ea1}   # Hackathon 2610 project
  TOKEN=${AMSC_TOKEN:-$(cat "${AMSC_TOKEN_FILE:-$HOME/.amsc/pat}" 2>/dev/null)}
  [ -n "$TOKEN" ] || { echo "no AmSC PAT: set AMSC_TOKEN or AMSC_TOKEN_FILE" >&2; exit 2; }
  api() { local m=$1 p=$2; shift 2; curl -sS -m 60 -X "$m" -H "Authorization: Bearer $TOKEN" "$@" "$BASE$p"; }
  jpost() { api POST "$1" -H "Content-Type: application/json" --data "$2"; }
  body() { jq -nc --arg p "$1" '{path: $p}'; }   # jq, not a literal: bash brace expansion splits {"a": 1, "b": 2}
  task() {  # poll a filesystem task until it ends; print the task JSON
    local tid st out; tid=$(jq -r '.task_id // empty' <<<"$1")
    [ -n "$tid" ] || { echo "$1" >&2; return 1; }
    for _ in $(seq 1 60); do
      out=$(api GET "/task/$tid"); st=$(jq -r .status <<<"$out")
      case "$st" in pending|active) sleep 2 ;; *) echo "$out"; return 0 ;; esac
    done
    echo "task $tid still $st" >&2; return 1
  }
  state() { api GET "/compute/status/$COMPUTE/$1" | jq -r '"\(.status.state) \(.status.exit_code) \(.status.message)"'; }
  case "$cmd" in
    whoami) api GET /account/whoami; echo ;;
    alloc) api GET "/account/projects/$PROJECT/project_allocations" \
           | jq -r '.[].entries[] | "\(.usage|floor) / \(.allocation) \(.unit|split(":")|last)"' ;;
    submit) jpost "/compute/job/$COMPUTE" "@$1"; echo ;;
    state) state "$1" ;;
    wait) wait_loop "$1" "${2:-600}" state ;;
    cancel) api DELETE "/compute/cancel/$COMPUTE/$1" -o /dev/null -w "HTTP %{http_code}\n" ;;
    mkdir) task "$(jpost "/filesystem/mkdir/$STORAGE" "$(jq -nc --arg p "$1" '{path: $p, parent: true}')")" | jq -c '{status, result}' ;;
    ls) task "$(jpost "/filesystem/ls/$STORAGE" "$(body "$1")")" \
        | jq -r 'if (.result.output|type)=="array" then .result.output[] | "\(.size) \(.last_modified) \(.name)" else .result end' ;;
    cat) task "$(jpost "/filesystem/view/$STORAGE" "$(jq -nc --arg p "$1" --argjson n "${VIEW_SIZE:-200000}" '{path: $p, size: $n, offset: 0}')")" \
         | jq -r '.result.output.content // .result' ;;
    rm) task "$(jpost "/filesystem/rm/$STORAGE" "$(jq -nc --arg p "$1" '{path: $p}')")" | jq -c '{status, result}' ;;
    upload) task "$(api POST "/filesystem/upload/$STORAGE?path=$2" -F "file=@$1")" | jq -c '{status, result}' ;;
    download) task "$(jpost "/filesystem/download/$STORAGE" "$(body "$1")")" \
              | jq -r '.result.output | if type=="object" then (.content // .data) else . end' \
              | base64 -d > "$2" && ls -l "$2" ;;   # IRI caps downloads at 2 MiB: use Globus (globus_esnet.py) above that
    *) echo "esnet: whoami | alloc | submit | state | wait | cancel | mkdir | ls | cat | rm | upload | download"; exit 2 ;;
  esac ;;
*) echo "IRI_FACILITY must be alcf or esnet"; exit 2 ;;
esac
