#!/bin/bash
# IRI driver for the fastsim endpoints: submit, poll, list and read files. Facility from IRI_FACILITY (default alcf).
# usage: [IRI_FACILITY=alcf] iri.sh submit <jobspec.json> | wait <jobid> [maxsec] | state <jobid> | cancel <jobid>
#        | cat <path> | ls <path>
# alcf:  genesis iri-api scripts + Globus token (site.env: IRI, IRI_PY); Polaris + Eagle. Only submit/wait/state/cat/ls.
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
*) echo "IRI_FACILITY must be alcf"; exit 2 ;;
esac
