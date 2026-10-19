#!/bin/bash
# Thin wrapper over the genesis iri-api scripts for ALCF: submit, poll, cat a file on Eagle.
# usage: iri.sh submit <jobspec.json> | wait <jobid> [maxsec] | cat <eagle path> | ls <eagle path>
# Needs site.env: IRI (the genesis-skills iri-api skill directory) and IRI_PY (a python with globus-sdk).
IRI=${IRI:?source site.env}
PY=${IRI_PY:-python3}
POLARIS=55c1c993-1124-47f9-b823-514ba3849a9a
EAGLE=1c3ad9d4-2e91-42bc-becb-72b1fde1235c
OPENAPI=${IRI_OPENAPI_URL:-https://api.alcf.anl.gov/api/v2/openapi.json}  # ALCF moved the API to /api/v2 by 2026-10-19
call() { "$PY" "$IRI/scripts/iri_api_call.py" --facility alcf --openapi-url "$OPENAPI" call "$@" --ensure-token; }
task() {  # run a filesystem op and print its result once the task completes
  local tid; tid=$(call "$@" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['task_id'])") || return 1
  for _ in $(seq 1 30); do
    out=$(call --operation-id getTask --path-param task_id="$tid")
    st=$(echo "$out" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['status'])")
    [ "$st" = completed ] || [ "$st" = failed ] && { echo "$out"; return 0; }
    sleep 2
  done
  echo "task $tid still $st" >&2; return 1
}
case "$1" in
  submit) call --operation-id launchJob --path-param resource_id=$POLARIS --json-file "$2" ;;
  wait)
    end=$(( $(date +%s) + ${3:-280} ))
    while :; do
      st=$(call --operation-id getJob --path-param resource_id=$POLARIS --path-param job_id="$2" \
           | "$PY" -c "import json,sys;d=json.load(sys.stdin)['status'];print(d['state'],d.get('exit_code'))")
      echo "$(date +%T) $st"
      case "$st" in completed*|failed*|canceled*) break ;; esac
      [ "$(date +%s)" -ge "$end" ] && { echo "still running; call wait again"; break; }
      sleep 15
    done ;;
  cat) task --operation-id view --path-param resource_id=$EAGLE --json-body "{\"path\": \"$2\"}" \
       | "$PY" -c "import json,sys;r=json.load(sys.stdin)['result'];print(r['output']['content'] if 'output' in r else 'NOFILE '+json.dumps(r)[:300])" ;;
  ls) task --operation-id ls --path-param resource_id=$EAGLE --json-body "{\"path\": \"$2\"}" \
       | "$PY" -c "import json,sys;[print(f\"{e['size']:>10} {e['last_modified']} {e['name']}\") for e in json.load(sys.stdin)['result']['output']]" ;;
  *) echo "usage: $0 submit <json> | wait <jobid> [maxsec] | cat <path> | ls <path>"; exit 2 ;;
esac
