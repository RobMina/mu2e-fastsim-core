#!/bin/bash
# Check wrapper probes and contained execution without building or unpacking an image.
set -euo pipefail
PRIVATE=${1:?usage: bash native/cry/check-extraction.sh PRIVATE_DIR}
WRAPPER="$PRIVATE/bin/unsquashfs"
REAL=$(env -i _WRAPPER_EXEC_CMD=echo "$WRAPPER")
test -f "$REAL"
ROOT=$(mktemp -d "${APPTAINER_TMPDIR:?set APPTAINER_TMPDIR}/extract-check.XXXXXX")
trap 'rm -rf "$ROOT"' EXIT
mkdir -p "$ROOT/proc" "$ROOT/sys" "$ROOT/dev"
ARGS=(exec --no-home --no-nv --no-rocm -C --no-init --writable)
add_bind() {
    mkdir -p "$ROOT$(dirname "$2")"
    touch "$ROOT$2"
    ARGS+=(-B "$1:$2:ro")
}
add_bind "$WRAPPER" "$WRAPPER"
add_bind "$REAL" "$REAL"
add_bind /bin/bash /bin/bash
for ELF in "$REAL" /bin/bash; do
    INTERP=$(readelf -l "$ELF" | sed -n 's/.*Requesting program interpreter: \([^]]*\).*/\1/p')
    test -n "$INTERP"
    while read -r SRC DEST; do
        add_bind "$SRC" "$DEST"
    done < <(env -i "$INTERP" --list "$ELF" | awk '
      $1 ~ /^\// {print $1, $1}
      $2 == "=>" && $3 ~ /^\// {dest=$3; sub(/[^/]*$/, $1, dest); print $3, dest}')
done
unset APPTAINER_CONFIG_FILE
STATUS=0
apptainer "${ARGS[@]}" "$ROOT" "$WRAPPER" -version > "$ROOT/version.txt" 2>&1 || STATUS=$?
cat "$ROOT/version.txt"
test "$STATUS" -le 1
grep -q '^CRY build: unsquashfs capped at 4 processors' "$ROOT/version.txt"
grep -q '^unsquashfs version ' "$ROOT/version.txt"
printf 'Contained extractor version check passed (version exit status %s).\n' "$STATUS"
