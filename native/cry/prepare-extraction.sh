#!/bin/bash
# Prepare a private extraction wrapper and Apptainer config; never start a build.
set -euo pipefail
DEST=${1:?usage: bash native/cry/prepare-extraction.sh DEST}
case "$DEST" in /*) ;; *) echo 'DEST must be absolute' >&2; exit 2 ;; esac
case "$DEST" in *[[:space:]]*) echo 'DEST must not contain whitespace' >&2; exit 2 ;; esac
REAL=$(command -v unsquashfs)
SYSCONFDIR=$(apptainer buildcfg | sed -n 's/^SYSCONFDIR=//p')
SOURCE="$SYSCONFDIR/apptainer/apptainer.conf"
test -r "$SOURCE"
test ! -e "$DEST"
mkdir -p "$DEST/bin"
printf '#!/bin/bash\nREAL=%q\nif [[ ${_WRAPPER_EXEC_CMD:-} == echo ]]; then\n  printf "%%s\\n" "$REAL"\n  exit 0\nelif [[ -n ${_WRAPPER_EXEC_CMD:-} ]]; then\n  read -r -a probe <<< "$_WRAPPER_EXEC_CMD"\n  exec "${probe[@]}" "$REAL"\nfi\necho "CRY build: unsquashfs capped at 4 processors, 64 MiB queues" >&2\nexec "$REAL" -processors 4 -data-queue 64 -frag-queue 64 "$@"\n' "$REAL" > "$DEST/bin/unsquashfs"
chmod 755 "$DEST/bin/unsquashfs"
sed '/^[[:space:]]*binary path[[:space:]]*=/d' "$SOURCE" > "$DEST/apptainer.conf"
printf '\nbinary path = %s/bin:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n' "$DEST" >> "$DEST/apptainer.conf"
printf 'Prepared %s/apptainer.conf\nReal extractor: %s\n' "$DEST" "$REAL"
