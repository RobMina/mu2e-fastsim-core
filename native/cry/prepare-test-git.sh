#!/bin/bash
# Copy the read-only host git runtime for the container's static test selector.
set -euo pipefail
DEST=${1:?usage: bash native/cry/prepare-test-git.sh DEST}
test ! -e "$DEST"
mkdir -p "$DEST/bin" "$DEST/lib"
GIT=$(command -v git)
cp "$GIT" "$DEST/bin/git.real"
INTERP=$(readelf -l "$GIT" | sed -n 's/.*Requesting program interpreter: \([^]]*\).*/\1/p')
test -n "$INTERP"
cp "$INTERP" "$DEST/lib/ld.so"
while read -r LIB NAME; do
    test -f "$LIB"
    cp -L "$LIB" "$DEST/lib/$NAME"
done < <(env -i "$INTERP" --list "$GIT" | awk '$2 == "=>" && $3 ~ /^\// {print $3, $1}')
cat > "$DEST/bin/git" <<'EOF'
#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
exec "$HERE/lib/ld.so" --library-path "$HERE/lib" "$HERE/bin/git.real" "$@"
EOF
chmod 755 "$DEST/bin/git"
printf 'Prepared read-only local git runtime at %s (no network/clone support promised).\n' "$DEST"
