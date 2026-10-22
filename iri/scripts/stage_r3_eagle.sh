#!/bin/bash
# Stage the R3 inputs GPVM -> Eagle by ssh stream through the laptop (no laptop copy), md5-checked, resumable.
# Run on the laptop with a live ssh master to a Polaris login node (MobilePASS+); the GPVM cannot reach ALCF.
#   stage_r3_eagle.sh [--dry-run] [--with-g4ref]
# Env: GPVM (mu2egpvm03), POLARIS (ssh alias, default polaris), EAGLE_R3 (destination), CRVX_REPO (crvextrap repo on the GPVM).
# Layout on Eagle (mode 700): steps/crvsteps.*.root, [g4ref/crvsteps.*.root], cond/{crvcalib,crvphoton}.csv,
# crvextrap_<sha>/crvextrap/ (git archive of the committed package) + crvextrap_current symlink, MANIFEST.txt.
set -euo pipefail
G=${GPVM:-mu2egpvm03}; P=${POLARIS:-polaris}
DEST=${EAGLE_R3:-/eagle/GenesisHackathonOct26/ram2aq/crvfastsim/r3_inputs}
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
CRVX=${CRVX_REPO:-/exp/mu2e/app/users/ram2aq/analysis/mu2e_track_crv_mcresponse}
DRY=false; G4REF=false
for a in "$@"; do case "$a" in --dry-run) DRY=true ;; --with-g4ref) G4REF=true ;; *) echo "unknown $a"; exit 2 ;; esac; done

ssh -O check "$P" 2>/dev/null || ssh "$P" true </dev/null || { echo "no ssh to $P: open the master first (ssh -fN polaris-master)"; exit 3; }
ssh "$P" "mkdir -p $DEST/steps $DEST/cond $DEST/g4ref && chmod 700 $DEST $DEST/cond" </dev/null

# pairs "src dst"
LIST=$(ssh "$G" "ls $D/prod_as_tables/crvsteps.*.root" </dev/null | sed "s|.*/\(crvsteps\..*\)|$D/prod_as_tables/\1 steps/\1|")
LIST+=$'\n'"$D/geometry/crvcalib_sim_best_v1_5_run1400.csv cond/crvcalib.csv"
LIST+=$'\n'"$D/geometry/crvphoton_sim_best_v1_5_run1400.csv cond/crvphoton.csv"
if $G4REF; then
  LIST+=$'\n'$(ssh "$G" "ls $D/g4ref_v1_tables/crvsteps.*.root" </dev/null | sed "s|.*/\(crvsteps\..*\)|$D/g4ref_v1_tables/\1 g4ref/\1|")
fi
echo "$LIST" | awk '{print "  "$1" -> "$2}'
$DRY && exit 0

MAN=$(mktemp); trap 'rm -f $MAN' EXIT
while read -r SRC DST; do
  [ -n "$SRC" ] || continue
  SUM=$(ssh "$G" "md5sum $SRC" </dev/null | cut -c1-32)
  HAVE=$(ssh "$P" "md5sum $DEST/$DST 2>/dev/null" </dev/null | cut -c1-32 || true)
  if [ "$HAVE" != "$SUM" ]; then
    ssh "$G" "cat $SRC" </dev/null | ssh "$P" "cat > $DEST/$DST.part && mv $DEST/$DST.part $DEST/$DST"
    HAVE=$(ssh "$P" "md5sum $DEST/$DST" </dev/null | cut -c1-32)
    [ "$HAVE" = "$SUM" ] || { echo "checksum mismatch $DST"; exit 4; }
    echo "streamed $DST $SUM"
  else
    echo "present  $DST $SUM"
  fi
  echo "$SUM  $DST  <- $G:$SRC" >> "$MAN"
done <<< "$LIST"

# crvextrap: committed package only, pinned by commit
[ -z "$(ssh "$G" "git -C $CRVX status --porcelain -- crvextrap" </dev/null)" ] || { echo "crvextrap has uncommitted changes on $G"; exit 5; }
SHA=$(ssh "$G" "git -C $CRVX rev-parse --short HEAD" </dev/null)
if ssh "$P" "test -d $DEST/crvextrap_$SHA/crvextrap" </dev/null; then
  echo "present  crvextrap_$SHA"
else
  ssh "$G" "git -C $CRVX archive --format=tar HEAD crvextrap" </dev/null \
    | ssh "$P" "mkdir -p $DEST/crvextrap_$SHA && tar -x -C $DEST/crvextrap_$SHA" 
  echo "unpacked crvextrap_$SHA"
fi
ssh "$P" "ln -sfn crvextrap_$SHA $DEST/crvextrap_current" </dev/null
echo "crvextrap $SHA  crvextrap_current -> crvextrap_$SHA  <- $G:$CRVX" >> "$MAN"
ssh "$P" "cat > $DEST/MANIFEST.txt && chmod -R go-rwx $DEST && du -sh $DEST" < "$MAN"
echo "manifest $DEST/MANIFEST.txt ($(grep -c . "$MAN") entries)"
