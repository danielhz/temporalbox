#!/usr/bin/env bash
# Step 1 — extract the temporally-qualified predicate subset from the pinned
# Wikidata RDF dump. Run on kg1 (where the dump lives). One streaming pass.
#
#   bash 01_extract_subset.sh [DUMP] [OUTDIR]
#
# Defaults reproduce the paper's subset from the pinned 2026-04 dump.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP="${1:-/data/ki_nfs_data0/ac/wikidata-dump-2026-04/latest-all.nt.bz2}"
OUTDIR="${2:-$HOME/wd-extract}"
CORES="${CORES:-6}"      # be polite on the shared host

mkdir -p "$OUTDIR"
echo "[extract] dump=$DUMP"
echo "[extract] patterns=$HERE/patterns.txt"
echo "[extract] out=$OUTDIR/subset.nt.gz   cores=$CORES"
echo "[extract] START $(date -Iseconds)"

# lbzip2 parallel decompress -> fixed-string predicate filter -> recompress
lbzip2 -dc -n "$CORES" "$DUMP" \
  | grep -F -f "$HERE/patterns.txt" \
  | pigz -p 4 > "$OUTDIR/subset.nt.gz"

echo "[extract] END $(date -Iseconds)"
echo "[extract] lines=$(pigz -dc "$OUTDIR/subset.nt.gz" | wc -l)  size=$(du -h "$OUTDIR/subset.nt.gz" | cut -f1)"
