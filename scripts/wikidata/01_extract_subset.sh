#!/usr/bin/env bash
# Step 1 — extract the temporally-qualified predicate subset from the pinned
# Wikidata RDF dump. Run wherever the dump copy lives. One streaming pass.
#
#   WD_DUMP=/path/to/latest-all.nt.bz2 bash 01_extract_subset.sh [OUTDIR]
#
# Pinned source (public): Wikidata "latest-all" RDF dump of 2026-04
# (latest-all.nt.bz2, 191,612,079,272 bytes, dumped 2026-04-08/10), from
# https://dumps.wikimedia.org/wikidatawiki/entities/ . Use a local copy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP="${WD_DUMP:?set WD_DUMP to the local path of latest-all.nt.bz2}"
OUTDIR="${1:-$HOME/wd-extract}"
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
