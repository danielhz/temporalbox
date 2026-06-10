#!/usr/bin/env bash
# Step 3 — export the office-membership and subclass TSVs from the oxigraph
# store via SPARQL. Run on kg1.
#
#   bash 03_export.sh [OUTDIR]
#
# Writes OUTDIR/offices/{memberships,p279}.tsv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:-$HOME/wd-extract}"
OXIGRAPH="${OXIGRAPH:-$HOME/.cargo/bin/oxigraph_server}"
STORE="$OUTDIR/store"
DEST="$OUTDIR/offices"
mkdir -p "$DEST"

echo "[export] memberships -> $DEST/memberships.tsv"
"$OXIGRAPH" -l "$STORE" query --query-file "$HERE/queries/export_memberships.rq" \
  --results-format tsv > "$DEST/memberships.tsv"
echo "[export]   rows=$(wc -l < "$DEST/memberships.tsv")"

echo "[export] P279 edges -> $DEST/p279.tsv"
"$OXIGRAPH" -l "$STORE" query --query-file "$HERE/queries/export_p279.rq" \
  --results-format tsv > "$DEST/p279.tsv"
echo "[export]   rows=$(wc -l < "$DEST/p279.tsv")"
