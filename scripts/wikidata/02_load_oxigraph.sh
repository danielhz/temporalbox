#!/usr/bin/env bash
# Step 2 — load the extracted subset into a local oxigraph store. Run on the
# data server.
#
#   bash 02_load_oxigraph.sh [OUTDIR]
#
# OUTDIR must contain subset.nt.gz from step 1; the store is written to
# OUTDIR/store. Uses the user-local oxigraph (no permission issues, unlike the
# shared TDB2 index).
set -euo pipefail

OUTDIR="${1:-$HOME/wd-extract}"
OXIGRAPH="${OXIGRAPH:-$HOME/.cargo/bin/oxigraph_server}"

rm -rf "$OUTDIR/store"
echo "[load] loading $OUTDIR/subset.nt.gz into $OUTDIR/store"
pigz -dc "$OUTDIR/subset.nt.gz" \
  | "$OXIGRAPH" -l "$OUTDIR/store" load --format nt --lenient
echo "[load] store size: $(du -sh "$OUTDIR/store" | cut -f1)"
