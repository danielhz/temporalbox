#!/usr/bin/env bash
# Run ELK (reasoner of record for the RQ1 oracle cross-check) on an OWL
# functional-syntax ontology produced by ltel_box.oracle.export_functional_owl.
#
#   bash run_elk.sh input.ofn output_taxonomy.ofn
#
# Pinned version: ELK 0.6.0 (elk-distribution-cli, Apache-2.0), fetched once
# from Maven Central into ./tools/. Requires Java 11+.
set -euo pipefail

ELK_VERSION="0.6.0"
ELK_URL="https://repo1.maven.org/maven2/io/github/liveontologies/elk-distribution-cli/${ELK_VERSION}/elk-distribution-cli-${ELK_VERSION}.jar"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/tools"
JAR="$TOOLS/elk-${ELK_VERSION}.jar"

IN="${1:?usage: run_elk.sh input.ofn output_taxonomy.ofn}"
OUT="${2:?usage: run_elk.sh input.ofn output_taxonomy.ofn}"

if [ ! -f "$JAR" ]; then
    mkdir -p "$TOOLS"
    echo "[elk] fetching ELK ${ELK_VERSION} from Maven Central"
    curl -fsSL -o "$JAR" "$ELK_URL"
fi

java -jar "$JAR" -i "$IN" -c -o "$OUT"
echo "[elk] taxonomy written to $OUT"
