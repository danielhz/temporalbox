#!/usr/bin/env bash
# Run ELK (reasoner of record for the RQ1 oracle cross-check) on an OWL
# functional-syntax ontology produced by ltel_box.oracle.export_functional_owl.
#
#   bash run_elk.sh input.ofn output_taxonomy.ofn
#
# Pinned version: ELK 0.6.0 (elk-distribution-cli, Apache-2.0), fetched once
# from Maven Central into ./tools/. Requires Java 11+.
set -euo pipefail

# The runnable CLI ships inside the distribution .zip (the bare -cli .jar on
# Maven is only an aggregator pom), so we fetch and unpack the zip once.
ELK_VERSION="0.6.0"
ELK_ZIP_URL="https://repo1.maven.org/maven2/io/github/liveontologies/elk-distribution-cli/${ELK_VERSION}/elk-distribution-cli-${ELK_VERSION}.zip"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/tools"
JAR="$TOOLS/elk-distribution-cli-${ELK_VERSION}/elk.jar"

IN="${1:?usage: run_elk.sh input.ofn output_taxonomy.ofn}"
OUT="${2:?usage: run_elk.sh input.ofn output_taxonomy.ofn}"

if [ ! -f "$JAR" ]; then
    mkdir -p "$TOOLS"
    echo "[elk] fetching ELK ${ELK_VERSION} from Maven Central"
    curl -fsSL -o "$TOOLS/elk-${ELK_VERSION}.zip" "$ELK_ZIP_URL"
    unzip -o -q "$TOOLS/elk-${ELK_VERSION}.zip" -d "$TOOLS"
fi

java -jar "$JAR" -i "$IN" -c -o "$OUT"
echo "[elk] taxonomy written to $OUT"
