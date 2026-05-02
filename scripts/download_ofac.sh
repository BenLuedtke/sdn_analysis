#!/usr/bin/env bash
# Download OFAC sanctions list files to data/raw/.
# Run from the project root: bash scripts/download_ofac.sh

set -euo pipefail

RAW="data/raw"
BASE="https://sanctionslistservice.ofac.treas.gov/api/download"

mkdir -p "$RAW"

echo "Downloading SDN Advanced XML..."
curl -sSL "$BASE/SDN_ADVANCED.XML" -o "$RAW/SDN_ADVANCED.XML"

echo "Downloading Consolidated Advanced XML..."
curl -sSL "$BASE/CONS_ADVANCED.XML" -o "$RAW/CONS_ADVANCED.XML"

echo "Downloading SDN flat CSV (for comparison)..."
curl -sSL "$BASE/SDN.CSV" -o "$RAW/SDN.CSV"

echo "Done. Files written to $RAW/"
ls -lh "$RAW/"
