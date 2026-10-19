#!/bin/bash
# Downloads the bitstreams used for the measurements from the GitHub release "bitstreams-v1" of this repository and
# places each at the path listed in bitstreams.sha256 (the path the scripts expect), checking its SHA256.
# Bitstreams are kept out of git because of their size. The release asset name is the path with "/" replaced by "__".
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$R"
URL=https://github.com/nasir26/u55c-statevector-30q/releases/download/bitstreams-v1
while read -r sha path; do
  [ -z "$sha" ] && continue
  if [ -f "$path" ] && echo "$sha  $path" | sha256sum -c --quiet - 2>/dev/null; then echo "ok (present) $path"; continue; fi
  mkdir -p "$(dirname "$path")"
  curl -fL --retry 3 -o "$path" "$URL/${path//\//__}"
  echo "$sha  $path" | sha256sum -c -
done < bitstreams.sha256
