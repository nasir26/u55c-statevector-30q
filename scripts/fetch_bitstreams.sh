#!/bin/bash
# Downloads the bitstreams used for the measurements from the GitHub release "bitstreams-v1" of this repository and
# places each at the path listed in bitstreams.sha256 (the path the scripts expect), checking its SHA256.
# Bitstreams are kept out of git because of their size. The release asset name is the path with "/" replaced by "__".
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$R"
URL=https://github.com/nasir26/u55c-statevector-30q/releases/download/bitstreams-v1
get() {   # curl, wget or Python, whichever the host has
  if command -v curl >/dev/null; then curl -fL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null; then wget -q -O "$2" "$1"
  else python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"; fi
}
while read -r sha path; do
  [ -z "$sha" ] && continue
  if [ -f "$path" ] && echo "$sha  $path" | sha256sum -c --quiet - 2>/dev/null; then echo "ok (present) $path"; continue; fi
  mkdir -p "$(dirname "$path")"
  get "$URL/${path//\//__}" "$path"
  echo "$sha  $path" | sha256sum -c -
done < bitstreams.sha256
