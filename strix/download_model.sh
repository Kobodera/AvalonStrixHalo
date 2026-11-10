#!/usr/bin/env bash
# Resumable, SHA256-verified download of the Avalon model set.
# Re-run any time after interruption; verified files are skipped.
set -u
DEST="${AVALON_MODEL_DIR:-$(dirname "$(dirname "$0")")/models/Qwen3.8-Flash-Next-Avalon-GGUF}"
MANIFEST="$(dirname "$0")/model_manifest.tsv"
mkdir -p "$DEST"
ok=0; fail=0
while IFS=$'\t' read -r path sha size url; do
  [ -z "$path" ] && continue
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then
    echo "[skip] $path (verified)"; ok=$((ok+1)); continue
  fi
  for attempt in 1 2 3 4 5; do
    echo "[get ] $path attempt $attempt ($(date -u +%H:%M:%S))"
    curl -fL -C - --retry 15 --retry-delay 10 --retry-all-errors \
         --connect-timeout 30 -o "$out" "$url" && break
    echo "[warn] $path attempt $attempt failed"; sleep 30
  done
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then
    echo "[ok  ] $path"; ok=$((ok+1))
  else
    echo "[bad ] $path corrupt or wrong size, removing"; rm -f "$out"; fail=$((fail+1))
  fi
done < "$MANIFEST"
echo "DONE: $ok verified, $fail failed ($(date -u))"
[ "$fail" = 0 ]
