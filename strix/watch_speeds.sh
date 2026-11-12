#!/usr/bin/env bash
# Live token speeds of a running Avalon server, straight from its log.
# While a request runs the server prints one line per 3 s: tg = lifetime average,
# tg_3s = the last 3 s window; finished requests add eval / prompt eval / draft lines.
# usage: strix/watch_speeds.sh <logfile>
# tip:   strix/launch_vision_mtp.sh > ~/avalon.log 2>&1  (then watch ~/avalon.log)
set -euo pipefail
log="${1:?usage: watch_speeds.sh <logfile>, the file you redirected the server to}"
[ -r "$log" ] || { printf 'cannot read %s\n' "$log" >&2; exit 1; }
tail -n +1 -f "$log" | grep --line-buffered -E "tg =|eval time =|prompt eval time =|draft acceptance ="
