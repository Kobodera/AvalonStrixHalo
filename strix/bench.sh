#!/usr/bin/env bash
# Quick benchmark on the live server: warm generation with MTP, then prefill at
# long prompts. Prints the numbers for the README table.
set -euo pipefail
BASE="${1:-http://localhost:8080}"

python3 - "$BASE" <<'EOF'
import json, sys, time, urllib.request

base = sys.argv[1]

def chat(body):
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r, time.time() - t0

# warm generation: 512 tokens greedy, 3 passes, print each and the best
unit = ("The history of computational memory hierarchies begins with the very first "
        "stored-program machines, and every generation since has had to feed its "
        "processors faster than the last. ")
for trial in range(1, 4):
    r, el = chat({"model": "t", "max_tokens": 512, "temperature": 0, "stream": False,
                  "messages": [{"role": "user",
                                "content": "Explain how expert mixture routing works, in detail."}]})
    n = r["usage"]["completion_tokens"]
    t = r["choices"][0]["message"]["content"]
    print(f"GEN trial {trial}: {n} tokens in {el:6.1f}s = {n/el:6.1f} t/s")

# prefill at growing prompt sizes: n_predict=1, measure prompt throughput
for target in (8192, 32768, 131072):
    reps = target // 22 + 2
    prompt = unit * reps
    r, el = chat({"model": "t", "max_tokens": 1, "temperature": 0, "stream": False,
                  "messages": [{"role": "user", "content": prompt}]})
    pt = r["usage"]["prompt_tokens"]
    print(f"PREFILL {pt:6d} tokens: {pt/el:8.1f} t/s ({el:.1f}s)")
EOF
