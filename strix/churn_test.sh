#!/usr/bin/env bash
# Mixed IMG+CHAT+TOOLS churn test (stability gate), port of the 3090 ops script.
# Alternates vision / plain chat / tool-call requests, checks /health between each.
set -euo pipefail
BASE="${1:-http://localhost:8080}"
ROUNDS="${2:-4}"

python3 - "$BASE" "$ROUNDS" <<'EOF'
import base64, io, json, sys, time, urllib.request, urllib.error

base, rounds = sys.argv[1], int(sys.argv[2])

def post(body, timeout=300):
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        msg = r["choices"][0]["message"]
        return True, time.time() - t0, msg.get("content") or "", msg.get("tool_calls")
    except Exception as e:
        return False, time.time() - t0, str(e), None

def health():
    try:
        urllib.request.urlopen(f"{base}/health", timeout=10).read()
        return True
    except Exception:
        return False

try:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (512, 160), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=32)   # bigger glyphs: OCR reads small bitmaps with stray spaces
    except TypeError:
        font = ImageFont.load_default()
    d.text((10, 15), "CHURN ROUND MARKER 7749", fill="black", font=font)
    d.text((10, 70), "alpha bravo charlie delta", fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
except ImportError:
    sys.exit("python3-pil missing: sudo apt install -y python3-pil")

img_body = lambda: {"model": "t", "max_tokens": 120, "temperature": 0.1, "messages": [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Transcribe all text in this image."}]}]}

tools = [{"type": "function", "function": {
    "name": "calculator", "description": " arithmetic",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "expression to evaluate"}},
        "required": ["expression"]}}}]

npass = nfail = 0
for r in range(1, rounds + 1):
    if not health():
        print(f"ROUND {r} PRE: HEALTH FAIL - possible wedge"); nfail += 1; continue

    ok, s, txt, _ = post(img_body())
    tight = "".join(txt.split())   # OCR inserts stray spaces in digit runs
    good = ok and "7749" in tight and "alphabravo" in tight
    print(f"ROUND {r} VISION: ok={ok} {s:.1f}s match={'7749' in tight}")
    if good: npass += 1
    else: nfail += 1; print("  text:", txt[:120])

    ok, s, txt, _ = post({"model": "t", "max_tokens": 64, "temperature": 0.2,
                          "messages": [{"role": "user",
                                        "content": f"Reply with exactly: CHAT-OK-{r}"}]})
    good = ok and f"CHAT-OK-{r}" in txt
    print(f"ROUND {r} CHAT:   ok={ok} {s:.1f}s match={f'CHAT-OK-{r}' in txt}")
    if good: npass += 1
    else: nfail += 1; print("  text:", txt[:120])

    ok, s, txt, tc = post({"model": "t", "max_tokens": 128, "temperature": 0.1,
                           "messages": [{"role": "user",
                                         "content": "What is 25 times 4? You MUST call the calculator tool."}],
                           "tools": tools, "tool_choice": "required"})
    good = ok and bool(tc)
    print(f"ROUND {r} TOOLS:  ok={ok} {s:.1f}s tool_call={bool(tc)}")
    if good: npass += 1
    else: nfail += 1; print("  text:", txt[:120])

print(f"CHURN: {npass} pass, {nfail} fail of {rounds * 3}")
print("CHURN PASS" if nfail == 0 else "CHURN CHECK NEEDED")
EOF
