#!/usr/bin/env bash
# Vision test: render text to a PNG, POST it to /v1/chat/completions, score the OCR.
# Needs python3-pil to draw the image.
set -euo pipefail
BASE="${1:-http://localhost:8080}"

python3 - "$BASE" <<'EOF'
import base64, io, json, sys, time, urllib.request
try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("python3-pil missing: sudo apt install -y python3-pil")

base = sys.argv[1]
img = Image.new("RGB", (640, 200), "white")
d = ImageDraw.Draw(img)
d.text((10, 20), "AVALON VISION TEST STRIX", fill="black")
d.text((10, 80), "The quick brown fox jumps over the lazy dog.", fill="black")
d.text((10, 120), "Expert cache + MTP + mmproj = 42", fill="black")
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

body = json.dumps({
    "model": "test", "max_tokens": 200, "temperature": 0.1,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Transcribe all text you see in this image, line by line."},
    ]}],
}).encode()

req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
resp = json.load(urllib.request.urlopen(req, timeout=600))
el = time.time() - t0
text = resp["choices"][0]["message"]["content"]
print(f"elapsed: {el:.1f}s")
print("response:", text)
score = sum(1 for pat in ("AVALON", "quick brown fox", "42") if pat in text)
print(f"OCR score: {score} / 3")
print("VISION PASS" if score == 3 else "VISION CHECK NEEDED")
EOF
