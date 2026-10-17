"""Contact sheet per case: recent window (new run) and uniformly sampled stored history."""
import glob, json, sys
from pathlib import Path
import cv2
import numpy as np

M = Path("/home/zhangnuohua/medhorizon")
N = {}
for p in glob.glob(str(M / "video2-fill-full-context-20261015/results/ovo-routed-vcheck-2gpu-20261017/attempts/*/predictions.jsonl")):
    N.update({json.loads(l)["sample_id"]: json.loads(l) for l in open(p) if l.endswith("\n")})
OLDF = {json.loads(l)["sample_id"]: json.loads(l) for l in open(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")}
W = 300

def grab(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    if not ok:
        return np.zeros((int(W * 0.75), W, 3), np.uint8)
    h, w = frame.shape[:2]
    return cv2.resize(frame, (W, int(h * W / w)))

def tile(cap, fps, times, label_fn):
    tiles = []
    for t in times:
        img = grab(cap, fps, t).copy()
        cv2.rectangle(img, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(img, label_fn(t), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    return np.vstack(rows)

for sid in sys.argv[1:]:
    r = N[sid]
    cap = cv2.VideoCapture(r["media"]); fps = cap.get(cv2.CAP_PROP_FPS)
    hist = r["history_timestamps"]; old = set(OLDF[sid]["history_timestamps"])
    k = min(24, len(hist))
    pick = [hist[round(i * (len(hist) - 1) / max(k - 1, 1))] for i in range(k)] if hist else []
    recent = r["recent_timestamps"][-8:]
    parts = [tile(cap, fps, recent, lambda t: f"RECENT {t:.1f}s")]
    if pick:
        parts.append(np.full((10, parts[0].shape[1], 3), 255, np.uint8))
        parts.append(tile(cap, fps, pick, lambda t: f"H {t:.0f}s" + (" (oldF)" if t in old else "")))
    cv2.imwrite(f"{sid}.jpg", np.vstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(sid, "ok", len(hist), "history frames")
