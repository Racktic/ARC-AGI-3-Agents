"""Turn a MemoryAgent run's per-turn ACT frames into an animated GIF.

Usage:
    python make_gif.py <run_name>                 # memruns/<run_name>/
    python make_gif.py <images_dir> [--log LOG]    # explicit dir
    python make_gif.py <run_name> --ms 600 --out foo.gif

It strings together t1_act.png, t2_act.png, ... (the grid the model SAW each
turn) and captions each frame with "turn N -> <action chosen>" (read from the
run's io.log when available).
"""
import argparse
import glob
import json
import os
import re
import sys

from PIL import Image, ImageDraw


def turn_of(path: str) -> int:
    m = re.search(r"t(\d+)_act", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_actions(log_path: str) -> dict[int, str]:
    """Map turn -> action string by parsing the io.log ACT calls."""
    actions: dict[int, str] = {}
    if not log_path or not os.path.exists(log_path):
        return actions
    cur = None
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        m = re.search(r"\[ACT\]\s+turn~(\d+)", line)
        if m:
            cur = int(m.group(1))
            continue
        if cur is not None and "tool_use input:" in line:
            try:
                blob = line.split("tool_use input:", 1)[1].strip()
                d = json.loads(blob)
                a = d.get("action", "?")
                if "x" in d:
                    a += f"({d['x']},{d['y']})"
                actions[cur] = a
            except Exception:
                pass
            cur = None
    return actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under memruns/, or a directory of t*_act.png")
    ap.add_argument("--log", default=None, help="path to io.log (for action captions)")
    ap.add_argument("--ms", type=int, default=800, help="ms per frame")
    ap.add_argument("--out", default=None, help="output gif path")
    args = ap.parse_args()

    # resolve images dir + run dir
    if os.path.isdir(args.run):
        if glob.glob(os.path.join(args.run, "t*_act.png")):
            imgs_dir = args.run
            run_dir = os.path.dirname(imgs_dir.rstrip("/")) or "."
        else:
            run_dir = args.run
            imgs_dir = os.path.join(run_dir, "images")
    else:
        run_dir = os.path.join("memruns", args.run)
        imgs_dir = os.path.join(run_dir, "images")

    files = sorted(glob.glob(os.path.join(imgs_dir, "t*_act.png")), key=turn_of)
    if not files:
        sys.exit(f"no t*_act.png found in {imgs_dir}")

    log_path = args.log or os.path.join(run_dir, "io.log")
    actions = load_actions(log_path)

    frames = []
    for f in files:
        t = turn_of(f)
        im = Image.open(f).convert("RGB")
        W, H = im.size
        bar = 26
        canvas = Image.new("RGB", (W, H + bar), (255, 255, 255))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        cap = f"turn {t}   ->   {actions.get(t, '?')}"
        d.text((6, H + 7), cap, fill=(0, 0, 0))
        frames.append(canvas)

    out = args.out or os.path.join(run_dir, "actions.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.ms, loop=0)
    print(f"wrote {out}  ({len(frames)} frames @ {args.ms}ms, "
          f"captions from {'log' if actions else 'turn# only'})")


if __name__ == "__main__":
    main()
