#!/usr/bin/env python3
"""
Quick sanity-check + optional visual preview for validation/data.json

Usage
-----
# just list status
python check_validation.py --data_root data

# list + show every valid pair (may open many windows!)
python check_validation.py --data_root data --show
"""
import argparse, json, sys
from pathlib import Path

try:
    import cv2, matplotlib.pyplot as plt   # only needed when --show
except ImportError:
    cv2 = plt = None                       # handled later

def preview(inp_path: Path, out_path: Path, colour: str) -> None:
    """Pop up a side-by-side preview using matplotlib + OpenCV."""
    if cv2 is None or plt is None:         # packages not installed
        return
    g  = cv2.imread(str(inp_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(cv2.imread(str(out_path)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(g,   cmap="gray"); ax[0].axis("off"); ax[0].set_title("polygon")
    ax[1].imshow(rgb); ax[1].axis("off"); ax[1].set_title(f"output ({colour})")
    plt.tight_layout(); plt.show()

def main(root: Path, show: bool) -> None:
    json_path = root / "validation" / "data.json"
    if not json_path.exists():
        sys.exit(f"❌  {json_path} not found")

    records = json.loads(json_path.read_text())
    problems = 0

    for i, rec in enumerate(records):
        # schema check
        for k in ("input_polygon", "output_image", "colour"):
            if k not in rec:
                print(f"[{i:04d}] ❌ missing '{k}' field → {rec}")
                problems += 1
                break
        else:
            inp, out, col = rec["input_polygon"], rec["output_image"], rec["colour"]
            inp_path = root / "validation" / "inputs"  / inp
            out_path = root / "validation" / "outputs" / out
            ok_inp, ok_out = inp_path.exists(), out_path.exists()
            status = "✓" if ok_inp and ok_out else "❌"
            if not (ok_inp and ok_out):
                problems += 1
            print(f"[{i:04d}] {status}  {inp} → {out}  ({col})")
            if show and ok_inp and ok_out:
                preview(inp_path, out_path, col)

    msg = f"\n✅  All {len(records)} validation samples look good!" \
          if problems == 0 else \
          f"\n⚠️  Found {problems} problem(s) in {len(records)} records."
    print(msg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data",
                    help="Folder with training/ and validation/ sub-folders")
    ap.add_argument("--show", action="store_true",
                    help="Also display each valid input/output pair")
    args = ap.parse_args()
    main(Path(args.data_root), args.show)
