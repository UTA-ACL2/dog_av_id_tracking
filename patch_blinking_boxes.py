#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "01_visual" / "score_bark_events.py"

NEW_FUNCTION = '''def interpolate_box(
    track_rows: pd.DataFrame,
    target_frame: int,
    max_gap_frames: int,
) -> tuple[float, float, float, float] | None:
    """Fill short tracking gaps to reduce blinking boxes."""
    frames = track_rows["frame"].to_numpy(dtype=np.int64)
    boxes = track_rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)

    if len(frames) == 0:
        return None

    position = int(np.searchsorted(frames, target_frame))

    if position < len(frames) and int(frames[position]) == target_frame:
        return tuple(float(value) for value in boxes[position])

    left_index = position - 1 if position > 0 else None
    right_index = position if position < len(frames) else None

    if left_index is not None and right_index is not None:
        left_frame = int(frames[left_index])
        right_frame = int(frames[right_index])
        gap = right_frame - left_frame

        if (
            left_frame < target_frame < right_frame
            and gap > 0
            and gap <= 2 * max_gap_frames
        ):
            alpha = (target_frame - left_frame) / gap
            box = boxes[left_index] * (1.0 - alpha) + boxes[right_index] * alpha
            return tuple(float(value) for value in box)

    candidates = [i for i in (left_index, right_index) if i is not None]
    if not candidates:
        return None

    nearest = min(candidates, key=lambda i: abs(int(frames[i]) - target_frame))
    if abs(int(frames[nearest]) - target_frame) > max_gap_frames:
        return None

    return tuple(float(value) for value in boxes[nearest])
'''


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit(f"Target not found: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")
    pattern = re.compile(
        r"def interpolate_box\(\n.*?(?=\n\ndef crop_and_letterbox\()",
        flags=re.DOTALL,
    )
    match = pattern.search(original)
    if not match:
        raise SystemExit("Could not locate interpolate_box()")

    updated = original[:match.start()] + NEW_FUNCTION + original[match.end():]
    updated = updated.replace(
        'parser.add_argument("--max-box-gap-sec", type=float, default=0.5)',
        'parser.add_argument("--max-box-gap-sec", type=float, default=1.0)',
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"{TARGET.name}.before_blink_patch_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8")

    result = subprocess.run([sys.executable, "-m", "py_compile", str(TARGET)])
    if result.returncode != 0:
        shutil.copy2(backup, TARGET)
        raise SystemExit("Compilation failed; original restored.")

    print("Patched:", TARGET)
    print("Backup:", backup)
    print("Default max box gap is now 1.0 second.")


if __name__ == "__main__":
    main()
