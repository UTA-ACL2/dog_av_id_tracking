#!/usr/bin/env python3
"""
run_mot_eval.py

Runs TrackEval (HOTA/CLEAR/Identity) on a MOT-style eval folder WITHOUT
needing seqinfo.ini files or the standard MOTChallenge directory layout.

Expected input layout (exactly what you described):

    MOT_eval/
        gt/
            1.txt
            2.txt
            ...
            40.txt
        trackers/
            profile_tracker/
                1.txt
                2.txt
                ...
            raw_tracker/
                1.txt
                2.txt
                ...

Sequences whose gt/<seq>.txt is missing or empty are skipped automatically.

gt.txt / tracker .txt files should be comma-separated MOT format rows:
    frame,id,x,y,w,h,conf,class,visibility      (gt - class/vis optional)
    frame,id,x,y,w,h,conf,-1,-1,-1               (tracker output)

Missing trailing columns (conf/class/visibility) are auto-filled.
Frame numbers should start at 1 (standard MOT convention).

Usage:
    python run_mot_eval.py --root /path/to/MOT_eval
    python run_mot_eval.py --root /path/to/MOT_eval --trackers profile_tracker raw_tracker
    python run_mot_eval.py --root /path/to/MOT_eval --out results.csv

Requires: pip install trackeval --break-system-packages
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile

import numpy as np


def parse_mot_file(src_path):
    """Read a possibly-short MOT txt file, dedupe, and shift frames to be
    1-indexed if needed. Returns (rows, max_frame) where rows is a list of
    (frame_i, oid_i, x, y, w, h, conf, cls, vis) tuples, sorted by
    (frame, id). Track IDs that aren't plain integers (e.g. 'dog_1') are
    mapped to consistent integers within this file."""
    id_map = {}

    def to_int_id(raw_id):
        try:
            return int(float(raw_id))
        except ValueError:
            if raw_id not in id_map:
                id_map[raw_id] = len(id_map) + 1
            return id_map[raw_id]

    parsed_rows = []
    min_frame = None
    with open(src_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip() != ""]
            if len(parts) < 6:
                raise ValueError(
                    f"Line in {src_path} has fewer than 6 columns "
                    f"(need at least frame,id,x,y,w,h): {line}"
                )
            frame, oid, x, y, w, h = parts[0:6]
            conf = parts[6] if len(parts) > 6 else "1"
            cls = parts[7] if len(parts) > 7 else "1"
            vis = parts[8] if len(parts) > 8 else "1"

            frame_i = int(float(frame))
            min_frame = frame_i if min_frame is None else min(min_frame, frame_i)
            oid_i = to_int_id(oid)
            parsed_rows.append((frame_i, oid_i, x, y, w, h, conf, cls, vis))

    # MOT format is 1-indexed; shift up if this file starts at 0
    shift = 1 if min_frame == 0 else 0

    # Deduplicate: a given (frame, id) must appear at most once. If a tracker
    # (or gt) emits duplicates for the same id in the same frame, keep only
    # the highest-confidence box and drop the rest.
    best_by_key = {}
    dup_count = 0
    for frame_i, oid_i, x, y, w, h, conf, cls, vis in parsed_rows:
        frame_i += shift
        key = (frame_i, oid_i)
        try:
            conf_f = float(conf)
        except ValueError:
            conf_f = 1.0
        if key in best_by_key:
            dup_count += 1
            if conf_f <= best_by_key[key][1]:
                continue
        best_by_key[key] = ((frame_i, oid_i, x, y, w, h, conf, cls, vis), conf_f)

    if dup_count > 0:
        print(
            f"WARNING: {src_path} had {dup_count} duplicate (frame, id) row(s); "
            f"kept the highest-confidence box for each and dropped the rest"
        )

    rows = [r for r, _ in sorted(best_by_key.values(), key=lambda r: (r[0][0], r[0][1]))]
    max_frame = max((r[0] for r in rows), default=0)
    return rows, max_frame


def write_mot_rows(rows, dst_path, is_gt):
    lines = []
    for frame_i, oid_i, x, y, w, h, conf, cls, vis in rows:
        if is_gt:
            lines.append(f"{frame_i},{oid_i},{x},{y},{w},{h},{conf},{cls},{vis}")
        else:
            lines.append(f"{frame_i},{oid_i},{x},{y},{w},{h},{conf},-1,-1,-1")
    with open(dst_path, "w") as f:
        f.write("\n".join(lines))
    return max((r[0] for r in rows), default=0)


def tracker_file_path(trackers_src_dir, tracker, seq):
    """Tracker txt files may live directly under trackers/<tracker>/<seq>.txt
    or under trackers/<tracker>/data/<seq>.txt - support both."""
    data_sub = os.path.join(trackers_src_dir, tracker, "data", f"{seq}.txt")
    if os.path.isfile(data_sub):
        return data_sub
    return os.path.join(trackers_src_dir, tracker, f"{seq}.txt")


def stage_dataset(root, trackers, work_dir, only_sampled_frames=False):
    """Copy/normalize gt + tracker files into a TrackEval-friendly staging
    directory and return (gt_dir, trackers_dir, seq_info).

    If only_sampled_frames is True, GT rows are filtered down to only the
    frame numbers that appear anywhere in that sequence's tracker output
    (union across all trackers being evaluated) - i.e. only frames the
    tracker actually ran inference on, instead of every annotated GT frame."""
    gt_src_dir = os.path.join(root, "gt")
    trackers_src_dir = os.path.join(root, "trackers")

    if not os.path.isdir(gt_src_dir):
        sys.exit(f"ERROR: could not find gt folder at {gt_src_dir}")
    if not os.path.isdir(trackers_src_dir):
        sys.exit(f"ERROR: could not find trackers folder at {trackers_src_dir}")

    seqs = sorted(
        [
            os.path.splitext(fn)[0]
            for fn in os.listdir(gt_src_dir)
            if fn.endswith(".txt") and os.path.isfile(os.path.join(gt_src_dir, fn))
        ],
        key=lambda s: (len(s), s),
    )
    if not seqs:
        sys.exit(f"ERROR: no gt .txt files found directly in {gt_src_dir} (expected e.g. gt/1.txt, gt/2.txt, ...)")

    if trackers is None:
        trackers = sorted(
            [d for d in os.listdir(trackers_src_dir) if os.path.isdir(os.path.join(trackers_src_dir, d))]
        )
    if not trackers:
        sys.exit(f"ERROR: no tracker subfolders found in {trackers_src_dir}")

    gt_dir = os.path.join(work_dir, "gt")
    trackers_dir = os.path.join(work_dir, "trackers")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(trackers_dir, exist_ok=True)

    seq_info = {}
    used_seqs = []
    total_gt_rows_before = 0
    total_gt_rows_after = 0

    for seq in seqs:
        gt_file = os.path.join(gt_src_dir, f"{seq}.txt")

        if os.path.getsize(gt_file) == 0:
            print(f"WARNING: skipping sequence {seq}, gt/{seq}.txt is empty")
            continue

        # every tracker must have a file for this sequence
        missing = [t for t in trackers if not os.path.isfile(tracker_file_path(trackers_src_dir, t, seq))]
        if missing:
            print(f"WARNING: skipping sequence {seq}, missing tracker file(s) for: {missing}")
            continue

        # parse all tracker files for this sequence first (we need their
        # frame sets if only_sampled_frames is on, and their rows either way)
        tracker_parsed = {}
        sampled_frames = set()
        for t in trackers:
            src = tracker_file_path(trackers_src_dir, t, seq)
            rows, _ = parse_mot_file(src)
            tracker_parsed[t] = rows
            sampled_frames.update(r[0] for r in rows)

        gt_rows, _ = parse_mot_file(gt_file)
        total_gt_rows_before += len(gt_rows)

        if only_sampled_frames:
            gt_rows = [r for r in gt_rows if r[0] in sampled_frames]

        total_gt_rows_after += len(gt_rows)

        if not gt_rows:
            print(f"WARNING: skipping sequence {seq}, no gt rows left after filtering")
            continue

        gt_max_frame = write_mot_rows(gt_rows, os.path.join(gt_dir, f"{seq}.txt"), is_gt=True)

        seq_max_frame = gt_max_frame
        for t in trackers:
            t_out_dir = os.path.join(trackers_dir, t)
            os.makedirs(t_out_dir, exist_ok=True)
            t_max_frame = write_mot_rows(tracker_parsed[t], os.path.join(t_out_dir, f"{seq}.txt"), is_gt=False)
            seq_max_frame = max(seq_max_frame, t_max_frame)

        seq_info[seq] = seq_max_frame
        used_seqs.append(seq)

    if only_sampled_frames and total_gt_rows_before > 0:
        dropped = total_gt_rows_before - total_gt_rows_after
        pct = 100.0 * dropped / total_gt_rows_before
        print(
            f"only-sampled-frames: kept {total_gt_rows_after}/{total_gt_rows_before} "
            f"GT rows ({pct:.1f}% dropped as 'not sampled by tracker')"
        )

    if not used_seqs:
        sys.exit("ERROR: no valid sequences to evaluate (check warnings above)")

    return gt_dir, trackers_dir, seq_info, trackers


def run_eval(gt_dir, trackers_dir, seq_info, trackers, out_dir):
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["OUTPUT_DETAILED"] = False
    eval_config["PLOT_CURVES"] = False
    eval_config["OUTPUT_SUMMARY"] = False
    eval_config["USE_PARALLEL"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": gt_dir,
            "TRACKERS_FOLDER": trackers_dir,
            "OUTPUT_FOLDER": out_dir,
            "TRACKERS_TO_EVAL": trackers,
            "TRACKER_SUB_FOLDER": "",
            "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt",
            "CLASSES_TO_EVAL": ["pedestrian"],  # internal label only, not semantically used
            "SPLIT_TO_EVAL": "all",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": False,  # don't filter/relabel dets - evaluate everything as-is
            "SKIP_SPLIT_FOL": True,  # skip MOTxx-train style subfolders
            "SEQ_INFO": seq_info,  # <-- this is what avoids needing seqinfo.ini
        }
    )

    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator = trackeval.Evaluator(eval_config)
    results, _messages = evaluator.evaluate(dataset_list, metrics_list)
    return results


def summarize(results, trackers, out_csv=None):
    dataset_name = "MotChallenge2DBox"
    rows = []
    for tracker in trackers:
        seq_res = results[dataset_name][tracker]
        combined = seq_res["COMBINED_SEQ"]["pedestrian"]

        hota = np.mean(combined["HOTA"]["HOTA"])
        deta = np.mean(combined["HOTA"]["DetA"])
        assa = np.mean(combined["HOTA"]["AssA"])
        mota = combined["CLEAR"]["MOTA"]
        fp = combined["CLEAR"]["CLR_FP"]
        fn = combined["CLEAR"]["CLR_FN"]
        idsw = combined["CLEAR"]["IDSW"]
        idf1 = combined["Identity"]["IDF1"]

        rows.append(
            {
                "tracker": tracker,
                "HOTA": round(float(hota) * 100, 3),
                "DetA": round(float(deta) * 100, 3),
                "AssA": round(float(assa) * 100, 3),
                "MOTA": round(float(mota) * 100, 3),
                "IDF1": round(float(idf1) * 100, 3),
                "FP": int(fp),
                "FN": int(fn),
                "IDs": int(idsw),
            }
        )

    headers = ["tracker", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "FP", "FN", "IDs"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in headers}

    print("\n===== Summary (COMBINED over all sequences) =====")
    print("".join(h.ljust(widths[h]) for h in headers))
    for r in rows:
        print("".join(str(r[h]).ljust(widths[h]) for h in headers))

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV summary to {out_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run TrackEval (HOTA/MOTA/IDF1/...) with no seqinfo.ini needed.")
    parser.add_argument("--root", required=True, help="Path to MOT_eval folder (contains gt/ and trackers/)")
    parser.add_argument("--trackers", nargs="*", default=None, help="Tracker names to evaluate (default: all subfolders of trackers/)")
    parser.add_argument("--out", default=None, help="Path to write CSV summary (default: <root>/eval_results.csv)")
    parser.add_argument("--keep-staging", action="store_true", help="Keep the normalized staging folder for inspection")
    parser.add_argument(
        "--only-sampled-frames",
        action="store_true",
        help="Filter GT down to only frames present in tracker output for that sequence "
        "(use this if your tracker runs at reduced fps and skips frames on purpose)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_csv = args.out or os.path.join(root, "eval_results.csv")

    if args.keep_staging:
        work_dir = os.path.join(root, "_trackeval_staging")
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp(prefix="trackeval_staging_")

    try:
        gt_dir, trackers_dir, seq_info, trackers = stage_dataset(
            root, args.trackers, work_dir, only_sampled_frames=args.only_sampled_frames
        )
        print(f"Staged {len(seq_info)} sequence(s): {sorted(seq_info.keys())}")
        print(f"Evaluating tracker(s): {trackers}")

        eval_out_dir = os.path.join(work_dir, "eval_output")
        os.makedirs(eval_out_dir, exist_ok=True)
        results = run_eval(gt_dir, trackers_dir, seq_info, trackers, eval_out_dir)
        summarize(results, trackers, out_csv=out_csv)
    finally:
        if not args.keep_staging:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()