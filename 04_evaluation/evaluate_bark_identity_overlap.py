#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def pick_col(df, explicit, candidates, required=True):
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Missing column {explicit}. Available: {df.columns.tolist()}")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise KeyError(f"Could not detect column from {candidates}. Available: {df.columns.tolist()}")
    return None


def tiou(a0, a1, b0, b1):
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap, overlap / union if union > 0 else 0.0


def normalize_gt(df, args):
    s = pick_col(df, args.gt_start_col, ["start_time_sec","start_time","start"])
    e = pick_col(df, args.gt_end_col, ["end_time_sec","end_time","end"])
    d = pick_col(df, args.gt_dog_col, ["dog_id","annotated_dog_id","identity","barking_dog_id","label"])
    v = pick_col(df, args.gt_video_col, ["video_id","video_identifier","video"], required=False)
    ev = pick_col(df, args.gt_event_col, ["event_id","bark_id","id"], required=False)
    out = pd.DataFrame({
        "video_id": df[v].astype(str) if v else "__single_video__",
        "gt_event_id": df[ev].astype(str) if ev else [f"gt_{i:06d}" for i in range(len(df))],
        "gt_start": pd.to_numeric(df[s], errors="coerce"),
        "gt_end": pd.to_numeric(df[e], errors="coerce"),
        "gt_dog_id": df[d].astype(str),
    }).dropna(subset=["gt_start","gt_end","gt_dog_id"])
    return out.reset_index(drop=True)


def normalize_pred(df, args):
    s = pick_col(df, args.pred_start_col, ["start_time_sec","start_time","start"])
    e = pick_col(df, args.pred_end_col, ["end_time_sec","end_time","end"])
    p = pick_col(df, args.pred_profile_col, ["all_dog_profile_id","persistent_profile_id","profile_id"])
    v = pick_col(df, args.pred_video_col, ["video_id","video_identifier","video"], required=False)
    ev = pick_col(df, args.pred_event_col, ["event_id","bark_id","id"], required=False)
    out = pd.DataFrame({
        "video_id": df[v].astype(str) if v else "__single_video__",
        "pred_event_id": df[ev].astype(str) if ev else [f"pred_{i:06d}" for i in range(len(df))],
        "pred_start": pd.to_numeric(df[s], errors="coerce"),
        "pred_end": pd.to_numeric(df[e], errors="coerce"),
        "pred_profile_id": df[p].astype(str),
    }).dropna(subset=["pred_start","pred_end","pred_profile_id"])
    return out.reset_index(drop=True)


def match_events(gt, pred, threshold):
    if gt.empty or pred.empty:
        return pd.DataFrame(), gt.copy(), pred.copy()
    scores = np.zeros((len(gt), len(pred)))
    overlaps = np.zeros_like(scores)
    for i, g in gt.iterrows():
        for j, p in pred.iterrows():
            overlaps[i,j], scores[i,j] = tiou(g.gt_start, g.gt_end, p.pred_start, p.pred_end)
    rr, cc = linear_sum_assignment(-scores)
    matches, used_g, used_p = [], set(), set()
    for i, j in zip(rr, cc):
        if scores[i,j] < threshold:
            continue
        g, p = gt.iloc[i], pred.iloc[j]
        matches.append({
            "video_id": str(g.video_id),
            "gt_event_id": g.gt_event_id,
            "pred_event_id": p.pred_event_id,
            "gt_start": g.gt_start,
            "gt_end": g.gt_end,
            "pred_start": p.pred_start,
            "pred_end": p.pred_end,
            "overlap_sec": overlaps[i,j],
            "temporal_iou": scores[i,j],
            "gt_dog_id": g.gt_dog_id,
            "pred_profile_id": p.pred_profile_id,
        })
        used_g.add(i); used_p.add(j)
    ug = gt.loc[[i for i in range(len(gt)) if i not in used_g]].copy()
    up = pred.loc[[j for j in range(len(pred)) if j not in used_p]].copy()
    return pd.DataFrame(matches), ug, up


def align_ids(matches):
    if matches.empty:
        return {}, pd.DataFrame()
    preds = sorted(matches.pred_profile_id.unique())
    gts = sorted(matches.gt_dog_id.unique())
    mat = np.zeros((len(preds), len(gts)))
    for i,p in enumerate(preds):
        for j,g in enumerate(gts):
            mat[i,j] = ((matches.pred_profile_id == p) & (matches.gt_dog_id == g)).sum()
    n = max(mat.shape)
    padded = np.zeros((n,n)); padded[:mat.shape[0], :mat.shape[1]] = mat
    rr, cc = linear_sum_assignment(-padded)
    mapping, rows = {}, []
    for i,j in zip(rr,cc):
        if i >= len(preds):
            continue
        mapped = gts[j] if j < len(gts) and mat[i,j] > 0 else "__unmapped__"
        mapping[preds[i]] = mapped
        rows.append({"pred_profile_id": preds[i], "mapped_gt_dog_id": mapped,
                     "matched_event_count": mat[i,j] if j < len(gts) else 0})
    return mapping, pd.DataFrame(rows)


def div(a,b):
    return a/b if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.30)
    ap.add_argument("--gt-start-col"); ap.add_argument("--gt-end-col")
    ap.add_argument("--gt-dog-col"); ap.add_argument("--gt-video-col")
    ap.add_argument("--gt-event-col")
    ap.add_argument("--pred-start-col"); ap.add_argument("--pred-end-col")
    ap.add_argument("--pred-profile-col"); ap.add_argument("--pred-video-col")
    ap.add_argument("--pred-event-col")
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    gt = normalize_gt(pd.read_csv(args.annotations), args)
    pred = normalize_pred(pd.read_csv(args.predictions), args)

    if set(gt.video_id) == {"__single_video__"} and len(set(pred.video_id)-{"__single_video__"}) == 1:
        gt["video_id"] = next(iter(set(pred.video_id)-{"__single_video__"}))
    if set(pred.video_id) == {"__single_video__"} and len(set(gt.video_id)-{"__single_video__"}) == 1:
        pred["video_id"] = next(iter(set(gt.video_id)-{"__single_video__"}))

    videos = sorted(set(gt.video_id) | set(pred.video_id))
    all_m, all_map, all_ug, all_up, metrics = [], [], [], [], []

    for vid in videos:
        g = gt[gt.video_id == vid].reset_index(drop=True)
        p = pred[pred.video_id == vid].reset_index(drop=True)
        m, ug, up = match_events(g,p,args.iou_threshold)
        mapping, map_df = align_ids(m)
        if not m.empty:
            m["mapped_gt_dog_id"] = m.pred_profile_id.map(mapping).fillna("__unmapped__")
            m["identity_correct"] = m.mapped_gt_dog_id == m.gt_dog_id
            all_m.append(m)
        if not map_df.empty:
            map_df.insert(0,"video_id",vid); all_map.append(map_df)
        if not ug.empty:
            ug.insert(0,"evaluation_video_id",vid); all_ug.append(ug)
        if not up.empty:
            up.insert(0,"evaluation_video_id",vid); all_up.append(up)

        matched = len(m)
        correct = int(m.identity_correct.sum()) if not m.empty else 0
        prec, rec = div(matched,len(p)), div(matched,len(g))
        f1 = 2*prec*rec/(prec+rec) if matched and prec+rec else 0.0
        metrics.append({
            "video_id": vid,
            "gt_events": len(g),
            "predicted_events": len(p),
            "matched_events": matched,
            "false_negatives": len(ug),
            "false_positives": len(up),
            "event_precision": prec,
            "event_recall": rec,
            "event_f1": f1,
            "mean_temporal_iou": float(m.temporal_iou.mean()) if not m.empty else np.nan,
            "identity_correct": correct,
            "identity_accuracy_on_matched": div(correct,matched),
            "end_to_end_identity_accuracy": div(correct,len(g)),
            "gt_unique_dogs": g.gt_dog_id.nunique(),
            "pred_unique_profiles": p.pred_profile_id.nunique(),
        })

    matches = pd.concat(all_m, ignore_index=True) if all_m else pd.DataFrame()
    mappings = pd.concat(all_map, ignore_index=True) if all_map else pd.DataFrame()
    ug = pd.concat(all_ug, ignore_index=True) if all_ug else pd.DataFrame()
    up = pd.concat(all_up, ignore_index=True) if all_up else pd.DataFrame()
    per_video = pd.DataFrame(metrics)

    total_gt = int(per_video.gt_events.sum())
    total_pred = int(per_video.predicted_events.sum())
    total_match = int(per_video.matched_events.sum())
    total_correct = int(per_video.identity_correct.sum())
    prec, rec = div(total_match,total_pred), div(total_match,total_gt)
    f1 = 2*prec*rec/(prec+rec) if total_match and prec+rec else 0.0

    aggregate = pd.DataFrame([{
        "videos": len(videos),
        "gt_events": total_gt,
        "predicted_events": total_pred,
        "matched_events": total_match,
        "false_negatives": total_gt-total_match,
        "false_positives": total_pred-total_match,
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": f1,
        "mean_temporal_iou": float(matches.temporal_iou.mean()) if not matches.empty else np.nan,
        "identity_correct": total_correct,
        "identity_accuracy_on_matched": div(total_correct,total_match),
        "end_to_end_identity_accuracy": div(total_correct,total_gt),
        "iou_threshold": args.iou_threshold,
    }])

    confusion = pd.DataFrame()
    if not matches.empty:
        confusion = (matches.groupby(
            ["video_id","gt_dog_id","pred_profile_id","mapped_gt_dog_id","identity_correct"]
        ).size().reset_index(name="event_count"))

    matches.to_csv(out/"event_matches.csv", index=False)
    mappings.to_csv(out/"identity_mapping.csv", index=False)
    per_video.to_csv(out/"per_video_metrics.csv", index=False)
    aggregate.to_csv(out/"aggregate_metrics.csv", index=False)
    confusion.to_csv(out/"identity_confusion.csv", index=False)
    ug.to_csv(out/"unmatched_annotations.csv", index=False)
    up.to_csv(out/"unmatched_predictions.csv", index=False)

    r = aggregate.iloc[0]
    summary = f"""BARK IDENTITY OVERLAP EVALUATION

IoU threshold: {r.iou_threshold:.2f}
GT events: {int(r.gt_events)}
Predicted events: {int(r.predicted_events)}
Matched events: {int(r.matched_events)}
Event precision: {100*r.event_precision:.2f}%
Event recall: {100*r.event_recall:.2f}%
Event F1: {100*r.event_f1:.2f}%
Mean temporal IoU: {r.mean_temporal_iou:.4f}
Identity accuracy on matched events: {100*r.identity_accuracy_on_matched:.2f}%
End-to-end identity accuracy: {100*r.end_to_end_identity_accuracy:.2f}%
"""
    (out/"evaluation_summary.txt").write_text(summary)
    print(summary)
    print("Saved:", out)


if __name__ == "__main__":
    main()
