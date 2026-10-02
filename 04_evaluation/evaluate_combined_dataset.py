#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-a-annotations-root', type=Path, required=True)
    p.add_argument('--dataset-a-workspace-root', type=Path, required=True)
    p.add_argument('--dataset-b-annotations-root', type=Path, required=True)
    p.add_argument('--dataset-b-manifest-root', type=Path, required=True)
    p.add_argument('--dataset-b-pipeline-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--spatial-iou-threshold', type=float, default=0.30)
    p.add_argument('--max-track-gap-sec', type=float, default=1.0)
    p.add_argument('--samples-per-event', type=int, default=15)
    return p.parse_args()


def pick(df, names, required=True):
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f'Missing one of {names}; columns={list(df.columns)}')
    return None


def div(a, b):
    return float(a / b) if b else float('nan')


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def state(v):
    s = (v or '').strip().lower().replace(' ', '_').replace('-', '_')
    if s in {'bark', 'barking'}:
        return 'barking'
    if s in {'not_barking', 'notbarking', 'silent'}:
        return 'not_barking'
    return s


def parse_xml(path, offset=0):
    root = ET.parse(path).getroot()
    tracks = []
    for t in root.findall('track'):
        rows = []
        for b in t.findall('box'):
            if b.attrib.get('outside', '0') == '1':
                continue
            st = ''
            for a in b.findall('attribute'):
                if a.attrib.get('name', '').strip().lower() == 'vocalization':
                    st = state(a.text)
                    break
            rows.append((
                int(b.attrib['frame']) + offset,
                float(b.attrib['xtl']), float(b.attrib['ytl']),
                float(b.attrib['xbr']), float(b.attrib['ybr']), st,
            ))
        if rows:
            rows.sort(key=lambda x: x[0])
            tracks.append({'id': int(t.attrib.get('id', len(tracks))), 'rows': rows})
    return tracks


def gt_at(track, frame):
    rows = track['rows']
    frames = [r[0] for r in rows]
    pos = int(np.searchsorted(frames, frame))
    if pos < len(rows) and rows[pos][0] == frame:
        r = rows[pos]
        return r[5], (r[1], r[2], r[3], r[4])
    if pos == 0 or pos >= len(rows):
        return '', None
    l, r = rows[pos-1], rows[pos]
    if not (l[0] < frame < r[0]) or r[0] - l[0] > 2:
        return '', None
    if l[5] != r[5]:
        return '', None
    alpha = (frame-l[0])/(r[0]-l[0])
    box = tuple((1-alpha)*l[i] + alpha*r[i] for i in range(1, 5))
    return l[5], box


def load_tracks(path):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        'track_id': pd.to_numeric(df[pick(df, ['track_id','track','id'])], errors='raise').astype(int),
        'frame': pd.to_numeric(df[pick(df, ['frame','frame_idx','frame_id'])], errors='raise').round().astype(int),
        'x1': pd.to_numeric(df[pick(df, ['x1','xtl','left'])], errors='raise'),
        'y1': pd.to_numeric(df[pick(df, ['y1','ytl','top'])], errors='raise'),
        'x2': pd.to_numeric(df[pick(df, ['x2','xbr','right'])], errors='raise'),
        'y2': pd.to_numeric(df[pick(df, ['y2','ybr','bottom'])], errors='raise'),
    })
    if 'time_sec' in df.columns:
        out['time_sec'] = pd.to_numeric(df['time_sec'], errors='coerce')
    return out.sort_values(['track_id','frame']).drop_duplicates(['track_id','frame'])


def load_preds(path):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        'event_id': df[pick(df, ['event_id','barkseq_id','bark_id'])].astype(str),
        'start_time': pd.to_numeric(df[pick(df, ['start_time_sec','start_time','start_sec','start'])], errors='raise'),
        'end_time': pd.to_numeric(df[pick(df, ['end_time_sec','end_time','end_sec','end'])], errors='raise'),
    })
    sf = pick(df, ['start_frame','frame_start'], False)
    ef = pick(df, ['end_frame','frame_end'], False)
    tr = pick(df, ['predicted_track_id','track_id','predicted_id'], False)
    st = pick(df, ['status','prediction_status'], False)
    out['start_frame'] = pd.to_numeric(df[sf], errors='coerce') if sf else np.nan
    out['end_frame'] = pd.to_numeric(df[ef], errors='coerce') if ef else np.nan
    out['predicted_track_id'] = pd.to_numeric(df[tr], errors='coerce') if tr else np.nan
    out['status'] = df[st].astype(str) if st else 'predicted'
    return out


def fps_of(preds, tracks):
    valid = preds[(preds.start_frame.notna()) & (preds.start_time > 0)]
    if not valid.empty:
        vals = valid.start_frame / valid.start_time
        vals = vals[(vals > 1) & (vals < 240)]
        if len(vals): return float(vals.median())
    if 'time_sec' in tracks.columns:
        valid = tracks[(tracks.time_sec.notna()) & (tracks.time_sec > 0)]
        if not valid.empty:
            vals = valid.frame / valid.time_sec
            vals = vals[(vals > 1) & (vals < 240)]
            if len(vals): return float(vals.median())
    return 30.0


def pred_box(rows, frame, max_gap):
    frames = rows.frame.to_numpy(int)
    boxes = rows[['x1','y1','x2','y2']].to_numpy(float)
    pos = int(np.searchsorted(frames, frame))
    if pos < len(frames) and frames[pos] == frame:
        return tuple(boxes[pos])
    li = pos-1 if pos > 0 else None
    ri = pos if pos < len(frames) else None
    if li is not None and ri is not None:
        gap = frames[ri]-frames[li]
        if frames[li] < frame < frames[ri] and 0 < gap <= 2*max_gap:
            a = (frame-frames[li])/gap
            return tuple((1-a)*boxes[li] + a*boxes[ri])
    cand = [x for x in [li,ri] if x is not None]
    if not cand: return None
    idx = min(cand, key=lambda x: abs(frames[x]-frame))
    return tuple(boxes[idx]) if abs(frames[idx]-frame) <= max_gap else None


def sample_frames(row, fps, n):
    if pd.notna(row.start_frame) and pd.notna(row.end_frame):
        a, b = int(round(row.start_frame)), int(round(row.end_frame))
    else:
        a, b = int(math.floor(row.start_time*fps)), int(math.ceil(row.end_time*fps))
    if b <= a: return [a]
    return sorted({int(round(x)) for x in np.linspace(a, b, min(n, b-a+1))})


def resolve_dataset_a(work, vid):
    bases = [work/f'{vid}_base', work/f'v2_results_{vid}', work/f'results_{vid}']
    profs = [work/f'{vid}_profiles', work/f'v2_4_1_profiles_{vid}']
    fuses = [work/f'{vid}_fusion', work/f'v2_4_1_fused_{vid}']
    pred = next((d/'predictions.csv' for d in bases if (d/'predictions.csv').is_file()), None)
    tr = next((d/'tracks.csv' for d in bases if (d/'tracks.csv').is_file()), None)
    pm = next((d/'track_to_all_dog_profile.csv' for d in profs if (d/'track_to_all_dog_profile.csv').is_file()), None)
    fs = next((d/'pipeline_summary_v2.csv' for d in fuses if (d/'pipeline_summary_v2.csv').is_file()), None)
    return pred, tr, pm, fs


def resolve_dataset_b(root, vid):
    pred = root/f'{vid}_base/predictions.csv'
    tr = root/f'{vid}_base/tracks.csv'
    pm = root/f'{vid}_profiles/track_to_all_dog_profile.csv'
    fs = root/f'{vid}_fusion/pipeline_summary_v2.csv'
    return pred, tr, pm if pm.is_file() else None, fs if fs.is_file() else None


def evaluate(dataset, vid, pred_path, track_path, gt_provider, profile_map, fusion_summary, threshold, gap_sec, samples):
    preds, tracks = load_preds(pred_path), load_tracks(track_path)
    fps = fps_of(preds, tracks)
    grouped = {int(k): v.reset_index(drop=True) for k,v in tracks.groupby('track_id')}
    max_gap = max(1, int(round(gap_sec*fps)))
    events = []
    for _, e in preds.iterrows():
        gt_tracks = gt_provider(e.event_id)

        # Skip source bark events that were not manually annotated in CVAT.
        if gt_tracks is None:
            continue

        visible_frames, scores = 0, []
        pred_visible = pd.notna(e.predicted_track_id) and str(e.status).lower() not in {'offscreen','outside','outside/offscreen','outside_or_offscreen'}
        for frame in sample_frames(e, fps, samples):
            gt_boxes = []
            for gt in gt_tracks:
                st, box = gt_at(gt, frame)
                if st == 'barking' and box is not None:
                    gt_boxes.append(box)
            if gt_boxes: visible_frames += 1
            if not pred_visible or not gt_boxes: continue
            rows = grouped.get(int(e.predicted_track_id))
            if rows is None: continue
            pb = pred_box(rows, frame, max_gap)
            if pb is not None: scores.append(max(iou(pb, gb) for gb in gt_boxes))
        gt_visible = visible_frames > 0
        mean_iou = float(np.mean(scores)) if scores else float('nan')
        events.append({
            'dataset': dataset, 'video_id': vid, 'event_id': e.event_id,
            'gt_visible_barker': int(gt_visible), 'gt_offscreen': int(not gt_visible),
            'predicted_visible_barker': int(pred_visible), 'predicted_offscreen': int(not pred_visible),
            'mean_spatial_iou': mean_iou,
            'localization_correct': int(gt_visible and pred_visible and scores and mean_iou >= threshold),
        })
    temp = int(tracks.track_id.nunique())
    persistent = merged = np.nan
    if profile_map and profile_map.is_file():
        pm = pd.read_csv(profile_map)
        col = pick(pm, ['all_dog_profile_id','persistent_profile_id','profile_id','all_dog_id'], False)
        if col:
            persistent = int(pm[col].nunique()); merged = temp-persistent
    fusion = {'dataset': dataset, 'video_id': vid}
    if fusion_summary and fusion_summary.is_file():
        df = pd.read_csv(fusion_summary)
        if not df.empty: fusion.update(df.iloc[0].to_dict())
    return events, {'dataset':dataset,'video_id':vid,'temporary_tracks':temp,'persistent_profiles':persistent,'fragments_merged':merged}, fusion


def summarize(events, tracks, fusion, label):
    e = events[events.dataset.eq(label)] if label != 'combined' else events
    t = tracks[tracks.dataset.eq(label)] if label != 'combined' else tracks
    f = fusion[fusion.dataset.eq(label)] if label != 'combined' else fusion
    tp = int(((e.gt_visible_barker==1)&(e.predicted_visible_barker==1)).sum())
    fp = int(((e.gt_visible_barker==0)&(e.predicted_visible_barker==1)).sum())
    fn = int(((e.gt_visible_barker==1)&(e.predicted_visible_barker==0)).sum())
    tn = int(((e.gt_visible_barker==0)&(e.predicted_visible_barker==0)).sum())
    p, r = div(tp,tp+fp), div(tp,tp+fn)
    visible = int((e.gt_visible_barker==1).sum())
    out = {
        'group':label, 'videos_evaluated':int(e.video_id.nunique()), 'bark_events_total':len(e),
        'gt_visible_bark_events':visible, 'gt_offscreen_bark_events':int((e.gt_offscreen==1).sum()),
        'correctly_localized_visible_events':int(e.localization_correct.sum()),
        'localization_accuracy':div(e.localization_correct.sum(), visible),
        'mean_spatial_iou':float(e.loc[e.gt_visible_barker.eq(1),'mean_spatial_iou'].mean()),
        'visible_precision':p, 'visible_recall':r,
        'visible_f1':div(2*p*r,p+r) if np.isfinite(p) and np.isfinite(r) else np.nan,
        'visible_offscreen_accuracy':div(tp+tn,tp+fp+fn+tn),
        'temporary_tracks_total':float(t.temporary_tracks.sum()),
        'persistent_profiles_total':float(t.persistent_profiles.sum()),
        'fragments_merged_total':float(t.fragments_merged.sum()),
    }
    fusion_names = {
        "bark_events": "fusion_bark_events_total",
        "assigned_bark_events": "assigned_bark_events_total",
        "audio_visual_conflicts": "audio_visual_conflicts_total",
        "offscreen_audio_only_profiles": "offscreen_audio_only_profiles_total",
    }

    for col, output_name in fusion_names.items():
        if col in f.columns:
            out[output_name] = float(
                pd.to_numeric(f[col], errors="coerce").sum()
            )

    if (
        "fusion_bark_events_total" in out
        and "assigned_bark_events_total" in out
    ):
        out["fusion_assignment_rate"] = div(
            out["assigned_bark_events_total"],
            out["fusion_bark_events_total"],
        )
    if 'audio_visual_conflicts_total' in out and 'assigned_bark_events_total' in out:
        out['audio_visual_conflict_rate'] = div(out['audio_visual_conflicts_total'], out['assigned_bark_events_total'])
    return out


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    event_rows=[]; track_rows=[]; fusion_rows=[]; failures=[]

    for d in sorted([x for x in a.dataset_a_annotations_root.iterdir() if x.is_dir() and x.name.isdigit()], key=lambda x:int(x.name)):
        vid=d.name; pred,tr,pm,fs=resolve_dataset_a(a.dataset_a_workspace_root,vid)
        if not pred or not tr or not (d/'annotations.xml').is_file():
            failures.append({'dataset':'dataset_a','video_id':vid,'error':'missing inputs'}); continue
        try:
            gt=parse_xml(d/'annotations.xml')
            ev,st,fu=evaluate('dataset_a',vid,pred,tr,lambda _id,gt=gt:gt,pm,fs,a.spatial_iou_threshold,a.max_track_gap_sec,a.samples_per_event)
            event_rows += ev; track_rows.append(st); fusion_rows.append(fu)
            print(f'DATASET_A {vid}: complete')
        except Exception as ex:
            failures.append({'dataset':'dataset_a','video_id':vid,'error':repr(ex)})

    for d in sorted([x for x in a.dataset_b_annotations_root.iterdir() if x.is_dir() and x.name.isdigit()], key=lambda x:int(x.name)):
        vid=d.name; pred,tr,pm,fs=resolve_dataset_b(a.dataset_b_pipeline_root,vid)
        mf=a.dataset_b_manifest_root/vid/f'{vid}_cvat_manifest.csv'
        if not pred.is_file() or not tr.is_file() or not mf.is_file():
            failures.append({'dataset':'dataset_b','video_id':vid,'error':'missing inputs'}); continue
        try:
            manifest=pd.read_csv(mf)
            def provider(event_id, vid=vid, manifest=manifest):
                bark=int(str(event_id).rsplit('_',1)[-1])
                row=manifest[pd.to_numeric(manifest.bark_id,errors='coerce').eq(bark)].iloc[0]
                expected_name = f"{vid}_bark_{bark:03d}.mp4"
                video_annotation_root = a.dataset_b_annotations_root / vid

                matching_videos = list(
                    video_annotation_root.glob(f"*/{expected_name}")
                )

                if len(matching_videos) == 0:
                    # This bark event exists in the manifest but was not
                    # included among the manually annotated CVAT clips.
                    return None

                if len(matching_videos) > 1:
                    raise RuntimeError(
                        f"Multiple annotation clips found for "
                        f"video={vid}, bark={bark}: {matching_videos}"
                    )

                xml_path = matching_videos[0].parent / "annotations.xml"

                return parse_xml(
                    xml_path,
                    int(row.clip_start_frame),
                )
            ev,st,fu=evaluate('dataset_b',vid,pred,tr,provider,pm,fs,a.spatial_iou_threshold,a.max_track_gap_sec,a.samples_per_event)
            event_rows += ev; track_rows.append(st); fusion_rows.append(fu)
            print(f'DATASET_B {vid}: complete')
        except Exception as ex:
            failures.append({'dataset':'dataset_b','video_id':vid,'error':repr(ex)})

    events=pd.DataFrame(event_rows); tracks=pd.DataFrame(track_rows); fusion=pd.DataFrame(fusion_rows); fail=pd.DataFrame(failures)
    events.to_csv(a.output_dir/'per_event_metrics.csv',index=False)
    tracks.to_csv(a.output_dir/'track_profile_statistics.csv',index=False)
    fusion.to_csv(a.output_dir/'fusion_statistics.csv',index=False)
    fail.to_csv(a.output_dir/'failures.csv',index=False)
    summaries=[summarize(events,tracks,fusion,x) for x in ['dataset_a','dataset_b','combined'] if x=='combined' or (events.dataset==x).any()]
    pd.DataFrame(summaries).to_csv(a.output_dir/'overall_summary_by_dataset.csv',index=False)
    (a.output_dir/'overall_summary_by_dataset.json').write_text(json.dumps(summaries,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False)); print('Saved:',a.output_dir)


if __name__ == '__main__':
    main()
