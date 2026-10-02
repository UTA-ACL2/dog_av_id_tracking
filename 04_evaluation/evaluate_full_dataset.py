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
    p.add_argument('--annotations-root', type=Path, required=True)
    p.add_argument('--pipeline-output-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--spatial-iou-threshold', type=float, default=0.30)
    p.add_argument('--max-track-gap-sec', type=float, default=1.0)
    p.add_argument('--samples-per-event', type=int, default=15)
    return p.parse_args()


def pick(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
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
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def norm_state(v):
    v = (v or '').strip().lower().replace(' ', '_').replace('-', '_')
    if v in {'bark', 'barking'}:
        return 'barking'
    if v in {'silent', 'not_barking', 'notbarking'}:
        return 'not_barking'
    return v


def parse_cvat(xml_path):
    root = ET.parse(xml_path).getroot()
    tracks = []
    for t in root.findall('track'):
        rows = []
        for b in t.findall('box'):
            if b.attrib.get('outside', '0') == '1':
                continue
            state = ''
            for a in b.findall('attribute'):
                if a.attrib.get('name', '').strip().lower() == 'vocalization':
                    state = norm_state(a.text)
                    break
            rows.append((
                int(b.attrib['frame']),
                float(b.attrib['xtl']), float(b.attrib['ytl']),
                float(b.attrib['xbr']), float(b.attrib['ybr']),
                state,
            ))
        if rows:
            rows.sort(key=lambda x: x[0])
            tracks.append({'id': int(t.attrib.get('id', len(tracks))), 'rows': rows})
    return tracks


def cvat_at(track, frame):
    rows = track['rows']
    frames = [r[0] for r in rows]
    pos = int(np.searchsorted(frames, frame))
    if pos < len(rows) and rows[pos][0] == frame:
        r = rows[pos]
        return (r[1], r[2], r[3], r[4]), r[5]
    if pos == 0 or pos >= len(rows):
        return None, ''
    l, r = rows[pos-1], rows[pos]
    if not (l[0] < frame < r[0]):
        return None, ''
    gap = r[0] - l[0]
    if gap <= 0:
        return None, ''
    alpha = (frame-l[0]) / gap
    box = tuple(float(l[i]*(1-alpha) + r[i]*alpha) for i in range(1, 5))
    state = l[5] if l[5] == r[5] else ''
    return box, state


def normalize_tracks(path):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out['track_id'] = pd.to_numeric(df[pick(df, ['track_id','track','id'])], errors='raise').astype(int)
    out['frame'] = pd.to_numeric(df[pick(df, ['frame','frame_idx','frame_id'])], errors='raise').round().astype(int)
    for c, names in {
        'x1':['x1','xtl','left'], 'y1':['y1','ytl','top'],
        'x2':['x2','xbr','right'], 'y2':['y2','ybr','bottom']
    }.items():
        out[c] = pd.to_numeric(df[pick(df, names)], errors='raise').astype(float)
    if 'time_sec' in df.columns:
        out['time_sec'] = pd.to_numeric(df['time_sec'], errors='coerce')
    return out.dropna(subset=['track_id','frame','x1','y1','x2','y2']).sort_values(['track_id','frame'])


def infer_fps(raw_tracks):
    if 'time_sec' in raw_tracks.columns:
        valid = raw_tracks['time_sec'] > 0
        est = raw_tracks.loc[valid, 'frame'] / raw_tracks.loc[valid, 'time_sec']
        est = est[np.isfinite(est) & (est > 1)]
        if len(est):
            return float(np.median(est))
    return 30.0


def track_box(rows, frame, max_gap):
    frames = rows['frame'].to_numpy(np.int64)
    boxes = rows[['x1','y1','x2','y2']].to_numpy(float)
    if not len(frames):
        return None
    pos = int(np.searchsorted(frames, frame))
    if pos < len(frames) and int(frames[pos]) == frame:
        return tuple(map(float, boxes[pos]))
    li = pos-1 if pos > 0 else None
    ri = pos if pos < len(frames) else None
    if li is not None and ri is not None:
        lf, rf = int(frames[li]), int(frames[ri])
        gap = rf-lf
        if lf < frame < rf and 0 < gap <= 2*max_gap:
            a = (frame-lf)/gap
            return tuple(map(float, boxes[li]*(1-a)+boxes[ri]*a))
    candidates = [i for i in (li,ri) if i is not None]
    if not candidates:
        return None
    ni = min(candidates, key=lambda i: abs(int(frames[i])-frame))
    if abs(int(frames[ni])-frame) > max_gap:
        return None
    return tuple(map(float, boxes[ni]))


def normalize_predictions(path):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out['event_id'] = df[pick(df, ['event_id','barkseq_id','bark_id'])].astype(str)
    out['start'] = pd.to_numeric(df[pick(df, ['start_time_sec','start_time','start_sec','start'])])
    out['end'] = pd.to_numeric(df[pick(df, ['end_time_sec','end_time','end_sec','end'])])
    tc = pick(df, ['predicted_track_id','track_id','predicted_id'], required=False)
    sc = pick(df, ['status','prediction_status'], required=False)
    out['predicted_track_id'] = pd.to_numeric(df[tc], errors='coerce') if tc else np.nan
    out['status'] = df[sc].astype(str) if sc else 'predicted'
    return out


def sample_frames(start, end, fps, n):
    a, b = int(math.floor(start*fps)), int(math.ceil(end*fps))
    if b <= a:
        return [a]
    return sorted({int(round(x)) for x in np.linspace(a, b, min(n, b-a+1))})


def profile_count(path):
    if not path.is_file():
        return float('nan')
    df = pd.read_csv(path)
    c = pick(df, ['all_dog_profile_id','persistent_profile_id','profile_id','all_dog_id'], required=False)
    return int(df[c].nunique()) if c else float('nan')


def evaluate_video(video_id, ann, base, profiles, fusion, threshold, max_gap_sec, samples):
    cvat = parse_cvat(ann)
    pred = normalize_predictions(base/'predictions.csv')
    tracks = normalize_tracks(base/'tracks.csv')
    fps = infer_fps(tracks)
    max_gap = max(1, int(round(max_gap_sec*fps)))
    grouped = {int(t): g.reset_index(drop=True) for t,g in tracks.groupby('track_id')}

    event_rows = []
    for _, e in pred.iterrows():
        frames = sample_frames(float(e.start), float(e.end), fps, samples)
        status = str(e.status).strip().lower()
        pt = e.predicted_track_id
        predicted_visible = pd.notna(pt) and status not in {'outside_or_offscreen','offscreen','outside'}
        gt_visible_frames = 0
        ious = []
        matched_ids = []
        for f in frames:
            gt = []
            for t in cvat:
                box, state = cvat_at(t, f)
                if box is not None and state == 'barking':
                    gt.append((t['id'], box))
            if gt:
                gt_visible_frames += 1
            if not (predicted_visible and gt):
                continue
            prow = grouped.get(int(pt))
            if prow is None:
                continue
            pbox = track_box(prow, f, max_gap)
            if pbox is None:
                continue
            best_id, best_iou = max(((gid, iou(pbox, gbox)) for gid,gbox in gt), key=lambda x: x[1])
            ious.append(float(best_iou)); matched_ids.append(best_id)
        gt_visible = gt_visible_frames > 0
        mean_iou = float(np.mean(ious)) if ious else float('nan')
        correct = bool(gt_visible and predicted_visible and ious and mean_iou >= threshold)
        dominant = int(pd.Series(matched_ids).mode().iloc[0]) if matched_ids else ''
        event_rows.append({
            'video_id':video_id, 'event_id':e.event_id,
            'start_time_sec':float(e.start), 'end_time_sec':float(e.end),
            'prediction_status':e.status,
            'predicted_track_id':int(pt) if pd.notna(pt) else '',
            'gt_visible_barker':int(gt_visible), 'gt_offscreen':int(not gt_visible),
            'predicted_visible_barker':int(predicted_visible), 'predicted_offscreen':int(not predicted_visible),
            'mean_spatial_iou':mean_iou, 'max_spatial_iou':float(np.max(ious)) if ious else float('nan'),
            'localization_correct':int(correct), 'matched_gt_track_id':dominant,
            'sampled_frames':len(frames), 'gt_visible_sampled_frames':gt_visible_frames,
        })

    ev = pd.DataFrame(event_rows)
    tp = int(((ev.gt_visible_barker==1)&(ev.predicted_visible_barker==1)).sum())
    fp = int(((ev.gt_visible_barker==0)&(ev.predicted_visible_barker==1)).sum())
    fn = int(((ev.gt_visible_barker==1)&(ev.predicted_visible_barker==0)).sum())
    tn = int(((ev.gt_visible_barker==0)&(ev.predicted_visible_barker==0)).sum())
    precision, recall = div(tp,tp+fp), div(tp,tp+fn)
    f1 = div(2*precision*recall, precision+recall) if np.isfinite(precision) and np.isfinite(recall) else float('nan')
    visible = ev[ev.gt_visible_barker==1]
    vispred = visible[visible.predicted_visible_barker==1]

    video_row = {
        'video_id':video_id, 'events_total':len(ev), 'gt_visible_events':len(visible),
        'gt_offscreen_events':int(ev.gt_offscreen.sum()),
        'correctly_localized_visible_events':int(ev.localization_correct.sum()),
        'localization_accuracy_all_visible':div(ev.localization_correct.sum(), len(visible)),
        'localization_accuracy_given_visible_prediction':div(vispred.localization_correct.sum(), len(vispred)),
        'mean_spatial_iou':float(vispred.mean_spatial_iou.mean()) if len(vispred) else float('nan'),
        'visible_precision':precision, 'visible_recall':recall, 'visible_f1':f1,
        'visible_offscreen_accuracy':div(tp+tn,tp+fp+fn+tn),
        'tp':tp,'fp':fp,'fn':fn,'tn':tn,
    }

    temp_tracks = int(tracks.track_id.nunique())
    persistent = profile_count(profiles/'track_to_all_dog_profile.csv')
    merge_row = {
        'video_id':video_id, 'temporary_tracks':temp_tracks,
        'persistent_profiles_produced':persistent,
        'fragments_merged':temp_tracks-persistent if np.isfinite(persistent) else float('nan'),
    }

    fusion_row = {'video_id':video_id}
    fs = fusion/'pipeline_summary_v2.csv'
    if fs.is_file():
        d = pd.read_csv(fs)
        if not d.empty:
            fusion_row.update(d.iloc[0].to_dict())

    return event_rows, video_row, merge_row, fusion_row


def main():
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    event_rows=[]; video_rows=[]; merge_rows=[]; fusion_rows=[]; failures=[]
    dirs = sorted([p for p in a.annotations_root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p:int(p.name))
    for d in dirs:
        vid = d.name
        base = a.pipeline_output_root/f'{vid}_base'
        profiles = a.pipeline_output_root/f'{vid}_profiles'
        fusion = a.pipeline_output_root/f'{vid}_fusion'
        needed = [d/'annotations.xml', base/'predictions.csv', base/'tracks.csv']
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            failures.append({'video_id':vid,'error':'missing files','details':';'.join(missing)})
            print('SKIP', vid)
            continue
        try:
            er, vr, mr, fr = evaluate_video(vid, d/'annotations.xml', base, profiles, fusion, a.spatial_iou_threshold, a.max_track_gap_sec, a.samples_per_event)
            event_rows += er; video_rows.append(vr); merge_rows.append(mr); fusion_rows.append(fr)
            print(f"{vid}: {vr['correctly_localized_visible_events']}/{vr['gt_visible_events']} visible barks correct")
        except Exception as e:
            failures.append({'video_id':vid,'error':type(e).__name__,'details':str(e)})
            print('FAILED', vid, e)

    events=pd.DataFrame(event_rows); videos=pd.DataFrame(video_rows); merges=pd.DataFrame(merge_rows); fusion=pd.DataFrame(fusion_rows)
    events.to_csv(a.output_dir/'per_event_metrics.csv', index=False)
    videos.to_csv(a.output_dir/'per_video_metrics.csv', index=False)
    merges.to_csv(a.output_dir/'track_merge_statistics.csv', index=False)
    fusion.to_csv(a.output_dir/'fusion_statistics.csv', index=False)
    pd.DataFrame(failures).to_csv(a.output_dir/'failures.csv', index=False)
    if events.empty:
        raise SystemExit('No events evaluated')

    tp = int(((events.gt_visible_barker==1)&(events.predicted_visible_barker==1)).sum())
    fp = int(((events.gt_visible_barker==0)&(events.predicted_visible_barker==1)).sum())
    fn = int(((events.gt_visible_barker==1)&(events.predicted_visible_barker==0)).sum())
    tn = int(((events.gt_visible_barker==0)&(events.predicted_visible_barker==0)).sum())
    precision, recall = div(tp,tp+fp), div(tp,tp+fn)
    f1 = div(2*precision*recall,precision+recall) if np.isfinite(precision) and np.isfinite(recall) else float('nan')
    visible=events[events.gt_visible_barker==1]
    vispred=visible[visible.predicted_visible_barker==1]
    summary={
        'videos_discovered':len(dirs),'videos_evaluated':videos.video_id.nunique(),'videos_failed_or_skipped':len(failures),
        'bark_events_total':len(events),'gt_visible_bark_events':len(visible),'gt_offscreen_bark_events':int(events.gt_offscreen.sum()),
        'correctly_localized_visible_events':int(events.localization_correct.sum()),
        'micro_localization_accuracy_all_visible':div(events.localization_correct.sum(),len(visible)),
        'micro_localization_accuracy_given_visible_prediction':div(vispred.localization_correct.sum(),len(vispred)),
        'macro_localization_accuracy_all_visible':float(videos.localization_accuracy_all_visible.mean()),
        'mean_spatial_iou':float(vispred.mean_spatial_iou.mean()) if len(vispred) else float('nan'),
        'visible_detection_precision':precision,'visible_detection_recall':recall,'visible_detection_f1':f1,
        'visible_offscreen_classification_accuracy':div(tp+tn,tp+fp+fn+tn),
        'temporary_tracks_total':int(merges.temporary_tracks.sum()),
        'persistent_profiles_produced_total':float(merges.persistent_profiles_produced.sum()),
        'fragments_merged_total':float(merges.fragments_merged.sum()),
    }
    if not fusion.empty:
        for c in fusion.columns:
            if c!='video_id': fusion[c]=pd.to_numeric(fusion[c],errors='coerce')
        if 'bark_events' in fusion: summary['fusion_bark_events']=float(fusion.bark_events.sum())
        if 'assigned_bark_events' in fusion:
            summary['fusion_assigned_bark_events']=float(fusion.assigned_bark_events.sum())
            summary['fusion_assignment_rate']=div(fusion.assigned_bark_events.sum(),fusion.bark_events.sum()) if 'bark_events' in fusion else float('nan')
        if 'audio_visual_conflicts' in fusion:
            summary['audio_visual_conflicts_total']=float(fusion.audio_visual_conflicts.sum())
            summary['audio_visual_conflict_rate']=div(fusion.audio_visual_conflicts.sum(),fusion.assigned_bark_events.sum()) if 'assigned_bark_events' in fusion else float('nan')
        if 'offscreen_audio_only_profiles' in fusion:
            summary['offscreen_audio_only_profiles_total']=float(fusion.offscreen_audio_only_profiles.sum())

    pd.DataFrame([{'metric':k,'value':v} for k,v in summary.items()]).to_csv(a.output_dir/'overall_summary.csv',index=False)
    (a.output_dir/'overall_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nOVERALL SUMMARY')
    for k,v in summary.items():
        if isinstance(v,float) and any(x in k for x in ['accuracy','precision','recall','f1','rate']):
            print(f'{k}: {v*100:.2f}%')
        else:
            print(f'{k}: {v}')
    print('\nSaved:',a.output_dir)


if __name__ == '__main__':
    main()
