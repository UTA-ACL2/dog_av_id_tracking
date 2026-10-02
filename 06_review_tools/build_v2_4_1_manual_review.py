#!/usr/bin/env python3
"""Build a focused manual-review package for high-risk v2.4.1 videos.

Default videos: 42956, 51669, 3921, 79755, 80461.
Uses existing outputs only; no model inference is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

ROOT = Path('/path/to/outputs')
VIDEO_ROOT = Path('/path/to/videos')
DEFAULT_IDS = ['42956', '51669', '3921', '79755', '80461']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--ids', nargs='*', default=DEFAULT_IDS)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'v2_4_1_manual_review')
    p.add_argument('--skip-videos', action='store_true')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_video(video_id: str) -> Path:
    matches = sorted(VIDEO_ROOT.glob(f'{video_id}_*.mp4'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one video for {video_id}; found {matches}')
    return matches[0]


def color_for_profile(profile_id: str) -> tuple[int, int, int]:
    palette = [
        (0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,0,255),
        (255,255,0),(128,255,0),(255,128,0),(0,128,255),(180,180,255),
    ]
    n = 0
    for ch in str(profile_id):
        if ch.isdigit():
            n = n * 10 + int(ch)
    return palette[n % len(palette)]


def draw_box(frame, row: pd.Series, profile_id: str, thickness: int = 3) -> None:
    tid = int(row['track_id'])
    x1, y1, x2, y2 = [int(round(float(row[c]))) for c in ['x1','y1','x2','y2']]
    color = color_for_profile(profile_id)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    cv2.putText(frame, f'{profile_id} | track {tid}', (x1, max(24, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def build_annotated_video(video_path: Path, tracks: pd.DataFrame,
                          mapping: pd.DataFrame, output_path: Path) -> None:
    track_to_profile = dict(zip(mapping['track_id'].astype(int),
                                mapping['persistent_profile_id'].astype(str)))
    tracks = tracks.copy()
    tracks['profile_id'] = tracks['track_id'].map(track_to_profile)
    tracks = tracks.dropna(subset=['profile_id'])
    by_frame = {int(f): g for f, g in tracks.groupby('frame')}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not create {output_path}')

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows = by_frame.get(frame_index)
            active = []
            if rows is not None:
                for _, row in rows.iterrows():
                    draw_box(frame, row, str(row['profile_id']))
                active = sorted(set(rows['profile_id'].astype(str)))
            cv2.putText(frame, f'frame={frame_index} time={frame_index/fps:.2f}s',
                        (15,28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
            cv2.putText(frame, 'active profiles: ' + ', '.join(active),
                        (15,54), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255,255,255), 2)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def representative_frames(tracks: pd.DataFrame, left: int, right: int) -> list[int]:
    lf = sorted(set(tracks.loc[tracks['track_id']==left, 'frame'].astype(int)))
    rf = sorted(set(tracks.loc[tracks['track_id']==right, 'frame'].astype(int)))
    common = sorted(set(lf) & set(rf))
    if common:
        return [common[len(common)//2]]
    if not lf or not rf:
        return []
    best = min((abs(a-b), a, b) for a in lf for b in rf)
    return [best[1]] if best[1] == best[2] else [best[1], best[2]]


def write_merge_screenshots(video_path: Path, tracks: pd.DataFrame,
                            mapping: pd.DataFrame, merges: pd.DataFrame,
                            output_dir: Path, force: bool) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    track_to_profile = dict(zip(mapping['track_id'].astype(int),
                                mapping['persistent_profile_id'].astype(str)))
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    records = []
    try:
        for idx, merge in merges.iterrows():
            left = int(merge['left_track_id'])
            right = int(merge['right_track_id'])
            frames = representative_frames(tracks, left, right)
            paths = []
            for frame_index in frames:
                dest = output_dir / f'merge_{idx:03d}_tracks_{left}_{right}_frame_{frame_index}.jpg'
                if not dest.exists() or force:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    rows = tracks[(tracks['frame']==frame_index) & tracks['track_id'].isin([left,right])]
                    for _, row in rows.iterrows():
                        tid = int(row['track_id'])
                        draw_box(frame, row, track_to_profile.get(tid, 'unmapped'), 4)
                    cv2.putText(frame,
                                f"{merge.get('merge_reason','')} | tracks {left},{right} | frame {frame_index} | {frame_index/fps:.2f}s",
                                (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                (255,255,255), 2)
                    cv2.imwrite(str(dest), frame)
                if dest.exists():
                    paths.append(str(dest))
            rec = merge.to_dict()
            rec.update({
                'representative_frames': ','.join(map(str, frames)),
                'screenshot_paths': ' | '.join(paths),
                'manual_same_physical_dog': '',
                'manual_merge_correct': '',
                'manual_failure_type': '',
                'manual_notes': '',
            })
            records.append(rec)
    finally:
        cap.release()
    return records


def main() -> None:
    args = parse_args()
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    video_rows = []
    merge_rows = []

    for video_id in sorted(set(args.ids), key=int):
        print('\n' + '='*72)
        print('REVIEW PACKAGE', video_id)
        print('='*72)

        video = find_video(video_id)
        base = ROOT / f'v2_results_{video_id}'
        profile = ROOT / f'v2_4_1_profiles_{video_id}'
        fused = ROOT / f'v2_4_1_fused_{video_id}'
        review = out_root / video_id
        review.mkdir(parents=True, exist_ok=True)

        tracks = safe_read_csv(base / 'tracks.csv')
        mapping = safe_read_csv(profile / 'track_to_all_dog_profile.csv')
        merges = safe_read_csv(profile / 'visual_track_merges.csv')
        quality = safe_read_csv(profile / 'visual_track_quality.csv')
        bark = safe_read_csv(fused / 'bark_to_profile_v2.csv')
        pipeline = safe_read_csv(fused / 'pipeline_summary_v2.csv')

        if tracks.empty or mapping.empty:
            video_rows.append({'video_id': video_id, 'status': 'missing_required_outputs'})
            print('SKIP: missing tracks or mapping')
            continue

        annotated = review / f'{video_id}_profile_review.mp4'
        if not args.skip_videos:
            if args.force or not annotated.exists():
                print('Creating', annotated)
                build_annotated_video(video, tracks, mapping, annotated)
            else:
                print('Reusing', annotated)

        local_merge_rows = write_merge_screenshots(
            video, tracks, mapping, merges, review / 'merge_screenshots', args.force)
        for r in local_merge_rows:
            r['video_id'] = video_id
        merge_rows.extend(local_merge_rows)

        merge_counts = merges['merge_reason'].value_counts().to_dict() if (not merges.empty and 'merge_reason' in merges) else {}
        temp = int(tracks['track_id'].nunique())
        persistent = int(mapping['persistent_profile_id'].nunique())
        reduction = temp - persistent

        intro = uncertain = 0
        if not quality.empty and 'track_classification' in quality:
            classes = quality['track_classification'].astype(str)
            intro = int((classes == 'likely_intro_outro_image').sum())
            uncertain = int(classes.isin(['static_uncertain','likely_intro_outro_image']).sum())

        summary = pipeline.iloc[0].to_dict() if not pipeline.empty else {}
        conflicts = 0
        if not bark.empty and 'audio_visual_agreement' in bark:
            conflicts = int((bark['audio_visual_agreement'] == False).sum())  # noqa: E712

        row = {
            'video_id': video_id,
            'status': 'complete',
            'temporary_tracks': temp,
            'persistent_visual_profiles': persistent,
            'track_fragments_reduced': reduction,
            'reduction_fraction': reduction/temp if temp else float('nan'),
            'part_based_duplicate_merges': int(merge_counts.get('candidate_part_based_duplicate',0)),
            'fragment_reid_merges': int(merge_counts.get('candidate_fragment_reid',0)),
            'overlap_duplicate_merges': int(merge_counts.get('candidate_duplicate_overlap',0)),
            'likely_intro_outro_tracks': intro,
            'uncertain_tracks': uncertain,
            'total_persistent_dogs': summary.get('total_persistent_dogs'),
            'barking_dogs': summary.get('barking_dogs'),
            'silent_dogs': summary.get('silent_dogs'),
            'bark_events': summary.get('bark_events'),
            'assigned_bark_events': summary.get('assigned_bark_events'),
            'audio_visual_conflicts': summary.get('audio_visual_conflicts', conflicts),
            'annotated_video': str(annotated) if annotated.exists() else '',
            'manual_visible_physical_dogs': '',
            'manual_estimated_barking_dogs': '',
            'manual_all_merges_correct': '',
            'manual_any_overmerge': '',
            'manual_any_undermerge': '',
            'manual_identity_switch_present': '',
            'manual_intro_outro_false_profile': '',
            'manual_silent_profiles_are_real': '',
            'manual_overall_result': '',
            'manual_notes': '',
        }
        video_rows.append(row)
        pd.DataFrame([row]).to_csv(review / 'video_review_summary.csv', index=False)
        pd.DataFrame(local_merge_rows).to_csv(review / 'merge_review_sheet.csv', index=False)

    video_df = pd.DataFrame(video_rows)
    merge_df = pd.DataFrame(merge_rows)
    video_path = out_root / 'manual_video_review_sheet.csv'
    merge_path = out_root / 'manual_merge_review_sheet.csv'
    video_df.to_csv(video_path, index=False)
    merge_df.to_csv(merge_path, index=False)

    readme = f'''# v2.4.1 Manual Review Package

Main video review sheet:
{video_path}

Main merge review sheet:
{merge_path}

For each video, inspect the annotated MP4 and merge screenshots, then fill in columns beginning with `manual_`.

Questions:
1. Do all merged tracks belong to the same physical dog?
2. Were any clearly different dogs incorrectly merged?
3. Are any remaining profiles duplicates of one physical dog?
4. Does any temporary track switch physical dogs?
5. Are likely intro/outro tracks actually static graphics?
6. Do silent profiles correspond to real visible dogs?
7. Is the final profile count plausible?
'''
    (out_root / 'README_REVIEW.md').write_text(readme)

    print('\nSaved:', video_path)
    print('Saved:', merge_path)
    print('Saved:', out_root / 'README_REVIEW.md')
    cols = [c for c in ['video_id','status','temporary_tracks','persistent_visual_profiles',
                        'track_fragments_reduced','part_based_duplicate_merges',
                        'fragment_reid_merges','audio_visual_conflicts'] if c in video_df.columns]
    print('\nSummary:')
    print(video_df[cols].to_string(index=False))


if __name__ == '__main__':
    main()
