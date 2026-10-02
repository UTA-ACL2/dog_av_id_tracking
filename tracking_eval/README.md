# Dog Tracking Evaluation Pipeline

## Setup

```bash
python3 -m venv venv && source venv/bin/activate

# GPU (recommended — the pipeline is designed to run on GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# CPU-only (works, but slow — mainly useful for testing)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you skip the `--index-url` step, `pip install -r requirements.txt` still works,
but on Linux the default `torch` wheel pulls in the full CUDA runtime (~2GB) even
without a GPU — use the `cpu` index above to avoid that.

**First-run downloads:** OSNet ReID weights (via `torchreid`) and `rtdetr-l.pt` (via
Ultralytics, only if it's missing) both auto-download on first use — needs internet
once.

**⚠️ Check your `rtdetr-l.pt`:** it should be a complete Ultralytics RT-DETR
checkpoint (~63MB). If yours is much smaller or fails to load, it's likely a partial
download — delete it and either let it auto-download (see above) or re-fetch it from
Ultralytics. Quick check:
```bash
python3 -c "import zipfile; zipfile.ZipFile('rtdetr-l.pt').testzip(); print('OK')"
```
(`BadZipFile` = corrupted/truncated.)

## Folder layout

```
run_full_pipeline.py    # the only script at the top level - the entry point
requirements.txt
README.md
cvat_to_trackeval.py
run_all_tracking.py
predictions_to_mot.py
run_all_profile_merging.py
merge_profiles_into_tracks.py
tracks_to_trackeval.py
run_alt_trackers.py
run_mot_eval.py
track_dogs.py
all_dog_profile_manager.py
analyze_gt.py
smoke_test.py
dataset/annotations/*.xml       # CVAT exports, one per video, id matching dataset/<id>.mp4
dataset/vids/*.mp4           # source videos, named <id>.mp4
rtdetr-l.pt
```

## Running the pipeline

Run everything with one command:

```bash
python run_full_pipeline.py                # GPU (default)
python run_full_pipeline.py --device cpu    # CPU
```

Or run each stage yourself (from the project root):

```bash
# 1. CVAT XML annotations -> MOT ground truth
python scripts/cvat_to_trackeval.py
#    annotations/*.xml -> MOT_eval/gt/<id>.txt

# 2. Detect + track every video
python scripts/run_all_tracking.py [--device cpu]
#    dataset/*.mp4 -> raw-outputs/<id>/tracks.csv

# 3. Build persistent dog profiles from those tracks
python scripts/predictions_to_mot.py [--device cpu]
#    raw-outputs/<id>/tracks.csv -> merged-outputs/<id>/track_to_all_dog_profile.csv
#    (calls all_dog_profile_manager.py per video)

# 4. Merge the profile mapping back onto the raw tracks
python scripts/merge_profiles_into_tracks.py
#    -> merged-outputs/<id>/tracks_with_profiles.csv

# 5. Convert both raw and profile-merged tracks into MOT format
python scripts/tracks_to_trackeval.py
#    -> MOT_eval/trackers/raw_tracker/data/<id>.txt
#    -> MOT_eval/trackers/profile_tracker/data/<id>.txt

# 6. Generate SORT/ByteTrack/BoT-SORT baselines for comparison
python scripts/run_alt_trackers.py --videos-dir dataset --root MOT_eval [--device cpu]
#    -> MOT_eval/trackers/{sort_tracker,bytetrack_tracker,botsort_tracker}

# 7. Final table
python scripts/run_mot_eval.py --root MOT_eval --only-sampled-frames --out MOT_eval/eval_results.csv
```

`--device` (default `cuda:0`) is accepted by `run_all_tracking.py`,
`predictions_to_mot.py`/`run_all_profile_merging.py`, and `run_alt_trackers.py`, and
by `run_full_pipeline.py`, which forwards it to all three. Each of these **raises an
error rather than silently falling back to CPU** if you ask for `cuda` and it isn't
available — pass `--device cpu` explicitly on a CPU-only machine.

Stages 2 and 3 support resume: they skip a video if its output file
(`raw-outputs/<id>/tracks.csv` or `merged-outputs/<id>/track_to_all_dog_profile.csv`)
already exists. Delete the file (or pass `--force`) to redo it.

## Testing

```bash
python scripts/smoke_test.py            # fast — no GPU/torch needed
python scripts/smoke_test.py --full     # also exercises the real model code, on CPU
```

The default run builds small synthetic fixtures in an isolated temp directory and
really executes the non-GPU stages — `cvat_to_trackeval.py`,
`merge_profiles_into_tracks.py`, `tracks_to_trackeval.py`, and `run_mot_eval.py`
(a real TrackEval HOTA/CLEAR/Identity computation) — to confirm the parsing/
merging/scoring logic is all wired together correctly, without needing a GPU or any
of the heavy ML dependencies installed.

`--full` additionally generates a short synthetic video and, if
torch/ultralytics/torchreid are installed, runs `run_all_tracking.py`,
`predictions_to_mot.py`, and `run_alt_trackers.py` against it with `--device cpu`.
This exercises the real RT-DETR/OSNet code paths on CPU. Expect zero dog detections
(it's a synthetic video, not a real dog) — the point is confirming nothing crashes
and files land where later stages expect them.

To check real detection accuracy, run the pipeline on one real short video with GPU
(or `--device cpu`, slow) and inspect `raw-outputs/<id>/tracks.csv`.

## Output layout after a full run

```
MOT_eval/
  gt/<id>.txt
  trackers/
    raw_tracker/data/<id>.txt
    profile_tracker/data/<id>.txt
    sort_tracker/<id>.txt
    bytetrack_tracker/<id>.txt
    botsort_tracker/<id>.txt
  eval_results.csv          # final HOTA / MOTA / IDF1 scoreboard
raw-outputs/<id>/tracks.csv
merged-outputs/<id>/track_to_all_dog_profile.csv
merged-outputs/<id>/tracks_with_profiles.csv
```

## Acknowledgments

Metric computation (stage 7) uses [TrackEval](https://github.com/JonathonLuiten/TrackEval)
by Jonathon Luiten and Arne Hoffhues (MIT License), installed as a normal dependency
via `requirements.txt` — no need to vendor the repo itself.

```bibtex
@misc{luiten2020trackeval,
  author = {Jonathon Luiten, Arne Hoffhues},
  title = {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year = {2020}
}
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31}, year={2020}, publisher={Springer}
}
```

Detection uses [Ultralytics RT-DETR](https://github.com/ultralytics/ultralytics)
(AGPL-3.0), and appearance/ReID embeddings use
[torchreid](https://github.com/KaiyangZhou/deep-person-reid) (OSNet), both also
installed via `requirements.txt` rather than vendored.

## Other notes

- `run_alt_trackers.py` re-runs detection from scratch for every baseline method
  using the same detector/ReID weights as `track_dogs.py`; slow on CPU. Use
  `--tracking-fps` to subsample and `--methods sort` (etc.) to run one baseline at a
  time while testing.
- CVAT track labels are matched case-insensitively against `"dog"` in
  `cvat_to_trackeval.py`.