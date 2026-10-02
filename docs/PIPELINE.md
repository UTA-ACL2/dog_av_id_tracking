# Complete Pipeline

Input:
- video.mp4
- timestamps.csv

Stage 1:
predict_multimodal_barker_v2.py

Stage 2:
all_dog_profile_manager_v2_4_1.py

Stage 3:
audio_fingerprint.py
fingerprint_model.py

Stage 4:
fuse_visual_audio_v2.py

Stage 5:
finalize_v2_4_1_evaluation.py

Outputs:
- Visual profiles
- Audio fingerprints
- Bark assignments
- Final multimodal identities
- Evaluation summaries
