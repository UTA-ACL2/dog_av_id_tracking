#!/usr/bin/env python3
"""Reusable audio fingerprint encoder extracted from the online notebook."""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SmallResNetTSP(nn.Module):
    def __init__(self, emb_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.block1 = BasicBlock(64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block2 = BasicBlock(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block3 = BasicBlock(256)
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = x.mean(dim=3)
        pooled = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return F.normalize(self.proj(pooled), dim=1)


class HybridFusionNet(nn.Module):
    def __init__(self, in_dim: int = 1104, emb_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class AudioFingerprintEncoder:
    def __init__(
        self,
        *,
        dog2vec_model: Path,
        fairseq_path: Path,
        resnet_checkpoint: Path,
        hybrid_checkpoint: Path,
        device: str = "cuda:0",
        sample_rate: int = 16000,
    ) -> None:
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.n_mfcc = 80
        self.n_mels = 80
        self.n_fft = 400
        self.hop_length = 160
        self.logmel_frames = 200

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
        for path in [dog2vec_model, resnet_checkpoint, hybrid_checkpoint]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)

        self.resnet = SmallResNetTSP().to(self.device)
        res_ckpt = torch.load(resnet_checkpoint, map_location=self.device, weights_only=False)
        self.resnet.load_state_dict(res_ckpt["model_state"])
        self.resnet.eval()

        self.hybrid = HybridFusionNet().to(self.device)
        hyb_ckpt = torch.load(hybrid_checkpoint, map_location=self.device, weights_only=False)
        self.hybrid.load_state_dict(hyb_ckpt["model_state"])
        self.hybrid.eval()

        fairseq_text = str(Path(fairseq_path).resolve())
        if fairseq_text not in sys.path:
            sys.path.insert(0, fairseq_text)
        import fairseq  # type: ignore

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(Path(dog2vec_model).resolve())]
        )
        self.dog2vec = models[0].to(self.device)
        self.dog2vec.eval()

    @staticmethod
    def normalize_waveform(wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        return wav / (np.max(np.abs(wav)) + 1e-8)

    def mfcc_feat(self, wav: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=wav, sr=self.sample_rate, n_mfcc=self.n_mfcc
        ).mean(axis=1).astype("float32")

    def logmel_feat(self, wav: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
        )
        x = librosa.power_to_db(mel, ref=np.max).T.astype("float32")
        if len(x) >= self.logmel_frames:
            start = (len(x) - self.logmel_frames) // 2
            x = x[start : start + self.logmel_frames]
        else:
            x = np.concatenate(
                [
                    x,
                    np.zeros(
                        (self.logmel_frames - len(x), self.n_mels), dtype="float32"
                    ),
                ],
                axis=0,
            )
        return x

    def dog2vec_feat(self, wav: np.ndarray) -> np.ndarray:
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feat, _ = self.dog2vec.extract_features(
                source=x, padding_mask=None, mask=False, output_layer=9
            )
        return feat.mean(dim=1).squeeze(0).cpu().numpy().astype("float32")

    def resnet_feat(self, wav: np.ndarray) -> np.ndarray:
        lm = (
            torch.from_numpy(self.logmel_feat(wav))
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        with torch.inference_mode():
            return self.resnet(lm).squeeze(0).cpu().numpy().astype("float32")

    def encode(self, wav: np.ndarray) -> np.ndarray:
        wav = self.normalize_waveform(wav)
        fused = np.concatenate(
            [self.mfcc_feat(wav), self.dog2vec_feat(wav), self.resnet_feat(wav)]
        )[None, :].astype("float32")
        if fused.shape[1] != 1104:
            raise ValueError(f"Expected 1104-D fused feature, got {fused.shape}.")
        with torch.inference_mode():
            embedding = (
                self.hybrid(torch.from_numpy(fused).to(self.device))
                .squeeze(0)
                .cpu()
                .numpy()
                .astype("float32")
            )
        return embedding / (np.linalg.norm(embedding) + 1e-12)
