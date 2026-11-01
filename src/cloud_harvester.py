"""
Autonomous Kaggle Free Cloud Runner for Acoustic-RAG.

Runs inside Kaggle Free Dual T4 GPUs / P100 environment:
1. Multi-genre target streaming via proxy.
2. Demucs v4 GPU stem separation.
3. Linear 2,000-bin STFT & INT16 tokenization.
4. On-the-fly Invariant Master SQLite Codebook synthesis.
5. Immediate sync to Hugging Face Git-Xet storage (zero local disk bloat).
"""

import os
import sys
import shutil
import tempfile
import sqlite3
import numpy as np
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from huggingface_hub import HfApi

# Cloud & Storage Configurations
HF_TOKEN = os.getenv("HF_TOKEN", "")
REPO_ID = os.getenv("HF_DATASET_REPO", "rmaapk/music")
PROXY = os.getenv("PROXY_URL", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical sub-genre acoustic category definitions
HARVEST_TARGETS = [
    (412, "ytsearch1:UK drill vocal acapella dry studio"),
    (413, "ytsearch1:808 bass slide heavy sub 140bpm"),
    (414, "ytsearch1:Phonk drift bass heavy distortion cowbell"),
    (415, "ytsearch1:Detroit techno 909 kick acoustic transient"),
    (416, "ytsearch1:Korean ballad high dynamic vocal belt"),
    (417, "ytsearch1:Trap hi-hat rolls clean transient 32nd"),
    (418, "ytsearch1:Heavy metal distorted guitar chug drop D"),
    (419, "ytsearch1:RnB soul vocal harmonies dry room"),
]


def harvest_category(category_id: int, query: str, demucs_model, hf_api: HfApi):
    """
    Acquires, splits, and tokenizes audio for a single category with zero disk residue.
    """
    scratch_dir = tempfile.mkdtemp(prefix=f"harvest_{category_id}_")
    raw_wav = os.path.join(scratch_dir, "input.wav")
    bin_path = os.path.join(scratch_dir, f"cat_{category_id}_tokens.bin")

    try:
        print(f"\n=======================================================")
        print(f"[HARVESTING] Category {category_id} | Target: {query}")
        print(f"=======================================================")

        # 1. Download via proxy
        proxy_arg = f'--proxy "{PROXY}"' if PROXY else ""
        cmd = f'yt-dlp {proxy_arg} -x --audio-format wav "{query}" -o "{raw_wav}" --quiet'
        ret = os.system(cmd)
        if ret != 0 or not os.path.exists(raw_wav):
            print(f"[Skip] Target {query} could not be acquired.")
            return None

        # 2. Resample to 40 kHz
        wav, sr = torchaudio.load(raw_wav)
        if sr != 40000:
            wav = torchaudio.transforms.Resample(sr, 40000)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        wav_gpu = wav.to(DEVICE).unsqueeze(0)

        # 3. Demucs separation in VRAM
        with torch.no_grad():
            sources = apply_model(demucs_model, wav_gpu)[0]

        # Stem selection (3: vocal, 1: bass)
        stem = sources[3].mean(dim=0)

        # 4. 2,000 Linear Bins STFT (10 Hz/bin up to 20,000 Hz)
        n_fft = 4000
        hop_length = 800
        window = torch.hann_window(n_fft).to(DEVICE)

        stft = torch.stft(stem, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        stft_2000 = stft[0:2000, :]

        mag = torch.abs(stft_2000)
        mag_db = 20.0 * torch.log10(torch.clamp(mag, min=1e-6))
        db_energy = mag_db + 120.0

        single_ints = torch.where(
            db_energy < 10.0,
            torch.zeros_like(db_energy, dtype=torch.int16),
            torch.round(db_energy).to(torch.int16)
        )
        frames_array = single_ints.t().cpu().numpy()

        with open(bin_path, "wb") as f:
            f.write(frames_array.tobytes())

        # 5. Push individual token binary to Hugging Face
        if HF_TOKEN:
            hf_api.upload_file(
                path_or_fileobj=bin_path,
                path_in_repo=f"tokens/cat_{category_id}.bin",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Acoustic RAG Ingest Cat {category_id} ({len(frames_array)} frames)"
            )
            print(f"[HF Sync] Cat {category_id} pushed to dataset.")

        return frames_array

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_pipeline():
    print(f"[Acoustic RAG Cloud Pipeline] Live on: {DEVICE}")
    hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None
    demucs_model = get_model('htdemucs').to(DEVICE).eval()
    print("[Pipeline] Demucs v4 loaded in GPU memory.")

    output_db = "/tmp/acoustic_rag_master.db"
    conn = sqlite3.connect(output_db)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settled_codebook (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL,
            state_idx INTEGER NOT NULL,
            band_payload BLOB NOT NULL,
            total_samples INTEGER DEFAULT 1,
            UNIQUE(category_id, state_idx)
        );
    """)
    conn.commit()

    for cat_id, target in HARVEST_TARGETS:
        frames = harvest_category(cat_id, target, demucs_model, hf_api)
        if frames is None:
            continue

        print(f"[Master Settler] Settling Category {cat_id} across 1,500 states...")
        cur.execute("BEGIN TRANSACTION;")
        for s in range(1500):
            step_indices = [i for i in range(len(frames)) if i % 1500 == s]
            if step_indices:
                avg_bands = np.mean(frames[step_indices], axis=0)
                settled_int16 = np.where(avg_bands < 10.0, 0, np.round(avg_bands)).astype(np.int16)
            else:
                settled_int16 = np.zeros(2000, dtype=np.int16)

            cur.execute("""
                INSERT OR REPLACE INTO settled_codebook (category_id, state_idx, band_payload)
                VALUES (?, ?, ?)
            """, (cat_id, s, settled_int16.tobytes()))
        conn.commit()

    conn.close()

    if HF_TOKEN and os.path.exists(output_db):
        print(f"[Pipeline] Uploading consolidated master database to Hugging Face...")
        hf_api.upload_file(
            path_or_fileobj=output_db,
            path_in_repo="databases/acoustic_rag_master.db",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Consolidated Acoustic RAG Master SQLite Codebook"
        )
        print(">>> [SUCCESS] Full Master Database committed to Hugging Face!")


if __name__ == "__main__":
    run_pipeline()

