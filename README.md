# Acoustic-RAG: Zero-Cost Discrete Harmonic Codebook Synthesis and Parametric Memory Grafting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://zenodo.org)
[![Verification CI](https://github.com/RMAAPK/MusicAI/actions/workflows/ci.yml/badge.svg)](https://github.com/RMAAPK/MusicAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Codebook](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Codebook%20Repository-blue)](https://huggingface.co/datasets/rmaapk/music)

A mathematical architecture and cloud execution paradigm for **Retrieval-Augmented Generation (RAG)** on high-frequency continuous acoustic waveforms. Acoustic-RAG bridges the gap between discrete semantic tokens and phase-settled high-frequency spectral synthesis ($12\text{ kHz} - 20\text{ kHz}$) without multi-million parameter diffusion post-processors.

---

## 1. Core Architectural Pillars

### 1.1 40 kHz Linear Nyquist Grid (2,000 Bands)
Standard Mel filterbanks compress high frequencies logarithmically, destroying the harmonic detail required for clean vocal belting and crisp transients. Acoustic-RAG utilizes a uniform linear STFT space:
$$\text{Sample Rate } F_s = 40,000 \text{ Hz} \implies f_{\text{Nyquist}} = 20,000 \text{ Hz}$$
$$N_{\text{FFT}} = 4,000, \quad H = 800 \implies \Delta f = \frac{40000}{4000} = 10.0 \text{ Hz / bin}$$
The positive frequency spectrum maps exactly to $k \in [0, 1999]$ covering $0 \text{ Hz}$ to $19,990 \text{ Hz}$ linearly.

### 1.2 Deterministic INT16 Decibel Clamping
To convert continuous spectrogram magnitudes into fast, indexable discrete tokens:
$$X_{\text{dB}}(k, t) = 20 \log_{10} \max(|X(k, t)|, 10^{-6}) + 120.0 \text{ dB}$$
$$T(k, t) = \begin{cases} 0, & \text{if } X_{\text{dB}}(k, t) < 10.0 \\ \lfloor X_{\text{dB}}(k, t) \rceil, & \text{otherwise} \end{cases} \quad \in \mathbb{Z}^{2000}$$
Each time frame occupies exactly **4 KB** ($2,000 \times 2$ bytes).

### 1.3 1,500-State Modulo Cyclic Harmonic Memory ($\mathcal{S}_{1500}$)
Musical rhythm, formants, and transients follow harmonic orbit cycles. Frames are mapped via modulo arithmetic:
$$s(t) = t \pmod{1500}$$
$$\mathbf{\Phi}_{c, s} = \frac{1}{|\mathcal{T}_s|} \sum_{t \in \mathcal{T}_s} \mathbf{T}(:, t)$$
Settling thousands of frames into an invariant 1,500-state codebook filters out transient anomalies while retaining true sub-genre timbral identities.

### 1.4 Parametric Memory Grafting (Post-Processing)
Eliminates robotic high-frequency "metallic husking" in voice models (e.g. RVC) via linear harmonic grafting:
$$Y_{\text{refined}}(k, t) = (1 - \alpha) \cdot Y_{\text{synth}}(k, t) + \alpha \cdot \mathbf{P}_{c, s(t)}(k), \quad \alpha \in [0.02, 0.08]$$

---

## 2. Zero-Cost Cloud Harvester (Kaggle T4 / P100)

Training and populating these codebooks runs autonomously inside **Kaggle's free GPU tier** without hitting ephemeral disk or session limits:
* Audio is streamed through proxy into RAM.
* Demucs v4 performs stem isolation in GPU VRAM (`cuda:0`).
* Discrete STFT frames are calculated and committed directly into an embedded SQLite database (`PRAGMA synchronous = OFF; journal_mode = WAL;`).
* Raw audio WAVs are deleted immediately after tokenization, keeping local disk footprint $< 100 \text{ MB}$.
* Artifacts and token binaries are committed directly to Hugging Face Git-Xet (`rmaapk/music`).

---

## 3. Repository Structure

```
├── .github/workflows/
│   └── ci.yml               # Automated verification and CI tests
├── paper/
│   └── paper.md             # Complete academic preprint for Zenodo / arXiv
├── src/
│   ├── acoustic_rag.py      # Core mathematical engine & SQLite codebook store
│   └── cloud_harvester.py   # Autonomous Kaggle/Cloud GPU crawler
├── tests/
│   └── test_acoustic_rag.py # Mathematical invariant verification tests
├── LICENSE                  # MIT License
└── README.md
```

---

## 4. Quickstart

### Installation
```bash
git clone https://github.com/RMAAPK/MusicAI.git
cd MusicAI
pip install numpy
```

### Verification
```bash
python3 -m unittest discover tests -v
```

### Basic Usage
```python
import numpy as np
from src.acoustic_rag import AcousticRAGEngine, CodebookStore, AcousticRAGGrafter

# 1. Initialize engine
engine = AcousticRAGEngine(sample_rate=40000, num_bins=2000)

# 2. Extract discrete tokens (40 kHz, 2000 linear bins)
audio = np.random.randn(40000).astype(np.float32)
tokens = engine.extract_discrete_tokens(audio)
print("Token shape:", tokens.shape) # [frames, 2000]

# 3. Settle into 1,500 cyclic states
settled_matrix = engine.aggregate_cyclic_states(tokens)
print("Settled matrix:", settled_matrix.shape) # [1500, 2000]

# 4. Save to embedded SQLite codebook
store = CodebookStore("codebook.db")
store.insert_category_matrix(category_id=412, matrix_1500=settled_matrix)
```

---

## 5. Citation & Academic Reference

```bibtex
@article{asghar2026acousticrag,
  title={Acoustic-RAG: Zero-Cost Discrete Harmonic Codebook Synthesis and Parametric Memory Grafting on Constrained Cloud Accelerators},
  author={Asghar, Raja Muhammad Ali},
  journal={arXiv preprint},
  year={2026},
  publisher={Zenodo},
  url={https://github.com/RMAAPK/MusicAI}
}
```

---

## 6. License
MIT License. Open-source research by **RMAA PK / Ali CNC** (Raja Muhammad Ali Asghar).
