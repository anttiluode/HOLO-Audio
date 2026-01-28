# HOLO-Audio — Geometric Neural Audio Codec

![GUI](./holo-studio.png)

HOLO-Audio is a PyTorch-based **neural audio codec** that treats audio as a **geometric / harmonic interference pattern** rather than a raw waveform.

Instead of storing PCM samples or using fixed transforms (MP3/DCT), HOLO-Audio trains a **harmonic neural encoder–decoder** that learns a compact *Phase & Amplitude representation* ("Harmonic DNA") from which audio can be reconstructed.

This repository contains both:
- a **model factory** (trainer)
- a **GUI studio** for encoding, decoding, and playback

---

## 🧠 Core Ideas

### Harmonic Neural Representation
- Uses **sine activations** instead of ReLU to force the network to model **periodicity**
- Encoder maps audio chunks → `(phase, amplitude)` harmonic coefficients
- Vocoder reconstructs waveform chunks from those coefficients

### Holographic Compression
- Audio is stored as **neural instructions** (quantized latent vectors)
- `.holo` files contain:
  - Harmonic tier
  - Sample rate
  - Quantized latent DNA
- Decoder model reconstructs waveform on demand

### Variable Harmonic Tiers (VBR)
Lower tiers = extreme compression, higher tiers = fidelity.

| Harmonics | Typical Result |
|----------:|----------------|
| 4–8       | Sine-wave speech / abstract tones |
| 16        | Robotic but rhythmic |
| 32        | Intelligible speech |
| 64        | Clear speech |
| 128       | High fidelity |
| 256       | Near-lossless texture |

---

## 📊 Compression Characteristics

- Compression depends on **harmonic tier**, not bit-rate
- Lower tiers rely on the brain’s ability to infer speech (Remez-style sine speech)
- Higher tiers capture breath, transients, and texture
- `.holo` files are zlib-compressed latent tensors

> Note: This is **not entropy-optimal compression** yet. It is a *neural geometric codec*.

---

## 📂 Repository Structure

```
.
├── holostudio.py          # GUI studio (encode / decode / playback)
├── holomodeltrainer.py    # Model factory (trains all harmonic tiers)
├── holo_model_h*.pth      # Trained decoder/encoder checkpoints
├── training_data/         # WAV/MP3 training audio
└── *.holo                 # Compressed holographic audio files
```

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install torch numpy soundfile sounddevice matplotlib
```

CUDA is **highly recommended** for training.

---

### 2. Train the Models

Put `.wav` or `.mp3` files into:

```
training_data/
```

Then run:

```bash
python holomodeltrainer.py
```

This will generate:

```
holo_model_h4.pth
holo_model_h8.pth
...
holo_model_h256.pth
```

Each file is a **decoder/encoder pair** for a specific harmonic tier.

---

### 3. Run the Studio

```bash
python holostudio.py
```

Features:
- Load WAV / MP3
- Select harmonic tier
- Encode + reconstruct
- Save `.holo`
- Load `.holo` with automatic tier switching
- Playback original vs neural reconstruction
- Export reconstructed WAV

---

## 🧬 File Format: `.holo` (V2)

Each `.holo` file contains:

- Magic header (`HOLO`)
- Format version
- Harmonic tier
- Sample rate
- Quantization range
- Latent tensor shape
- zlib-compressed DNA payload

The file **does not store audio**, only reconstruction instructions.

---

## 🔬 Theoretical Background

This project draws inspiration from:

- **Sine-Wave Speech** (Remez et al., 1981)
- **Implicit Neural Representations**
- **SIREN networks**
- Harmonic / interference-based views of cognition and perception

The working hypothesis is that **intelligence and perception are fundamentally harmonic systems**, and that constraining neural networks to oscillatory bases forces them to learn *structure* instead of noise.

---

## ⚠️ What This Is / Is Not

**This IS:**
- A neural vocoder + codec experiment
- A variable-rate harmonic representation
- A perceptual / geometric audio model

**This is NOT:**
- A drop-in replacement for MP3/Opus
- Bit-perfect lossless compression
- Entropy-optimized (yet)

---

## 📜 License

MIT License
