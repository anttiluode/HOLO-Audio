"""
HOLO-Audio Metrics Suite
========================
Automated quality assessment for the HOLO neural audio codec.
Generates spectral, perceptual, and compression metrics.

Usage:
    python holo_metrics.py                    # Run on test files in current dir
    python holo_metrics.py --audio test.wav   # Test specific file
    python holo_metrics.py --tiers 32 64 128  # Test specific tiers only
    python holo_metrics.py --report           # Generate markdown report
    
Requirements:
    pip install torch numpy soundfile scipy matplotlib

Author: Antti (HOLO project)
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import os
import glob
import argparse
import json
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq
import struct
import zlib

# ==============================================================================
# NEURAL ARCHITECTURE (Must match trainer/studio)
# ==============================================================================
class SineActivation(nn.Module):
    def forward(self, x): return torch.sin(x)

class HoloVocoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), SineActivation(),
            nn.Linear(hidden_dim, hidden_dim), SineActivation(),
            nn.Linear(hidden_dim, hidden_dim), SineActivation(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.net(x)

class LogHoloEncoder(nn.Module):
    def __init__(self, chunk_size, num_harmonics):
        super().__init__()
        self.encoder = nn.Linear(chunk_size, num_harmonics * 2) 
    def forward(self, x): return self.encoder(x)


# ==============================================================================
# METRIC FUNCTIONS
# ==============================================================================

def signal_to_noise_ratio(original, reconstructed):
    """Calculate SNR in dB"""
    noise = original - reconstructed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def peak_signal_to_noise_ratio(original, reconstructed):
    """Calculate PSNR in dB"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    max_val = np.max(np.abs(original))
    return 20 * np.log10(max_val / np.sqrt(mse))


def root_mean_square_error(original, reconstructed):
    """Calculate RMSE"""
    return np.sqrt(np.mean((original - reconstructed) ** 2))


def spectral_convergence(original, reconstructed, n_fft=2048):
    """
    Spectral convergence - measures how well frequency content is preserved.
    Lower is better. Returns percentage.
    """
    orig_spec = np.abs(np.fft.rfft(original, n=n_fft))
    recon_spec = np.abs(np.fft.rfft(reconstructed, n=n_fft))
    
    # Use 2-norm (Euclidean) for 1D arrays
    numerator = np.linalg.norm(orig_spec - recon_spec)
    denominator = np.linalg.norm(orig_spec)
    
    if denominator < 1e-10:
        return 0.0
    return (numerator / denominator) * 100


def log_spectral_distance(original, reconstructed, n_fft=2048, eps=1e-10):
    """
    Log Spectral Distance - perceptually weighted frequency error.
    Lower is better. Measured in dB.
    """
    orig_spec = np.abs(np.fft.rfft(original, n=n_fft)) + eps
    recon_spec = np.abs(np.fft.rfft(reconstructed, n=n_fft)) + eps
    
    log_diff = 20 * np.log10(orig_spec) - 20 * np.log10(recon_spec)
    return np.sqrt(np.mean(log_diff ** 2))


def envelope_correlation(original, reconstructed, frame_size=1024):
    """
    Correlation between amplitude envelopes.
    Higher is better (1.0 = perfect).
    """
    def get_envelope(sig, frame_size):
        n_frames = len(sig) // frame_size
        envelope = np.zeros(n_frames)
        for i in range(n_frames):
            frame = sig[i*frame_size:(i+1)*frame_size]
            envelope[i] = np.sqrt(np.mean(frame ** 2))
        return envelope
    
    orig_env = get_envelope(original, frame_size)
    recon_env = get_envelope(reconstructed, frame_size)
    
    if len(orig_env) < 2:
        return 1.0
    
    corr = np.corrcoef(orig_env, recon_env)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def transient_preservation_score(original, reconstructed, threshold_percentile=95):
    """
    Measures how well transients (sharp attacks) are preserved.
    Higher is better (0-100).
    """
    def detect_transients(sig):
        diff = np.abs(np.diff(sig))
        threshold = np.percentile(diff, threshold_percentile)
        return diff > threshold
    
    orig_trans = detect_transients(original)
    recon_trans = detect_transients(reconstructed)
    
    # Use shorter array
    min_len = min(len(orig_trans), len(recon_trans))
    orig_trans = orig_trans[:min_len]
    recon_trans = recon_trans[:min_len]
    
    if np.sum(orig_trans) == 0:
        return 100.0
    
    # How many original transients were captured?
    captured = np.sum(orig_trans & recon_trans)
    total = np.sum(orig_trans)
    
    return (captured / total) * 100


def phase_coherence(original, reconstructed, n_fft=2048):
    """
    Phase coherence metric - measures phase alignment.
    Higher is better (0-1).
    """
    orig_spec = np.fft.rfft(original, n=n_fft)
    recon_spec = np.fft.rfft(reconstructed, n=n_fft)
    
    orig_phase = np.angle(orig_spec)
    recon_phase = np.angle(recon_spec)
    
    # Circular correlation for phase
    phase_diff = orig_phase - recon_phase
    coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return coherence


def frequency_band_analysis(original, reconstructed, sr, bands=None):
    """
    Analyze SNR in different frequency bands.
    Returns dict of band: SNR pairs.
    """
    if bands is None:
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, min(20000, sr//2 - 1))
        }
    
    results = {}
    
    for band_name, (low, high) in bands.items():
        if high > sr // 2:
            continue
            
        # Design bandpass filter
        nyq = sr / 2
        low_norm = low / nyq
        high_norm = min(high / nyq, 0.99)
        
        if low_norm >= high_norm:
            continue
            
        try:
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            orig_band = signal.filtfilt(b, a, original)
            recon_band = signal.filtfilt(b, a, reconstructed)
            
            snr = signal_to_noise_ratio(orig_band, recon_band)
            results[band_name] = snr
        except:
            results[band_name] = None
    
    return results


def compression_stats(original_audio, dna_tensor, sr, harmonics):
    """
    Calculate compression statistics.
    """
    # Original size (16-bit PCM)
    original_bytes = len(original_audio) * 2
    original_bitrate = (sr * 16) / 1000  # kbps
    
    # DNA size (quantized + compressed)
    flat = dna_tensor.detach().cpu().numpy().flatten()
    mn, mx = flat.min(), flat.max()
    scale = 255.0 / (mx - mn) if mx > mn else 1.0
    q = np.round((flat - mn) * scale).astype(np.uint8)
    
    # Header size
    header_size = 29  # V2 format
    compressed = zlib.compress(q.tobytes(), level=9)
    holo_bytes = header_size + len(compressed)
    
    # Duration
    duration = len(original_audio) / sr
    holo_bitrate = (holo_bytes * 8) / duration / 1000  # kbps
    
    return {
        'original_bytes': original_bytes,
        'holo_bytes': holo_bytes,
        'compression_ratio': original_bytes / holo_bytes,
        'original_bitrate_kbps': original_bitrate,
        'holo_bitrate_kbps': holo_bitrate,
        'duration_seconds': duration,
        'harmonics': harmonics
    }


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(harmonics, chunk_size=1024, hidden=256):
    """Load a trained HOLO model tier."""
    fname = f"holo_model_h{harmonics}.pth"
    if not os.path.exists(fname):
        return None
    
    ckpt = torch.load(fname, map_location='cpu')
    
    chunk_size = ckpt.get('c', 1024)
    hidden = ckpt.get('hid', 256)
    
    enc = LogHoloEncoder(chunk_size, harmonics)
    voc = HoloVocoder(harmonics * 2, chunk_size, hidden)
    
    enc.load_state_dict(ckpt['encoder'])
    voc.load_state_dict(ckpt['vocoder'])
    
    enc.eval()
    voc.eval()
    
    return {'enc': enc, 'voc': voc, 'c': chunk_size, 'h': harmonics}


def encode_decode(audio, bundle):
    """Run audio through encoder-decoder pipeline."""
    C = bundle['c']
    pad = (C - (len(audio) % C)) % C
    chunks = torch.tensor(np.pad(audio, (0, pad)).reshape(-1, C), dtype=torch.float32)
    
    with torch.no_grad():
        dna = bundle['enc'](chunks)
        recon = bundle['voc'](dna)
    
    return recon.flatten().numpy()[:len(audio)], dna


# ==============================================================================
# MAIN METRICS RUNNER
# ==============================================================================

def run_metrics(audio_path, tiers=None, verbose=True):
    """
    Run full metrics suite on an audio file across all available tiers.
    """
    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HOLO METRICS: {os.path.basename(audio_path)}")
        print(f"Duration: {len(audio)/sr:.2f}s | Sample Rate: {sr}Hz")
        print(f"{'='*60}")
    
    # Find available tiers
    if tiers is None:
        files = glob.glob("holo_model_h*.pth")
        tiers = []
        for f in files:
            try:
                h = int(f.split('_h')[-1].replace('.pth', ''))
                tiers.append(h)
            except:
                pass
        tiers.sort()
    
    if not tiers:
        print("ERROR: No trained models found!")
        return None
    
    results = {
        'file': os.path.basename(audio_path),
        'duration': len(audio) / sr,
        'sample_rate': sr,
        'timestamp': datetime.now().isoformat(),
        'tiers': {}
    }
    
    for h in tiers:
        bundle = load_model(h)
        if bundle is None:
            if verbose:
                print(f"\n[{h}H] Model not found, skipping...")
            continue
        
        if verbose:
            print(f"\n--- Tier {h} Harmonics ---")
        
        # Encode/decode
        recon, dna = encode_decode(audio, bundle)
        
        # Ensure same length
        min_len = min(len(audio), len(recon))
        orig = audio[:min_len]
        rec = recon[:min_len]
        
        # Calculate all metrics
        tier_results = {
            # Core metrics
            'snr_db': signal_to_noise_ratio(orig, rec),
            'psnr_db': peak_signal_to_noise_ratio(orig, rec),
            'rmse': root_mean_square_error(orig, rec),
            
            # Spectral metrics
            'spectral_convergence_pct': spectral_convergence(orig, rec),
            'log_spectral_distance_db': log_spectral_distance(orig, rec),
            
            # Perceptual metrics
            'envelope_correlation': envelope_correlation(orig, rec),
            'transient_preservation_pct': transient_preservation_score(orig, rec),
            'phase_coherence': phase_coherence(orig, rec),
            
            # Frequency band analysis
            'band_snr': frequency_band_analysis(orig, rec, sr),
            
            # Compression stats
            'compression': compression_stats(orig, dna, sr, h)
        }
        
        results['tiers'][h] = tier_results
        
        if verbose:
            print(f"  SNR:           {tier_results['snr_db']:.2f} dB")
            print(f"  PSNR:          {tier_results['psnr_db']:.2f} dB")
            print(f"  Spectral Conv: {tier_results['spectral_convergence_pct']:.2f}%")
            print(f"  Log Spec Dist: {tier_results['log_spectral_distance_db']:.2f} dB")
            print(f"  Envelope Corr: {tier_results['envelope_correlation']:.4f}")
            print(f"  Transients:    {tier_results['transient_preservation_pct']:.1f}%")
            print(f"  Phase Coher:   {tier_results['phase_coherence']:.4f}")
            print(f"  Compression:   {tier_results['compression']['compression_ratio']:.1f}x "
                  f"({tier_results['compression']['holo_bitrate_kbps']:.1f} kbps)")
    
    return results


def generate_comparison_table(results):
    """Generate a markdown comparison table."""
    if not results or 'tiers' not in results:
        return "No results available."
    
    tiers = sorted(results['tiers'].keys())
    
    lines = []
    lines.append(f"# HOLO Metrics Report: {results['file']}")
    lines.append(f"")
    lines.append(f"**Duration:** {results['duration']:.2f}s | **Sample Rate:** {results['sample_rate']}Hz")
    lines.append(f"**Generated:** {results['timestamp']}")
    lines.append(f"")
    
    # Main comparison table
    lines.append("## Quality vs Compression")
    lines.append("")
    lines.append("| Harmonics | SNR (dB) | PSNR (dB) | Spec Conv (%) | Envelope Corr | Bitrate (kbps) | Ratio |")
    lines.append("|-----------|----------|-----------|---------------|---------------|----------------|-------|")
    
    for h in tiers:
        t = results['tiers'][h]
        snr = t['snr_db']
        psnr = t['psnr_db']
        sc = t['spectral_convergence_pct']
        ec = t['envelope_correlation']
        br = t['compression']['holo_bitrate_kbps']
        ratio = t['compression']['compression_ratio']
        
        snr_str = f"{snr:.1f}" if snr != float('inf') else "∞"
        psnr_str = f"{psnr:.1f}" if psnr != float('inf') else "∞"
        
        lines.append(f"| {h:>9} | {snr_str:>8} | {psnr_str:>9} | {sc:>13.1f} | {ec:>13.4f} | {br:>14.1f} | {ratio:>5.1f}x |")
    
    lines.append("")
    
    # Perceptual metrics table
    lines.append("## Perceptual Metrics")
    lines.append("")
    lines.append("| Harmonics | Log Spec Dist (dB) | Transients (%) | Phase Coherence |")
    lines.append("|-----------|-------------------|----------------|-----------------|")
    
    for h in tiers:
        t = results['tiers'][h]
        lines.append(f"| {h:>9} | {t['log_spectral_distance_db']:>17.2f} | {t['transient_preservation_pct']:>14.1f} | {t['phase_coherence']:>15.4f} |")
    
    lines.append("")
    
    # Frequency band analysis for highest tier
    if tiers:
        best_tier = max(tiers)
        bands = results['tiers'][best_tier].get('band_snr', {})
        if bands:
            lines.append(f"## Frequency Band SNR (Tier {best_tier})")
            lines.append("")
            lines.append("| Band | Frequency Range | SNR (dB) |")
            lines.append("|------|-----------------|----------|")
            
            band_ranges = {
                'sub_bass': '20-60 Hz',
                'bass': '60-250 Hz',
                'low_mid': '250-500 Hz',
                'mid': '500-2000 Hz',
                'high_mid': '2000-4000 Hz',
                'presence': '4000-6000 Hz',
                'brilliance': '6000+ Hz'
            }
            
            for band, snr in bands.items():
                if snr is not None:
                    snr_str = f"{snr:.1f}" if snr != float('inf') else "∞"
                    lines.append(f"| {band.replace('_', ' ').title()} | {band_ranges.get(band, 'N/A')} | {snr_str} |")
    
    lines.append("")
    
    # Interpretation guide
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- **SNR > 20 dB**: Generally good quality")
    lines.append("- **SNR > 30 dB**: High quality, hard to distinguish from original")
    lines.append("- **Spectral Convergence < 20%**: Good frequency preservation")
    lines.append("- **Envelope Correlation > 0.95**: Excellent dynamic preservation")
    lines.append("- **Transient Preservation > 80%**: Good attack/impact retention")
    lines.append("")
    lines.append("### Comparison Reference")
    lines.append("- MP3 128 kbps: ~15-20 dB SNR typical")
    lines.append("- MP3 320 kbps: ~25-30 dB SNR typical")
    lines.append("- Opus 64 kbps: ~20-25 dB SNR typical")
    
    return "\n".join(lines)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="HOLO Audio Metrics Suite")
    parser.add_argument('--audio', type=str, help='Path to audio file to test')
    parser.add_argument('--tiers', type=int, nargs='+', help='Specific tiers to test')
    parser.add_argument('--report', action='store_true', help='Generate markdown report')
    parser.add_argument('--json', action='store_true', help='Output JSON results')
    parser.add_argument('--output', type=str, default='holo_metrics_report.md', help='Output file path')
    
    args = parser.parse_args()
    
    # Find test audio
    if args.audio:
        audio_files = [args.audio]
    else:
        # Look for test files
        audio_files = glob.glob("*.wav") + glob.glob("test_audio/*.wav")
        audio_files = [f for f in audio_files if os.path.exists(f)]
        
        if not audio_files:
            print("No audio files found. Specify with --audio or place .wav files in current directory.")
            print("\nUsage: python holo_metrics.py --audio your_file.wav")
            return
    
    all_results = []
    
    for audio_path in audio_files:
        results = run_metrics(audio_path, tiers=args.tiers, verbose=True)
        if results:
            all_results.append(results)
    
    if all_results and args.report:
        # Generate markdown report
        report = generate_comparison_table(all_results[0])
        
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    
    if all_results and args.json:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        json_out = args.output.replace('.md', '.json')
        with open(json_out, 'w') as f:
            json.dump(convert_types(all_results), f, indent=2)
        print(f"JSON saved to: {json_out}")
    
    print("\n" + "="*60)
    print("METRICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()