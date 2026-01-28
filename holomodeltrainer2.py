import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import glob
import random

# ==============================================================================
# 0. CUDA SETUP (The Speed Boost)
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- RUNNING ON: {DEVICE} ---")
if DEVICE.type == 'cpu':
    print("WARNING: Training on CPU will be slow. Install CUDA if possible.")

# ==============================================================================
# 1. PERCEPTUAL LOSS
# ==============================================================================
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 512, 256], hop_sizes=[120, 60, 30], win_lengths=[600, 300, 150]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft_size, hop_size, win_length):
        x = x.view(-1, x.size(-1))
        # Ensure window is on the same device as input x (GPU)
        window = torch.hann_window(win_length).to(x.device) 
        return torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, 
                          window=window, return_complex=True)

    def forward(self, x_fake, x_real):
        loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            real_stft = self.stft(x_real, fft_size, hop_size, win_length)
            fake_stft = self.stft(x_fake, fft_size, hop_size, win_length)
            
            real_mag = torch.abs(real_stft)
            fake_mag = torch.abs(fake_stft)
            
            loss += F.l1_loss(torch.log(real_mag + 1e-7), torch.log(fake_mag + 1e-7))
            loss += F.l1_loss(real_mag, fake_mag)
            
        return loss

# ==============================================================================
# 2. MODELS
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.uniform_(-np.sqrt(6/m.in_features), np.sqrt(6/m.in_features))

    def forward(self, x):
        return self.net(x)

class LogHoloEncoder(nn.Module):
    def __init__(self, chunk_size, num_harmonics):
        super().__init__()
        self.encoder = nn.Linear(chunk_size, num_harmonics * 2) 
    def forward(self, x): return self.encoder(x)

# ==============================================================================
# 3. DATA LOADER
# ==============================================================================
def load_data(chunk_size=1024, folder="training_data"):
    chunks = []
    if not os.path.exists(folder): os.makedirs(folder)
    
    files = glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.mp3"))
    print(f"Found {len(files)} files in {folder}")
    
    for f in files:
        try:
            d, sr = sf.read(f)
            if len(d.shape)>1: d=d.mean(axis=1)
            d = d / (np.max(np.abs(d)) + 1e-6)
            
            num = len(d)//chunk_size
            for i in range(num):
                c = d[i*chunk_size:(i+1)*chunk_size]
                if np.max(np.abs(c)) > 0.02:
                    chunks.append(c)
        except: pass
    
    if len(chunks) > 0:
        random.shuffle(chunks)
        print(f"Loaded {len(chunks)} chunks.")
        return np.array(chunks)
    return np.array([])

# ==============================================================================
# 4. THE FACTORY LOOP
# ==============================================================================
if __name__ == "__main__":
    print("--- HOLO-VARIABLE MODEL FACTORY (CUDA) ---")
    
    CHUNK = 1024
    # UPDATED: Added Low-Frequency "Sine Speech" Tiers
    TIERS = [4, 8, 16, 32, 64, 128, 256] 
    BATCH = 64 
    EPOCHS = 5
    HIDDEN = 256
    
    print("Loading audio data...")
    data_np = load_data(CHUNK)
    if len(data_np) == 0:
        print("Error: No audio found.")
        exit()
        
    # Keep dataset on CPU initially to save VRAM, move batches to GPU later
    dataset = torch.tensor(data_np, dtype=torch.float32)
    
    # Num_workers=0 is safer on Windows, use >0 on Linux
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, pin_memory=True)
    
    loss_fn = nn.L1Loss()
    
    for harmonics in TIERS:
        print(f"\n>>> TRAINING TIER: {harmonics} HARMONICS <<<")
        
        # Init models and move to GPU
        encoder = LogHoloEncoder(CHUNK, harmonics).to(DEVICE)
        vocoder = HoloVocoder(input_dim=harmonics*2, output_dim=CHUNK, hidden_dim=HIDDEN).to(DEVICE)
        
        opt = torch.optim.Adam(list(encoder.parameters()) + list(vocoder.parameters()), lr=0.0002)
        
        for ep in range(EPOCHS):
            total_loss = 0
            for batch in loader:
                # MOVE BATCH TO GPU
                batch = batch.to(DEVICE)
                
                dna = encoder(batch)
                recon = vocoder(dna)
                loss = loss_fn(recon, batch)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            if ep % 5 == 0 or ep == EPOCHS-1:
                print(f"   Epoch {ep}: Loss {total_loss/len(loader):.4f}")
                
        fname = f"holo_model_h{harmonics}.pth"
        print(f"   Saving {fname}...")
        
        # Save CPU versions of weights so the Radio (which might run on CPU) can load them easily
        torch.save({
            'encoder': encoder.cpu().state_dict(),
            'vocoder': vocoder.cpu().state_dict(),
            'h': harmonics,
            'c': CHUNK,
            'hid': HIDDEN
        }, fname)

    print("\nFACTORY COMPLETE. All models generated.")