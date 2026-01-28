import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import os
import zlib
import struct
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# 1. THE NEURAL ARCHITECTURE (Must match Trainer)
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
# 2. VBR PACKER (V2 Format with Header)
# ==============================================================================
class HoloPackerV2:
    @staticmethod
    def pack(dna_tensor, sample_rate, harmonics):
        flat = dna_tensor.detach().cpu().numpy().flatten()
        mn, mx = flat.min(), flat.max()
        scale = 255.0 / (mx - mn) if mx > mn else 1.0
        q = np.round((flat - mn) * scale).astype(np.uint8)
        
        # HEADER V2: [Magic:4] [Ver:1] [Harmonics:4] [SR:4] [Min:4] [Max:4] [Shape0:4] [Shape1:4]
        # Total Header = 29 Bytes
        header = struct.pack('4sBIIffII', b'HOLO', 2, int(harmonics), int(sample_rate), float(mn), float(mx), dna_tensor.shape[0], dna_tensor.shape[1])
        
        return header + zlib.compress(q.tobytes(), level=9)

    @staticmethod
    def unpack(b_data):
        # Check Magic
        if b_data[:4] != b'HOLO':
            raise ValueError("Not a valid HOLO file.")
        
        version = b_data[4]
        if version == 2:
            # Parse V2
            magic, ver, h, sr, mn, mx, s0, s1 = struct.unpack('4sBIIffII', b_data[:29])
            compressed_data = b_data[29:]
        else:
            # Fallback for old files (assuming standard V1 struct)
            raise ValueError("Old format detected. Use legacy player.")

        raw = zlib.decompress(compressed_data)
        q = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        rec = (q * ((mx - mn) / 255.0)) + mn
        return torch.tensor(rec.reshape(s0, s1)), sr, h

# ==============================================================================
# 3. MODEL MANAGER (The Switchboard)
# ==============================================================================
class ModelManager:
    def __init__(self):
        self.models = {} # Cache of loaded models
        self.available_tiers = []
        self.scan_models()
        
    def scan_models(self):
        # Look for holo_model_h*.pth
        files = glob.glob("holo_model_h*.pth")
        self.available_tiers = []
        for f in files:
            try:
                # Parse filename "holo_model_h32.pth" -> 32
                h_str = f.split('_h')[-1].replace('.pth', '')
                h = int(h_str)
                self.available_tiers.append(h)
            except: pass
        self.available_tiers.sort()
        print(f"Found Quality Tiers: {self.available_tiers}")

    def load_tier(self, harmonics):
        if harmonics in self.models:
            return self.models[harmonics]
        
        fname = f"holo_model_h{harmonics}.pth"
        if not os.path.exists(fname):
            return None
        
        print(f"Loading Tier {harmonics} from disk...")
        ckpt = torch.load(fname)
        
        chunk_size = ckpt.get('c', 1024)
        hidden = ckpt.get('hid', 256)
        
        enc = LogHoloEncoder(chunk_size, harmonics)
        voc = HoloVocoder(harmonics*2, chunk_size, hidden)
        
        enc.load_state_dict(ckpt['encoder'])
        voc.load_state_dict(ckpt['vocoder'])
        
        # Store tuple
        bundle = {'enc': enc, 'voc': voc, 'c': chunk_size, 'h': harmonics}
        self.models[harmonics] = bundle
        return bundle

# ==============================================================================
# 4. THE STUDIO GUI
# ==============================================================================
class HoloStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Holo-Studio // Variable Bitrate Workstation")
        self.root.geometry("1100x850")
        self.root.configure(bg="#111")
        
        self.mgr = ModelManager()
        self.raw_audio = None; self.recon_audio = None
        self.current_bundle = None # Currently loaded model bundle
        self.sr = 44100
        
        self.setup_ui()
        
        # Load default tier if exists
        if self.mgr.available_tiers:
            self.tier_var.set(self.mgr.available_tiers[0])
            self.change_tier()
        else:
            self.lbl_status.configure(text="No Models Found! Run Trainer.", foreground="red")

    def setup_ui(self):
        style = ttk.Style(); style.theme_use('clam')
        style.configure("TFrame", background="#111"); style.configure("TLabel", background="#111", foreground="#ddd")
        style.configure("TButton", font=('Segoe UI', 9, 'bold'))
        
        # HEADER
        head = ttk.Frame(self.root, padding=15); head.pack(fill=tk.X)
        ttk.Label(head, text="HOLO STUDIO", font=('Segoe UI', 18, 'bold'), foreground="#00e5ff").pack(side=tk.LEFT)
        self.lbl_status = ttk.Label(head, text="Ready.", foreground="white"); self.lbl_status.pack(side=tk.RIGHT)

        # MAIN CONTROL STRIP
        ctrl = ttk.Frame(self.root, padding=10); ctrl.pack(fill=tk.X)
        
        # File Ops
        ttk.Button(ctrl, text="📂 LOAD WAV", command=self.load_audio).pack(side=tk.LEFT, padx=5)
        
        # Quality Selector
        q_frame = ttk.LabelFrame(ctrl, text="Quality Chooser (Encoder)", padding=5)
        q_frame.pack(side=tk.LEFT, padx=20)
        
        self.tier_var = tk.IntVar()
        self.combo_tier = ttk.Combobox(q_frame, textvariable=self.tier_var, values=self.mgr.available_tiers, state="readonly", width=10)
        self.combo_tier.pack(side=tk.LEFT, padx=5)
        self.combo_tier.bind("<<ComboboxSelected>>", lambda e: self.change_tier())
        ttk.Label(q_frame, text="Harmonics").pack(side=tk.LEFT)

        # Actions
        self.btn_proc = ttk.Button(ctrl, text="⚡ ENCODE & RECONSTRUCT", command=self.run_process)
        self.btn_proc.pack(side=tk.LEFT, padx=15)

        # Save/Load
        ttk.Button(ctrl, text="💾 SAVE .HOLO", command=self.save_holo).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="📂 LOAD .HOLO (Smart)", command=self.load_holo).pack(side=tk.LEFT, padx=5)
        
        # VISUALIZATION
        vis = ttk.Frame(self.root); vis.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.fig = plt.Figure(figsize=(8, 5), dpi=100, facecolor="#111")
        self.ax1 = self.fig.add_subplot(211); self.ax1.set_facecolor("#222"); self.ax1.tick_params(colors='white'); self.ax1.set_title("Original", color='white')
        self.ax2 = self.fig.add_subplot(212); self.ax2.set_facecolor("#222"); self.ax2.tick_params(colors='white'); self.ax2.set_title("Neural Reconstruction", color='white')
        self.canvas = FigureCanvasTkAgg(self.fig, vis); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # PLAYBACK
        play = ttk.Frame(self.root, padding=15); play.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(play, text="▶ ORIGINAL", command=lambda: self.play(self.raw_audio)).pack(side=tk.LEFT, padx=20)
        ttk.Button(play, text="⏹ STOP", command=sd.stop).pack(side=tk.LEFT, padx=20)
        ttk.Button(play, text="▶ NEURAL DECODE", command=lambda: self.play(self.recon_audio)).pack(side=tk.LEFT, padx=20)
        ttk.Button(play, text="📤 EXPORT WAV", command=self.export_wav).pack(side=tk.RIGHT, padx=20)

    # --- LOGIC ---
    def change_tier(self):
        h = self.tier_var.get()
        bundle = self.mgr.load_tier(h)
        if bundle:
            self.current_bundle = bundle
            self.lbl_status.configure(text=f"Active Model: {h} Harmonics (VBR)", foreground="#00ff00")
        else:
            self.lbl_status.configure(text=f"Failed to load Tier {h}", foreground="red")

    def load_audio(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if not p: return
        try:
            d, sr = sf.read(p); self.sr = sr
            if len(d.shape)>1: d = d.mean(axis=1)
            # No limit
            self.raw_audio = d.astype(np.float32)
            self.ax1.clear(); self.ax1.plot(self.raw_audio[::100], 'gray'); self.ax1.set_title("Original", color='white')
            self.canvas.draw()
            self.lbl_status.configure(text=f"Loaded WAV ({len(d)/sr:.1f}s)")
        except Exception as e: messagebox.showerror("Error", str(e))

    def run_process(self):
        if self.raw_audio is None: return
        if self.current_bundle is None: 
            messagebox.showwarning("Error", "No Model Selected/Loaded")
            return
            
        threading.Thread(target=self._process_thread).start()

    def _process_thread(self):
        self.btn_proc.configure(state="disabled")
        self.lbl_status.configure(text="Encoding & Dreaming...", foreground="cyan")
        try:
            b = self.current_bundle
            C = b['c']
            pad = (C - (len(self.raw_audio)%C)) % C
            chunks = torch.tensor(np.pad(self.raw_audio, (0,pad)).reshape(-1, C), dtype=torch.float32)
            
            with torch.no_grad():
                dna = b['enc'](chunks)
                recon = b['voc'](dna)
            
            self.recon_audio = recon.flatten().numpy()[:len(self.raw_audio)]
            self.root.after(0, self._update_ui_post_process)
            
        except Exception as e:
            print(e)
            self.lbl_status.configure(text="Error", foreground="red")
        finally:
            self.btn_proc.configure(state="normal")

    def _update_ui_post_process(self):
        self.ax2.clear(); self.ax2.plot(self.recon_audio[::100], 'cyan'); self.ax2.set_title("Reconstruction", color='white')
        self.canvas.draw()
        self.lbl_status.configure(text="Done.", foreground="#00ff00")

    def save_holo(self):
        if self.raw_audio is None or self.current_bundle is None: return
        p = filedialog.asksaveasfilename(defaultextension=".holo")
        if not p: return
        
        b = self.current_bundle
        C = b['c']
        pad = (C - (len(self.raw_audio)%C)) % C
        chunks = torch.tensor(np.pad(self.raw_audio, (0,pad)).reshape(-1, C), dtype=torch.float32)
        with torch.no_grad(): dna = b['enc'](chunks)
        
        # Save V2 Format
        bin_data = HoloPackerV2.pack(dna, self.sr, b['h'])
        with open(p, 'wb') as f: f.write(bin_data)
        
        self.lbl_status.configure(text=f"Saved {os.path.basename(p)}")

    def load_holo(self):
        p = filedialog.askopenfilename(filetypes=[("Holo", "*.holo")])
        if not p: return
        try:
            with open(p, 'rb') as f: bin_data = f.read()
            dna, sr, h = HoloPackerV2.unpack(bin_data)
            
            # SMART SWITCHING
            print(f"File requires {h} Harmonics...")
            if h not in self.mgr.available_tiers:
                messagebox.showerror("Missing Model", f"This file needs Tier {h}, but you don't have that model trained.")
                return
            
            # Switch Tier
            self.tier_var.set(h)
            self.change_tier()
            
            # Decode
            with torch.no_grad():
                recon = self.current_bundle['voc'](dna)
            
            self.recon_audio = recon.flatten().numpy()
            self.sr = sr
            
            self.ax1.clear(); self.ax1.text(0.5,0.5,"(Holo Loaded)",color='white',ha='center')
            self.ax2.clear(); self.ax2.plot(self.recon_audio[::100], 'cyan')
            self.canvas.draw()
            self.lbl_status.configure(text=f"Loaded VBR Holo ({h}H)")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def play(self, a): 
        if a is not None: sd.stop(); sd.play(a, self.sr)

    def export_wav(self):
        if self.recon_audio is None: return
        p = filedialog.asksaveasfilename(defaultextension=".wav")
        if not p: return
        sf.write(p, self.recon_audio, self.sr)

if __name__ == "__main__":
    root = tk.Tk()
    app = HoloStudio(root)
    root.mainloop()