import os
import json
import time
import random
import numpy as np
import librosa
import pyroomacoustics as pra
import multiprocessing

# Global Core Utils
from config import *
from dsp.main import spherical_to_cartesian

class AudioStreamGenerator:
    """
    Simulates a continuous 3D soundscape. 
    Writes 1-second chunks to a shared buffer/queue along with GT metadata.
    """
    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        self.sample_rate = SAMPLE_RATE
        self.sample_files = []
        # Only include sounds from valid trigger classes, skip 'background'
        trigger_classes = set(LOCALIZATION_TRIGGERS.keys())
        
        for cls in trigger_classes:
            cls_dir = os.path.join(samples_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for root, _, files in os.walk(cls_dir):
                for file in files:
                    if file.endswith(('.mp3', '.wav', '.flac')):
                        self.sample_files.append(os.path.join(root, file))
        
        if not self.sample_files:
            print(f"Warning: No audio files found recursively in {samples_dir}")
        
        # Room setup (constant for a stream session) — from config
        self.room_dim   = ROOM_DIM.copy()
        self.t60        = 0.4
        self.mic_center = MIC_ARRAY_CENTER.copy()
        
    def generate_infinite(self, queue, block_duration_options=(10, 15, 20), chunk_duration=5.0):
        """
        Infinite loop generating spatialized audio.
        Pushes chunk_duration chunks to the queue.
        Each chunk is a dict: {'audio': np.ndarray(4, SR*chunk_duration), 'events': list of events}
        """
        print(f"[Stream Generator] Starting continuous generation...", flush=True)
        total_time_generated = 0.0
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        while True:
            # Randomly choose 5s or 10s block
            block_duration = random.choice(block_duration_options)
            
            # Setup Room for this block
            vol = np.prod(self.room_dim)
            surf = 2 * (self.room_dim[0]*self.room_dim[1] + self.room_dim[1]*self.room_dim[2] + self.room_dim[2]*self.room_dim[0])
            avg_absorption = np.clip(1 - np.exp(-0.161 * vol / (surf * self.t60)), 0.05, 0.95)
            
            room = pra.ShoeBox(self.room_dim, fs=self.sample_rate, max_order=12, absorption=avg_absorption)
            mic_pos = MIC_POSITIONS_LOCAL + self.mic_center[:, None]
            room.add_microphone_array(mic_pos)
            
            # Decide Events for this block
            # EVENT_PROBABILITY chance of an event; guaranteed to land in the FIRST chunk
            # so the SED always processes a chunk that actually contains the sound.
            block_events = []
            has_event = random.random() < EVENT_PROBABILITY
            num_events = 1 if has_event else 0

            for _ in range(num_events):
                # Place event squarely within the first 5-second chunk
                offset_sec = random.uniform(0.5, chunk_duration - 1.5)
                
                src_file = random.choice(self.sample_files)
                y, _ = librosa.load(src_file, sr=self.sample_rate, mono=True)
                # Keep up to 1.5s of the sound
                y = y[:int(1.5 * self.sample_rate)]
                
                # Spatial position
                az = np.deg2rad(random.uniform(-180, 180))
                el = np.deg2rad(random.uniform(-30, 30))
                dist = random.uniform(1.2, 4.0)
                pos = self.mic_center + spherical_to_cartesian(az, el) * dist
                pos = np.clip(pos, 0.1, self.room_dim - 0.1)
                
                room.add_source(pos, signal=y, delay=offset_sec)
                
                # Save event metadata
                # The event type is the class name (immediate subdirectory under samples)
                rel_src = os.path.relpath(src_file, self.samples_dir)
                event_type = rel_src.split(os.sep)[0]
                block_events.append({
                    "time_start": offset_sec,
                    "time_end": offset_sec + (len(y) / self.sample_rate),
                    "event_type": event_type,
                    "az_deg": np.rad2deg(az),
                    "el_deg": np.rad2deg(el),
                })
            
            # Process Block
            if num_events > 0:
                room.simulate()
                signals = room.mic_array.signals.astype(np.float32)
            else:
                # Silence block: just flat noise, no sources
                target_len = int(block_duration * self.sample_rate)
                signals = np.zeros((4, target_len), dtype=np.float32)

            # Pad/Trim to exact block length
            target_len = int(block_duration * self.sample_rate)
            if signals.shape[1] < target_len:
                signals = np.pad(signals, ((0,0), (0, target_len - signals.shape[1])))
            else:
                signals = signals[:, :target_len]
                
            # Add background noise (SNR 25-40dB)
            snr_db = random.uniform(25, 40)
            sig_pwr = np.mean(signals**2) + 1e-12
            noise_pwr = sig_pwr / (10**(snr_db/10))
            signals += np.random.normal(0, np.sqrt(noise_pwr), signals.shape).astype(np.float32)
            
            # Slice into chunk_duration chunks and map events
            num_chunks = int(block_duration / chunk_duration)
            for i in range(num_chunks):
                chunk_start_sec = i * chunk_duration
                chunk_end_sec = (i + 1) * chunk_duration
                
                chunk_audio = signals[:, int(chunk_start_sec * self.sample_rate) : int(chunk_end_sec * self.sample_rate)]
                
                # Find events that occur in this 1s chunk
                active_events = []
                for ev in block_events:
                    # Check overlap
                    if max(chunk_start_sec, ev["time_start"]) < min(chunk_end_sec, ev["time_end"]):
                        active_events.append(ev)
                        
                chunk_data = {
                    "timestamp": total_time_generated + chunk_start_sec,
                    "audio": chunk_audio,
                    "events": active_events
                }
                
                queue.put(chunk_data)
                
            total_time_generated += block_duration
            
            # Throttle if queue is getting full to prevent unbounded memory growth
            while queue.qsize() > 20:
                time.sleep(0.5)

def run_generator(queue):
    generator = AudioStreamGenerator(SAMPLES_DIR)
    generator.generate_infinite(queue)

if __name__ == "__main__":
    q = multiprocessing.Queue()
    run_generator(q)
