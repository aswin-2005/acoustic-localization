import multiprocessing
import time
import numpy as np

import os
import sys

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from stream.generator import run_generator
from sed.api import SEDPredictor
from doa.api import DOAPredictor
from dsp.main import compute_gcc_phat

def run_pipeline(audio_queue, viz_queue=None):
    """
    Core pipeline consumer: reads audio chunks, runs SED + DOA, 
    and optionally pushes results to viz_queue for the dashboard.
    """
    print("[Pipeline] Initializing pipeline models...")
    sed_model = SEDPredictor(SED_MODEL_WEIGHTS, SED_LABELS_CSV)
    doa_model = DOAPredictor(DOA_MODEL_WEIGHTS)
    print("[Pipeline] Models loaded. Waiting for stream data...")
    
    while True:
        try:
            # Block until data is available
            chunk_data = audio_queue.get(timeout=5.0)
        except multiprocessing.queues.Empty:
            print("[Pipeline] Queue is empty, waiting...")
            continue
            
        timestamp = chunk_data["timestamp"]
        audio_chunk = chunk_data["audio"]
        gt_events = chunk_data["events"]
        
        chunk_duration = audio_chunk.shape[1] / SAMPLE_RATE
        
        # Simulate real-time stream consumption by sleeping
        time.sleep(chunk_duration)
        
        print(f"\n--- [T={timestamp:.1f}s] Processing Chunk ---")
        
        if gt_events:
            print(f"Ground Truth Events in this chunk: {', '.join([e['event_type'] for e in gt_events])}")
        else:
            print("Ground Truth: Silence")
            
        # 1. SED (Use Mic 1)
        sed_results = sed_model.predict_from_buffer(audio_chunk[0], top_k=3)
        
        # Check if any detected class is a target trigger and above its specific threshold
        detected_event = None
        detected_prob = 0.0
        for label, prob in sed_results:
            if label in LOCALIZATION_TRIGGERS and prob > LOCALIZATION_TRIGGERS[label]:
                detected_event = label
                detected_prob = prob
                break
        
        pred_az = None
        pred_el = None
                
        if detected_event:
            print(f"-> SED HIT: '{detected_event}' (Conf: {detected_prob:.2f})")
            
            # 2. DOA
            features = []
            for m in range(1, 4):
                cc = compute_gcc_phat(audio_chunk[m], audio_chunk[0], SAMPLE_RATE, MAX_LAG)
                features.append(cc)
            feature_tensor = np.array(features)
            
            pred_az, pred_el = doa_model.predict(feature_tensor)
            print(f"-> DOA PREDICTION : Az={pred_az:+.1f}°  El={pred_el:+.1f}°")
            
            # Print Ground truth coords and angular error
            if gt_events:
                gt = gt_events[0]
                gt_az = gt['az_deg'];  gt_el = gt['el_deg']
                # Shortest-path azimuth error
                az_err = abs(((pred_az - gt_az) + 180) % 360 - 180)
                el_err = abs(pred_el - gt_el)
                print(f"-> GROUND TRUTH   : Az={gt_az:+.1f}°  El={gt_el:+.1f}°")
                print(f"-> ANGULAR ERROR  : dAz={az_err:.1f}°  dEl={el_err:.1f}°")
            
        else:
            # Print top SED result just for debugging
            top_label, top_prob = sed_results[0]
            print(f"-> No trigger detected. Top sound: '{top_label}' ({top_prob:.2f})")
        
        # Push result to visualization queue if provided
        if viz_queue is not None:
            viz_result = {
                'timestamp': timestamp,
                'audio': audio_chunk[0],  # Send first channel for playback
                'gt_events': gt_events,
                'sed_results': sed_results,
                'detected_event': detected_event,
                'detected_prob': detected_prob,
                'pred_az': pred_az,
                'pred_el': pred_el,
            }
            try:
                viz_queue.put_nowait(viz_result)
            except:
                pass  # If viz queue full, skip (non-blocking)

def main():
    """Run pipeline without visualization (terminal only)."""
    multiprocessing.set_start_method('spawn')
    
    stream_queue = multiprocessing.Queue(maxsize=50)
    
    gen_proc = multiprocessing.Process(target=run_generator, args=(stream_queue,))
    gen_proc.start()
    
    try:
        run_pipeline(stream_queue)
    except KeyboardInterrupt:
        print("\n[Main] Stopping processes...")
        gen_proc.terminate()
        gen_proc.join()
        print("[Main] Exited cleanly.")

def main_with_viz():
    """Run pipeline WITH the live matplotlib dashboard."""
    multiprocessing.set_start_method('spawn')
    
    stream_queue = multiprocessing.Queue(maxsize=50)
    viz_queue = multiprocessing.Queue(maxsize=100)
    
    # Start Generator Process
    gen_proc = multiprocessing.Process(target=run_generator, args=(stream_queue,))
    gen_proc.start()
    
    # Start Pipeline Process (pushes results to viz_queue)
    pipe_proc = multiprocessing.Process(target=run_pipeline, args=(stream_queue, viz_queue))
    pipe_proc.start()
    
    # Run dashboard
    from web_visualization.dashboard import WebLiveDashboard
    
    print("[Main] Launching Web 3D visualization... Check browser.")
    dashboard = WebLiveDashboard(viz_queue)
    
    try:
        dashboard.start()           # blocks until window is closed
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[Main] Stopping all processes...")
        gen_proc.terminate()
        pipe_proc.terminate()
        gen_proc.join(timeout=3)
        pipe_proc.join(timeout=3)
        print("[Main] Exited cleanly.")
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Acoustic Surveillance Pipeline")
    parser.add_argument('--viz', action='store_true', help='Launch with live matplotlib dashboard')
    args = parser.parse_args()
    
    if args.viz:
        main_with_viz()
    else:
        main()

