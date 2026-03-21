import multiprocessing
import requests
import time
import os
import sys
from datetime import datetime

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOM_DIM, MIC_ARRAY_CENTER
from web_visualization.server import start_server

class WebLiveDashboard:
    """
    Drop-in replacement for the old LiveDashboard.
    It launches the Flask/Socket.IO backend in a separate process, 
    makes the initialization API call, consumes the viz_queue to send
    periodic ping API calls, AND accumulates a full session log so that
    a detailed analysis report can be written on exit.
    """
    def __init__(self, viz_queue):
        self.q = viz_queue
        self.server_url = "http://127.0.0.1:5000"
        self.server_proc = None

        # Session bookkeeping
        self.session_start = None
        self.frames = []           # one dict per processed chunk

    def _start_server(self):
        import logging
        log = logging.getLogger('werkzeug')
        log.disabled = True
        start_server(host='127.0.0.1', port=5000)

    def start(self):
        print("[WebViz] Launching Web Visualization Server...")
        self.server_proc = multiprocessing.Process(target=self._start_server)
        self.server_proc.start()
        
        time.sleep(3)   # Give server time to spin up
        self.session_start = datetime.now()
        
        # ── 1. Initialization API ──────────────────────────────────────
        payload = {
            "room_w": ROOM_DIM[0],
            "room_l": ROOM_DIM[1],
            "room_h": ROOM_DIM[2],
            "mic_pos": MIC_ARRAY_CENTER.tolist()
        }
        
        initialized = False
        for _ in range(5):
            try:
                requests.post(f"{self.server_url}/api/init", json=payload, timeout=2.0)
                initialized = True
                break
            except Exception:
                time.sleep(1.0)
                
        if initialized:
            print("=" * 60)
            print("[WebViz] Dashboard is running! Open your browser at:")
            print("[WebViz] http://127.0.0.1:5000")
            print("=" * 60)
        else:
            print("[WebViz] Failed to init server after retries.")
        
        # ── 2. Main loop ───────────────────────────────────────────────
        try:
            while True:
                r = self.q.get()

                ts          = r.get('timestamp', 0.0)
                det         = r.get('detected_event')
                paz         = r.get('pred_az')
                pel         = r.get('pred_el')
                gts         = r.get('gt_events', [])
                sed_results = r.get('sed_results', [])
                det_prob    = r.get('detected_prob', 0.0)

                gt_az = gts[0]['az_deg'] if gts else None
                gt_el = gts[0]['el_deg'] if gts else None
                gt_type = gts[0]['event_type'] if gts else None

                # Angular error (shortest-path)
                az_err = el_err = None
                if paz is not None and gt_az is not None:
                    az_err = abs(((paz - gt_az) + 180) % 360 - 180)
                    el_err = abs(pel - gt_el)

                # Accumulate frame record
                self.frames.append({
                    'timestamp':     ts,
                    'gt_event':      gt_type,
                    'gt_az':         gt_az,
                    'gt_el':         gt_el,
                    'detected_event': det,
                    'detected_prob':  det_prob,
                    'pred_az':       float(paz) if paz is not None else None,
                    'pred_el':       float(pel) if pel is not None else None,
                    'az_err':        az_err,
                    'el_err':        el_err,
                    'sed_top3':      sed_results[:3],   # list of (label, prob)
                })

                # Forward to visualization
                event_data = {
                    'event_type':     det if det else 'Silence',
                    'true_azimuth':   gt_az,
                    'true_elevation': gt_el,
                    'pred_azimuth':   float(paz) if paz is not None else None,
                    'pred_elevation': float(pel) if pel is not None else None,
                }
                try:
                    requests.post(f"{self.server_url}/api/ping", json=event_data, timeout=1.0)
                except Exception as req_err:
                    print("[WebViz] Ping error:", req_err)
                        
        except KeyboardInterrupt:
            # Generate report before dying
            if self.frames:
                from web_visualization.report import generate_report
                generate_report(
                    frames=self.frames,
                    session_start=self.session_start,
                    room_dim=ROOM_DIM,
                    mic_pos=MIC_ARRAY_CENTER,
                )

            if self.server_proc:
                self.server_proc.terminate()
                self.server_proc.join()
            print("[WebViz] Stopped.")

