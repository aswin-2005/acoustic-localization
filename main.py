# main.py
import time
import numpy as np

import config
from stream import load_config, simulate_one_second
from models.doa import analyze_doa
from models.sed import detect_impacts


def ground_truth_doa(source_pos, mic_positions):
    mic_center = mic_positions.mean(axis=0)
    v = source_pos - mic_center
    v /= np.linalg.norm(v)

    az = np.degrees(np.arctan2(v[1], v[0])) % 360
    el = np.degrees(np.arcsin(v[2]))
    return az, el


# --------------------------------------------------
# Init
# --------------------------------------------------
print("Initializing system...")
load_config(config)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    try:
        # -------------------------
        # Simulate one frame
        # -------------------------
        frame, audio_name, src_pos = simulate_one_second()
        az_gt, el_gt = ground_truth_doa(
            src_pos, config.MIC_POSITIONS
        )
        print('-'*60)
        print("\nSimulated Frame : ")
        print("  audio : ", audio_name)
        print(f"  Simulated Azimuth : {az_gt:.2f}")
        print(f"  Simulated Alevation : {el_gt:.2f}")
        

        # -------------------------
        # Run SED
        # -------------------------
        mono = frame.mean(axis=0)
        sed_hits, _ = detect_impacts(mono)

        if not sed_hits:
            time.sleep(1.0)
            continue

        for label, score in sed_hits:
            thresh = config.SED_THRESHOLDS.get(
                label, config.SED_THRESHOLD
            )

            if score < thresh:
                continue

            # -------------------------
            # DOA estimation
            # -------------------------
            az_est, el_est = analyze_doa(
                fs=config.FS,
                mic_positions=config.MIC_POSITIONS,
                signals=frame,
                room_dims=config.ROOM_DIMS,
            )

            # -------------------------
            # Output
            # -------------------------
            print("\nSED Detected : ", label)
            print(f"  Confidence : {score:.2f}")
            print(f"  Predicted Azimuth : {az_est:.2f}")
            print(f"  Predicted Elevation : {el_est:.2f}")
            
            #---------------------------
            # Error Analysis
            #---------------------------
            az_err = ((az_est - az_gt + 180) % 360) - 180
            el_err = el_est - el_gt
            print(f"\nAzimuth Error : {az_err:.2f}")
            print(f"Elevation Error : {el_err:.2f}")

            break

        print()

        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped.")
        break
