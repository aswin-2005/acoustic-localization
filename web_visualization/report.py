"""
web_visualization/report.py
────────────────────────────
Generates a plain-text simulation analysis log written to
  <project-root>/reports/sim_<timestamp>.log
after a session is stopped with Ctrl-C.
"""

import os
import math
import collections
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None

def _std(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))

def _fmt(val, decimals=1):
    if val is None:
        return "    N/A"
    return f"{val:+{6+decimals}.{decimals}f}"

def _fmtf(val, decimals=3):
    """Format a plain fraction/float without sign."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"

def _bar(frac, width=30, fill="#", empty="."):
    if frac is None:
        return "." * width
    n = round(frac * width)
    return fill * n + empty * (width - n)

def _angular3d(az_err, el_err):
    if az_err is None or el_err is None:
        return None
    return math.degrees(
        math.acos(
            max(-1.0, min(1.0,
                math.cos(math.radians(az_err)) * math.cos(math.radians(el_err))
            ))
        )
    )

SEP  = "=" * 72
SEP2 = "-" * 72
SEP3 = "." * 72


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(frames, session_start, room_dim, mic_pos):
    """
    Parameters
    ----------
    frames       : list[dict]  – one per chunk, as built in dashboard.py
    session_start: datetime    – when the session started
    room_dim     : array-like  – [W, L, H] in metres
    mic_pos      : array-like  – [x, y, z] mic centre in metres
    """
    session_end = datetime.now()
    duration_s  = (session_end - session_start).total_seconds()

    total_chunks = len(frames)
    if total_chunks == 0:
        print("[Report] No frames recorded — skipping report.")
        return

    # ── Partition frames ─────────────────────────────────────────────────────
    event_frames   = [f for f in frames if f['gt_event'] is not None]
    silence_frames = [f for f in frames if f['gt_event'] is None]

    tp_frames = [f for f in event_frames   if f['detected_event'] is not None]
    fn_frames = [f for f in event_frames   if f['detected_event'] is None]
    fp_frames = [f for f in silence_frames if f['detected_event'] is not None]
    tn_frames = [f for f in silence_frames if f['detected_event'] is None]

    tp = len(tp_frames);  fn = len(fn_frames)
    fp = len(fp_frames);  tn = len(tn_frames)
    n_events = len(event_frames)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1        = (2 * precision * recall / (precision + recall)
                 if precision and recall else None)
    accuracy  = (tp + tn) / total_chunks

    # ── DOA metrics ──────────────────────────────────────────────────────────
    doa_frames = [f for f in tp_frames if f['pred_az'] is not None]
    az_errs = [f['az_err'] for f in doa_frames]
    el_errs = [f['el_err'] for f in doa_frames]
    ang3d   = [_angular3d(f['az_err'], f['el_err']) for f in doa_frames]

    mae_az = _mean(az_errs);  std_az = _std(az_errs)
    mae_el = _mean(el_errs);  std_el = _std(el_errs)
    mae_3d = _mean(ang3d)

    hit15 = (sum(1 for e in ang3d if e is not None and e <= 15) / len(ang3d)
             if ang3d else None)
    hit30 = (sum(1 for e in ang3d if e is not None and e <= 30) / len(ang3d)
             if ang3d else None)

    gt_counter  = collections.Counter(f['gt_event']        for f in event_frames)
    fp_counter  = collections.Counter(f['detected_event']  for f in fp_frames if f['detected_event'])
    fn_top      = [f['sed_top3'][0][0] for f in fn_frames if f.get('sed_top3')]
    fn_counter  = collections.Counter(fn_top)

    # ── Build text ───────────────────────────────────────────────────────────
    L = []
    A = L.append

    A(SEP)
    A("  ACOUSTIC SURVEILLANCE  —  SESSION ANALYSIS REPORT")
    A(SEP)
    A(f"  Session Start  : {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"  Session End    : {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"  Duration       : {int(duration_s//60)}m {int(duration_s%60)}s")
    A(f"  Room           : {room_dim[0]} x {room_dim[1]} x {room_dim[2]} m")
    A(f"  Mic Centre     : [{mic_pos[0]:.2f}, {mic_pos[1]:.2f}, {mic_pos[2]:.2f}] m")
    A(f"  Chunks Processed: {total_chunks}")
    A(SEP)
    A("")

    # ── 1. Event Timeline ────────────────────────────────────────────────────
    A(SEP2)
    A("  SECTION 1 : EVENT TIMELINE")
    A(SEP2)
    hdr = (f"{'#':>4}  {'Time':>7}  {'GT Event':<15}  {'Detected':<15}  "
           f"{'Conf':>5}  {'PredAz':>7}  {'PredEl':>7}  "
           f"{'AzErr':>7}  {'ElErr':>7}  {'3DErr':>7}")
    A(hdr)
    A(SEP3)
    for i, f in enumerate(frames, 1):
        gt_l  = (f['gt_event']        or "silence")[:15]
        det_l = (f['detected_event']  or "—")[:15]
        ang   = _angular3d(f.get('az_err'), f.get('el_err'))
        conf  = f"{f['detected_prob']:.2f}" if f['detected_prob'] else "  —  "
        A(f"  {i:>3}  {f['timestamp']:>7.1f}"
          f"  {gt_l:<15}  {det_l:<15}"
          f"  {conf:>5}"
          f"  {_fmt(f.get('pred_az'))}"
          f"  {_fmt(f.get('pred_el'))}"
          f"  {_fmt(f.get('az_err'))}"
          f"  {_fmt(f.get('el_err'))}"
          f"  {_fmt(ang)}")
    A("")

    # ── 2. SED Performance ───────────────────────────────────────────────────
    A(SEP2)
    A("  SECTION 2 : SOUND EVENT DETECTION (SED) PERFORMANCE")
    A(SEP2)
    A("")
    A("  Confusion Matrix:")
    A(f"                       PREDICTED")
    A(f"                   Event     Silence")
    A(f"  GT   Event   |  {tp:^7}  |  {fn:^7}  |")
    A(f"       Silence  |  {fp:^7}  |  {tn:^7}  |")
    A("")
    A(f"  True Positives  : {tp}")
    A(f"  False Negatives : {fn}")
    A(f"  False Positives : {fp}")
    A(f"  True Negatives  : {tn}")
    A(SEP3)
    A(f"  Precision       : {_fmtf(precision)}")
    A(f"  Recall          : {_fmtf(recall)}")
    A(f"  F1 Score        : {_fmtf(f1)}")
    A(f"  Accuracy        : {_fmtf(accuracy)}")
    A(f"  Event Chunks    : {n_events} / {total_chunks}")
    A("")
    if f1 is not None:
        if f1 >= 0.8:
            A("  [GOOD]  Excellent F1 — SED model discriminates events well.")
        elif f1 >= 0.5:
            A("  [OK]    Moderate F1 — review thresholds for improvement.")
        else:
            A("  [WARN]  Low F1 — consider tuning LOCALIZATION_TRIGGERS thresholds.")
    A("")

    A("  Ground-Truth Event Distribution:")
    A(f"  {'Event Class':<18}  {'GT Count':>9}  {'Detected':>9}  {'Miss Rate':>10}")
    A("  " + "-" * 54)
    for cls, cnt in sorted(gt_counter.items(), key=lambda x: -x[1]):
        detected = sum(1 for f_ in tp_frames if f_['gt_event'] == cls)
        miss     = 1.0 - (detected / cnt if cnt else 0)
        A(f"  {cls:<18}  {cnt:>9}  {detected:>9}  {miss:>9.0%}")
    A("")

    if fp_counter:
        A("  False Positive Breakdown:")
        A(f"  {'Falsely Detected Class':<25}  {'Count':>6}")
        A("  " + "-" * 33)
        for cls, cnt in sorted(fp_counter.items(), key=lambda x: -x[1]):
            A(f"  {cls:<25}  {cnt:>6}")
        A("")

    if fn_counter:
        A("  Top SED Label During Missed Events:")
        A(f"  {'Label':<25}  {'Count':>6}")
        A("  " + "-" * 33)
        for lbl, cnt in fn_counter.most_common(10):
            A(f"  {lbl:<25}  {cnt:>6}")
        A("")

    # ── 3. DOA Performance ───────────────────────────────────────────────────
    A(SEP2)
    A("  SECTION 3 : DIRECTION OF ARRIVAL (DOA) PERFORMANCE")
    A(SEP2)
    A("")
    if not doa_frames:
        A("  No true-positive events with DOA predictions were recorded.")
        A("")
    else:
        A(f"  Evaluated on {len(doa_frames)} correctly-detected events.")
        A("")
        A(f"  {'Metric':<14}  {'Azimuth':>10}  {'Elevation':>10}  {'3D Combined':>12}")
        A("  " + "-" * 52)
        A(f"  {'MAE':<14}  {_fmt(mae_az).strip():>10}°  {_fmt(mae_el).strip():>10}°  {_fmt(mae_3d).strip():>12}°")
        A(f"  {'Std Dev':<14}  {_fmt(std_az).strip():>10}°  {_fmt(std_el).strip():>10}°  {'—':>12}")
        A(f"  {'Min Error':<14}  {_fmt(min(az_errs)).strip():>10}°  {_fmt(min(el_errs)).strip():>10}°  {_fmt(min(ang3d)).strip():>12}°")
        A(f"  {'Max Error':<14}  {_fmt(max(az_errs)).strip():>10}°  {_fmt(max(el_errs)).strip():>10}°  {_fmt(max(ang3d)).strip():>12}°")
        A("")
        A(f"  Hit Rate  <=15deg (tight) : {_fmtf(hit15, 1) if hit15 is not None else 'N/A':>6}%  [{_bar(hit15 or 0)}]")
        A(f"  Hit Rate  <=30deg (loose) : {_fmtf(hit30, 1) if hit30 is not None else 'N/A':>6}%  [{_bar(hit30 or 0)}]")
        A("")
        if mae_3d is not None:
            if mae_3d <= 15:
                A("  [GOOD]  Excellent DOA accuracy — mean 3D error <= 15deg.")
            elif mae_3d <= 30:
                A("  [OK]    Acceptable DOA accuracy — mean 3D error within 30deg.")
            else:
                A("  [WARN]  High DOA error — mean 3D error > 30deg.")
        A("")
        A("  Per-Event DOA Results:")
        A(f"  {'Time':>7}  {'GT Event':<15}  "
          f"{'GT Az':>7}  {'GT El':>7}  "
          f"{'Pred Az':>8}  {'Pred El':>8}  "
          f"{'Az Err':>7}  {'El Err':>7}  {'3D Err':>7}")
        A("  " + "-" * 82)
        for f in doa_frames:
            ang = _angular3d(f['az_err'], f['el_err'])
            A(f"  {f['timestamp']:>7.1f}"
              f"  {(f['gt_event'] or ''):<15}"
              f"  {_fmt(f['gt_az'])}"
              f"  {_fmt(f['gt_el'])}"
              f"  {_fmt(f['pred_az'])}"
              f"  {_fmt(f['pred_el'])}"
              f"  {_fmt(f['az_err'])}"
              f"  {_fmt(f['el_err'])}"
              f"  {_fmt(ang)}")
        A("")

    # ── 4. Summary ───────────────────────────────────────────────────────────
    A(SEP)
    A("  EXECUTIVE SUMMARY")
    A(SEP)
    A(f"  Duration         : {int(duration_s//60)}m {int(duration_s%60)}s  "
      f"({total_chunks} chunks)")
    A(f"  Event Chunks     : {n_events} / {total_chunks}")
    A(f"  SED Precision    : {_fmtf(precision)}")
    A(f"  SED Recall       : {_fmtf(recall)}")
    A(f"  SED F1 Score     : {_fmtf(f1)}")
    A(f"  DOA MAE (3D)     : {_fmtf(mae_3d) if mae_3d is not None else 'N/A'} deg  "
      f"({len(doa_frames)} localized events)")
    A(f"  DOA Hit <= 15deg : {f'{hit15*100:.1f}%' if hit15 is not None else 'N/A'}")
    A(f"  DOA Hit <= 30deg : {f'{hit30*100:.1f}%' if hit30 is not None else 'N/A'}")
    A(SEP)
    A(f"  Report generated : {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
    A(SEP)

    # ── Write file ───────────────────────────────────────────────────────────
    root_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    fname = os.path.join(
        reports_dir,
        f"sim_{session_start.strftime('%Y%m%d_%H%M%S')}.log"
    )

    with open(fname, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"\n[Report] Simulation analysis saved -> {fname}")
    return fname
