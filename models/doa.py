import numpy as np
from scipy.signal import butter, filtfilt

SPEED_OF_SOUND = 343.0


# --------------------------------------------------
# Bandpass filter
# --------------------------------------------------
def bandpass(x, fs, lo=500, hi=6000):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


# --------------------------------------------------
# GCC-PHAT
# --------------------------------------------------
def gcc_phat(sig, refsig, fs, max_tau, interp=1):
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-15

    cc = np.fft.irfft(R, n=interp * n)

    max_shift = int(interp * fs * max_tau)
    max_shift = min(max_shift, cc.shape[0] // 2)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    return tau


# --------------------------------------------------
# TDOA matrix estimation
# --------------------------------------------------
def estimate_tdoa(mic_signals, fs, mic_positions, interp=1):
    """
    mic_signals: (num_mics, samples)
    """
    num_mics = mic_signals.shape[0]
    tdoa_matrix = np.zeros((num_mics, num_mics))

    max_mic_dist = np.max(
        np.linalg.norm(mic_positions - mic_positions[0], axis=1)
    )
    max_tau = max_mic_dist / SPEED_OF_SOUND

    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            sig = bandpass(mic_signals[i], fs)
            ref = bandpass(mic_signals[j], fs)

            tau = gcc_phat(
                sig,
                ref,
                fs=fs,
                max_tau=max_tau,
                interp=interp,
            )

            tdoa_matrix[i, j] = tau
            tdoa_matrix[j, i] = -tau

    return tdoa_matrix


# --------------------------------------------------
# Plane-wave DOA estimation (far-field)
# --------------------------------------------------
def estimate_direction_vector(mic_positions, tdoa_matrix):
    """
    Solves (p_i - p_0) · u = c * Δt_i0
    """
    ref = 0
    A = []
    b = []

    for i in range(1, mic_positions.shape[0]):
        A.append(mic_positions[i] - mic_positions[ref])
        b.append(SPEED_OF_SOUND * tdoa_matrix[i, ref])

    A = np.asarray(A)
    b = np.asarray(b)

    u, *_ = np.linalg.lstsq(A, b, rcond=None)

    norm = np.linalg.norm(u)
    if norm < 1e-6:
        return None

    return u / norm



# --------------------------------------------------
# Vector → angles
# --------------------------------------------------
def unit_vector_to_angles(u):
    azimuth = np.degrees(np.arctan2(u[1], u[0])) % 360
    elevation = np.degrees(np.arcsin(u[2]))
    return azimuth, elevation


# --------------------------------------------------
# Public API
# --------------------------------------------------
def analyze_doa(
    fs,
    mic_positions,
    signals,
    room_dims,
    interp=1,
):
    """
    signals: (num_mics, samples)
    """
    assert signals.shape[0] == mic_positions.shape[0], \
        "Mic count mismatch"

    tdoa_matrix = estimate_tdoa(
        signals,
        fs=fs,
        mic_positions=mic_positions,
        interp=interp,
    )

    u = estimate_direction_vector(
        mic_positions,
        tdoa_matrix
    )

    if u is None:
        return None, None

    
    u = -u
    return unit_vector_to_angles(u)

