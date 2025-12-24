import numpy as np



def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-15  # PHAT weighting

    cc = np.fft.irfft(R, n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    return tau, cc



def estimate_tdoa(mic_signals, fs=1, max_tau=None, interp=1):
    num_mics = mic_signals.shape[0]
    tdoa_matrix = np.zeros((num_mics, num_mics))

    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            tau, _ = gcc_phat(mic_signals[i], mic_signals[j], fs, max_tau, interp)
            tdoa_matrix[i, j] = tau
            tdoa_matrix[j, i] = -tau

    return tdoa_matrix



def estimate_direction_vector(
    mic_positions,
    tdoa_matrix,
    sound_speed=343.0
):
    ref = 0
    A = []
    b = []

    for i in range(1, mic_positions.shape[0]):
        A.append(mic_positions[i] - mic_positions[ref])
        b.append(sound_speed * tdoa_matrix[i, ref])

    A = np.array(A)
    b = np.array(b)

    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    u = u / np.linalg.norm(u)

    return u