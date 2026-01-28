import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

class RollingSpectrogram:
    def __init__(
        self,
        fs,
        n_fft=1024,
        hop_length=256,
        spec_frames=5,
        history_seconds=30,
    ):
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop_length

        # ---- Spectrogram window (short-term) ----
        self.spec_cols = spec_frames * int(fs / hop_length)
        self.spec = None
        self.freqs = None

        # ---- Error history (long-term) ----
        self.max_err_pts = int((history_seconds * fs) / hop_length)
        self.az_err_hist = []
        self.el_err_hist = []
        self.err_display_pts = 10

        # ---- Plot setup ----
        plt.ion()
        self.fig, (self.ax_spec, self.ax_err) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=False,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # Spectrogram
        self.img = None

        # Error plots
        self.az_line, = self.ax_err.plot([], [], "c-", label="Az Error (deg)")
        self.el_line, = self.ax_err.plot([], [], "m--", label="El Error (deg)")
        self.avg_line, = self.ax_err.plot([], [], "y:", linewidth=2,
                                          label="Avg |Az Error|")

        self.ax_err.set_ylabel("Degrees")
        self.ax_err.set_ylim(-30, 30)
        self.ax_err.legend(loc="upper right")

    def update(self, mono_signal, az_err=None, el_err=None):
        # ---- STFT ----
        f, _, Zxx = stft(
            mono_signal,
            fs=self.fs,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop,
            padded=False,
            boundary=None,
        )

        S = 20 * np.log10(np.abs(Zxx) + 1e-8)

        if self.spec is None:
            self.freqs = f
            self.spec = S
        else:
            self.spec = np.concatenate([self.spec, S], axis=1)
            self.spec = self.spec[:, -self.spec_cols:]

        # ---- Error history ----
        if az_err is not None:
            self.az_err_hist.append(az_err)
            self.el_err_hist.append(el_err)

            if len(self.az_err_hist) > self.max_err_pts:
                self.az_err_hist = self.az_err_hist[-self.max_err_pts:]
                self.el_err_hist = self.el_err_hist[-self.max_err_pts:]

        self._draw()

    def _draw(self):
        # ---- Spectrogram plot ----
        if self.img is None:
            self.img = self.ax_spec.imshow(
                self.spec,
                origin="lower",
                aspect="auto",
                cmap="magma",
            )
            self.ax_spec.set_ylabel("Frequency (Hz)")
            self.ax_spec.set_title("Spectrogram (Last 5 Frames)")
            self.fig.colorbar(self.img, ax=self.ax_spec)
        else:
            self.img.set_data(self.spec)

        # ---- Error plot ----
        hist_len = len(self.az_err_hist)
        if hist_len > 0:
            start = max(0, hist_len - self.err_display_pts)
            az = self.az_err_hist[start:hist_len]
            el = self.el_err_hist[start:hist_len]
            x = np.arange(len(az))
            self.az_line.set_data(x, az)
            self.el_line.set_data(x, el)
            avg = np.mean(np.abs(az))
            self.avg_line.set_data(x, [avg] * len(x))
            self.ax_err.set_xlim(0, max(1, len(x) - 1))

        plt.pause(0.001)
