# Helper functions for the SPE assignment
# The defined functions and Class are written to be used in a jupyter-notebook

# import libraries
import numpy as np

from scipy.signal import argrelextrema, find_peaks, butter, sosfilt, peak_prominences, \
decimate, correlate, correlation_lags

import matplotlib as mpl
import matplotlib.pyplot as plt

# relative path for the data files, currently not implemented

data_path = ''

def load_signals_ppg() -> tuple:
    """Load raw PPG signal data from numpy .ndy files.
    First load then flatten the data. 
    """

    signal_ppg_1 = np.load(data_path + 'first_ppg.npy', mmap_mode=None, allow_pickle=False, \
        fix_imports=True)
    signal_ppg_1 = signal_ppg_1.flatten()

    signal_ppg_2 = np.load(data_path + 'second_ppg.npy', mmap_mode=None, allow_pickle=False, \
        fix_imports=True)
    signal_ppg_2 = signal_ppg_2.flatten()

    signal_ppg_3 = np.load(data_path + 'third_ppg.npy', mmap_mode=None, allow_pickle=False, \
        fix_imports=True)
    signal_ppg_3 = signal_ppg_3.flatten()

    return signal_ppg_1, signal_ppg_2, signal_ppg_3


def load_signals_ecg() -> tuple:
    """Load raw ECG signal data from numpy .ndy files.
    First load then flatten the data. 
    """

    signal_ecg_1 = np.load(data_path + 'first_ecg.npy', mmap_mode=None, allow_pickle=False, \
        fix_imports=True)
    signal_ecg_1 = signal_ecg_1.flatten()

    signal_ecg_2 = np.load(data_path + 'second_ecg.npy', mmap_mode=None, allow_pickle=False, \
        fix_imports=True)
    signal_ecg_2 = signal_ecg_2.flatten()

    signal_ecg_3 = np.load(data_path + 'third_ecg.npy', mmap_mode=None, allow_pickle=False, \
        fix_imports=True)
    signal_ecg_3 = signal_ecg_3.flatten()

    return signal_ecg_1, signal_ecg_2, signal_ecg_3


def plot_all_ppg_signals(signal_ppg_1, signal_ppg_2, signal_ppg_3, interval_begin: int = 0, \
    interval_end: int = 0) -> None:  
    
    """Plot all three PPG signals.
    The interval_begin and interval_end parameters are optional, but can be used to plot a signal segment.
    """

    fig, ppg = plt.subplots(1, 3)
    fig.suptitle('PPG Signals', fontsize=20)

    if not interval_end:
        interval_end = signal_ppg_1.size
    ppg[0].plot(signal_ppg_1[interval_begin:interval_end])
    ppg[0].set_title('Signal PPG 1')
    ppg[0].set_xlabel('sample')
    ppg[0].set_ylabel('magnitutde')

    if not interval_end:
        interval_end = signal_ppg_1.size
    ppg[1].plot(signal_ppg_2[interval_begin:interval_end])
    ppg[1].set_title('Signal PPG 2')
    ppg[1].set_xlabel('sample')
    ppg[1].set_ylabel('magnitutde')

    if not interval_end:
        interval_end = signal_ppg_1.size
    ppg[2].plot(signal_ppg_3[interval_begin:interval_end])
    ppg[2].set_title('Signal PPG 3')
    ppg[2].set_xlabel('sample')
    ppg[2].set_ylabel('magnitutde')

    plt.show()


def plot_all_ecg_signals(signal_ecg_1, signal_ecg_2, signal_ecg_3, interval_begin: int = 0, \
    interval_end: int = 0) -> None:
    
    """Plot all three ECG signals.
    The interval_begin and interval_end parameters are optional, but can be used to plot a signal segment.
    """

    fig, ppg = plt.subplots(1, 3)
    fig.suptitle('ECG Signals', fontsize=20)

    if not interval_end:
        interval_end = signal_ecg_1.size
    ppg[0].plot(signal_ecg_1[interval_begin:interval_end])
    ppg[0].set_title('Signal ECG 1')
    ppg[0].set_xlabel('sample')
    ppg[0].set_ylabel('magnitutde')

    if not interval_end:
        interval_end = signal_ecg_2.size
    ppg[1].plot(signal_ecg_2[interval_begin:interval_end])
    ppg[1].set_title('Signal ECG 2')
    ppg[1].set_xlabel('sample')
    ppg[1].set_ylabel('magnitutde')

    if not interval_end:
        interval_end = signal_ecg_3.size
    ppg[2].plot(signal_ecg_3[interval_begin:interval_end])
    ppg[2].set_title('Signal ECG 3')
    ppg[2].set_xlabel('sample')
    ppg[2].set_ylabel('magnitutde')

    plt.show()


def get_sampling_rate(signal: float, signal_type: str = 'ppg', signal_number: int = 0) -> None:
    """Get sampling rate from the raw signal by simply computing the distance between two extrema
    """

    if signal_type == 'ppg':
        local_extrema = argrelextrema(signal[540:600], np.less)     # Compute extrema within a signal segment
        local_extrema = np.diff(local_extrema)                      # Compute distance between two extrema
        fs = np.max(local_extrema)                                  # Sampling frequency

    if signal_type == 'ecg':       
        signal_cut = signal[20000:22000]                            # Specify signal sement
        peak_1 = np.max(signal_cut[0:1000])                         # Locate first maxima
        peak_2 = np.max(signal_cut[1000::]) + 1000                  # Locate second maxima
        fs = peak_2 - peak_1                                        # Sampling frequency

    ts = np.round(1 / fs, decimals=8)                               # Sampling time

    print('Signal: ' + str.upper(signal_type) + f' #{signal_number}')
    print(f'Sampling frequency: {fs}')
    print(f'Sampling time: {ts} \n')


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> float:
    """Initialize Butterworth bandpass filter.
    """

    nyq = 0.5 * fs                                                         # Nyquist frequency
    low = lowcut / nyq                                                     # Lower cutoff frequency
    high = highcut / nyq                                                   # Higher cutoff frequency
    sos = butter(order, [low, high], analog=False, btype='bandpass', \
        output='sos')  

    return sos


def butter_bandpass_filter(data: float, lowcut: float, highcut: float, fs: float, \
    order: int = 5) -> float:
    
    """Filter signal with a Butterworth bandpass filter """

    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)                       # Filter output

    return y


def downsample_by_40(signal):
    """Downsamaple signal by a factor of 40.
    Due to stability issues, we apply decimation (downsampling) multiple times.
    """

    signal_downsampled = decimate(signal, 5, ftype='iir', zero_phase=True)
    signal_downsampled = decimate(signal_downsampled, 4, ftype='iir', zero_phase=True)
    signal_downsampled = decimate(signal_downsampled, 2, ftype='iir', zero_phase=True)
    return signal_downsampled.flatten()


class Signal:
    """Signal object for the raw PPG and ECG signals
    To initialize an instance, three inputs are required.

    signal:         raw signal numpy array
    signal_type:    'ecg' or 'ppg'
    signal_number:  corresponding signal number

    """
    def __init__(self, signal, signal_type, signal_number):
        self.signal = signal                         # Raw signaly array
        self.N = self.signal.size                    # Signal length
        self.signal_type = signal_type               # Signal type
        self.signal_number = signal_number           # Signal number
        self.fs = None                               # Sampling frequency
        self.ts = None                               # Sampling time
        self.t = None                                # time vector/array
        self.lowcut = None                           # Lower cutoff frequency
        self.highcut = None                          # Higher cutoff frequency
        self.peaks = None                            # Signal peak indices
        self.prominences = None                      # Peak prominences
        self.anomalies = None                        # Anomaly points
        self.anomaly_segments = None                 # Anomaly segments
        self.score = None                            # Signal quality score


    def initialize(self):
        """Initializes signal by setting sampling frequency, sampling time,
        lower and higher cutoff frequencies and the time vector
        """

        self.fs = 25
        self.ts = 0.04
        self.lowcut = 0.5
        self.highcut = 1.5
        if self.signal_type == 'ecg':
            self.fs = 1000
            self.ts = 0.001
            self.lowcut = 0.5
            self.highcut = 50

        self.t = np.linspace(0, (self.N - 1) * self.ts, self.N)


    def plot_signal(self, interval_begin: int = 0, interval_end: int = 0):
        """Plot signal with respect to time and magnitude
        """

        fig, sig = plt.subplots(1, 1)
        fig.suptitle(str.upper(self.signal_type) + ' Signal #' + str(self.signal_number), fontsize=20)

        if not interval_end:
            interval_end = self.signal.size

        sig.plot(self.t[interval_begin:interval_end], self.signal[interval_begin:interval_end])
        # sig.set_title('Signal PPG 1')
        sig.set_xlabel('t in seconds')
        sig.set_ylabel('magnitude')


    def preprocess_signal(self):
        """Preprocess signal with a bandpass filter.
        Removes trend, higher and lower frequences, offset
        """
        self.signal = butter_bandpass_filter(self.signal, self.lowcut, self.highcut, self.fs, order=5) 

        if self.signal_type == 'ppg':
            self.signal = self.signal[300::]   # Remove offset
            self.t = self.t[300::]             # Adjust time vector


    def plot_signal_spectrum(self):
        """Plot signal spectrum in frequency domain.
        """
        N = self.signal.size                           # Sample size
        fstep = self.fs / N                            # Frequency step
        f = np.linspace(0, (N - 1) * fstep, N)         # Frequency vector

        # Detrend signal first using a polynomial
        # without this operation, it is difficult to obtain the signal spectrum

        poly_coeff = np.poly1d(np.polyfit(self.t, self.signal, 8))    # Obtain polynomial coefficients
        detrended_signal = self.signal - poly_coeff(self.t)           # Fit signal and detrend

        X = np.fft.fft(detrended_signal)            # Perform fft
        X_mag = np.abs(X) / N                       # magnitude of X

        f_plot = f[0:int(N / 2 + 1)]                # Frequency vector for plotting

        X_mag_plot = 2 * X_mag[0:int(N/2 + 1)]

        X_mag_plot[0] = X_mag_plot[0]/2             # DC component does not have to be multiplied by 2

        # Plot spectrum
        fig, sig = plt.subplots(1, 1)
        fig.suptitle(str.upper(self.signal_type) + ' Signal #' + str(self.signal_number) + ' Spectrum', fontsize=20)
        sig.plot(f_plot, X_mag_plot)
        sig.set_xlabel('f in Hz')
        sig.set_ylabel('magnitude')


    def get_peak_indices(self):
        """Obtain indices for signal peaks.
        Distances (pulse width) for PPG and ECG signals were
        obtained by inspection
        """      

        distance = 20
        if self.signal_type == 'ecg':
            distance = 750
        self.peaks, _ = find_peaks(self.signal, distance=distance)


    def plot_peaks(self):
        """Plot signal peaks
        """  

        fig, sig = plt.subplots(1, 1)
        fig.suptitle(str.upper(self.signal_type) + ' Signal #' + str(self.signal_number), fontsize=20)
        sig.set_xlabel('time in seconds')
        sig.set_ylabel('magnitude')
        sig.plot(self.t, self.signal)
        sig.plot(self.t[self.peaks], self.signal[self.peaks], 'X')


    def get_peak_prominences(self):
        """Obtain signal peak prominences.
        """  
        self.prominences = peak_prominences(self.signal, self.peaks)[0]


    def plot_peak_prominences(self):
        """Plot signal peak prominences.
        """  

        contour_heights = self.signal[self.peaks] - self.prominences
        fig, sig = plt.subplots(1, 1)
        fig.suptitle(str.upper(self.signal_type) + ' Signal #' + str(self.signal_number) + \
            ' with Prominences', fontsize=20)
        sig.plot(self.t, self.signal)
        sig.plot(self.t[self.peaks], self.signal[self.peaks], "X")
        sig.vlines(x=self.t[self.peaks], ymin=contour_heights, ymax=self.signal[self.peaks])
        sig.set_xlabel('time in seconds')
        sig.set_ylabel('magnitude')


    def detect_anomalies(self, prominence_threshold: float = 0.2):
        """Detect anomalies by processing at signal peaks.
        """  

        anomaly_segments = []
        # Use threshold = 0.2 or user input to find prominces that live outside the region
        prominence_above_threshold = \
            np.where(self.prominences > np.median(self.prominences) * \
                (1 + prominence_threshold))[0]
        prominence_below_threshold = \
            np.where(self.prominences < np.median(self.prominences) * \
             (1 - prominence_threshold))[0]

        anomalies = np.concatenate((prominence_above_threshold, prominence_below_threshold))
        anomalies = np.sort(anomalies)     # Sort indices 

        local_segment = []

        for i in range(0, anomalies.size - 1):
            # check if neighboring indices only differ by 1 and append to local segment
            if anomalies[i + 1] - anomalies[i] == 1:
                local_segment.append(anomalies[i])
            # if next neighbor index is greater 1 and local_segment is not empty append
            # index of segment end point and reset local_segment 
            elif len(local_segment) > 0:
                local_segment.append(anomalies[i])
                # append local segment to anomaly list
                anomaly_segments.append(self.peaks[local_segment])  
                local_segment = []

        anomalies = self.peaks[anomalies]           # Get correct anomaly indices
        self.anomalies = anomalies.flatten()
        self.anomaly_segments = anomaly_segments


    def plot_anomalies(self):
        """Plot signal anomalies
        """  

        fig, sig = plt.subplots(1, 1)
        fig.suptitle(str.upper(self.signal_type) + ' Signal #' + str(self.signal_number) + \
            ' Anomalies / Outliers', fontsize=20)
        sig.plot(self.t, self.signal)
        sig.plot(self.t[self.anomalies], self.signal[self.anomalies], "X")
        sig.set_xlabel('time in seconds')
        sig.set_ylabel('magnitude')


    def plot_anomaly_segments(self):
        """Plot signal anomalies and anomaly segments
        """  

        fig, sig = plt.subplots(1, 1)
        fig.suptitle(
            str.upper(self.signal_type) + ' Signal #' + str(self.signal_number) + 
            ' Anomalies / Anomaly Segments', fontsize=20)
        sig.plot(self.t, self.signal)
        sig.plot(self.t[self.anomalies], self.signal[self.anomalies], "X", color='black')
        sig.set_xlabel('time in seconds')
        sig.set_ylabel('magnitude')
        for segment in self.anomaly_segments:
            sig.axvline(x=self.t[segment[0]], ymin=0.05, ymax=0.95, color='black', ls='--')
            sig.axvline(x=self.t[segment[-1]], ymin=0.05, ymax=0.95, color='black', ls='--')
            sig.fill_between(self.t[[segment[0], segment[-1]]], np.min(self.signal), \
                np.max(self.signal), color='red', alpha=0.3)


    def compute_lag(self, signal_2):
        """Compute lag between two signals by computing cross-correlation
        """  

        sig_1 = self.signal
        sig_2 = signal_2.signal

        # downsample if ECG signal
        if self.signal_type == 'ecg' and signal_2.signal_type == 'ppg':
            sig_1 = downsample_by_40(sig_1)
        if self.signal_type == 'ppg' and signal_2.signal_type == 'ecg':
            sig_2 = downsample_by_40(sig_2)

        # remove mean value
        mean_signal_1 = np.mean(sig_1);
        mean_signal_2 = np.mean(sig_2);

        signal_1_0 = sig_1 - mean_signal_1
        signal_2_0 = sig_2 - mean_signal_2

        # rescale to the same power
        rescale_1 = np.sqrt(np.mean(np.power(np.abs(signal_1_0), 2)))
        rescale_2 = np.sqrt(np.mean(np.power(np.abs(signal_2_0), 2)))
        # rescale_ppg = 1
        # rescale_ecg = 1

        signal_1_0 = signal_1_0/rescale_1;
        signal_2_0 = signal_2_0/rescale_2;

        # compute correlation and lags
        correlation = correlate(signal_1_0, signal_2_0, mode="full")
        lags = correlation_lags(signal_1_0.size, signal_2_0.size, mode="full")
        lag = lags[np.argmax(correlation)]

        print(f'Signal 1 length: {sig_1.size}')
        print(f'Signal 2 length: {sig_2.size}')
        print(f'Lag (in samples): {lag} \n')

        return correlation

    def plot_signal_correlation(self, signal_2):
        """Plot signal correlation.
        """  
        correlation = self.compute_lag(signal_2)
        fig, sig = plt.subplots(1, 1)
        fig.suptitle(str.upper(self.signal_type) + ' Signal #' + str(self.signal_number) + ' and ' +
                     str.upper(signal_2.signal_type) + ' Signal #' + str(signal_2.signal_number) +
                     ' Correlation', fontsize=20)
        sig.plot(correlation)
        sig.set_xlabel('samples')
        sig.set_ylabel('magnitude')

    def get_score(self):
        """Compute signal quality scire
        """  
        
        self.score = np.round(1 - self.anomalies.size / self.peaks.size, decimals=4)
        
