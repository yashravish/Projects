import numpy as np
import pyaudio
import scipy.signal
import time
from collections import deque

class SwaraDetector:
    def __init__(self, base_pitch_name: str = None):
        # ========= Audio Settings =========
        self.CHUNK = 2048  # smaller chunk for lower latency
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        
        # ========= Frequency / Thresholds =========
        self.MIN_FREQ = 25  # Lowered to better catch lower octave
        self.MAX_FREQ = 1000
        self.AMPLITUDE_THRESHOLD = 0.002  # RMS below this is "silence"
        
        # ========= Stability Logic =========
        self.swara_history = deque(maxlen=5)  # store last 5 swaras
        self.required_consistent_frames = 3   # require 3 in a row
        self.last_swara = None
        self.last_stable_time = time.time()
        self.stable_duration = 0.1  # must wait 0.1s before reprinting
        
        # ========= Note Frequencies =========
        self.note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # ========= Swara Ratios =========
        # We include lower, middle, and upper swaras
        self.swara_ratios = {
            # Lower octave
            'sa_': 0.5,  # Lower Sa
            'ri_': 256/486,  # Lower ri
            'Ri_': 9/16,  # Lower Ri
            'ga_': 32/54,  # Lower ga
            'Ga_': 81/128,  # Lower Ga
            'ma_': 2/3,  # Lower ma
            'Ma_': 729/1024,  # Lower Ma
            'pa_': 3/4,  # Lower Pa
            'dha_': 128/162,  # Lower dha
            'Dha_': 27/32,  # Lower Dha
            'ni_': 16/18,  # Lower ni
            'Ni_': 243/256,  # Lower Ni
            
            # Middle octave
            'Sa': 1.0,
            'ri': 256/243,
            'Ri': 9/8,
            'ga': 32/27,
            'Ga': 81/64,
            'ma': 4/3,
            'Ma': 729/512,
            'Pa': 3/2,
            'dha': 128/81,
            'Dha': 27/16,
            'ni': 16/9,
            'Ni': 243/128,
            'SA': 2.0  # Upper Sa
        }
        
        if base_pitch_name:
            self.base_freq = self.note_frequencies[base_pitch_name]
            self.calculate_swara_frequencies()

    def calculate_swara_frequencies(self):
        self.swara_ranges = {}
        tolerance_low = 0.98
        tolerance_high = 1.02
        for swara, ratio in self.swara_ratios.items():
            center_freq = self.base_freq * ratio
            low_freq = center_freq * tolerance_low
            high_freq = center_freq * tolerance_high
            self.swara_ranges[swara] = (low_freq, high_freq)

    def apply_bandpass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        nyquist = self.RATE / 2
        low_cut = self.MIN_FREQ / nyquist
        high_cut = self.MAX_FREQ / nyquist
        
        # 4th-order Butterworth bandpass
        b, a = scipy.signal.butter(4, [low_cut, high_cut], btype='band')
        filtered_data = scipy.signal.filtfilt(b, a, audio_data)
        return filtered_data

    def is_voice_present(self, audio_data: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > self.AMPLITUDE_THRESHOLD

    def get_fundamental_frequency(self, audio_data: np.ndarray) -> float:
        if not self.is_voice_present(audio_data):
            return 0.0

        filtered = self.apply_bandpass_filter(audio_data)
        windowed = filtered * scipy.signal.windows.hann(len(filtered))
        
        # Enhanced autocorrelation for better low frequency detection
        corr = np.correlate(windowed, windowed, mode='full')
        corr = corr[len(corr)//2:]
        
        # Adaptive peak threshold based on signal strength
        peak_threshold = 0.15 * np.max(corr)  # Slightly lower threshold for better low freq detection
        
        # Increased minimum peak distance for better low frequency resolution
        min_peak_distance = int(self.RATE / self.MAX_FREQ)
        
        peaks = scipy.signal.find_peaks(
            corr,
            height=peak_threshold,
            distance=min_peak_distance
        )[0]
        
        if len(peaks) > 0:
            peak_values = corr[peaks]
            peak_freqs = self.RATE / peaks
            
            valid_mask = (peak_freqs >= self.MIN_FREQ) & (peak_freqs <= self.MAX_FREQ)
            if np.any(valid_mask):
                # Prioritize stronger peaks in the valid frequency range
                weighted_peaks = peak_values * valid_mask
                best_idx = np.argmax(weighted_peaks)
                return peak_freqs[best_idx]
        return 0.0

    def is_swara_stable(self, current_swara: str) -> bool:
        if current_swara in ["silence", "undefined"]:
            return False
        
        self.swara_history.append(current_swara)
        
        if len(self.swara_history) < self.required_consistent_frames:
            return False
            
        recent = list(self.swara_history)[-self.required_consistent_frames:]
        return all(s == recent[0] for s in recent)

    def map_frequency_to_swara(self, freq: float) -> str:
        if freq <= 0:
            return "silence"
            
        # Direct matching in all octave ranges
        for swara, (low, high) in self.swara_ranges.items():
            if low <= freq <= high:
                return swara
        
        # Check if the frequency might be in a different octave
        freq_octaves = [freq/2, freq, freq*2]  # Check one octave up and down
        
        best_match = "undefined"
        min_diff = float('inf')
        
        # Find the closest matching swara across all octaves
        for test_freq in freq_octaves:
            for swara, (low, high) in self.swara_ranges.items():
                if low <= test_freq <= high:
                    # Calculate how close we are to the center frequency
                    center = (low + high) / 2
                    diff = abs(test_freq - center)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = swara
        
        return best_match

    def process_audio_stream(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        try:
            print("\nCalibrating... please remain silent.")
            time.sleep(1.5)
            print("Ready to detect swaras. Press Ctrl+C to stop.")
            print("Note: '_' suffix indicates lower octave (e.g., 'sa_')\n")
            
            while True:
                data = np.frombuffer(
                    stream.read(self.CHUNK, exception_on_overflow=False),
                    dtype=np.float32
                )
                
                freq = self.get_fundamental_frequency(data)
                current_swara = self.map_frequency_to_swara(freq)
                
                now = time.time()
                
                if (current_swara not in ["silence", "undefined"] and
                    current_swara != self.last_swara and
                    self.is_swara_stable(current_swara) and
                    now - self.last_stable_time > self.stable_duration):
                    
                    print(f"{current_swara} ({freq:.1f} Hz)")
                    self.last_swara = current_swara
                    self.last_stable_time = now
        
        except KeyboardInterrupt:
            print("\nStopping...")
            stream.stop_stream()
            stream.close()
            audio.terminate()

def select_base_pitch() -> str:
    valid_notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    print("\nWelcome to Swara Detector!")
    print("Available pitches:", ", ".join(valid_notes))
    while True:
        choice = input("Enter your pitch (e.g. 'C', 'G#'): ").upper().strip()
        if choice in valid_notes:
            return choice
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    pitch_name = select_base_pitch()
    detector = SwaraDetector(pitch_name)
    detector.process_audio_stream()