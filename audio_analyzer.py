import csv
import json

import librosa
from pydub import AudioSegment
import numpy as np
import os
import matplotlib.pyplot as plt
from pydub.silence import detect_silence
from scipy.signal import get_window, welch


class AudioProcessor:
    def __init__(self, folder_path, segment_length=200, overlap=100, silence_threshold=-40):
        self.folder_path = folder_path
        self.segment_length = segment_length
        self.overlap = overlap
        self.silence_threshold = silence_threshold  # Đơn vị dBFS

    def load_audio_files(self):
        """Load audio files from the specified folder or file and return them as a list of dictionaries."""
        audio_data = []
        if os.path.isdir(self.folder_path):  # Nếu folder_path là một thư mục
            for filename in os.listdir(self.folder_path):
                if filename.endswith('.wav'):
                    path = os.path.join(self.folder_path, filename)
                    audio = AudioSegment.from_file(path)
                    # Append each file as a dictionary containing the filename and the AudioSegment object
                    audio_data.append({"filename": filename, "audio": audio})
        elif os.path.isfile(self.folder_path) and self.folder_path.endswith(
                '.wav'):  # Nếu folder_path là một tệp âm thanh
            filename = os.path.basename(self.folder_path)
            audio = AudioSegment.from_file(self.folder_path)
            audio_data.append({"filename": filename, "audio": audio})
        else:
            raise ValueError("folder_path không phải là một thư mục hoặc tệp âm thanh hợp lệ.")
        return audio_data

    def split_audio(self, audio):
        """Split a single audio file into overlapping segments."""
        segments = []
        start_time = 0
        while start_time + self.segment_length <= len(audio):
            end_time = start_time + self.segment_length
            segment = audio[start_time:end_time]
            segments.append(segment)
            start_time += self.overlap
        return segments

    def calculate_silence_percentage(self, segment):
        """Calculate the percentage of silence in the segment based on the silence threshold."""
        silence = detect_silence(segment, min_silence_len=1, silence_thresh=self.silence_threshold)
        total_silence = sum([end - start for start, end in silence])  # Tính tổng số milliseconds của các khoảng lặng
        silence_percentage = (total_silence / len(segment)) * 100
        return silence_percentage

    def calculate_audio_features(self, segment):
        """Calculate various audio features from a single audio segment."""
        samples = np.array(segment.get_array_of_samples())
        samples_normalized = samples / np.max(np.abs(samples))
        samples_float = librosa.util.buf_to_float(samples_normalized, n_bytes=2, dtype=np.float32)
        mfccs = librosa.feature.mfcc(y=samples_float, sr=segment.frame_rate, n_mfcc=13)
        # Tính toán độ dài tín hiệu sau khi đệm số 0
        N = len(samples)
        next_power_of_2 = 2 ** (N.bit_length() + 1)

        # Áp dụng Zero-padding
        padded_samples = np.pad(samples, (0, next_power_of_2 - N), mode='constant')

        # Tính toán phổ FFT với Zero-padding
        windowed_samples = padded_samples * get_window('hann', len(padded_samples))
        fft_spectrum, frequencies = welch(windowed_samples, segment.frame_rate, nperseg=2048)
        samples_normalized = samples / np.max(np.abs(samples))  # Normalize samples
        half_spectrum = fft_spectrum
        # Calculate Bandwidth
        peak_freq = frequencies[np.argmax(half_spectrum)]
        bandwidth = np.sqrt(np.sum((frequencies - peak_freq) ** 2 * half_spectrum) / np.sum(half_spectrum))

        # Energy Distribution
        energy_distribution = half_spectrum / np.sum(half_spectrum)

        # Harmonicity
        harmonicity = np.sum(half_spectrum ** 2) / np.max(half_spectrum)

        # Pitch
        pitch = peak_freq

        # Average Energy
        average_energy = np.mean(samples_normalized ** 2)

        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples_normalized)))) / 2
        zcr = zero_crossings / len(samples_normalized)

        # Silence Percentage
        silence_percentage = self.calculate_silence_percentage(segment)

        return {
            "bandwidth": bandwidth,
            "harmonicity": harmonicity,
            "pitch": pitch,
            "mfccs":mfccs.tolist(),
            "average_energy": average_energy,
            "zero_crossing_rate": zcr,
            "silence_percentage": silence_percentage,
        }

    def process_audios(self):
        """Process all audio files to extract various properties from segments."""
        audio_files = self.load_audio_files()
        data_audio = []
        for data in audio_files:
            file_name = data["filename"]
            audio = data["audio"]
            segments = self.split_audio(audio)
            results=[]
            for segment in segments:
                features = self.calculate_audio_features(segment)
                results.append(features)
            data_audio.append({"file_name": file_name, "file_results": results})
        return data_audio

    def save_to_json(self, data_audio, output="audio_features.json"):
        """Save the audio features of each segment of each file to a JSON file."""
        with open(output, mode='w', encoding='utf-8') as file:
            json.dump(data_audio, file, indent=4)


# Usage example:
audio_folder = './files'
processor = AudioProcessor(audio_folder)
results = processor.process_audios()
processor.save_to_json(results)
