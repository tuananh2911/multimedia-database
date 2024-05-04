import json
import csv
import sys
import numpy as np
from pydub import AudioSegment

from audio_analyzer import AudioProcessor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class AudioAnalyzer:
    def __init__(self, json_file):
        self.json_file = json_file
        self.segment_data = self.load_segment_features()

    def load_segment_features(self):
        data = []
        with open(self.json_file, 'r', encoding='utf-8') as file:
            # Đọc file JSON
            data = json.load(file)

        # Optional: Nếu dữ liệu JSON không phải là list và bạn muốn thao tác với nó như một list
        if not isinstance(data, list):
            data = [data]

        return data

    def calculate_distance(self, new_segment, existing_segment):
        # Khoảng cách cho pitch dùng giá trị tuyệt đối của hiệu số
        pitch_distance = abs(new_segment['pitch'] - existing_segment['pitch'])

        # Khoảng cách Euclidean cho các đặc trưng số
        bandwidth_distance = abs(new_segment['bandwidth'] - existing_segment['bandwidth'])
        harmonicity_distance = abs(new_segment['harmonicity'] - existing_segment['harmonicity'])
        average_energy_distance = abs(new_segment['average_energy'] - existing_segment['average_energy'])
        zcr_distance = abs(new_segment['zero_crossing_rate'] - existing_segment['zero_crossing_rate'])
        silence_distance = abs(new_segment['silence_percentage'] - existing_segment['silence_percentage'])

        # Khoảng cách sử dụng Dynamic Time Warping cho MFCCs

        mfcc_distance, path = fastdtw(new_segment['mfccs'], existing_segment['mfccs'], dist=euclidean)
        # Kết hợp các khoảng cách
        # Có thể áp dụng trọng số cho từng khoảng cách tùy thuộc vào mức độ quan trọng của từng đặc trưng
        combined_distance = (pitch_distance + bandwidth_distance + harmonicity_distance +
                             average_energy_distance + zcr_distance + silence_distance + mfcc_distance)

        return combined_distance

    def find_most_similar_segments(self, new_audio_path, top_n=10):
        audio_analyzer = AudioProcessor(new_audio_path)
        data_segment_input = audio_analyzer.process_audios()
        similarities = []
        for file in self.segment_data:
            values = file['file_results']
            list_dis = []
            for value in values:
                sub_distance = 2 ** 31 - 1
                for segment_input in data_segment_input[0]['file_results']:
                    distance = self.calculate_distance(segment_input, value)
                    if sub_distance > distance:
                        sub_distance = distance
                list_dis.append(sub_distance)
            similarities.append({"file_name": file['file_name'], "average_distance": np.average(list_dis)})
        # Sort by distance, smaller is more similar
        # Return the top N similar files
        # Sắp xếp danh sách các dictionary dựa trên 'average_distance'
        similarities_sorted = sorted(similarities, key=lambda x: x['average_distance'])
        return similarities_sorted[:10]


# Usage example
csv_path = './audio_features.json'
analyzer = AudioAnalyzer(csv_path)
new_audio_path = './test'
top_similar_files = analyzer.find_most_similar_segments(new_audio_path)

