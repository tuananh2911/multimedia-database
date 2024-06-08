import concurrent.futures
import json

import numpy as np

from audio_analyzer import AudioProcessor
from classification import AudioClusterer


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

    def compare_mfcc(self, mfcc_1, mfcc_2):
        sum = 0
        for i in range(0, len(mfcc_1)):
            sum += abs(mfcc_1[i] - mfcc_2[i])
        distance = sum / len(mfcc_1)
        return distance

    def calculate_distance(self, new_segment, existing_segment):
        # Khoảng cách cho pitch dùng giá trị tuyệt đối của hiệu số
        pitch_distance = abs(new_segment['pitch'] - existing_segment['pitch']) / existing_segment['pitch']

        # Khoảng cách Euclidean cho các đặc trưng số
        bandwidth_distance = abs(new_segment['bandwidth'] - existing_segment['bandwidth']) / existing_segment[
            'bandwidth']
        harmonicity_distance = abs(new_segment['harmonicity'] - existing_segment['harmonicity']) / existing_segment[
            'harmonicity']
        average_energy_distance = abs(new_segment['average_energy'] - existing_segment['average_energy']) / \
                                  existing_segment['average_energy']
        zcr_distance = abs(new_segment['zero_crossing_rate'] - existing_segment['zero_crossing_rate']) / \
                       existing_segment['zero_crossing_rate']
        silence_distance = abs(new_segment['silence_percentage'] - existing_segment['silence_percentage'])
        if (existing_segment['silence_percentage'] != 0):
            silence_distance /= existing_segment['silence_percentage']

        mfcc_distance = self.compare_mfcc(new_segment['mfccs'], existing_segment['mfccs'])

        combined_distance = (pitch_distance + bandwidth_distance + harmonicity_distance +
                             average_energy_distance + zcr_distance + silence_distance + mfcc_distance) / 7

        return combined_distance

    def find_most_similar_segments(self, new_audio_path, top_n=10):
        audio_analyzer = AudioProcessor(new_audio_path)
        data_segment_input = audio_analyzer.process_audios("abc")
        similarities = []
        kmean_clusters = AudioClusterer(4, 42)
        file_names_similar = kmean_clusters.get_files_in_same_cluster(data_segment_input[0])
        data_segment_processed = []
        for row in self.segment_data:
            if row['file_name'] in file_names_similar:
                data_segment_processed.append(row)
        def process_file(file):
            values = file['file_results']
            list_dis = []
            for value in values:
                sub_distance = min(self.calculate_distance(segment_input, value) for segment_input in
                                   data_segment_input[0]['file_results'])
                list_dis.append(sub_distance)
            return {"file_name": file['file_name'], "average_distance": np.average(list_dis)}

        # Sử dụng ThreadPoolExecutor để song song hóa xử lý
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, file) for file in data_segment_processed]
            for future in concurrent.futures.as_completed(futures):
                similarities.append(future.result())

        # Sắp xếp kết quả và trả về top n
        print(len(similarities))
        similarities_sorted = sorted(similarities, key=lambda x: x['average_distance'])
        return similarities_sorted[:]

# Usage example
# csv_path = './audio_features.json'
# analyzer = AudioAnalyzer(csv_path)
# new_audio_path = './test'
# top_similar_files = analyzer.find_most_similar_segments(new_audio_path)
