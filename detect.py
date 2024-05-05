import json
import csv
import sys
import concurrent.futures
import cv2
import numpy as np
from pydub import AudioSegment

from audio_analyzer import AudioProcessor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim

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

    def compare_images_ssim(self,image_path_1, image_path_2):
        """
        So sánh hai ảnh dựa trên chỉ số SSIM.

        Args:
        image_path_1 (str): Đường dẫn đến ảnh thứ nhất.
        image_path_2 (str): Đường dẫn đến ảnh thứ hai.

        Returns:
        float: Giá trị SSIM giữa hai ảnh.
        """
        # Đọc ảnh và chuyển đổi sang grayscale
        img1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
        # Đảm bảo hai ảnh có cùng kích thước
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for SSIM comparison.")

        # Tính toán SSIM
        ssim_index, ssim_map = ssim(img1, img2, full=True)

        return ssim_index

    def calculate_distance(self, new_segment, existing_segment):
        # Khoảng cách cho pitch dùng giá trị tuyệt đối của hiệu số
        pitch_distance = abs(new_segment['pitch'] - existing_segment['pitch'])/existing_segment['pitch']

        # Khoảng cách Euclidean cho các đặc trưng số
        bandwidth_distance = abs(new_segment['bandwidth'] - existing_segment['bandwidth'])/ existing_segment['bandwidth']
        harmonicity_distance = abs(new_segment['harmonicity'] - existing_segment['harmonicity'])/existing_segment['harmonicity']
        average_energy_distance = abs(new_segment['average_energy'] - existing_segment['average_energy'])/existing_segment['average_energy']
        zcr_distance = abs(new_segment['zero_crossing_rate'] - existing_segment['zero_crossing_rate'])/existing_segment['zero_crossing_rate']
        silence_distance = abs(new_segment['silence_percentage'] - existing_segment['silence_percentage'])
        if(existing_segment['silence_percentage'] != 0):
            silence_distance/=existing_segment['silence_percentage']

        image_distance= self.compare_images_ssim(new_segment['image_path'], existing_segment['image_path'])
        # Kết hợp các khoảng cách
        # Có thể áp dụng trọng số cho từng khoảng cách tùy thuộc vào mức độ quan trọng của từng đặc trưng
        combined_distance = (pitch_distance + bandwidth_distance + harmonicity_distance +
                             average_energy_distance + zcr_distance + silence_distance + 1-image_distance)/7

        return combined_distance

    def find_most_similar_segments(self, new_audio_path, top_n=10):
        audio_analyzer = AudioProcessor(new_audio_path)
        data_segment_input = audio_analyzer.process_audios("abc")
        similarities = []

        def process_file(file):
            values = file['file_results']
            list_dis = []
            for value in values:
                sub_distance = min(self.calculate_distance(segment_input, value) for segment_input in data_segment_input[0]['file_results'])
                list_dis.append(sub_distance)
            return {"file_name": file['file_name'], "average_distance": np.average(list_dis)}

        # Sử dụng ThreadPoolExecutor để song song hóa xử lý
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, file) for file in self.segment_data]
            for future in concurrent.futures.as_completed(futures):
                similarities.append(future.result())

        # Sắp xếp kết quả và trả về top n
        similarities_sorted = sorted(similarities, key=lambda x: x['average_distance'])
        return similarities_sorted[:top_n]


# Usage example
# csv_path = './audio_features.json'
# analyzer = AudioAnalyzer(csv_path)
# new_audio_path = './test'
# top_similar_files = analyzer.find_most_similar_segments(new_audio_path)

