import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle


class AudioClusterer:
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = None
        self.kmeans = None
        self.file_cluster_mapping = {}

    def train_and_save_model(self, data, scaler_path='scaler.pkl', kmeans_path='kmeans_model.pkl'):
        feature_list = []
        file_segments = []
        for item in data:
            file_name = item["file_name"]
            for segment in item["file_results"]:
                features = [
                    segment['bandwidth'],
                    segment['harmonicity'],
                    segment['pitch'],
                    segment['average_energy'],
                    segment['zero_crossing_rate'],
                    segment['silence_percentage']
                ]
                features.extend(segment['mfccs'])
                feature_list.append(features)
                file_segments.append(file_name)

        X = np.array(feature_list)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X_scaled)

        with open(scaler_path, 'wb') as file:
            pickle.dump(self.scaler, file)

        with open(kmeans_path, 'wb') as file:
            pickle.dump(self.kmeans, file)

        # Save file-cluster mapping
        labels = self.kmeans.labels_
        file_cluster_count = {file_name: [] for file_name in file_segments}
        for label, file_name in zip(labels, file_segments):
            file_cluster_count[file_name].append(label)

        self.file_cluster_mapping = {}
        for file_name, clusters in file_cluster_count.items():
            most_common_cluster = Counter(clusters).most_common(1)[0][0]
            self.file_cluster_mapping[file_name] = most_common_cluster

        with open('file_cluster_mapping.pkl', 'wb') as file:
            pickle.dump(self.file_cluster_mapping, file)

    def load_model(self, scaler_path='scaler.pkl', kmeans_path='kmeans_model.pkl',
                   mapping_path='file_cluster_mapping.pkl'):
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(kmeans_path, 'rb') as file:
            self.kmeans = pickle.load(file)

        with open(mapping_path, 'rb') as file:
            self.file_cluster_mapping = pickle.load(file)

    def predict_cluster(self, new_file_data):
        if self.scaler is None or self.kmeans is None:
            self.load_model()

        new_feature_list = []
        for segment in new_file_data["file_results"]:
            features = [
                segment['bandwidth'],
                segment['harmonicity'],
                segment['pitch'],
                segment['average_energy'],
                segment['zero_crossing_rate'],
                segment['silence_percentage']
            ]
            features.extend(segment['mfccs'])
            new_feature_list.append(features)

        new_X = np.array(new_feature_list)
        new_X_scaled = self.scaler.fit_transform(new_X)  # Ensure scaler is already loaded
        new_labels = self.kmeans.predict(new_X_scaled)  # Ensure kmeans is already loaded
        most_common_cluster = Counter(new_labels).most_common(1)[0][0]

        return {"filename": new_file_data["file_name"], "cluster": most_common_cluster}

    def get_files_in_same_cluster(self, new_file_data):
        cluster_info = self.predict_cluster(new_file_data)
        target_cluster = cluster_info["cluster"]
        same_cluster_files = [file_name for file_name, cluster in self.file_cluster_mapping.items() if
                              cluster == target_cluster]

        return same_cluster_files

# Ví dụ sử dụng
# data = [
#     {
#         "file_name": "xc25754_edited.wav",
#         "file_results": [
#             {
#                 "bandwidth": 122.59247416515468,
#                 "harmonicity": 7537428.5888671875,
#                 "pitch": 2.436902863745284e-06,
#                 "average_energy": 0.08378693802467821,
#                 "zero_crossing_rate": 0.1327437641723356,
#                 "silence_percentage": 0.0,
#                 "mfccs": [
#                     107.16460418701172, -2.66084361076355, 3.1058545112609863, -1.2818577289581299,
#                     -3.50656795501709, -2.068669319152832, -5.873672008514404, -2.2702248096466064,
#                     -1.773881435394287, 0.3004336357116699, 4.212522506713867, 1.7627588510513306,
#                     5.727506160736084, 1.6900935173034668, 3.2763915061950684, -0.6967756152153015,
#                     -2.497934341430664, -2.803633213043213, -2.776250123977661, -1.1516880989074707,
#                     -0.8918963074684143, -0.22050370275974274, 2.444385290145874, 1.831943154335022,
#                     3.586146593093872, 0.824165403842926, 0.9115421772003174, -1.5770375728607178,
#                     -0.9082900881767273, -1.6732409000396729, -1.0617657899856567, -1.4070546627044678,
#                     0.07683920860290527, 0.7567594647407532, 2.085357189178467, 1.39201021194458,
#                     1.3852136135101318, 0.6594146490097046, 0.20174644887447357, -1.1794112920761108
#                 ]
#             },
#             {
#                 "bandwidth": 79.637112816368,
#                 "harmonicity": 7537428.5888671875,
#                 "pitch": 1.733253667039767e-05,
#                 "average_energy": 0.05014064125436269,
#                 "zero_crossing_rate": 0.13582766439909297,
#                 "silence_percentage": 0.0,
#                 "mfccs": [
#                     106.79939270019531, -2.7904999256134033, 3.9518790245056152, -1.0493766069412231,
#                     -2.356008768081665, -2.01843523979187, -5.584344387054443, -2.073428153991699,
#                     -2.0748114585876465, -0.6645901799201965, 3.00003719329834, 1.2574090957641602,
#                     5.789688587188721, 1.860946536064148, 3.555258274078369, -0.2986317276954651,
#                     -1.6513980627059937, -1.8856974840164185, -2.962256669998169, -1.5795400142669678,
#                     -0.9375020265579224, -0.3910398781299591, 1.9579362869262695, 1.1021252870559692,
#                     3.207838773727417, 0.8126444220542908, 1.767061471939087, 0.12975405156612396,
#                     0.42596808075904846, -1.028469204902649, -1.5227817296981812, -1.287777304649353,
#                     -0.2310679852962494, -0.4935380816459656, 1.3148188591003418, 0.7142027020454407,
#                     1.016571044921875, 0.17138266563415527, 0.17677748203277588, -0.3850070536136627
#                 ]
#             }
#         ]
#     }
# ]

# # Huấn luyện và lưu mô hình
# clusterer = AudioClusterer()
# clusterer.train_and_save_model(data)
#
# # Dự đoán cụm và lấy các file trong cùng cụm cho file mới
# new_file_data = {
#     "file_name": "xc25754_edited.wav",
#     "file_results": [
#         {
#             "bandwidth": 122.59247416515468,
#             "harmonicity": 7537428.5888671875,
#             "pitch": 2.436902863745284e-06,
#             "average_energy": 0.08378693802467821,
#             "zero_crossing_rate": 0.1327437641723356,
#             "silence_percentage": 0.0,
#             "mfccs": [
#                 107.16460418701172, -2.66084361076355, 3.1058545112609863, -1.2818577289581299,
#                 -3.50656795501709, -2.068669319152832, -5.873672008514404, -2.2702248096466064,
#                 -1.773881435394287, 0.3004336357116699, 4.212522506713867, 1.7627588510513306,
#                 5.727506160736084, 1.6900935173034668, 3.2763915061950684, -0.6967756152153015,
#                 -2.497934341430664, -2.803633213043213, -2.776250123977661, -1.1516880989074707,
#                 -0.8918963074684143, -0.22050370275974274, 2.444385290145874, 1.831943154335022,
#                 3.586146593093872, 0.824165403842926, 0.9115421772003174, -1.5770375728607178,
#                 -0.9082900881767273, -1.6732409000396729, -1.0617657899856567, -1.4070546627044678,
#                 0.07683920860290527, 0.7567594647407532, 2.085357189178467, 1.39201021194458,
#                 1.3852136135101318, 0.6594146490097046, 0.20174644887447357, -1.1794112920761108
#             ]
#         },
#         {
#             "bandwidth": 79.637112816368,
#             "harmonicity": 7537428.5888671875,
#             "pitch": 1.733253667039767e-05,
#             "average_energy": 0.05014064125436269,
#             "zero_crossing_rate": 0.13582766439909297,
#             "silence_percentage": 0.0,
#             "mfccs": [
#                 106.79939270019531, -2.7904999256134033, 3.9518790245056152, -1.0493766069412231,
#                 -2.356008768081665, -2.01843523979187, -5.584344387054443, -2.073428153991699,
#                 -2.0748114585876465, -0.6645901799201965, 3.00003719329834, 1.2574090957641602,
#                 5.789688587188721, 1.860946536064148, 3.555258274078369, -0.2986317276954651,
#                 -1.6513980627059937, -1.8856974840164185, -2.962256669998169, -1.5795400142669678,
#                 -0.9375020265579224, -0.3910398781299591, 1.9579362869262695, 1.1021252870559692,
#                 3.207838773727417, 0.8126444220542908, 1.767061471939087, 0.12975405156612396,
#                 0.42596808075904846, -1.028469204902649, -1.5227817296981812, -1.287777304649353,
#                 -0.2310679852962494, -0.4935380816459656, 1.3148188591003418, 0.7142027020454407,
#                 1.016571044921875, 0.17138266563415527, 0.17677748203277588, -0.3850070536136627
#             ]
#         }
#     ]
# }
#
# clusterer.load_model()
# result = clusterer.predict_cluster(new_file_data)
# print(result)
#
# files_in_same_cluster = clusterer.get_files_in_same_cluster(new_file_data)
# print(files_in_same_cluster)
