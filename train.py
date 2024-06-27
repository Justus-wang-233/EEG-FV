import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import json
from collections import Counter

class WBCkNN:
    def __init__(self, k):
        self.k = k  # 设置k值，表示最近邻的数量

    def fit(self, X_train, y_train):
        self.X_train = X_train  # 保存训练数据
        self.y_train = y_train  # 保存训练标签

    def predict(self, X_test):
        # 对每个测试样本进行预测
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # 计算所有训练样本与测试样本之间的Bray-Curtis距离
        distances = [self._bray_curtis_distance(x, x_train) for x_train in self.X_train]

        # 找到距离最近的k个训练样本的索引
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 计算贝叶斯后验概率
        class_probs = self._calculate_class_probabilities(k_nearest_labels)

        # 返回具有最高概率的类别
        return max(class_probs, key=class_probs.get)

    def _bray_curtis_distance(self, x1, x2):
        # 计算两个样本之间的Bray-Curtis距离
        return np.sum(np.abs(x1 - x2)) / np.sum(np.abs(x1 + x2))

    def _calculate_class_probabilities(self, k_nearest_labels):
        # 计算k个最近邻样本中每个类别的频率
        class_counts = Counter(k_nearest_labels)
        total_count = sum(class_counts.values())
        # 计算每个类别的概率（频率）
        class_probs = {cls: count / total_count for cls, count in class_counts.items()}
        return class_probs

# 定义计算功率谱密度（PSD）和频带能量的函数
def calculate_psd_and_band_power(signal, fs):
    nperseg = min(len(signal), 256)
    freqs, psd = sig.welch(signal, fs, nperseg=nperseg)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    band_power = {}
    for band, (low, high) in bands.items():
        band_idx = np.logical_and(freqs >= low, freqs <= high)
        band_power[band] = np.sum(psd[band_idx])
    return psd, band_power

# 定义处理单个段的函数
def process_segment(segment, fs, segment_start_time, segment_end_time, seizure_present, seizure_start_time, seizure_end_time):
    if len(segment) < fs:  # 忽略长度不足的段
        return None, None

    psd, band_powers = calculate_psd_and_band_power(segment, fs)

    # 选择前3个PSD值
    top_3_psd = psd[:3]

    # 合并Bandpower和前3个PSD特征
    features = list(band_powers.values()) + list(top_3_psd)

    # 判断该段内是否有癫痫发作
    if seizure_present == 1 and seizure_start_time < segment_end_time and seizure_end_time > segment_start_time:
        label = 1
        overlap_start_time = max(0, seizure_start_time - segment_start_time)
        overlap_end_time = min(segment_end_time - segment_start_time, seizure_end_time - segment_start_time)
    else:
        label = 0
        overlap_start_time = None
        overlap_end_time = None

    return features, (label, overlap_start_time, overlap_end_time)

# 加载并处理数据
training_folder = r"C:\Users\lyjwa\Desktop\EEG-FV\test"  # Wang
# training_folder = r"/Users/guanhanchen/Documents/EEG-FV/mini_mat_wki"  # Guan
# training_folder  = "../shared_data/training_mini"  # Jupyter

ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

features = []
labels = []

# 每个段的持续时间（以秒为单位）
segment_duration = 50

for i, _id in enumerate(ids):
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]

    print(f"EEG Label for file {_id}: {_eeg_label}")

    _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)

    for j, signal_name in enumerate(_montage):
        signal = _montage_data[j]
        # 应用Notch滤波器以衰减电源频率干扰
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
        # 应用带通滤波器以过滤掉噪声
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)

        # 计算总段数
        num_segments = int(len(signal_filter) / (_fs * segment_duration))

        # 使用ThreadPoolExecutor并行处理段
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for segment_idx in range(num_segments):
                start_idx = segment_idx * _fs * segment_duration
                end_idx = start_idx + _fs * segment_duration
                segment = signal_filter[start_idx:end_idx]
                segment_start_time = segment_idx * segment_duration
                segment_end_time = segment_start_time + segment_duration
                futures.append(executor.submit(process_segment, segment, _fs, segment_start_time, segment_end_time, *_eeg_label))

            for future in futures:
                result = future.result()
                if result[0] is not None:
                    features.append(result[0])
                    labels.append(result[1])

X = np.array(features)
Y = np.array(labels, dtype=object)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Label distribution:", np.bincount([label[0] for label in Y]))

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42, stratify=[label[0] for label in Y])

print("Train label distribution:", np.bincount([label[0] for label in y_train]))
print("Test label distribution:", np.bincount([label[0] for label in y_test]))

# 进行交叉验证以选择最佳的k值
best_k = None
best_score = 0
k_values = range(1, 8)

for k in k_values:
    model = WBCkNN(k=k)
    model.fit(X_train, [label[0] for label in y_train])
    y_pred = model.predict(X_test)
    score = f1_score([label[0] for label in y_test], y_pred)
    print(f'k={k}, F1 Score={score}')
    if score > best_score:
        best_score = score
        best_k = k

print(f'Best k value: {best_k} with F1 Score: {best_score}')

# 使用最佳k值训练最终模型
model = WBCkNN(k=best_k)
model.fit(X_scaled, [label[0] for label in Y])

# 保存分类模型
model_params = {
    'k': best_k,
    'X_train': X_scaled.tolist(),
    'y_train': [[label[0], label[1], label[2]] for label in Y],
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=4)
