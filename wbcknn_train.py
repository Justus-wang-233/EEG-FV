import numpy as np
from wettbewerb import get_3montages, EEGDataset
import mne
from scipy import signal as sig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter
from typing import List, Tuple, Dict
import json


class WBCkNN:
    def __init__(self, k):
        self.k = k  # 设置k值，表示最近邻的数量

    def fit(self, X_train, y_train):
        self.X_train = X_train  # 保存训练数据
        self.y_train = y_train  # 保存训练标签

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self._bray_curtis_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 计算贝叶斯后验概率
        class_probs = self._calculate_class_probabilities(k_nearest_labels)

        # 返回具有最高概率的类别
        return max(class_probs, key=class_probs.get)

    def _bray_curtis_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2)) / np.sum(np.abs(x1 + x2))

    def _calculate_class_probabilities(self, k_nearest_labels):
        class_counts = Counter(k_nearest_labels)
        total_count = sum(class_counts.values())
        class_probs = {cls: count / total_count for cls, count in class_counts.items()}
        return class_probs


def calculate_band_power(signal, fs):
    freqs, psd = sig.welch(signal, fs, nperseg=256)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    band_power = []
    for band, (low, high) in bands.items():
        band_idx = np.logical_and(freqs >= low, freqs <= high)
        band_power.append(np.sum(psd[band_idx]))
    return band_power


def process_signal(signal, fs, seizure_present, seizure_start_time, seizure_end_time):
    features = []
    for channel in signal:
        band_power = calculate_band_power(channel, fs)
        features.extend(band_power)

    if len(features) != 15:
        print(f"Feature length mismatch: {len(features)} != 15")
        return None, None

    if seizure_present == 1:
        label = 1
    else:
        label = 0

    return features, (label, seizure_start_time, seizure_end_time)


def amplify_signal(signal, factor=1e6):
    return signal * factor


def apply_highpass_filter(signal, fs, cutoff=1.0):
    return mne.filter.filter_data(data=signal, sfreq=fs, l_freq=cutoff, h_freq=None, filter_length='auto', n_jobs=2,
                                  verbose=False)


def apply_notch_filter(signal, fs):
    try:
        return np.array(
            [mne.filter.notch_filter(channel_data, Fs=fs, freqs=np.array([50., 100.]), filter_length='auto', n_jobs=2,
                                     verbose=False) for channel_data in signal]
        )
    except ValueError as e:
        print(f"Skipping notch filter due to: {e}")
        return signal


def apply_bandpass_filter(signal, fs):
    try:
        return np.array(
            [mne.filter.filter_data(channel_data, sfreq=fs, l_freq=0.5, h_freq=50.0, filter_length='auto', n_jobs=2,
                                    verbose=False) for channel_data in signal]
        )
    except ValueError as e:
        print(f"Skipping bandpass filter due to: {e}")
        return signal


training_folder = "../shared_data/training"
features_dict: Dict[float, List[List[float]]] = {}
labels_dict: Dict[float, List[Tuple[int, float, float]]] = {}
means_stds_dict: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

dataset = EEGDataset(training_folder)
block_size = 100
max_signals = 6213

for start_index in range(0, min(len(dataset), max_signals), block_size):
    end_index = min(start_index + block_size, len(dataset), max_signals)
    print(f"Processing signals from {start_index} to {end_index}")

    for i in range(start_index, end_index):
        _id, _channels, _eeg_signals, _fs, _reference_system, _eeg_label = dataset[i]
        print(f"EEG Label for file {_id}: {_eeg_label}")

        _montage, _montage_data, _is_missing = get_3montages(_channels, _eeg_signals)
        signal_data = np.array(_montage_data)
        signal_amplified = amplify_signal(signal_data)
        signal_highpass = apply_highpass_filter(signal_amplified, _fs)
        signal_notch = apply_notch_filter(signal_highpass, _fs)
        signal_filter = apply_bandpass_filter(signal_notch, _fs)

        feature, label = process_signal(signal_filter, _fs, *_eeg_label)
        if feature is not None:
            if _fs not in features_dict:
                features_dict[_fs] = []
                labels_dict[_fs] = []
            features_dict[_fs].append(feature)
            labels_dict[_fs].append(label)

# 对所有特征进行标准化
for fs in features_dict:
    X = np.array(features_dict[fs])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    means_stds_dict[fs] = (mean, std)
    features_dict[fs] = (X - mean) / std

# 训练并保存模型
for fs in features_dict:
    X = np.array(features_dict[fs])
    Y = np.array(labels_dict[fs], dtype=object)

    print(f"Processing data with sampling frequency {fs} Hz")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Label distribution:", np.bincount([label[0] for label in Y]))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,
                                                        stratify=[label[0] for label in Y])

    print("Train label distribution:", np.bincount([label[0] for label in y_train]))
    print("Test label distribution:", np.bincount([label[0] for label in y_test]))

    best_k = None
    best_score = 0
    k_values = range(1, 8)

    for k in k_values:
        model = WBCkNN(k=k)
        model.fit(X_train, [label[0] for label in y_train])
        y_pred = model.predict(X_test)
        score = f1_score([label[0] for label in y_test], y_pred)
        if score > best_score:
            best_score = score
            best_k = k

    print(f'Best k value: {best_k} with F1 Score: {best_score}')

    model = WBCkNN(k=best_k)
    model.fit(X, [label[0] for label in Y])

    model_params = {
        'k': best_k,
        'X_train': X.tolist(),
        'y_train': [[label[0], label[1], label[2]] for label in Y],
        'mean': means_stds_dict[fs][0].tolist(),
        'std': means_stds_dict[fs][1].tolist()
    }
    output_file = f'model_wbcknn_{fs}Hz.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(model_params, f, ensure_ascii=False, indent=4)
    print(f'Model for sampling frequency {fs} Hz saved to {output_file}')
