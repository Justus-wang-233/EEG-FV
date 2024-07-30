import numpy as np
import json
import traceback
from typing import List, Dict, Any
from wettbewerb import get_3montages
import mne
from scipy import signal as sig
from collections import Counter

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
        k_nearest_distances = [distances[i] for i in k_indices]

        weights = [1 / (dist + 1e-5) for dist in k_nearest_distances]

        seizure_probs = Counter()
        onset_times = []
        offset_times = []

        for label, weight in zip(k_nearest_labels, weights):
            seizure_probs[label[0]] += weight
            if label[0] == 1:
                onset_times.append(label[1])
                offset_times.append(label[2])

        seizure_present = max(seizure_probs, key=seizure_probs.get)
        seizure_confidence = self._calculate_seizure_confidence(k_nearest_distances)

        if seizure_present == 1:
            onset, onset_confidence = self._calculate_onset_offset_confidence(onset_times, weights[:len(onset_times)])
            offset, offset_confidence = self._calculate_onset_offset_confidence(offset_times, weights[:len(offset_times)])
        else:
            onset = None
            offset = None
            onset_confidence = 0.0
            offset_confidence = 0.0

        return seizure_present, seizure_confidence, onset, onset_confidence, offset, offset_confidence

    def _bray_curtis_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2)) / np.sum(np.abs(x1 + x2))

    def _calculate_seizure_confidence(self, distances):
        confidences = [1 / (d + 1e-5) for d in distances]
        return np.sum(confidences) / len(confidences)

    def _calculate_onset_offset_confidence(self, times, weights):
        if not times:
            return 0.0, 0.0
        weights = np.array(weights)
        weights /= np.sum(weights)
        weighted_time = np.sum(np.array(times) * weights)
        avg_inverse_distance = np.mean(weights)
        confidence = avg_inverse_distance / (np.std(weights) + 1e-5)
        confidence = 1 / (1 + np.exp(-confidence))
        return weighted_time, confidence

def amplify_signal(signal, factor=1e6):
    return signal * factor

def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str = 'model.json') -> Dict[str, Any]:
    print(f"Entering predict_labels with model: {model_name}")
    try:
        # 根据采样频率生成模型文件名
        wbcknn_model_name = f"model_wbcknn_{int(fs)}Hz.json"

        print(f"Loading WBCkNN model from {wbcknn_model_name}...")
        with open(wbcknn_model_name, 'r') as f:
            wbcknn_model_params = json.load(f)
            k = wbcknn_model_params['k']
            X_train = np.array(wbcknn_model_params['X_train'])
            y_train = np.array(wbcknn_model_params['y_train'])
            mean = np.array(wbcknn_model_params['mean'])
            std = np.array(wbcknn_model_params['std'])

        # 初始化并训练 WBCkNN 模型
        wbcknn_model = WBCkNN(k=k)
        wbcknn_model.fit(X_train, y_train)

        print("Processing montage data...")
        _montage, _montage_data, _is_missing = get_3montages(channels, data)
        if _montage is None or len(_montage_data) == 0:
            print("Error: Montage data is empty")
            return {}

        def calculate_band_power(signal, fs):
            nperseg = min(len(signal), 256)
            freqs, psd = sig.welch(signal, fs, nperseg=nperseg)
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            band_powers = []
            for band, (low, high) in bands.items():
                band_idx = np.logical_and(freqs >= low, freqs <= high)
                band_powers.append(np.sum(psd[band_idx]))
            return band_powers

        signal_data = amplify_signal(np.array(_montage_data))

        signal_notch = np.array(
            [mne.filter.notch_filter(channel_data, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False) for channel_data in signal_data]
        )

        signal_filter = np.array(
            [mne.filter.filter_data(channel_data, sfreq=fs, l_freq=0.5, h_freq=50.0, n_jobs=2, verbose=False) for channel_data in signal_notch]
        )

        features = []
        for channel in signal_filter:
            band_power = calculate_band_power(channel, fs)
            features.extend(band_power)

        if len(features) != 15:
            print(f"Error: Expected 15 features, but got {len(features)}")
            return {}

        features = np.array(features).flatten()
        print(f"Features before standardization: {features}")

        # 标准化特征
        features = (features - mean) / std
        print(f"Features after standardization: {features}")

        prediction, seizure_confidence, onset, onset_confidence, offset, offset_confidence = wbcknn_model._predict(features)

        print(f"Seizure present: {prediction}")
        print(f"Seizure confidence: {seizure_confidence}")
        print(f"Onset: {onset}, Onset confidence: {onset_confidence}")
        print(f"Offset: {offset}, Offset confidence: {offset_confidence}")

        result = {
            "seizure_present": bool(prediction),
            "seizure_confidence": seizure_confidence,
            "onset": onset,
            "onset_confidence": onset_confidence,
            "offset": offset,
            "offset_confidence": offset_confidence
        }

        print(f"Final prediction: {result}")

        return result

    except Exception as e:
        print(f"Error in predict_labels: {e}")
        traceback.print_exc()
        return {
            "seizure_present": False,
            "seizure_confidence": 0.0,
            "onset": None,
            "onset_confidence": 0.0,
            "offset": None,
            "offset_confidence": 0.0
        }
