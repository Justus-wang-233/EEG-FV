# -*- coding: utf-8 -*-
import numpy as np
import json
import math
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

    def _calculate_class_probabilities(self, k_nearest_labels):
        class_counts = Counter(tuple(label) for label in k_nearest_labels)
        total_count = sum(class_counts.values())
        class_probs = {cls: count / total_count for cls, count in class_counts.items()}
        return class_probs

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
        # 使用 Sigmoid 函数将置信度规范化到 0 到 1 之间
        confidence = 1 / (1 + np.exp(-confidence))
        return weighted_time, confidence

def amplify_signal(signal, factor=1e6):
    return signal * factor

def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str,
                   model_name: str = 'model.json') -> Dict[str, Any]:
    print(f"Loading model from {model_name}")

    seizure_present = False
    seizure_confidence = 0.0
    segment_predictions = []

    try:
        with open(model_name, 'r') as f:
            model_params = json.load(f)
            k = model_params['k']
            X_train = np.array(model_params['X_train'])
            y_train = np.array(model_params['y_train'])

        model = WBCkNN(k=k)
        model.fit(X_train, y_train)

        _montage, _montage_data, _is_missing = get_3montages(channels, data)
        if _montage is None or len(_montage_data) == 0:
            print("Error: Montage data is empty")
            return {
                "seizure_present": False,
                "seizure_confidence": 0.0,
                "onset": None,
                "onset_confidence": None,
                "offset": None,
                "offset_confidence": None
            }

        segment_duration = 25
        num_segments = math.ceil(len(data[0]) / (fs * segment_duration))

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

        def pad_segment(segment, target_length):
            if len(segment.shape) == 1:
                segment = segment[np.newaxis, :]
            pad_length = target_length - segment.shape[1]
            if pad_length > 0:
                segment = np.pad(segment, ((0, 0), (0, pad_length)), mode='edge')
            return segment

        for segment_idx in range(num_segments):
            start_idx = int(segment_idx * fs * segment_duration)
            end_idx = int(start_idx + fs * segment_duration)

            if end_idx > len(data[0]):
                segment_data = data[:, start_idx:]
                segment_data = pad_segment(segment_data, int(fs * segment_duration))
            else:
                segment_data = data[:, start_idx:end_idx]

            segment_start_time = segment_idx * segment_duration
            segment_end_time = segment_start_time + segment_duration

            segment_features = []

            for j, signal_name in enumerate(_montage):
                signal = _montage_data[j][start_idx:end_idx]
                if len(signal.shape) == 1:
                    signal = signal[np.newaxis, :]
                if len(signal) < fs * segment_duration:
                    signal = pad_segment(signal, int(fs * segment_duration))

                signal = amplify_signal(signal, factor=1e6)

                signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
                signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=50.0, n_jobs=2, verbose=False)

                band_powers = calculate_band_power(signal_filter[0], fs)
                segment_features.extend(band_powers)

            segment_features = np.array(segment_features).flatten()

            prediction, seizure_confidence, onset, onset_confidence, offset, offset_confidence = model._predict(segment_features)
            print(f"Segment {segment_idx} prediction: {prediction}")

            segment_predictions.append({
                "prediction": prediction,
                "start_time": segment_start_time,
                "end_time": segment_end_time,
                "seizure_confidence": seizure_confidence,
                "onset": onset,
                "onset_confidence": onset_confidence,
                "offset": offset,
                "offset_confidence": offset_confidence
            })

        seizure_present = any([seg["prediction"] == 1 for seg in segment_predictions])
        print(f"Segment predictions: {segment_predictions}")

        if seizure_present:
            print("Seizure detected. Processing onset and offset times.")
            first_onset_time = None
            last_offset_time = None
            onset_confidences = []
            offset_confidences = []
            for seg in segment_predictions:
                if seg["prediction"] == 1:
                    if first_onset_time is None:
                        first_onset_time = seg["start_time"] + seg["onset"]
                        onset_confidences.append(seg["onset_confidence"])
                    last_offset_time = seg["start_time"] + seg["offset"]
                    offset_confidences.append(seg["offset_confidence"])

            onset_confidence = np.mean(onset_confidences)
            offset_confidence = np.mean(offset_confidences)

            prediction = {
                "seizure_present": seizure_present,
                "seizure_confidence": 1.0,  # 置信度！！！，可以调整
                "onset": first_onset_time,
                "onset_confidence": onset_confidence,
                "offset": last_offset_time,
                "offset_confidence": offset_confidence
            }
        else:
            prediction = {
                "seizure_present": False,
                "seizure_confidence": 0.0,
                "onset": None,
                "onset_confidence": None,
                "offset": None,
                "offset_confidence": None
            }

        print(f"Final prediction: {prediction}")
        return prediction

    except Exception as e:
        print(f"Error in predict_labels: {e}")
        traceback.print_exc()
        return {
            "seizure_present": False,
            "seizure_confidence": 0.0,
            "onset": None,
            "onset_confidence": None,
            "offset": None,
            "offset_confidence": None
        }
