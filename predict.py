# -*- coding: utf-8 -*-
import numpy as np
import json
from typing import List, Dict, Any
from wettbewerb import get_3montages
import mne
from scipy import signal as sig
from collections import Counter
import math

class WBCkNN:
    def __init__(self, k):
        self.k = k  # 设置k值，表示最近邻的数量

    def fit(self, X_train, y_train):
        self.X_train = X_train  # 保存训练数据
        self.y_train = y_train  # 保存训练标签

    def predict(self, X_test):
        # 对每个测试样本进行预测
        predictions = [self._predict(x) for x in X_test]
        return np.array([pred[0] for pred in predictions]), np.array([pred[1] for pred in predictions])

    def _predict(self, x):
        # 计算所有训练样本与测试样本之间的Bray-Curtis距离
        distances = [self._bray_curtis_distance(x, x_train) for x_train in self.X_train]

        # 找到距离最近的k个训练样本的索引
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 计算贝叶斯后验概率
        class_probs = self._calculate_class_probabilities(k_nearest_labels)

        # 返回具有最高概率的类别和最近邻样本的索引及距离
        return max(class_probs, key=class_probs.get), k_indices, [distances[i] for i in k_indices]

    def _bray_curtis_distance(self, x1, x2):
        # 计算两个样本之间的Bray-Curtis距离
        return np.sum(np.abs(x1 - x2)) / np.sum(np.abs(x1 + x2))

    def _calculate_class_probabilities(self, k_nearest_labels):
        # 计算k个最近邻样本中每个类别的频率
        class_counts = Counter(tuple(label) for label in k_nearest_labels)  # 转换为元组
        total_count = sum(class_counts.values())
        # 计算每个类别的概率（频率）
        class_probs = {cls: count / total_count for cls, count in class_counts.items()}
        return class_probs
    
    # Guan Changes Function
    def _calculate_class_probabilities_with_weights(self, k_nearest_labels, distances):
        # 使用距离的倒数作为权重并增加一个小的常数以防止除以零
        weights = [1 / (d + 1e-5) for d in distances]
        class_counts = Counter(k_nearest_labels)
        total_weight = sum(weights)
        # 计算每个类别的加权概率
        class_probs = {}
        for cls in class_counts.keys():
            class_probs[cls] = sum(
                [weights[i] for i, label in enumerate(k_nearest_labels) if label == cls]) / total_weight

        return class_probs

    def _calculate_seizure_confidence(self, distances):
        # 通过距离的倒数来计算癫痫发作的置信度
        return np.sum([1 / (d + 1e-5) for d in distances]) / len(distances)

    def _calculate_onset_offset_confidence(self, times, distances):
        # 通过距离的倒数来计算起始时间和结束时间的置信度
        if not times:
            return 0.0
        # 使用距离的倒数作为权重
        weights = np.array([1 / (d + 1e-5) for d in distances])
        # 归一化权重
        weights = np.sum(weights)
        # 计算加权平均时间
        weighted_time = np.sum(np.array(times) * weights)
        # 计算置信度: 使用距离的倒数的平均值来表示一致性
        avg_inverse_distance = np.mean(weights)
        # 置信度基于样本间的距离差异进行调整
        # 当所有样本距离都较小时，置信度应当较高，当局差异较大时，置信度应当较低
        confidence = avg_inverse_distance / (np.std(weights) + 1e-5)
        confidence = min(max(confidence, 0.0), 1.0) # 将置信度限制在【0， 1】范围内
        return weighted_time, confidence
        # 固置信度

### Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str,
                   model_name: str = 'model.json') -> Dict[str, Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        提供的通道名称
    data : ndarray
        给定通道的EEG信号
    fs : float
        信号的采样频率
    reference_system :  str
        使用的参考系统
    model_name : str
        模型的名称，用于加载
    Returns
    -------
    prediction : Dict[str,Any]
        包含是否发作及发作时间（开始+结束）的预测结果
    '''

    print(f"Loading model from {model_name}")

    # 初始化返回结果
    seizure_present = False
    seizure_confidence = 0.0
    segment_predictions = []

    # 读取训练好的模型参数
    with open(model_name, 'r') as f:
        model_params = json.load(f)
        k = model_params['k']
        X_train = np.array(model_params['X_train'])
        y_train = np.array(model_params['y_train'])

    # 初始化 WBCkNN 分类器并加载训练数据
    model = WBCkNN(k=k)
    model.fit(X_train, y_train)

    # 提取3导联信号
    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    features = []

    # 设置段持续时间（秒）
    segment_duration = 25
    num_segments = math.ceil(len(data[0]) / (fs * segment_duration))

    print(f"Number of segments: {num_segments}")

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
        band_powers = []
        for band, (low, high) in bands.items():
            band_power = np.trapz(psd[(freqs >= low) & (freqs <= high)], freqs[(freqs >= low) & (freqs <= high)])
            band_powers.append(band_power)
        return band_powers

    def pad_segment(segment, target_length):
        pad_length = target_length - len(segment)
        if pad_length > 0:
            segment = np.concatenate([segment, segment[-pad_length:]])
        return segment

    # 处理每个段
    for segment_idx in range(num_segments):
        start_idx = segment_idx * fs * segment_duration
        end_idx = start_idx + fs * segment_duration

        # 确保最后一个段不会超出信号的总长度
        if end_idx > len(data[0]):
            segment_data = data[:, start_idx:]
            segment_data = pad_segment(segment_data, fs * segment_duration)
        else:
            segment_data = data[:, start_idx:end_idx]

        segment_start_time = segment_idx * segment_duration
        segment_end_time = segment_start_time + segment_duration

        segment_features = []

        for j, signal_name in enumerate(_montage):
            signal = _montage_data[j][start_idx:end_idx]
            if len(signal) < fs * segment_duration:
                signal = pad_segment(signal, fs * segment_duration)
            signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)

            band_powers = calculate_psd_and_band_power(signal_filter, fs)

            segment_features.append(band_powers)

        mean_features = np.mean(segment_features, axis=0).reshape(1, -1)

        prediction, k_indices, distances = model._predict(mean_features[0])
        print(f"Segment {segment_idx} prediction: {prediction}")  # 调试信息

        # 存储每次的预测结果，仅存储预测类别
        segment_predictions.append({
            "prediction": prediction[0],  # 只存储预测的类别（0或1）
            "start_time": segment_start_time,
            "end_time": segment_end_time
        })

    # 检查是否有任何一段预测为1
    seizure_present = any([seg["prediction"] == 1 for seg in segment_predictions])
    print(f"Segment predictions: {segment_predictions}")

    # 如果有癫痫发作的段，合并时间段
    if seizure_present:
        seizure_confidence = 1.0
        onset_times = [seg["start_time"] for seg in segment_predictions if seg["prediction"] == 1]
        offset_times = [seg["end_time"] for seg in segment_predictions if seg["prediction"] == 1]

        # 合并相邻的时间段
        merged_onset_times = []
        merged_offset_times = []
        current_onset = onset_times[0]
        current_offset = offset_times[0]
        for i in range(1, len(onset_times)):
            if onset_times[i] <= current_offset:  # 如果当前段的发作时间与上一个段重叠或相接
                current_offset = max(current_offset, offset_times[i])
            else:
                merged_onset_times.append(current_onset)
                merged_offset_times.append(current_offset)
                current_onset = onset_times[i]
                current_offset = offset_times[i]
        merged_onset_times.append(current_onset)
        merged_offset_times.append(current_offset)

        onset = merged_onset_times[0]
        onset_confidence = 0.99
        offset = merged_offset_times[0]
        offset_confidence = 0.99
        if len(merged_onset_times) > 1:
            for i in range(1, len(merged_onset_times)):
                onset = min(onset, merged_onset_times[i])
                offset = max(offset, merged_offset_times[i])
    else:
        onset = None
        onset_confidence = None
        offset = None
        offset_confidence = None

    # --------------------------------------------------------------------------
    prediction = {
        "seizure_present": seizure_present,
        "seizure_confidence": seizure_confidence,
        "onset": onset,
        "onset_confidence": onset_confidence,
        "offset": offset,
        "offset_confidence": offset_confidence
    }

    print(f"Final prediction: {prediction}")  # 调试信息

    return prediction
