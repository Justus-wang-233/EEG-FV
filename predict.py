# -*- coding: utf-8 -*-

import numpy as np
import json
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
        class_counts = Counter(k_nearest_labels)
        total_count = sum(class_counts.values())
        # 计算每个类别的概率（频率）
        class_probs = {cls: count / total_count for cls, count in class_counts.items()}
        return class_probs

### Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str,
                   model_name: str = 'model.json') -> Dict[str, Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models, das ihr beispielsweise bei Abgabe genannt habt.
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

    # 初始化返回结果
    seizure_present = False
    seizure_confidence = 0.0
    onset = 0.0
    onset_confidence = 0.0
    offset = 0.0
    offset_confidence = 0.0

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
    segment_duration = 10
    num_segments = int(len(data[0]) / (fs * segment_duration))

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

    # 处理每个段
    for segment_idx in range(num_segments):
        start_idx = segment_idx * fs * segment_duration
        end_idx = start_idx + fs * segment_duration

        segment_data = data[:, start_idx:end_idx]
        segment_start_time = segment_idx * segment_duration
        segment_end_time = segment_start_time + segment_duration

        segment_features = []

        for j, signal_name in enumerate(_montage):
            signal = _montage_data[j][start_idx:end_idx]
            signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)

            band_powers = calculate_psd_and_band_power(signal_filter, fs)

            segment_features.append(band_powers)

        mean_features = np.mean(segment_features, axis=0).reshape(1, -1)

        prediction, k_indices, distances = model._predict(mean_features[0])
        if prediction == 1:
            seizure_present = True
            seizure_confidence = 1.0

            # 获取k个最近邻样本的开始和结束时间及其距离
            k_start_times = [y_train[i][1] for i in k_indices if y_train[i][1] is not None]
            k_end_times = [y_train[i][2] for i in k_indices if y_train[i][2] is not None]

            if k_start_times and k_end_times:
                # 距离的倒数作为权重
                weights = [1 / (d + 1e-5) for d in distances]  # 防止除以零
                weights = weights / np.sum(weights)  # 归一化权重

                # 计算加权平均值
                onset = np.sum(np.array(k_start_times) * weights)
                onset_confidence = 0.99
                offset = np.sum(np.array(k_end_times) * weights)
                offset_confidence = 0.99

    # --------------------------------------------------------------------------
    prediction = {"seizure_present": seizure_present, "seizure_confidence": seizure_confidence,
                  "onset": onset, "onset_confidence": onset_confidence, "offset": offset,
                  "offset_confidence": offset_confidence}

    return prediction
