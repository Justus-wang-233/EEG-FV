# -*- coding: utf-8 -*-

import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from wbcknn import WBCkNN  # 导入 WBCkNN 类
import json

# 定义计算功率谱密度（PSD）和频带能量的函数
def calculate_psd_and_band_power(signal, fs):
    nperseg = min(len(signal), 256)  # 调整nperseg以适应窗口大小
    freqs, psd = sig.welch(signal, fs, nperseg=nperseg)
    # 定义频带
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

# 定义处理单个段的函数
def process_segment(segment, fs, segment_start_time, segment_end_time, seizure_present, seizure_start_time, seizure_end_time):
    if len(segment) < fs:  # 忽略长度不足的段
        return None, None

    band_powers = calculate_psd_and_band_power(segment, fs)

    # 判断该段内是否有癫痫发作
    if seizure_present == 1 and seizure_start_time < segment_end_time and seizure_end_time > segment_start_time:
        label = 1
        overlap_start_time = max(0, seizure_start_time - segment_start_time)
        overlap_end_time = min(segment_end_time - segment_start_time, seizure_end_time - segment_start_time)
    else:
        label = 0
        overlap_start_time = None
        overlap_end_time = None

    return band_powers, (label, overlap_start_time, overlap_end_time)

# 加载并处理数据
training_folder = r"C:\Users\lyjwa\Desktop\EEG-FV\test"
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

features = []
labels = []

# 每个段的持续时间（以秒为单位）
segment_duration = 25

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

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=[label[0] for label in Y])

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
model.fit(X, [label[0] for label in Y])

# 保存分类模型
model_params = {
    'k': best_k,
    'X_train': X.tolist(),
    'y_train': [[label[0], label[1], label[2]] for label in Y]
}
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=4)
