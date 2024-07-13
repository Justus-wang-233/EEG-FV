import numpy as np
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import json

# 定义计算频带能量的函数
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

# 定义处理单个段的函数
def process_segment(segment, fs, segment_start_time, segment_end_time, seizure_present, seizure_start_time, seizure_end_time):
    if segment.shape[1] < fs:  # 忽略长度不足的段
        print("Segment too short, ignoring")
        return None, None

    features = []
    for channel in segment:
        band_power = calculate_band_power(channel, fs)
        features.extend(band_power)

    # 确保每段有15个特征
    if len(features) != 15:
        print(f"Feature length mismatch: {len(features)} != 15")
        return None, None

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

# 放大信号
def amplify_signal(signal, factor=1e6):
    return signal * factor

# 应用高通滤波器
def apply_highpass_filter(signal, fs, cutoff=1.0):
    return mne.filter.filter_data(data=signal, sfreq=fs, l_freq=cutoff, h_freq=None, n_jobs=2, verbose=False)

# 应用Notch滤波器
def apply_notch_filter(signal, fs):
    return np.array(
        [mne.filter.notch_filter(channel_data, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False) for channel_data in signal]
    )

# 应用带通滤波器
def apply_bandpass_filter(signal, fs):
    return np.array(
        [mne.filter.filter_data(channel_data, sfreq=fs, l_freq=0.5, h_freq=50.0, n_jobs=2, verbose=False) for channel_data in signal]
    )

# 加载并处理数据
training_folder  = r"C:\Users\lyjwa\Desktop\EEG-FV\test"  # Wang
# training_folder = r"/Users/guanhanchen/Documents/EEG-FV/mini_mat_wki"  # Guan
# training_folder  = "../shared_data/training_mini"  # Jupyter

ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

features = []
labels = []

# 每个段的持续时间（以秒为单位）
segment_duration = 25

for i, _id in enumerate(ids):
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]

    signal_length_seconds = _eeg_signals.shape[1] / _fs
    print(f"EEG Label for file {_id}: {_eeg_label}")
    print(f"Signal length: {signal_length_seconds} seconds")

    _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)

    # 合并三个通道的数据
    signal_data = np.array(_montage_data)
    # 放大信号
    signal_amplified = amplify_signal(signal_data)

    # 应用高通滤波器
    signal_highpass = apply_highpass_filter(signal_amplified, _fs)

    # 应用Notch滤波器以衰减电源频率干扰
    signal_notch = apply_notch_filter(signal_highpass, _fs)

    # 应用带通滤波器以过滤掉噪声
    signal_filter = apply_bandpass_filter(signal_notch, _fs)

    # 计算总段数
    num_segments = int(signal_filter.shape[1] / (_fs * segment_duration))

    print(f"Processing {num_segments} segments for file {_id}...")

    # 使用ThreadPoolExecutor并行处理段
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for segment_idx in range(num_segments):
            start_idx = segment_idx * _fs * segment_duration
            end_idx = start_idx + _fs * segment_duration
            segment = signal_filter[:, start_idx:end_idx]
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

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 进行超参数调优以选择最佳的参数
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

# Gradient Boosting
clf_gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(estimator=clf_gb, param_grid=param_grid_gb, cv=5, scoring='f1', n_jobs=-1)
grid_search_gb.fit(X_train, [label[0] for label in y_train])

best_clf_gb = grid_search_gb.best_estimator_
print(f'Best parameters for Gradient Boosting: {grid_search_gb.best_params_}')

# 在测试集上评估模型
y_pred_gb = best_clf_gb.predict(X_test)

f1_gb = f1_score([label[0] for label in y_test], y_pred_gb)
print(f'F1 Score on test set for Gradient Boosting: {f1_gb}')
print('Classification report for Gradient Boosting:')
print(classification_report([label[0] for label in y_test], y_pred_gb))

# 使用最佳参数训练最终模型
best_clf_gb.fit(X, [label[0] for label in Y])

# 提取模型和标准化器的参数
model_params = {
    'gb_model_params': best_clf_gb.get_params(),
    'scaler_mean_': scaler.mean_.tolist(),
    'scaler_scale_': scaler.scale_.tolist(),
    'X_train': X.tolist(),
    'y_train': [[label[0], label[1], label[2]] for label in Y]
}

# 将参数保存为JSON格式
with open('gbdt_model.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=4)
