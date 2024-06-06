# -*- coding: utf-8 -*-

import csv
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import json
from wbcknn import WBCkNN  # 导入 WBCkNN 类
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder = r"C:\Users\lyjwa\Desktop\EEG-FV\test"
# training_folder = "../shared_data/training_mini"

# 读取文件夹中的所有文件
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)
# 导入EEG文件、通道名称、采样频率和标签

# 初始化数据数组以保存所有的montage_data和标签
features = []
labels = []

for i, _id in enumerate(ids):  # 遍历每个文件
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]

    # 打印标签信息以确认其结构
    print(f"EEG Label for file {_id}: {_eeg_label}")

    # 计算Montage
    _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)

    for j, signal_name in enumerate(_montage):  # 遍历每个Montage
        # 获取当前Montage信号
        signal = _montage_data[j]
        # 应用Notch滤波器以衰减电源频率干扰
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
        # 应用带通滤波器以过滤掉噪声
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
                                               verbose=False)

        # 计算傅里叶变换
        fft_values = np.fft.fft(signal_filter)
        fft_magnitude = np.abs(fft_values)
        # 只取前N个频率分量作为特征
        num_features = 50
        fft_features = fft_magnitude[:num_features]

        # 计算癫痫发作的时间范围
        seizure_start_time = _eeg_label[1]
        seizure_end_time = _eeg_label[2]

        # 判断频域特征是否包含癫痫发作
        if seizure_start_time < len(signal_filter) / _fs and seizure_end_time > 0:
            label = 1
        else:
            label = 0

        # 将特征存储到features列表中
        features.append(fft_features)
        labels.append(label)

# 转换为NumPy数组
X = np.array(features)
Y = np.array(labels)

# 输出X和Y的形状
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# 检查标签分布
print("Label distribution:", np.bincount(Y))

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# 检查训练集和测试集的标签分布
print("Train label distribution:", np.bincount(y_train))
print("Test label distribution:", np.bincount(y_test))

# 进行交叉验证以选择最佳的k值
best_k = None
best_score = 0
k_values = range(1, 11)  # 可以调整k值的范围

for k in k_values:
    model = WBCkNN(k=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f'k={k}, F1 Score={score}')
    if score > best_score:
        best_score = score
        best_k = k

print(f'Best k value: {best_k} with F1 Score: {best_score}')

# 使用最佳k值训练最终模型
model = WBCkNN(k=best_k)
model.fit(X, Y)  # 使用所有数据训练模型

# 保存最佳模型参数和用于训练的样本
model_params = {
    'k': best_k,
    'X_train': X.tolist(),  # 将numpy数组转换为列表以便JSON序列化
    'y_train': Y.tolist()
}
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=4)



# # -*- coding: utf-8 -*-
#
# import csv
# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from wettbewerb import load_references, get_3montages
# import mne
# from scipy import signal as sig
# import ruptures as rpt
# import json
# from wbcknn import WBCkNN  # 导入 WBCkNN 类
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import f1_score
#
# ### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
#
# training_folder  = r"C:\Users\lyjwa\Desktop\EEG-FV\test"
# # training_folder  = "../shared_data/training_mini"
#
# ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)
# # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
#
# # 初始化数据数组以保存所有的montage_data和标签
# all_data_list = []
#
# segment_length = 250
#
# for i, _id in enumerate(ids):
#     _fs = sampling_frequencies[i]
#     _eeg_signals = data[i]
#     _eeg_label = eeg_labels[i]
#     print(f"EEG Label for file {_id}: {_eeg_label}")
#
#     # Berechne Montage
#     _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
#
#     for j, signal_name in enumerate(_montage):
#         # 获取当前Montage信号
#         signal = _montage_data[j]
#         # 应用Notch滤波器以衰减电源频率干扰
#         signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
#         # 应用带通滤波器以过滤掉噪声
#         signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
#                                                verbose=False)
#         _montage_data[j] = signal_filter
#
#     # 将montage_data从NumPy数组转换为列表
#     montage_data_list = _montage_data.tolist()
#
#     # 保存当前数据文件的montage_data和标签
#     all_data_list.append({
#         'montage_data': montage_data_list,
#         'label': _eeg_label[0]
#     })
#
# # 保存到文件
# with open('all_data.json', 'w', encoding='utf-8') as f:
#     json.dump(all_data_list, f, ensure_ascii=False, indent=4)
#
# exit()
#
#         # # 将信号分段，每段长度为250
#         # segments = [signal_filter[k:k + segment_length] for k in range(0, len(signal_filter), segment_length)]
#         # for segment in segments:
#         #     if len(segment) == segment_length:
#         #         features.append(segment)
#         #         labels.append(_eeg_label[0])  # 每个段的标签都与原信号一致
#         #     elif len(segment) < segment_length:
#         #         padded_segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
#         #         features.append(padded_segment)
#         #         labels.append(_eeg_label[0])
#
# # 转换为NumPy数组
# X = np.array(features)
# Y = np.array(labels)
#
# # 输出X和Y的形状
# print("X shape:", X.shape)
# print("Y shape:", Y.shape)
#
#         #signal_std[j] = np.std(signal_filter)
#
#     # Nur der Kanal mit der maximalen Standardabweichung wird berücksichtigt
#     #signal_std_max = signal_std.max()
#     #features.append(signal_std_max)
#
#
# exit()
#
# # 将数据分成训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # 进行交叉验证以选择最佳的k值
# best_k = None
# best_score = 0
# k_values = range(1, 11)  # 可以调整k值的范围
#
# for k in k_values:
#     model = WBCkNN(k=k)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     score = f1_score(y_test, y_pred)
#     print(f'k={k}, F1 Score={score}')
#     if score > best_score:
#         best_score = score
#         best_k = k
#
# print(f'Best k value: {best_k} with F1 Score: {best_score}')
#
# # 使用最佳k值训练最终模型
# model = WBCkNN(k=best_k)
# model.fit(X, Y)  # 使用所有数据训练模型
#
# # 使用模型进行预测
# predictions = model.predict(X)  # 使用模型对数据进行预测
#
# # 计算最终的 F1 分数
# TP = np.sum((predictions == Y) & (Y == 1))
# FP = np.sum((predictions == 1) & (Y == 0))
# FN = np.sum((predictions == 0) & (Y == 1))
# F1 = 2 * TP / (2 * TP + FP + FN)
# print('F1 Score auf Trainingsdaten:', F1)
#
# # 保存最佳模型参数
# model_params = {'k': best_k}
# with open('model.json', 'w', encoding='utf-8') as f:
#     json.dump(model_params, f, ensure_ascii=False, indent=4)
#     print('Seizure Detektionsmodell wurde gespeichert!')

# Onset Detektion (Der Beispielcode speichert hier kein Modell, da keine Parameter gelernt werden)
# Initialisiere Datenarrays
# onset_list_predict = []
# onset_list = []
# seizure_id_list = []
#
# for i, _id in enumerate(ids):
#     _fs = sampling_frequencies[i]
#     _eeg_signals = data[i]
#     _eeg_label = eeg_labels[i]
#     if _eeg_label[0]:
#         onset_list.append(_eeg_label[1])
#         seizure_id_list.append(_id)
#         # Berechne Montage
#         _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
#         for j, signal_name in enumerate(_montage):
#             # Ziehe erste Montage des EEG
#             signal = _montage_data[j]
#             # Wende Notch-Filter an um Netzfrequenz zu dämpfen
#             signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50., 100.]), n_jobs=2,
#                                                    verbose=False)
#             # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
#             signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
#                                                    verbose=False)
#
#             # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
#             # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
#             f, t, Zxx = sig.stft(signal_filter, _fs, nperseg=_fs * 3)
#             # Berechne Schrittweite der Frequenz
#             df = f[1] - f[0]
#             # Berechne Engergie (Betrag) basierend auf Real- und Imaginärteil der STFT
#             E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
#
#             # Erstelle neues Array in der ersten Iteration pro Patient
#             if j == 0:
#                 # Initilisiere Array mit Energiesignal des ersten Kanals
#                 E_array = np.array(E_Zxx)
#             else:
#                 # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
#                 E_array = np.vstack((E_array, np.array(E_Zxx)))
#
#         # Berechne Gesamtenergie aller Kanäle für jeden Zeitppunkt
#         E_total = np.sum(E_array, axis=0)
#         # Berechne Stelle der maximalen Energie
#         max_index = E_total.argmax()
#
#         # Berechne "changepoints" der Gesamtenergie
#         # Falls Maximum am Anfang des Signals ist muss der Onset ebenfalls am Anfang sein und wir können keinen "changepoint" berechnen
#         if max_index == 0:
#             onset_list_predict.append(0.0)
#         else:
#             # Berechne "changepoint" mit dem ruptures package
#             # Setup für  "linearly penalized segmentation method" zur Detektion von changepoints im Signal mi rbf cost function
#             algo = rpt.Pelt(model="rbf").fit(E_total)
#             # Berechne sortierte Liste der changepoints, pen = penalty value
#             result = algo.predict(pen=10)
#             # Indices sind ums 1 geshiftet
#             result1 = np.asarray(result) - 1
#             # Selektiere changepoints vor Maximum
#             result_red = result1[result1 < max_index]
#             # Falls es mindestens einen changepoint gibt nehmen wir den nächsten zum Maximum
#             if len(result_red) < 1:
#                 # Falls keine changepoint gefunden wurde raten wir, dass er "nahe" am Maximum ist
#                 print('No changepoint, taking maximum')
#                 onset_index = max_index
#             else:
#                 # Der changepoint entspricht gerade dem Onset
#                 onset_index = result_red[-1]
#             # Füge Onset zur Liste der Onsets hinzu
#             onset_list_predict.append(t[onset_index])
#
# # Compute absolute error between computed seizure onset and real onset based on doctor annotations
# prediction_error = np.abs(np.asarray(onset_list_predict) - np.asarray(onset_list))
# print('Mittlerer Onset Prädiktionsfehler Training:', np.mean(prediction_error))
#
# # Plot error per patient
# plt.figure(1)
# plt.scatter(np.arange(1, len(prediction_error) + 1), prediction_error)
# plt.ylabel('Error in s')
# plt.xlabel('Patients')
# plt.savefig('prediction_error_plot.png')
# #plt.show()


# # -*- coding: utf-8 -*-
# """
# Beispiel Code und  Spielwiese
#
# """
#
#
# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from wettbewerb import load_references, get_3montages
# import mne
# from scipy import signal as sig
# import ruptures as rpt
# import json
#
#
# ### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
#
# training_folder  = r"C:\Users\lyjwa\Desktop\wki-sose24\mini_mat_wki"
#
#
# ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)
# # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
#
#
#
# # Seizure Detektion (Der Beispielcode speichert hier ein Modell)
# # Initialisiere Datenarrays
# feature = []
# label = []
#
# for i,_id in enumerate(ids):
#     _fs = sampling_frequencies[i]
#     _eeg_signals = data[i]
#     _eeg_label = eeg_labels[i]
#     label.append(_eeg_label[0])
#     # Berechne Montage
#     _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
#     signal_std = np.zeros(len(_montage))
#     for j, signal_name in enumerate(_montage):
#         # Ziehe erste Montage des EEG
#         signal = _montage_data[j]
#         # Wende Notch-Filter an um Netzfrequenz zu dämpfen
#         signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
#         # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
#         signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
#
#
#         signal_std[j] = np.std(signal_filter)
#
#     # Nur der Kanal mit der maximalen Standardabweichung wird berücksichtigt
#     signal_std_max = signal_std.max()
#     feature.append(signal_std_max)
#
# # 转换为numpy数组
# X = np.array(feature)
# Y = np.array(label)
# best_f1 = 0
# th_opt = 0
# for th in np.arange(X.min(),X.max(),(X.max()-X.min())/1e5):
#     pred = X>th
#     TP = np.sum((pred==Y) & (Y==1))
#     FP = np.sum((pred==1) & (Y==0))
#     FN = np.sum((pred==0) & (Y==1))
#     F1 = 2*TP/(2*TP+FP+FN)
#     if F1 >best_f1:
#         th_opt = th
#         best_f1 = F1
# print('Optimaler Threshold ist', th_opt,' bei F1 auf Trainingsdaten von',best_f1)
#
# # Speichere Modell
# model_params = {'std_thresh':th_opt}
# with open('model.json', 'w', encoding='utf-8') as f:
#     json.dump(model_params, f, ensure_ascii=False, indent=4)
#     print('Seizure Detektionsmodell wurde gespeichert!')
#
#
#
# # Onset Detektion (Der Beispielcode speichert hier kein Modell, da keine Parameter gelernt werden)
# # Initialisiere Datenarrays
# onset_list_predict = []
# onset_list = []
# seizure_id_list = []
#
# for i,_id in enumerate(ids):
#     _fs = sampling_frequencies[i]
#     _eeg_signals = data[i]
#     _eeg_label = eeg_labels[i]
#     if _eeg_label[0]:
#         onset_list.append(_eeg_label[1])
#         seizure_id_list.append(_id)
#         # Berechne Montage
#         _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
#         for j, signal_name in enumerate(_montage):
#             # Ziehe erste Montage des EEG
#             signal = _montage_data[j]
#             # Wende Notch-Filter an um Netzfrequenz zu dämpfen
#             signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
#             # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
#             signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
#
#             # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
#             # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
#             f, t, Zxx = sig.stft(signal_filter, _fs, nperseg=_fs * 3)
#             # Berechne Schrittweite der Frequenz
#             df = f[1] - f[0]
#             # Berechne Engergie (Betrag) basierend auf Real- und Imaginärteil der STFT
#             E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
#
#
#
#             # Erstelle neues Array in der ersten Iteration pro Patient
#             if j == 0:
#                 # Initilisiere Array mit Energiesignal des ersten Kanals
#                 E_array = np.array(E_Zxx)
#             else:
#                 # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
#                 E_array = np.vstack((E_array, np.array(E_Zxx)))
#
#
#         # Berechne Gesamtenergie aller Kanäle für jeden Zeitppunkt
#         E_total = np.sum(E_array, axis=0)
#         # Berechne Stelle der maximalen Energie
#         max_index = E_total.argmax()
#
#         # Berechne "changepoints" der Gesamtenergie
#         # Falls Maximum am Anfang des Signals ist muss der Onset ebenfalls am Anfang sein und wir können keinen "changepoint" berechnen
#         if max_index == 0:
#             onset_list_predict.append(0.0)
#         else:
#             # Berechne "changepoint" mit dem ruptures package
#             # Setup für  "linearly penalized segmentation method" zur Detektion von changepoints im Signal mi rbf cost function
#             algo = rpt.Pelt(model="rbf").fit(E_total)
#             # Berechne sortierte Liste der changepoints, pen = penalty value
#             result = algo.predict(pen=10)
#             #Indices sind ums 1 geshiftet
#             result1 = np.asarray(result) - 1
#             # Selektiere changepoints vor Maximum
#             result_red = result1[result1 < max_index]
#             # Falls es mindestens einen changepoint gibt nehmen wir den nächsten zum Maximum
#             if len(result_red)<1:
#                 # Falls keine changepoint gefunden wurde raten wir, dass er "nahe" am Maximum ist
#                 print('No changepoint, taking maximum')
#                 onset_index = max_index
#             else:
#                 # Der changepoint entspricht gerade dem Onset
#                 onset_index = result_red[-1]
#             # Füge Onset zur Liste der Onsets hinzu
#             onset_list_predict.append(t[onset_index])
#
# # Compute absolute error between compute seizure onset and real onset based on doctor annotations
# prediction_error = np.abs(np.asarray(onset_list_predict) - np.asarray(onset_list))
# print('Mittlerer Onset Prädiktionsfehler Training:', np.mean(prediction_error))
#
# # Plot error per patient
# plt.figure(1)
# plt.scatter(np.arange(1, len(prediction_error)+1),prediction_error)
# #plt.hlines(10, 0, len(prediction_error)+1, colors='red')
# plt.ylabel('Error in s')
# plt.xlabel('Patients')
# plt.show()


