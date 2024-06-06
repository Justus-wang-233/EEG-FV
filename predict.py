# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any

from wbcknn import WBCkNN
from wettbewerb import get_3montages
from wettbewerb import load_references

# Pakete aus dem Vorlesungsbeispiel
import mne
from scipy import signal as sig
import ruptures as rpt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import braycurtis


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
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
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt.
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

    # ------------------------------------------------------------------------------
    # Euer Code ab hier

    training_data_folder = '../shared_data/training_mini'
    # sample2 Path


    # Initialisiere Return (Ergebnisse)
    #global distance, distance
    seizure_present = True  # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5  # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2  # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99  # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0  # gibt die Unsicherheit bezüglich des Endes an (optional)

    with open(model_name, 'r') as f:
        model_params = json.load(f)  # Lade das Modell
        k = model_params['k']
        X_train = np.array(model_params['X_train'])
        y_train = np.array(model_params['y_train'])

    # Initialisiere den WBCkNN-Klassifikator
    model = WBCkNN(k=k)
    model.fit(X_train, y_train)

    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    features = []

    for j, signal_name in enumerate(_montage):
        # Ziehe Montage des EEG
        signal = _montage_data[j]
        # Wende Notch-Filter an um Netzfrequenz zu dämpfen
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
        # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
                                               verbose=False)

        # Berechne die Fourier-Transformation
        fft_values = np.fft.fft(signal_filter)
        fft_magnitude = np.abs(fft_values)
        num_features = 50
        fft_features = fft_magnitude[:num_features]
        features.append(fft_features)

    # Durchschnitt der Features über alle Montagen
    mean_features = np.mean(features, axis=0).reshape(1, -1)

    # Klassifikation des Signals
    prediction = model.predict(mean_features)[0]

    if prediction == 1:
        seizure_present = True
        seizure_confidence = 1.0  # Sie können dies anpassen, um eine tatsächliche Zuversicht auszugeben

        # Bei der Detektion eines Anfalls Onset und Offset berechnen
        E_array = np.mean([sig.stft(signal_filter, fs, nperseg = fs * 3)[2] for signal_filter in _montage_data], axis=0)
        model = rpt.Pelt(model="rbf").fit(E_array.T)
        breakpoints = model.predict(pen=10)
        if len(breakpoints) > 1:
            onset = breakpoints[0] / fs
            onset_confidence = 0.99  # Hier können Sie eine tatsächliche Zuversicht ausgeben
            offset = breakpoints[1] / fs if len(breakpoints) > 1 else None
            offset_confidence = 0.99  # Hier können Sie eine tatsächliche Zuversicht ausgeben



    # ------------------------------------------------------------------------------
    prediction = {"seizure_present": seizure_present, "seizure_confidence": seizure_confidence,
                  "onset": onset, "onset_confidence": onset_confidence, "offset": offset,
                  "offset_confidence": offset_confidence}


    return prediction  # Dictionary mit prediction - Muss unverändert bleiben!




# # Hier könnt ihr euer vortrainiertes Modell laden (Kann auch aus verschiedenen Dateien bestehen)
#     with open(model_name, 'rb') as f:
#         parameters = json.load(f)  # Lade simples Model (1 Parameter)
#         k = parameters['k']
#
#     # Wende Beispielcode aus Vorlesung an
#
#     _montage, _montage_data, _is_missing = get_3montages(channels, data)
#     signal_std = np.zeros(len(_montage))
#     for j, signal_name in enumerate(_montage):
#         # Ziehe erste Montage des EEG
#         signal = _montage_data[j]
#         # Wende Notch-Filter an um Netzfrequenz zu dämpfen
#         signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
#         # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
#         signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
#                                                verbose=False)
#
#         # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
#         # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
#         f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
#         # Berechne Schrittweite der Frequenz
#         df = f[1] - f[0]
#         # Berechne Engergie (Betrag) basierend auf Real- und Imaginärteil der STFT
#         E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
#
#         signal_std[j] = np.std(signal_filter)
#
#         # Erstelle neues Array in der ersten Iteration pro Patient
#         if j == 0:
#             # Initilisiere Array mit Energiesignal des ersten Kanals
#             E_array = np.array(E_Zxx)
#         else:
#             # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
#             E_array = np.vstack((E_array, np.array(E_Zxx)))
#
#     # Berechne Feature zur Seizure Detektion
#     # signal_std_max = signal_std.max()
#     # Klassifiziere Signal
#     # seizure_present = signal_std_max >
#
#     def local_mean_vectors(data: np.ndarray, k: int) -> np.ndarray:
#         """
#         计算每个数据点的局部平均向量
#
#         参数类型:
#         data (ndarray): 数据集，形状为 (n_samples, n_features)
#         k (int): 最近邻的数量
#
#         返回:
#         ndarray: 局部平均向量，形状为 (n_samples, n_features)
#         """
#         # 训练最近邻模型
#         nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
#         distances, indices = nbrs.kneighbors(data)
#
#         # 初始化局部平均向量
#         local_means = np.zeros_like(data)
#
#         # 计算每个点的局部平均向量
#         for i in range(data.shape[0]):
#             # 获取 k 个最近邻（排除自身）
#             neighbors = data[indices[i][1:]]
#             # 计算平均值
#             local_means[i] = neighbors.mean(axis=0)
#
#         # 返回局部平均向量
#         return local_means
#
#
#     def generalized_mean_distance(data, p=2):
#         """
#         计算数据集的广义平均距离
#         参数:
#         data: Bray Curtis_Distance的数据组，形状为 (1_samples, n_features)
#         p (float): 广义平均的参数（默认为 2，即欧氏距离）
#
#         返回:
#         float: 数据集的广义平均距离
#         """
#         n_samples = data.shape[0]
#         if n_samples < 2:
#             raise ValueError("The dataset must contain at least two samples.")
#
#         # 初始化距离累加器
#         total_distance = 0
#         count = 0
#
#         # 计算所有样本对之间的距离
#         for i in range(k):
#             for j in range(i + 1, k):
#                 # 计算 Bray-Curtis 距离
#                 distance = braycurtis(data[i], data[j])
#                 # 计算距离的 p 次方
#                 total_distance += distance ** p
#                 count += 1
#
#         # 计算广义平均距离
#         mean_distance = (total_distance / count) ** (1 / p)
#
#         return mean_distance
#
#         # mean_distance: generalized_mean_distance的数据组，形状为 (1_samples, k_features)
#
#     def weighted_distance(_mean_distances: np.ndarray, _weights: np.ndarray) -> np.ndarray:
#         """
#         计算权重距离，假设所有的权重系数为1
#
#         参数:
#         mean_distances (ndarray): 样本的广义平均距离，形状为 (n_samples,)
#         weights (ndarray): 权重，形状为 (n_samples,)
#
#         返回:
#         ndarray: 权重距离，形状为 (n_samples,)
#         """
#         if _mean_distances.shape != _weights.shape:
#             raise ValueError("mean_distances 和 weights 必须具有相同的形状")
#
#         # 假设所有的权重系数 weights = 1
#
#         _weights = 1
#
#         weighted_distances = _mean_distances * _weights
#
#         return weighted_distances
#
#
#     local_mean_vector = local_mean_vectors(data, k)
#     # 数据类型；
#     # local_means: 数据集，形状为（k_samples，n_features）
#     split_arrays = np.split(local_mean_vector, local_mean_vector.shape[0], axis=0)
#
#
#
#     # 假设split_arrays 和 training_data_folder 已经定义
#     # 假设load_references函数已经定义
#
#     BC_distance = []  # 创建一个列表来存储 Bray-Curtis 距离
#
#     for i in range(k):
#         # 根据local_mean_vector生成sample1的相关数组
#
#         sample1 = split_arrays[i]
#         sample2 = load_references(training_data_folder, 0)
#
#         # 计算 Bray-Curtis 距离
#         distance = braycurtis(sample1, sample2)
#         BC_distance.append(distance)  # 将距离添加到列表中
#
#         # 打印 Bray-Curtis 距离
#         print(f"The Bray-Curtis distance between the samples is: {BC_distance[i]}")
#
#     # 计算广义平均距离 generalized_mean_distance
#     generalized_mean_distances = generalized_mean_distance(BC_distance, p = 2)
#
#     # 计算权重距离D
#     weighted_distance = weighted_distance(generalized_mean_distances, 1)
#
#     # 寻找D的最小值
#     min_weighted_distance = np.min(weighted_distance)
#
#     # 根据最小权重距离值更新seizure_present等变量，这里需要根据具体的分类条件进行判断和更新
#     threshold = 0.5  # 这里需要定义具体的阈值，根据实际情况调整
#     if min_weighted_distance < threshold:
#         seizure_present = True
#         seizure_confidence = 1 - min_weighted_distance  # 假设置信度与距离成反比
#
#         # 使用ruptures库进行变点检测，确定onset和offset
#         model = rpt.Pelt(model="rbf").fit(E_array.T)
#         breakpoints = model.predict(pen=10)
#         if len(breakpoints) > 1:
#             onset = breakpoints[0] / fs
#             onset_confidence = 0.99  # 这里可以根据具体情况调整
#             offset = breakpoints[1] / fs if len(breakpoints) > 1 else None
#             offset_confidence = 0.99  # 这里可以根据具体情况调整
#         else:
#             onset = breakpoints[0] / fs
#             onset_confidence = 0.99
#             offset = None
#             offset_confidence = 0.0
#     else:
#         seizure_present = False
#         seizure_confidence = min_weighted_distance  # 假设置信度与距离成正比





# # -*- coding: utf-8 -*-
# """
# 测试预训练模型的脚本
# Skript testet das vortrainierte Modell
#
#
# @author:  Maurice Rohr, Dirk Schweickard
# """
#
# import numpy as np
# import json
# import os
# from typing import List, Tuple, Dict, Any
# from wettbewerb import get_3montages
# from wbcknn import WBCkNN
#
# # Pakete aus dem Vorlesungsbeispiel
# import mne
# from scipy import signal as sig
# import ruptures as rpt
#
#
# ### Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
# # 函数的签名（参数和返回值的数量）不能更改
#
# def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str,
#                    model_name: str = 'model.json') -> Dict[str, Any]:
#     '''
#     Parameters
#     ----------
#     channels : List[str]
#         Namen der übergebenen Kanäle
#     data : ndarray
#         EEG-Signale der angegebenen Kanäle
#     fs : float
#         Sampling-Frequenz der Signale.
#     reference_system :  str
#         Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
#     model_name : str
#         Name eures Models,das ihr beispielsweise bei Abgabe genannt habt.
#         Kann verwendet werden um korrektes Model aus Ordner zu laden
#     Returns
#     -------
#     prediction : Dict[str,Any]
#         enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
#     '''
#
#     #------------------------------------------------------------------------------
#
#     # 加载模型参数
#     with open(model_name, 'rb') as f:
#         parameters = json.load(f)
#         k = parameters['k']
#
#     # 初始化 WBCkNN 模型
#     model = WBCkNN(k=k)
#
#     # Initialisiere Return (Ergebnisse) 初始化返回结果
#     seizure_present = True  # gibt an ob ein Anfall vorliegt
#     seizure_confidence = 0.5  # gibt die Unsicherheit des Modells an (optional)
#     onset = 4.2  # gibt den Beginn des Anfalls an (in Sekunden)
#     onset_confidence = 0.99  # gibt die Unsicherheit bezüglich des Beginns an (optional)
#     offset = 999999  # gibt das Ende des Anfalls an (optional)
#     offset_confidence = 0  # gibt die Unsicherheit bezüglich des Endes an (optional)
#
#     # 预处理和特征提取
#     _montage, _montage_data, _is_missing = get_3montages(channels, data)
#     signal_std = np.zeros(len(_montage))
#     for j, signal_name in enumerate(_montage):
#         # Ziehe erste Montage des EEG
#         signal = _montage_data[j]
#         # Wende Notch-Filter an um Netzfrequenz zu dämpfen
#         signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
#         # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
#         signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
#                                                verbose=False)
#
#         signal_std[j] = np.std(signal_filter)
#
#     # 提取特征
#     signal_std_max = signal_std.max()
#     feature = np.array(signal_std_max).reshape(-1, 1)  # 调整形状以匹配模型输入
#
#     # 使用模型进行预测
#     seizure_present = model.predict(feature)[0]  # 获取预测结果
#
#     # 计算总能量和变化点
#     _montage, _montage_data, _is_missing = get_3montages(channels, data)
#     for j, signal_name in enumerate(_montage):
#         signal = _montage_data[j]
#         signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
#         signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
#                                                verbose=False)
#         f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
#         df = f[1] - f[0]
#         E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
#
#         if j == 0:
#             E_array = np.array(E_Zxx)
#         else:
#             E_array = np.vstack((E_array, np.array(E_Zxx)))
#
#     E_total = np.sum(E_array, axis=0)
#     max_index = E_total.argmax()
#
#     if max_index == 0:
#         onset = 0.0
#         onset_confidence = 0.2
#     else:
#         algo = rpt.Pelt(model="rbf").fit(E_total)
#         result = algo.predict(pen=10)
#         result1 = np.asarray(result) - 1
#         result_red = result1[result1 < max_index]
#         if len(result_red) < 1:
#             print('No changepoint, taking maximum')
#             onset_index = max_index
#         else:
#             onset_index = result_red[-1]
#         onset = t[onset_index]
#
#     #------------------------------------------------------------------------------
#     prediction = {"seizure_present": seizure_present, "seizure_confidence": seizure_confidence,
#                   "onset": onset, "onset_confidence": onset_confidence, "offset": offset,
#                   "offset_confidence": offset_confidence}
#     print(prediction)
#
#     return prediction  # Dictionary mit prediction - Muss unverändert bleiben!


    # # Euer Code ab hier
    #
    # # Initialisiere Return (Ergebnisse) 初始化返回结果
    # seizure_present = True  # gibt an ob ein Anfall vorliegt
    # seizure_confidence = 0.5  # gibt die Unsicherheit des Modells an (optional)
    # onset = 4.2  # gibt den Beginn des Anfalls an (in Sekunden)
    # onset_confidence = 0.99  # gibt die Unsicherheit bezüglich des Beginns an (optional)
    # offset = 999999  # gibt das Ende des Anfalls an (optional)
    # offset_confidence = 0  # gibt die Unsicherheit bezüglich des Endes an (optional)
    #
    # # Hier könnt ihr euer vortrainiertes Modell laden (Kann auch aus verschiedenen Dateien bestehen)
    # # 加载预训练模型参数
    # with open(model_name, 'rb') as f:
    #     parameters = json.load(f)  # Lade simples Model (1 Parameter)
    #     th_opt = parameters['std_thresh']
    #
    # # Wende Beispielcode aus Vorlesung an
    # # 预处理和特征提取
    # _montage, _montage_data, _is_missing = get_3montages(channels, data)
    # signal_std = np.zeros(len(_montage))
    # for j, signal_name in enumerate(_montage):
    #     # Ziehe erste Montage des EEG
    #     signal = _montage_data[j]
    #     # Wende Notch-Filter an um Netzfrequenz zu dämpfen
    #     signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
    #     # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
    #     signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
    #                                            verbose=False)
    #
    #     # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
    #     # 短时傅立叶变换
    #     # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
    #     f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
    #     # Berechne Schrittweite der Frequenz
    #     df = f[1] - f[0]
    #     # Berechne Energie (Betrag) basierend auf Real- und Imaginärteil der STFT
    #     E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
    #
    #     signal_std[j] = np.std(signal_filter)
    #
    #     # Erstelle neues Array in der ersten Iteration pro Patient
    #     if j == 0:
    #         # Initialisiere Array mit Energiesignal des ersten Kanals
    #         E_array = np.array(E_Zxx)
    #     else:
    #         # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
    #         E_array = np.vstack((E_array, np.array(E_Zxx)))
    #
    # # 检测癫痫发作
    # # Berechne Feature zur Seizure Detektion
    # signal_std_max = signal_std.max()
    # # Klassifiziere Signal
    # seizure_present = signal_std_max > th_opt
    #
    # # 计算总能量和变化点
    # # Berechne Gesamtenergie aller Kanäle für jeden Zeitppunkt
    # E_total = np.sum(E_array, axis=0)
    # # Berechne Stelle der maximalen Energie
    # max_index = E_total.argmax()
    #
    # # Berechne "changepoints" der Gesamtenergie
    # # Falls Maximum am Anfang des Signals ist muss der Onset ebenfalls am Anfang sein und wir können keinen "changepoint" berechnen
    # if max_index == 0:
    #     onset = 0.0
    #     onset_confidence = 0.2
    #
    # else:
    #     # Berechne "changepoint" mit dem ruptures package
    #     # Setup für  "linearly penalized segmentation method" zur Detektion von changepoints im Signal mi rbf cost function
    #     algo = rpt.Pelt(model="rbf").fit(E_total)
    #     # Berechne sortierte Liste der changepoints, pen = penalty value
    #     result = algo.predict(pen=10)
    #     #Indices sind ums 1 geshiftet
    #     result1 = np.asarray(result) - 1
    #     # Selektiere changepoints vor Maximum
    #     result_red = result1[result1 < max_index]
    #     # Falls es mindestens einen changepoint gibt nehmen wir den nächsten zum Maximum
    #     if len(result_red) < 1:
    #         # Falls keine changepoint gefunden wurde raten wir, dass er "nahe" am Maximum ist
    #         print('No changepoint, taking maximum')
    #         onset_index = max_index
    #     else:
    #         # Der changepoint entspricht gerade dem Onset
    #         onset_index = result_red[-1]
    #     # Gebe Onset zurück
    #     onset = t[onset_index]

