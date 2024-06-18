# -*- coding: utf-8 -*-
"""
测试预训练模型的脚本
Skript testet das vortrainierte Modell

作者:  Maurice Rohr, Dirk Schweickard
"""

import numpy as np
import json
from typing import List, Dict, Any
from wettbewerb import get_3montages
from tensorflow.keras.models import load_model
from scipy import signal as sig
import ruptures as rpt
import mne


# 函数的签名（参数和返回值的数量）不能更改
def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str,
                   model_name: str = 'cnn_model.h5') -> Dict[str, Any]:
    '''
    参数
    ----------
    channels : List[str]
        提供的通道名称
    data : ndarray
        指定通道的EEG信号
    fs : float
        信号的采样频率
    reference_system :  str
        使用的参考系统，“参考电极”，不保证正确！
    model_name : str
        你们提交时命名的模型名称。
        可用于从文件夹中加载正确的模型

    返回
    -------
    prediction : Dict[str,Any]
        包含是否存在癫痫发作以及如果有，发作的开始和结束时间
    '''

    # 初始化返回结果
    seizure_present = False  # 指示是否存在癫痫发作
    seizure_confidence = 0.0  # 模型不确定性（可选）
    onset = 0.0  # 发作开始时间（秒）
    onset_confidence = 0.0  # 发作开始时间的不确定性（可选）
    offset = 999999  # 发作结束时间（可选）
    offset_confidence = 0.0  # 发作结束时间的不确定性（可选）

    # 加载预训练的CNN模型
    model = load_model(model_name)

    # 预处理和特征提取
    _montage, _montage_data, _is_missing = get_3montages(channels, data)

    # 处理所有的蒙太奇数据
    processed_data = []
    for j, signal_name in enumerate(_montage):
        signal = _montage_data[j]
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
                                               verbose=False)
        processed_data.append(signal_filter)

    # 转换为numpy数组并增加维度以匹配CNN输入格式
    processed_data = np.array(processed_data)
    processed_data = np.expand_dims(processed_data, axis=0)

    # 使用CNN模型进行预测
    prediction = model.predict(processed_data)
    seizure_present = bool(prediction[0][0] > 0.5)
    seizure_confidence = float(prediction[0][0])

    # 如果检测到癫痫发作，计算发作的开始时间
    if seizure_present:
        # 短时傅里叶变换
        f, t, Zxx = sig.stft(processed_data[0][0], fs, nperseg=fs * 3)
        df = f[1] - f[0]
        E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df

        # 计算总能量
        E_total = np.sum(E_Zxx, axis=0)
        max_index = E_total.argmax()

        if max_index == 0:
            onset = 0.0
            onset_confidence = 0.2
        else:
            algo = rpt.Pelt(model="rbf").fit(E_total)
            result = algo.predict(pen=10)
            result1 = np.asarray(result) - 1
            result_red = result1[result1 < max_index]

            if len(result_red) < 1:
                onset_index = max_index
            else:
                onset_index = result_red[-1]
            onset = t[onset_index]

    # 返回预测结果
    prediction = {"seizure_present": seizure_present, "seizure_confidence": seizure_confidence,
                  "onset": onset, "onset_confidence": onset_confidence, "offset": offset,
                  "offset_confidence": offset_confidence}

    return prediction  # 返回包含预测结果的字典 - 必须保持不变！
