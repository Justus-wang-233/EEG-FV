import numpy as np
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
        return np.array(predictions)

    def _predict(self, x):
        # 计算所有训练样本与测试样本之间的Bray-Curtis距离
        distances = [self._bray_curtis_distance(x, x_train) for x_train in self.X_train]

        # 找到距离最近的k个训练样本的索引
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 计算贝叶斯后验概率
        class_probs = self._calculate_class_probabilities(k_nearest_labels)

        # 返回具有最高概率的类别
        return max(class_probs, key=class_probs.get)

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
