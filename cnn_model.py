# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten
# from tensorflow.keras.utils import to_categorical
#
# # 假设你有训练数据 X_train 和 y_train
# X_train = np.random.rand(100, 3, 256, 256)  # 示例数据，100 个样本，每个样本为 3 通道 256x256 图像
# y_train = np.random.randint(2, size=100)  # 示例标签，二分类问题
#
# # 将标签转换为分类格式
# y_train = to_categorical(y_train, num_classes=2)
#
# # 构建简单的 CNN 模型
# model = Sequential([
#     Conv2D(32, kernel_size=3, activation='relu', input_shape=(3, 256, 256)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(2, activation='softmax')
# ])
#
# # 编译模型
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 训练模型
# model.fit(X_train, y_train, epochs=5)
#
# # 保存模型
# model.save('cnn_model.h5')