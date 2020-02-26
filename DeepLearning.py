import tensorflow as tf
import tensorflow_core.python.keras as keras
import tensorflow_core.python.keras.models as models
import tensorflow_core.python.keras.layers as layers
import tensorflow_core.python.keras.activations as activations
import numpy as np
import DataProcess

# 模型搭建
def tf_model():
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[4,]),
        layers.Dense(4,activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(8,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer=tf.optimizers.SGD(lr=0.001, decay=1e-5, nesterov=True, momentum=0.9),
              loss='binary_crossentropy',
              batch_size=256,
              metrics=['accuracy']
              )
    return model

# 训练模型 采用checkpoint回调函数，自动记录训练过程中的准确率最高模型
def tf_train():
    # 自动保存
    checkpoint = tf.keras.callbacks.ModelCheckpoint("checkPoint.h5",monitor='val_accuracy',verbose=1, save_best_only=True)

    (train_data, train_labels) = DataProcess.extractFlagForRate("train.csv")
    (verify_data, verify_labels) = DataProcess.extractFlagForRate("verify.csv")
    verify_tuple = (train_data, train_labels)
    print(train_labels)
    model = tf_model()
    model.fit(train_data, train_labels, epochs=100, validation_data = verify_tuple, callbacks=[checkpoint])
    model.evaluate(verify_data, verify_labels, verbose=2)
    is_save = input("[INFO] Save model? [y]/n")
    if is_save in ['Y', 'y']:
        print("[INFO] Saving...")
        keras.models.save_model(model,"model3.h5")
        print("[INFO] Save Complete!")

# 运行模型测试，自动随机抽样测试10次
def tf_model_test(path: str):
    model = keras.models.load_model(path)
    model.summary()
    for i in range(10):
        DataProcess.divideTrainAndVerify('comb_rate.csv', 'train.csv', 'verify.csv', 0.75)
        (verify_data, verify_labels) = DataProcess.extractFlagForRate("verify.csv")
        model.evaluate(verify_data, verify_labels, verbose=2)

if __name__ == "__main__":
    # tf_train()
    tf_model_test('checkPoint.h5')