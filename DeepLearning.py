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
        layers.BatchNormalization(input_shape=[40,]),
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

def tf_model_full():
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[40,]),
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
    verify_tuple = (verify_data, verify_labels)
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
    (verify_data, verify_labels) = DataProcess.extractFlagForRate("verify.csv")
    model.evaluate(verify_data, verify_labels, verbose=2)
    '''
    for i in range(10):
        DataProcess.divideTrainAndVerify('comb_rate.csv', 'train.csv', 'verify.csv', 0.75)
        (verify_data, verify_labels) = DataProcess.extractFlagForRate("verify.csv")
        model.evaluate(verify_data, verify_labels, verbose=2)
    '''

# 模型预测接口
def tf_predict(path: str, nparr: np.array):
    model = keras.models.load_model(path)
    return model.predict(np)

def tf_predict_na(path: str, saveTo:str):
    model = keras.models.load_model(path)
    model.summary()
    arr = DataProcess.ripLabels("alt_na_file.csv", ['flag', 'ID'])
    # arr = DataProcess.ripLabels("na_rate.csv", ['flag', 'ID'])
    # arr = model.predict(arr)
    arr = model.predict_classes(arr)
    np.savetxt(saveTo,arr,fmt = '%f',delimiter=',')

if __name__ == "__main__":
    tf_train()
    # tf_model_test('best_model_8vec.h5')
    # tf_predict_na(path = 'best_model_8vec.h5', saveTo = 'na_predict_8vec.csv')
    # tf_predict_na(path = 'best_model_8vec.h5', saveTo = 'na_predict.csv')

'''
模型说明：
best_model.h5           四变量dense net模型
best_model_8vec.h5      八变量dense net模型
'''