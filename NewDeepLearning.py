import tensorflow as tf
import tensorflow_core.python.keras as keras
import tensorflow_core.python.keras.models as models
import tensorflow_core.python.keras.layers as layers
import tensorflow_core.python.keras.activations as activations
import numpy as np
import FullDataProcess
import pandas as pd
import numpy as np

def getModel():
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[4,]),
        layers.Dense(4,activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(8,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    # lr=0.001, decay=1e-5, nesterov=True, momentum=0.9
    model.compile(optimizer=tf.optimizers.SGD(lr=0.001, decay=2e-5, nesterov=True, momentum=0.9),
              loss='binary_crossentropy',
              batch_size=32,
              metrics=['accuracy']
              )
    return model

# 训练模型 采用checkpoint回调函数，自动记录训练过程中的准确率最高模型
def tf_train():
    # 自动保存
    checkpoint = tf.keras.callbacks.ModelCheckpoint("checkPoint.h5",monitor='val_accuracy',verbose=1, save_best_only=True)

    (train_data, train_labels) = FullDataProcess.extractFlag("AllDataMLP/new.csv")
    (verify_data, verify_labels) = FullDataProcess.extractFlag("AllDataMLP/merge2_dropless_verify.csv")
    verify_tuple = (verify_data, verify_labels)
    model = getModel()
    history = model.fit(train_data, train_labels, epochs=500, validation_data = verify_tuple, callbacks=[checkpoint])
    print(history)
    model.evaluate(verify_data, verify_labels, verbose=2)
    is_save = input("[INFO] Save model? [y]/n")
    if is_save in ['Y', 'y']:
        print("[INFO] Saving...")
        keras.models.save_model(model,"TmpModels/tmp_model.h5")
        print("[INFO] Save Complete!")

# 运行模型测试
def tf_model_test(path: str, testfile: str):
    model = keras.models.load_model(path)
    model.summary()
    (verify_data, verify_labels) = FullDataProcess.extractFlag(testfile)
    model.evaluate(verify_data, verify_labels, verbose=2)
    '''
    for i in range(10):
        DataProcess.divideTrainAndVerify('comb_rate.csv', 'train.csv', 'verify.csv', 0.75)
        (verify_data, verify_labels) = DataProcess.extractFlagForRate("verify.csv")
        model.evaluate(verify_data, verify_labels, verbose=2)
    '''

if __name__ == "__main__":
    # np.savetxt(saveTo,arr,fmt = '%f',delimiter=',')
    # tf_train()
    tf_model_test('Models/checkPoint.h5', "AllDataMLP/merge2_dropless.csv")
    # tf_predict(path = 'AllDataMLP/checkPoint.h5', saveTo = 'AllDataMLP/anal.csv', testFile="AllDataMLP/all_onlyna.csv")
    # print(tf_predict_all('best4vec.h5', 'AllDataMLP/merge2_dropless.csv'))