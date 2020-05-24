import tensorflow as tf
import tensorflow_core.python.keras as keras
import tensorflow_core.python.keras.models as models
import tensorflow_core.python.keras.layers as layers
import tensorflow_core.python.keras.activations as activations
import numpy as np
import FullDataProcess
import pandas as pd
import numpy as np
import Logger

def getModel():
    model = keras.Sequential([
        layers.Input(shape=[4,]),
        layers.BatchNormalization(),
        layers.Dense(16,activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    # lr=0.001, decay=1e-5, nesterov=True, momentum=0.9
    model.compile(optimizer=tf.optimizers.SGD(lr=0.001, decay=1e-5, nesterov=True, momentum=0.9),
              loss='binary_crossentropy',
              batch_size=32,
              metrics=['accuracy']
              )
    '''
    model.compile(optimizer="rmsprop",
              loss='binary_crossentropy',
              metrics=['accuracy'],
              )
    '''
    return model

# 训练模型 采用checkpoint回调函数，自动记录训练过程中的准确率最高模型
def tf_train():
    # 自动保存
    logger = Logger.LossHistory()

    checkpoint = tf.keras.callbacks.ModelCheckpoint("TmpModels/tmp_model.h5",monitor='val_accuracy',verbose=1, save_best_only=True)

    (train_data, train_labels) = FullDataProcess.extractFlag("AllDataMLP/new.csv")
    (verify_data, verify_labels) = FullDataProcess.extractFlag("AllDataMLP/merge2_dropless_verify.csv")
    verify_tuple = (verify_data, verify_labels)
    model = getModel()
    model.fit(train_data, train_labels, epochs=500, validation_data = verify_tuple, callbacks=[checkpoint, logger], shuffle=True)
    model.evaluate(verify_data, verify_labels, verbose=1)
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

def tf_predict(path: str, testFile:str, saveTo:str, dropflag=True):
    model = keras.models.load_model(path)
    model.summary()
    df = pd.read_csv(testFile)
    tmp = df.drop(columns=['ID'])
    if dropflag == True:
        tmp = tmp.drop(columns=['flag'])
    df['result'] = model.predict_classes(tmp)
    if dropflag == True:
        df[['ID', 'result', 'flag']].to_csv(saveTo, index=False)
    else:
        df[['ID', 'result']].to_csv(saveTo, index=False)

if __name__ == "__main__":
    # print("Hello World!")
    # np.savetxt(saveTo,arr,fmt = '%f',delimiter=',')
    # tf_train()

    # tf_model_test('Models/best4vec-hybrid-ultra2.h5', "AllDataMLP/merge2_dropless_test.csv")
    # tf_predict(path = 'Models/best4vec-sgd.h5', saveTo = 'AllDataMLP/anal.csv', testFile="AllDataMLP/merge2_dropless_test.csv", dropflag=True)
    tf_predict(path = 'Models/best4vec-best.h5', saveTo = 'AllDataMLP/anal.csv', testFile="AllDataMLP/merge2_dropless_test.csv", dropflag=False)
    # print(tf_predict_all('best4vec.h5', 'AllDataMLP/merge2_dropless.csv'))
    pass