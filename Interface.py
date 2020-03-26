import pandas as pd
import tensorflow_core.python.keras as keras
import numpy as np

def extract(file:str, shuffle:bool=True):
    # 从csv中去除不需要的列
    # @param file csv文件
    # @param shuffle 是否对取得的数据重洗 
    df = pd.read_csv(file)
    if shuffle == True:
        df = df.take(np.random.permutation(len(df)))
    train_labels = df.as_matrix(columns=['flag'])
    df = df.drop(columns=['flag', 'ID'])
    train_data = df.as_matrix()
    return (train_data, train_labels)

def tf_predict_one(path: str, arr:np.ndarray):
    # 单个预测函数，模型：best4vec，四特征值 98.8% 模型
    # @param path 模型路径, xxx.h5
    # @param arr 输入数据 格式 [debt_perc, mainincome_perc, owner_perc, tax_perc]
    # 计算方式：
    # debt_perc = debt / allmoney 计算完每个比例后平均
    # mainincome_perc = mainincome / income 计算完每个比例后平均
    # tax_perc = tax / allmoney 计算完每个比例后平均
    # owner_perc = ownervalue / allmoney 计算完每个比例后平均
    model = keras.models.load_model(path)
    model.summary()
    return model.predict_classes(arr)

def tf_predict_all(path: str, predictFile: str):
    # 全部预测函数，模型：best4vec，四特征值 98.8% 模型
    # @param path 模型路径, xxx.h5
    # @param predictFile 用于预测的csv 其表头应为 [Id, debt_perc, mainincome_perc, owner_perc, tax_perc, flag]，除了Id 和 flag 顺序不能变
    # 结果：1 = 僵尸企业/ 0 = 正常
    model = keras.models.load_model(path)
    model.summary()
    arr = extract(predictFile, False)[0]
    return model.predict_classes(arr)