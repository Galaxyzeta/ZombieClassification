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
    replaceArr = [0,0,0,0]
    for i in range(len(arr)):
        if type(arr[i]) == str:
            arr[i] = replaceArr[i]
    arr = np.array(arr, dtype=np.float64)
    arr = np.reshape(arr, (1, 4))
    print(arr)
    model = keras.models.load_model(path)
    # model.summary()
    return model.predict_classes(arr).tolist()

def tf_predict_all(path: str, predictFile: str):
    # 全部预测函数，模型：best4vec，四特征值 98.8% 模型
    # @param path 模型路径, xxx.h5
    # @param predictFile 用于预测的csv 其表头应为 [Id, debt_perc, mainincome_perc, owner_perc, tax_perc, flag]，除了Id 和 flag 顺序不能变
    # @return 返回元组，第0维是预测结果汇总，第1维是准确度
    model = keras.models.load_model(path)
    # model.summary()
    arr = extract(predictFile, False)[0]
    pred = pd.DataFrame(model.predict_classes(arr)).rename(columns={0:'flag'})
    validate = pd.read_csv(predictFile)
    pred['ID'] = validate['ID']
    pred['trueflag'] = validate['flag']
    lost = pd.read_csv('AllDataMLP/anal.csv')[['flag', 'ID']]
    lost['trueflag'] = lost['flag']
    lost['flag'] = 0
    npd = pd.concat([lost, pred], axis=0).set_index('ID', drop=True).reset_index()
    npd['correct'] = npd['flag'] == npd['trueflag']
    size = npd.index.size+1
    acc = npd[npd['correct']==True]['ID'].count()/size
    return npd, acc

if __name__ == "__main__":
    # 这里是使用例子
    # 预测，第一维是预测情况，是pandas的dataframe，第二维是预测精度
    pd, acc = tf_predict_all('Models/best4vec.h5', 'AllDataMLP/merge2_dropless.csv')

    # 得到一个ID 与 预测结果 封装的元组
    zip_obj = zip(list(pd['ID']), list(pd['flag']))
    # 遍历，可以得到每一个ID 对应的 预测结果
    for i in zip_obj:
        print(i)
        
    # 单个预测，得到列表 [[1]] 或者 [[0]]
    print(tf_predict_one('Models/best4vec.h5', ['','','','']))