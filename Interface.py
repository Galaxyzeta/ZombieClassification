import pandas as pd
import tensorflow_core.python.keras as keras
import numpy as np
import matplotlib.pyplot as plt

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

# 选出预测概率大于threshold（默认0.1）的非僵尸企业的预测概率（均为小于0.5，可以 2*(0.5-k) 算出它变僵尸的概率）
# 格式：索引 预测概率 企业ID
def tf_predict_diff(path: str, predictFile: str, threshold: float=0.1):
    model = keras.models.load_model(path)
    # model.summary()
    csv = pd.read_csv(predictFile)[['ID', 'flag']]
    extracted = extract(predictFile, False)
    pred_res = model.predict_classes(extracted)
    pred_rate = model.predict(extracted)
    csv['pred_rate'] = pred_rate
    csv['pred_flag'] = pred_res
    ret = csv[(csv['pred_rate']>threshold)&(csv['pred_flag']==0)][['pred_rate', 'ID']]
    print(ret)
    return ret

# 根据预测结果得到每个僵尸企业所属的省份，实际上，如果需要，可以得到其他的数据，我会给出一些修改建议
# @param path: 基础数据来源
# @param pred_res: 预测结果
# @return: 返回僵尸企业对应的地区表格，以及每个地区僵尸企业的数量dict
def getProvince(pred_res:pd.DataFrame, path:str):
    ndf = pd.read_csv(path)
    ndf.drop(ndf[ndf['flag'] == 0].index, inplace=True)
    ndf = pd.merge(ndf, pred_res, on=['ID'])
    counter = dict(ndf.groupby('area').count()['ID'])
    # 这里可根据需要，加入service_type,area,ent_type,cont_type,cid,stock_rate等
    return ndf[['ID', 'area']], counter

# 将僵尸企业的特征值划分为若干档次
# 这里的代码有点不灵活，考虑到灵活性的问题，我建议根据需求自行修改，我会给出一些修改指导
# 建议tax_perc区间定少一点，因为大部分都是0
# @param pred_res: 预测得到的Dataframe
# @param pred_source: 预测的数据是从哪个csv预测出来的？
# @param division: 划分区间
# @param labels: 每个区间对应什么名字？
# @return：返回各企业划分结果的Dataframe（一个id对应多个特征值的划分结果 表格），以及每个特征值对应的企业数量dict{特征值A:{档次1：数量; 档次2：数量}， 特征值B：xxxxx，... ...}
def getAttributeMap(pred_res:pd.DataFrame, pred_source:str, division:list=[-np.inf,0,0.25,0.5,0.75,1,2.5,5,np.inf], labels:list=["LV1", "LV2", "LV3", "LV4", "LV5", "LV6", "LV7", "LV8"]):
    ndf = pd.read_csv(pred_source)
    ndf.drop(ndf[ndf['flag'] == 0].index, inplace=True)
    ndf.drop(columns=['flag'], inplace=True)
    ndf = pd.merge(pred_res, ndf, on=['ID'])[['ID', 'debt_perc', 'mainincome_perc', 'owner_perc', 'tax_perc', 'flag']]
    # 以下代码用于将值进行划分。可以不使用统一的division区间，即把下面的division硬编码成所想要的划分区间。labels是每一个档次对应的标签名，可根据需要硬编码。
    tmp1 = pd.cut(ndf['debt_perc'], division, labels=labels)
    tmp2 = pd.cut(ndf['mainincome_perc'], division, labels=labels)
    tmp3= pd.cut(ndf['owner_perc'], division, labels=labels)
    tmp4 = pd.cut(ndf['tax_perc'], division, labels=labels)
    res = ndf[['ID']]
    res = res.join(tmp1).join(tmp2).join(tmp3).join(tmp4)
    # 以下代码把输出结果存为csv，用于debug，正式环境下请移除。
    res.to_csv("Other/log.csv")
    # 以下代码统计结果
    counter = dict()
    counter[tmp1.name] = dict(tmp1.value_counts())
    counter[tmp2.name] = dict(tmp2.value_counts())
    counter[tmp3.name] = dict(tmp3.value_counts())
    counter[tmp4.name] = dict(tmp4.value_counts())
    return res, counter

if __name__ == "__main__":
    '''
    # 这里是使用例子
    # 预测，第一维是预测情况，是pandas的dataframe，第二维是预测精度
    ndf, acc = tf_predict_all('Models/best4vec.h5', 'AllDataMLP/merge2_dropless.csv')
    print(ndf, acc)

    # 得到一个ID 与 预测结果 封装的元组
    zip_obj = zip(list(pd['ID']), list(pd['flag']))
    # 遍历，可以得到每一个ID 对应的 预测结果
    for i in zip_obj:
        print(i)

    # 单个预测，得到列表 [[1]] 或者 [[0]]
    # print(tf_predict_one('Models/best4vec.h5', ['','','','']))
    
    # 预测的僵尸企业对应哪个省份 使用案例
    province, area = getProvince(pred_res=ndf, path='SourceData/base_verify1.csv')
    print(province)
    print(area)
    
    # 预测的僵尸企业的特征值档次划分表格与数量统计
    res, counter = getAttributeMap(pred_res=ndf, pred_source='AllDataMLP/merge2_dropless.csv')
    print(res)
    print(counter)
    '''
    tf_predict_diff('Models/best4vec-sgd.h5', 'AllDataMLP/merge2_dropless_test.csv', 0.10)
