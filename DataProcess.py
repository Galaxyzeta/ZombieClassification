import pandas as pd
import numpy as np

# Merge three csv. f1 = base.csv, f2 = year_report.csv, f3 = money_report.csv
# Then, apply drop na and wash data.
def mergeCsv(f1, f2, f3, out)->None:
    print("Combining 1/2...")
    c1 = pd.read_csv(f1)[["ID", "flag"]]
    c2 = pd.read_csv(f2)
    ndf:pd.DataFrame = pd.merge(c1, c2, on='ID')

    print("Combining 2/2...")
    c3 = pd.read_csv(f3)
    ndf = pd.merge(c3, ndf, on=['ID', 'year'])

    print("Washing Data")
    # HAS PROBLEM !
    # print("---Dropping negative(Optional)...")
    # ndf.drop(ndf[ndf<0].index, inplace=True)
    print("---Processing zq...")
    ndf.drop(ndf[ndf['zqlimit'].isna() & ndf['zqcost']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['zqcost'].isna() & ndf['zqlimit']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['zqcost'] > ndf['zqlimit']].index, inplace=True)
    print("---Processing gq...")
    ndf.drop(ndf[ndf['gqlimit'].isna() & ndf['gqcost']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['gqcost'].isna() & ndf['gqlimit']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['gqcost'] > ndf['gqlimit']].index, inplace=True)
    print("---Processing nbmy...")
    ndf.drop(ndf[ndf['nbmylimit'].isna() & ndf['nbmycost']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['nbmycost'].isna() & ndf['nbmylimit']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['nbmycost'] > ndf['nbmylimit']].index, inplace=True)
    print("---Processing xmzc...")
    ndf.drop(ndf[ndf['xmzclimit'].isna() & ndf['xmzccost']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['xmzccost'].isna() & ndf['xmzclimit']!=0].index, inplace=True)
    ndf.drop(ndf[ndf['xmzccost'] > ndf['xmzclimit']].index, inplace=True)
    print("Washing OK")
    ndf.dropna(subset= ['flag', 'year'] ,inplace=True)
    '''
    f = open("mid.csv", "w")
    ndf.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=ndf.columns, index=False)
    raise Exception
    '''

    print("Filling Na")
    ndf['zqlimit'].fillna(0, inplace=True)
    ndf['gqlimit'].fillna(0, inplace=True)
    ndf['nbmylimit'].fillna(0, inplace=True)
    ndf['xmzclimit'].fillna(0, inplace=True)
    ndf['zqcost'].fillna(0, inplace=True)
    ndf['gqcost'].fillna(0, inplace=True)
    ndf['nbmycost'].fillna(0, inplace=True)
    ndf['xmzccost'].fillna(0, inplace=True)
    for i in ndf.columns:
        ndf[i].fillna(ndf[i].mean(), inplace=True)
    print("Fill Na OK")

    print("Dropping Year...")
    ndf.drop('year', axis=1, inplace=True)
    print("Drop year OK")
    f = open(out, mode="w")
    # ndf.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=['ID', 'debt_perc', 'mainincome_perc', 'tax_perc', 'owner_perc', 'profit_perc', 'flag'], index=False)
    # ndf.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=['ID', 'debt_perc', 'mainincome_perc', 'tax_perc', 'owner_perc', 'nbmy_perc', 'zq_perc', 'gq_perc', 'xmzc_perc', 'flag'], index=False)
    ndf.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=ndf.columns, index=False)

# Group the data by ID and calculate mean value
def combinedTableAnalysis(src, dest):
    # wip
    df = pd.read_csv(src)
    print("Grouping(Slow)...")
    df_group = df.groupby("ID")
    print("Group OK")
    print("Calc Mean()...")
    tgt = df_group.mean()
    print("Calc OK")
    with open(dest, mode="w") as f:
        tgt.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=tgt.columns)
        print("Write OK")
'''
def combinedTableAnalysis2(src, dest):
    # wip
    df = pd.read_csv(src)
    print("Grouping(Slow)...")
    df_group = df.groupby("ID")
    print("Group OK")
    print("Calc Mean()...")
    tgt = df_group.apply(applyFunc)
    print(tgt)
    return
    print(tgt.columns)
    with open(dest, mode="w") as f:
        tgt.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=tgt.columns)
        print("Write OK")

def applyFunc(df):
    cols = ['ID', 'zq_perc', 'gq_perc', 'nbmy_perc', 'xmzc_perc', 'debt_perc', 'mainincome_perc', 'tax_perc', 'owner_perc']
    print('--Processing zq_perc...')
    df['zq_perc'] = (df['zqcost']/df['zqlimit']).mean()
    print('--Processing gq_perc...')
    df['gq_perc'] = (df['gqcost']/df['gqlimit']).mean()
    print('--Processing xmzc_perc...')
    df['xmzc_perc'] = (df['xmzccost']/df['xmzclimit']).mean()
    print('--Processing nbmy_perc...')
    df['nbmy_perc'] = (df['nbmycost']/df['nbmylimit']).mean()
    print('--Processing debt...')
    df['debt_perc'] = (df['debt']/df['allmoney']).mean()
    df['mainincome_perc'] = (df['mainincome']/df['income']).mean()
    df['tax_perc'] = (df['tax']/df['profit']).mean()
    df['owner_perc'] = (df['ownervalue']/df['profit']).mean()
    return df
'''

# Used in deepLearning.py
def extractFlag(file):
    df = pd.read_csv(file)
    df = df.take(np.random.permutation(len(df)))
    train_labels = df.as_matrix(columns=['flag'])
    # df = df.drop(columns=['flag', 'ID', 'year_provided', 'owner_perc'])
    df = df.drop(columns=['flag', 'ID', 'population'])
    train_data = df.as_matrix()
    return (train_data, train_labels)

def extractFlagForRate(file):
    df = pd.read_csv(file)
    df = df.take(np.random.permutation(len(df)))
    train_labels = df.as_matrix(columns=['flag'])
    df = df.drop(columns=['flag', 'ID', 'xmzc_perc', 'gq_perc', 'nbmy_perc', 'zq_perc'])
    # df = df.drop(columns=['flag', 'ID'])
    train_data = df.as_matrix()
    return (train_data, train_labels)

# Merge two similar csv into one.
def mergeTrainAndVerify(t, v, out):
    c1 = pd.read_csv(t)
    c2 = pd.read_csv(v)
    tgt = pd.concat([c1, c2], axis=0)
    f = open(out, mode="w")
    tgt.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=tgt.columns, index=False)
    print('Merge OK')

# Divide a csv into TRAIN and VERFIY by Ratio. Ratio = Train / Verify
def divideTrainAndVerify(f, tout, vout, ratio):
    inp = pd.read_csv(f)
    inp = inp.take(np.random.permutation(len(inp)))
    size = len(inp)
    ft = open(tout, mode="w")
    inp[0: int(size*ratio)].to_csv(path_or_buf=ft, line_terminator='\n', na_rep='NaN', columns=inp.columns, index=False)
    fv = open(vout, mode="w")
    inp[int(size*ratio) :size].to_csv(path_or_buf=fv, line_terminator='\n', na_rep='NaN', columns=inp.columns, index=False)
    print('Divide OK')

def ratioProcess(f, out):
    df = pd.read_csv(f)
    ndf = pd.DataFrame()
    print("Calculating ratio...")
    ndf['ID'] = df['ID']
    ndf['flag'] = df['flag']
    ndf['zq_perc'] = df['zqcost']/df['zqlimit']
    ndf['gq_perc'] = df['gqcost']/df['gqlimit']
    ndf['xmzc_perc'] = df['xmzccost']/df['xmzclimit']
    ndf['nbmy_perc'] = df['nbmycost']/df['nbmylimit']
    ndf['debt_perc'] = df['debt']/df['allmoney']
    ndf['mainincome_perc'] = df['mainincome']/df['income']
    ndf['tax_perc'] = df['tax']/df['profit']
    ndf['owner_perc'] = df['ownervalue']/df['profit']
    print("filling na...")
    ndf.fillna(0, inplace=True)
    print("replacing inf")
    # ndf.replace([np.inf, -np.inf], 0)
    with open(out, mode="w") as f:
        ndf.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=ndf.columns, index=False)
        print("Write OK")

if __name__ == "__main__":
    # 训练集有关的三个csv的合并，第一步先跑这两个
    mergeCsv('base_train_sum.csv', 'year_report_train_sum.csv', 'money_report_train_sum.csv', 'merge1.csv')
    combinedTableAnalysis('merge1.csv', 't1.csv')

    # 验证有关的三个csv的合并，第一步先跑这两个
    mergeCsv('base_verify1.csv', 'year_report_verify1.csv', 'money_information_verify1.csv', 'merge2.csv')
    combinedTableAnalysis('merge2.csv', 't2.csv')

    # 训练集和验证集合并成一个，第二步跑这个
    mergeTrainAndVerify('t1.csv', 't2.csv', 'comb.csv')

    # 计算均值
    ratioProcess("comb.csv", "comb_rate.csv")

    # 根据比例划分训练集和验证集，第三步跑这个
    # divideTrainAndVerify('verify.csv', 'verify1.csv', 'verify2.csv', 0.5)
    # divideTrainAndVerify('comb_rate.csv', 'train_rate.csv', 'verify_rate.csv', 0.8)
    
    # 划分验证集和数据集
    divideTrainAndVerify('comb_rate.csv', 'train.csv', 'verify.csv', 0.5)
    
    # 处理只包含比例的数据
    pass