import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
# Merge three csv. f1 = base.csv, f2 = year_report.csv, f3 = money_report.csv, f4 = knowledge.csv, out = out.csv
# Then, apply drop na and wash data.

# ID,zq_perc,xmzc_perc,gq_perc,nbmy_perc,debt_perc,mainincome_perc,owner_perc,tax_perc,s1_debt_perc,s1_owner_perc,s1_mainincome_perc,s2_debt_perc,s2_owner_perc,s2_mainincome_perc,reg_time,reg_money,stock_rate,flag,invention,logo,copyright,service_type_1,service_type_2,service_type_3,service_type_4,service_type_5,service_type_6,area_1,area_2,area_3,area_4,area_5,area_6,area_7,ent_type_1,ent_type_2,ent_type_3,ent_type_4,ent_type_5,cont_type_1,cont_type_2

# 数据预处理
def data_merge(f1:str, f2:str, f3:str, f4:str, out:str, fillna:bool=False, flagna:bool=False)->None:
    # --- 读取CSV ---
    c1:pd.DataFrame = pd.read_csv(f1)
    c2:pd.DataFrame = pd.read_csv(f2)
    c3:pd.DataFrame = pd.read_csv(f3)
    c4:pd.DataFrame = pd.read_csv(f4)

    # 中文数据替换
    repl_table = {'交通运输业': 'A', '商业服务业': 'B', '工业': 'C', '服务业': 'D', '社区服务': 'E', '零售业': 'F', '山东':'A', '广东':'B', '广西':'C', '江西':'D', '湖北': 'E', '湖南':'F', '福建':'G',
    '农民专业合作社':'A', '合伙企业':'B', '有限责任公司':'C', '股份有限公司':'D', '集体所有制企业':'E', '企业法人': 'A', '自然人': 'B'}
    c1.replace(repl_table, inplace=True)

    ## 如果ID是NA，不能训练，实际上这种情况是不存在的
    c1.dropna(subset=['ID'], inplace=True)
    c2.dropna(subset=['ID'], inplace=True)
    c3.dropna(subset=['ID'], inplace=True)
    c4.dropna(subset=['ID'], inplace=True)
    if fillna == False:
        pass
    else:
        ## 根据企业属性不同，计算平均值，并对年报表格填充NA值
        helper = ['service_type', 'ent_type', 'cont_type', 'area']
        ## Fake string 'NO VALUE' for classification help
        tmp: pd.DataFrame = pd.merge(c1[['ID']+helper].fillna('NO_VALUE'), c2, on=['ID'], how='left')
        ## Label Rename Process
        k = ['population', 'allmoney', 'debt', 'income', 'mainincome','profit', 'rawprofit', 'tax', 'ownervalue']
        tmpk = [x+'_avg' for x in k]
        rename = dict(zip(k, tmpk))
        ## Calculate average reference table, then rename. 
        repl_table = tmp.groupby(by=['service_type', 'ent_type', 'cont_type', 'area']).mean().rename(columns=rename).drop(columns=['year', 'ID'])
        ## 特征缩放
        c2 = pd.merge(tmp, repl_table, on=['service_type', 'ent_type', 'cont_type', 'area'], how='left')
        c2.dropna(subset=['ID'], inplace=True)
        for col in k:
            c2[col].fillna(c2[col+'_avg'] ,inplace=True)
        c2.drop(columns=tmpk+helper, inplace=True)

        for col in ['reg_time', 'reg_money', 'stock_rate']:
            c1[col].fillna(c1[col].mean(), inplace=True)
        c1['flag'].dropna(inplace=True)

        for col in ['invention','logo','copyright']:
            c4[col].fillna(0, inplace=True)
    
    c2['population'] = c2['population'] / c2['population'].max()
    ndf:pd.DataFrame = pd.merge(c2, c3, on=['ID','year'], how='left')
    
    if fillna == False:
        # 若不需要填充NA，采取严格模式，去掉以下情况：
        # 1 上限未知但存在当前值
        # 2 上限已知但不存在当前值
        # 3 当前值超过上限
        ndf.drop(ndf[ndf['zqlimit'].isna() & ndf['zqcost']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['zqcost'].isna() & ndf['zqlimit']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['zqcost'] > ndf['zqlimit']].index, inplace=True)

        ndf.drop(ndf[ndf['gqlimit'].isna() & ndf['gqcost']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['gqcost'].isna() & ndf['gqlimit']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['gqcost'] > ndf['gqlimit']].index, inplace=True)

        ndf.drop(ndf[ndf['nbmylimit'].isna() & ndf['nbmycost']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['nbmycost'].isna() & ndf['nbmylimit']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['nbmycost'] > ndf['nbmylimit']].index, inplace=True)

        ndf.drop(ndf[ndf['xmzclimit'].isna() & ndf['xmzccost']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['xmzccost'].isna() & ndf['xmzclimit']!=0].index, inplace=True)
        ndf.drop(ndf[ndf['xmzccost'] > ndf['xmzclimit']].index, inplace=True)
    else:
        pass

    # Calc property
    ndf.dropna(subset=['year'], inplace=True)

    ndf['zq_perc'] = ndf['zqcost']/ndf['zqlimit']
    ndf['gq_perc'] = ndf['gqcost']/ndf['gqlimit']
    ndf['xmzc_perc'] = ndf['xmzccost']/ndf['xmzclimit']
    ndf['nbmy_perc'] = ndf['nbmycost']/ndf['nbmylimit']

    ndf['debt_perc'] = ndf['debt']/ndf['allmoney']
    ndf['mainincome_perc'] = ndf['mainincome']/ndf['income']
    ndf['tax_perc'] = ndf['tax']/ndf['allmoney']
    ndf['owner_perc'] = ndf['ownervalue']/ndf['allmoney']

    if fillna == False:
        # 这里填充NA为0是安全的，因为fillNa=false时，不合要求的数据已经在前面被正确处理，剩下的NA是由除以0导致的，直接设置为0即可
        ndf['zq_perc'].fillna(0, inplace=True)
        ndf['gq_perc'].fillna(0, inplace=True)
        ndf['xmzc_perc'].fillna(0, inplace=True)
        ndf['nbmy_perc'].fillna(0, inplace=True)
    else:
        # 不然的话填充平均值
        ndf['zq_perc'].fillna(ndf['zq_perc'].mean(), inplace=True)
        ndf['gq_perc'].fillna(ndf['zq_perc'].mean(), inplace=True)
        ndf['xmzc_perc'].fillna(ndf['zq_perc'].mean(), inplace=True)
        ndf['nbmy_perc'].fillna(ndf['zq_perc'].mean(), inplace=True)

    if fillna==False:
        # 如果不填充na，则采取严格模式，给定年份不到3年直接丢弃！
        gp = ndf.groupby(by=['ID'])
        ndf = pd.merge((gp['year'].count()==3).reset_index().rename(columns={'year': 'judge'}), ndf, on=['ID'])
        ndf.drop(ndf[ndf['judge']==False].index, inplace=True)
        ndf.drop(columns=['judge', 'year'], inplace=True)
    
    # 均值计算
    gp = ndf.groupby(by=['ID'])
    mn = gp.mean()
    ndf = mn[['zq_perc', 'xmzc_perc', 'gq_perc', 'nbmy_perc', 'debt_perc', 'mainincome_perc', 'owner_perc', 'tax_perc']]

    # 计算三年数据每两年之间的差异，此操作会导致部分数据丢失，其中原因可能与年份少于3年有关！
    
    ndf = ndf.reset_index()
    ndf.index.rename('ind', inplace=True)
    rename = {'population':'population_change1', 'debt_perc': 's1_debt_perc', 'owner_perc': 's1_owner_perc', 'mainincome_perc': 's1_mainincome_perc', 'tax_perc': 's1_tax_perc'}
    calc = (gp['population','debt_perc', 'owner_perc', 'mainincome_perc', 'tax_perc'].shift(0)-gp['population','debt_perc', 'owner_perc', 'mainincome_perc', 'tax_perc'].shift(1)).dropna().reset_index().drop(columns=['index'])

    calc_s1 = calc[calc.index%2==0].reset_index().drop(columns=['index']).rename(columns=rename)
    calc_s1.index.rename('ind', inplace=True)
    ndf = pd.merge(ndf, calc_s1, on='ind')

    rename = {'population':'population_change2', 'debt_perc': 's2_debt_perc', 'owner_perc': 's2_owner_perc', 'mainincome_perc': 's2_mainincome_perc', 'tax_perc': 's2_tax_perc'}
    calc_s2 = calc[calc.index%2==1].reset_index().drop(columns=['index']).rename(columns=rename)
    calc_s2.index.rename('ind', inplace=True)
    ndf = pd.merge(ndf, calc_s2 ,on='ind')
    
    # 合并其余项
    try:
        c1.drop(columns=['cid'], inplace=True)
    except(Exception):
        pass
    
    ndf = pd.merge(ndf, c1, on=['ID'])
    ndf = pd.merge(ndf, c4, on=['ID'])

    # 是否仅保留 flag = NA 数据，主要用于半监督学习
    if flagna == True:
        ndf = ndf[ndf['flag'].isna()]
    else:
        ndf.dropna(subset=['flag'], inplace=True)

    # 新增one-hot编码列
    ndf = pd.get_dummies(ndf)
    
    if fillna == False:
        ndf['flag'].fillna(2 ,inplace=True)
        ndf.dropna(subset=ndf.columns, inplace=True)

    ################################
    # 以下是处理到最后，完整的表头标签
    ################################
    '''
    ['ID', 'zq_perc', 'xmzc_perc', 'gq_perc', 'nbmy_perc', 'debt_perc',
       'mainincome_perc', 'owner_perc', 'tax_perc', 's1_debt_perc',
       's1_owner_perc', 's1_mainincome_perc', 's1_tax_perc', 's2_debt_perc',
       's2_owner_perc', 's2_mainincome_perc', 's2_tax_perc', 'reg_time',
       'reg_money', 'stock_rate', 'invention', 'logo', 'copyright',
       'service_type_A', 'service_type_B', 'service_type_C', 'service_type_D',
       'service_type_E', 'service_type_F', 'area_A', 'area_B', 'area_C',
       'area_D', 'area_E', 'area_F', 'area_G', 'ent_type_A', 'ent_type_B',
       'ent_type_C', 'ent_type_D', 'ent_type_E', 'cont_type_A', 'cont_type_B',
       'population_change1', 'population_change2']
    '''
    ##############################################
    # 在下面去除你不需要的表头标签，我将提供一些预设
    ##############################################

    # @1 四特征值模型训练专用，特征值为[ debt_perc, mainincome_perc, owner_perc, tax_perc]

    ndf = ndf.drop(columns=['zq_perc', 'xmzc_perc', 'gq_perc', 'nbmy_perc', 's1_debt_perc',
       's1_owner_perc', 's1_mainincome_perc', 's1_tax_perc', 's2_debt_perc',
       's2_owner_perc', 's2_mainincome_perc', 's2_tax_perc', 'population_change1', 'population_change2',
        'reg_time','reg_money', 'stock_rate', 'invention', 'logo', 'copyright',
       'service_type_A', 'service_type_B', 'service_type_C', 'service_type_D',
       'service_type_E', 'service_type_F', 'area_A', 'area_B', 'area_C',
       'area_D', 'area_E', 'area_F', 'area_G', 'ent_type_A', 'ent_type_B',
       'ent_type_C', 'ent_type_D', 'ent_type_E', 'cont_type_A', 'cont_type_B',
       ], errors='ignore')

    # @2 几乎全特征模型，因以下几个数据与flag的相关性不高，因此可以扔掉
    '''
    ndf = ndf.drop(columns=[
       'reg_time','reg_money', 'stock_rate', 'invention', 'logo', 'copyright', 'population_change1', 'population_change2'
       ])
    '''
    # @3 全特征值模型，直接注释ndf.drop即可，但全部拿来训练肯定是存在问题的
    # 空白

    # @4 四特征+年份变化因素 模型
    '''
    ndf = ndf.drop(columns=['zq_perc', 'xmzc_perc', 'gq_perc', 'nbmy_perc', 's1_debt_perc',
       's1_owner_perc', 's1_mainincome_perc', 's1_tax_perc', 's2_debt_perc',
       's2_owner_perc', 's2_mainincome_perc', 's2_tax_perc', 'reg_time',
       'reg_money', 'stock_rate', 'invention', 'logo', 'copyright',
       'service_type_A', 'service_type_B', 'service_type_C', 'service_type_D',
       'service_type_E', 'service_type_F', 'area_A', 'area_B', 'area_C',
       'area_D', 'area_E', 'area_F', 'area_G', 'ent_type_A', 'ent_type_B',
       'ent_type_C', 'ent_type_D', 'ent_type_E', 'cont_type_A', 'cont_type_B',
       ])
    '''
    
    f = open(out, mode="w")
    ndf.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=ndf.columns, index=False)
    
    print('Write OK')

# 检查两个csv的index列的区别，通过两csv的合并，取出标志列，并写入out来检查
# 此方法在数据集处理完毕后必须运行，用于检查预处理数据与原始数据ID之间的差异，因为我们不能在数据处理完毕后，因一些未知原因导致一些ID的缺失，这在比赛中是不被允许的！
# 缺失的数据有如下几种处理方法：
# 1. 全部 flag = 0
# 2. 重新用一种简单的方法进行处理，然后写回，重新预测
# 目前还没做好
def diffReference(f1, f2, out):
    print("Analyzing lost data")
    fp1:pd.DataFrame = pd.read_csv(f1)
    fp2:pd.DataFrame = pd.read_csv(f2)
    fp1.dropna(subset=['flag'], inplace=True)
    print('Before:'+str(fp1.index.size))
    print('After:'+str(fp2.index.size))
    m = pd.merge(fp1[['ID', 'flag']], fp2['ID'], how="left", indicator=True, on=['ID'])
    m[m['_merge']=='left_only'].to_csv('AllDataMLP/anal.csv', index=False)
    print("Diff analysis OK")

# 合并两个表头相同的csv文件
def mergeTrainAndVerify(t, v, out):
    c1 = pd.read_csv(t)
    c2 = pd.read_csv(v)
    tgt = pd.concat([c1, c2], axis=0)
    f = open(out, mode="w")
    tgt.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=tgt.columns, index=False)
    print('Merge OK')

# 按比例划分一个csv文件，主要用于验证集与训练集划分、测试集和和验证集划分
def divideTrainAndVerify(f, tout, vout, ratio):
    inp = pd.read_csv(f)
    inp = inp.take(np.random.permutation(len(inp)))
    size = len(inp)
    ft = open(tout, mode="w")
    inp[0: int(size*ratio)].to_csv(path_or_buf=ft, line_terminator='\n', na_rep='NaN', columns=inp.columns, index=False)
    fv = open(vout, mode="w")
    inp[int(size*ratio) :size].to_csv(path_or_buf=fv, line_terminator='\n', na_rep='NaN', columns=inp.columns, index=False)
    print('Divide OK')

# 从 frm 拿出 count 数量的 flag = 0 记录，与 to 合并后，从 out 输出，操作会使 frm 减少取出的记录
def takeSomeFlag0(frm, to, out, count):
    df_to:pd.DataFrame = pd.read_csv(to)
    df_frm:pd.DataFrame = pd.read_csv(frm)
    df_frm = df_frm.take(np.random.permutation(len(df_frm)))
    ndf = df_frm[df_frm['flag']==0].iloc[0: count]
    print(df_frm.index.size)
    df_frm.drop(df_frm[df_frm['flag']==0].iloc[0: count].index, inplace=True)
    print(df_frm.index.size)
    tgt = pd.concat([df_to, ndf], axis=0, sort=False)
    f = open(out, mode="w")
    tgt.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=tgt.columns, index=False)
    f = open(frm, mode="w")
    df_frm.to_csv(path_or_buf=f, line_terminator='\n', na_rep='NaN', columns=df_frm.columns, index=False)
    print('Take OK')

# 此函数将在训练开始前被用到，用于提取数据及其标签
def extractFlag(file, shuffle=True):
    df = pd.read_csv(file)
    if shuffle == True:
        df = df.take(np.random.permutation(len(df)))
    train_labels = df.as_matrix(columns=['flag'])
    # df = df.drop(columns=['flag', 'ID', 'year_provided', 'owner_perc'])
    df = df.drop(columns=['flag', 'ID'])
    train_data = df.as_matrix()
    return (normalize(train_data), train_labels)

# 半监督学习用，将 nafile 的预测结果 flagfile 替换进 nafile，并输出 out 文件
def flagReplace(nafile:str, flagfile:str, out:str):
    tgtfp = pd.read_csv(nafile)
    flagfp = pd.read_csv(flagfile)
    tgtfp['flag'] = flagfp['flag']
    tgtfp.to_csv(out, index=False)

# 不要使用，因为我忘了当初为什么要写这个东西
def getNaFileCopyForAnotherModel(nafile:str, out:str):
    tgtfp = pd.read_csv(nafile)
    tgtfp[['ID', 'flag', 'zq_perc', 'gq_perc', 'xmzc_perc', 'nbmy_perc', 'debt_perc','mainincome_perc','tax_perc','owner_perc']].to_csv(out, index=False)

# 运行一次 main 中的前 7 条，可以进行一次完整的数据预处理。这几个函数全部运行完毕大概需要5秒的时间。
# 文件说明：
# merge1：由训练集处理，均为3年数据处理得到，处理过程中一旦有na就扔掉。
# merge2：由验证集处理，均为3年数据处理得到，处理过程中一旦有na就扔掉。
# merge1_onlyna：由训练集处理，只留下na，半监督学习用。
# merge2_onlyna：由验证集处理，只留下na，半监督学习用。
# merge2_dropless：由验证集处理，填补必要的na，全监督学习用。

if __name__ == "__main__":
    data_merge('SourceData/base_train_sum.csv', 'SourceData/year_report_train_sum.csv', 'SourceData/money_report_train_sum.csv', 'SourceData/knowledge_train_sum.csv', 'AllDataMLP/merge1.csv')
    data_merge('SourceData/base_train_sum.csv', 'SourceData/year_report_train_sum.csv', 'SourceData/money_report_train_sum.csv', 'SourceData/knowledge_train_sum.csv', 'AllDataMLP/merge1_onlyna.csv', flagna=True)
    data_merge('SourceData/base_verify1.csv', 'SourceData/year_report_verify1.csv', 'SourceData/money_information_verify1.csv', 'SourceData/paient_information_verify1.csv', 'AllDataMLP/merge2.csv')
    data_merge('SourceData/base_verify1.csv', 'SourceData/year_report_verify1.csv', 'SourceData/money_information_verify1.csv', 'SourceData/paient_information_verify1.csv', 'AllDataMLP/merge2_onlyna.csv', flagna=True)
    data_merge('SourceData/base_verify1.csv', 'SourceData/year_report_verify1.csv', 'SourceData/money_information_verify1.csv', 'SourceData/paient_information_verify1.csv', 'AllDataMLP/merge2_dropless.csv', fillna=True, flagna=False)
    takeSomeFlag0('AllDataMLP/merge2.csv', 'AllDataMLP/merge1.csv', 'AllDataMLP/new.csv', 3300)
    divideTrainAndVerify('AllDataMLP/merge2_dropless.csv', 'AllDataMLP/merge2_dropless_verify.csv', 'AllDataMLP/merge2_dropless_test.csv', 0.5)
    # WIP
    diffReference('SourceData/base_verify1.csv', 'AllDataMLP/merge2_dropless.csv', 'AllDataMLP/anal.csv')
    # mergeTrainAndVerify('AllDataMLP/merge1.csv', 'AllDataMLP/merge2.csv', 'AllDataMLP/all.csv')
    # flagReplace('AllDataMLP/all_onlyna.csv', 'AllDataMLP/anal.csv', 'AllDataMLP/na_flag_replaced.csv')
    pass
