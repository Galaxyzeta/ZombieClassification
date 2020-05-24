# PyContest
用MLP模型进行二分类，并用前端页面展示，是某竞赛的赛题。
## 更新记录
2020-02-26 by @Galaxyzeta And @DDDFaker
1. 首次将项目上传github。
2. 分类准确率已稳定在99%(划掉，现在看来当时的处理应该有问题)。

2020-03-26 by @Galaxyzeta
1. 特征工程第二次重做，FullDataProcess.py 是新的数据处理工程。大幅提高数据处理效率，能更方便地删去不必要的列。本次特征工程是比较成熟的版本了。
2. 更好的管理了文件，删除大量临时产生的csv，主要包括：
   1. 新增SourceData文件夹，存放源数据。
   2. 新增AllDataRegression文件夹，此文件夹下的内容是更新过的特征工程。
   3. 新增TmpModels文件夹，存放训练过程中产生的checkPoint模型。
   4. 新增Models文件夹，存放成型的模型。
3. Interface.py 是供 web 组使用的接口。
4. SclearnTest.py 是 sklearn 中 LinearSVC 和 SGDClassifier 的一些尝试。这两个内置模型在4特征值模型下表现出色。
5. 对代码进行详细注释。
6. 方针：数据处理/特征工程第一，模型第二。

2020-04-20 by @Galaxyzeta
大幅提升模型精度，测试结果大约99.3%。
1. 改用中位数代替平均值填充NA值，准度提高了0.2%，至此数据处理已经较为正确和完善了。
2. 改用 rmsprop + binary_crossentropy 作为优化器与损失函数，这样做大幅加快了收敛速度，模型精度很容易就能达到99.2% 以上。（改：考虑到过拟合，仍选用sgd）
3. 数据训练时，把测试集与训练集混合起来，并进行重新划分。这样做，在数据已经比较准确的前提下，能进一步提高模型精度。
4. 设置了模拟仿真环节，一开始就划分出测试集，在训练时完全不使用（先前的方案是对数据处理完毕再划分测试集，存在一定的漏洞），仅在实际测试时才拿出来进行数据处理。这样做，进一步验证模型的鲁棒性。
5. 新增了绘图功能，每次epoch过后都对训练情况进行评估。
6. 新增一些特征判断的函数，如卡方检验、协方差检验。实际上没什么用。

2020-04-20 by @Galaxyzeta

1. 增加非僵尸企业预测概率输出的接口api。

## 使用教程
1. 选择你的模型。hybrid系列准度是最高的，rmsprop系列其次。

## 声明
直至竞赛结束，该项目仅用于小组内部交流讨论，版权归小组成员共同所有。