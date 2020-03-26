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

@TodoList
1. 检查数据处理导致源数据缺失的问题，并设计缺失数据的验证方案。
2. 深入尝试半监督学习。

## 声明
直至竞赛结束，该项目仅用于小组内部交流讨论，版权归小组成员共同所有。

## 使用教程
1. 打开`FullDataProcess.py`，找到大约174行处，根据需要去掉不需要的特征标签。然后运行我写好的main方法的前7行代码，完成一次数据生成。
2. 打开`NewDeepLearning.py`，运行 tf_train 方法，开始一次训练，其中每次训练最好的模型将会被记录在 Models 文件夹内。
3. 运行tf_model_test方法，可以查看模型在某个csv上的准确度。
4. 如何帮我做特征工程：
   1. 改动特征标签，改动MLP模型，调节超参数。
   2. 运行tf_train开始炼丹。