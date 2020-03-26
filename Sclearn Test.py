import sklearn.linear_model as lin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn import svm
import FullDataProcess
import numpy as np

'''LOG
bestModel: 					full	fillna
1. lin.SGDClassifier		0.992	0.976-0.982
2. svm.LinerSVC				0.992	0.976
'''
def test():
	# preprocessing
	## Data read
	(train_data, train_labels) = FullDataProcess.extractFlag("AllDataRegression/new.csv")
	(verify_data, verify_labels) = FullDataProcess.extractFlag("AllDataRegression/merge2_dropless.csv")

	## Normalization
	train_data = normalize(train_data)
	verify_data = normalize(verify_data)
	## Data Inspection
	print(train_data.shape)
	print(verify_data.shape)
	# model operation
	model = svm.LinearSVC()
	for i in range(10):
		model.fit(X=train_data, y=train_labels)
		print('Score: %.5f' % model.score(verify_data, verify_labels))
		print('Score: %.5f' % model.score(train_data, train_labels))

if __name__ == "__main__":
	test()