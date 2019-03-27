from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np

import os

def load_data():
	train_features = pd.read_csv('./data/trainFeatures.csv')
	test_features = pd.read_csv('./data/testFeatures.csv')
	train_labels = pd.read_csv('./data/trainLabels.csv', header=None)

	return train_features, test_features, train_labels
"""
def map_age(x):
	if x>0 and x<=19:
		return 1
	elif x>19 and x<=30:
		return 2
	elif x>30 and x<=65:
		return 3
	else:
		return 4

def mapper_fnlwgt(x):
	if x< -61818.5:
		return -1
	elif x>-61818.5 and x<=117847:
		return 1
	elif x>117847 and x<=178449:
		return 2
	elif x>178449 and x<=237624:
		return 3
	elif x>237624 and x<=417289:
		return 4
	else:
		return 5

"""
def hash_variable(features_df, target_cols):
	"""
	@features_df: the complete features and its format is pd.DataFrame
	@target_cols: the cols to hash
	"""
	#Here we are going to transform age, fnwgt to categorical
	#features_df['age'] = features_df['age'].apply(lambda x:map_age(x))
	#features_df['fnlwgt'] = features_df['fnlwgt'].apply(lambda x:mapper_fnlwgt(x))
	
	# hash each col
	for each_col in target_cols:
		hash_dict = {}
		unique_element = np.unique(features_df[each_col])
		hash_val = 1
		# generate hash val
		for each_element in unique_element:
			hash_dict[each_element] = hash_val
			hash_val += 1
		features_df[each_col] = features_df[each_col].apply(lambda x:hash_dict[x])

	features_df['diff'] = features_df['capital-gain']-features_df['capital-loss']
		
	return features_df


def model(trainX, trainY, testX, testY):
	trainX = np.asarray(trainX)
	trainY = trainY.values
	trainY = np.reshape(trainY, (1,-1))[0]
	
	clf = RandomForestClassifier(n_estimators=160,
								n_jobs=-1,
								oob_score=True,
								max_depth=None,
								min_samples_split=8,
								min_samples_leaf=4,
								max_features='auto',
								max_leaf_nodes=None,
								bootstrap=True,
								warm_start=True,
								random_state=1,
								)

	kf = KFold(n_splits=5,shuffle=True)
	round = 1; total_acc=0.0

	#scores = cross_val_score(clf, trainX, trainY, cv = 5, scoring = 'accuracy')
	
	for train_index, test_index in kf.split(trainX):
		X_train, X_test = trainX[train_index], trainX[test_index]
		y_train, y_test = trainY[train_index], trainY[test_index]
		
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		train_y_pred = clf.predict(X_train)

		test_acc = float('%.4f'%accuracy_score(y_true=y_test, y_pred=y_pred))
		train_acc = '%.4f'%accuracy_score(y_true=y_train, y_pred=train_y_pred)
		
		print("K-fold:{}, train samples={}, test samples={}, train accuracy score={}\
			test accuracy score is={}".format(round,X_train.shape[0],X_test.shape[0],train_acc, test_acc))	
		
		total_acc += test_acc
		round += 1
	
	print('mean test accuracy on 5 K-folds={}'.format(total_acc/5.0))
	
	#print(scores)
	# test on whole test set
	
	whole_pred = clf.predict(testX)
	whole_acc = '%.4f'%accuracy_score(y_true=testY, y_pred=whole_pred)
	print('accuracy on complete test data is {}'.format(whole_acc))

	#save model
	joblib.dump(clf,'./RandomForestClassifier.pkl')
	print("model has been saved successfully!")

	
def predictor(X):
	if not os.path.exists('./RandomForestClassifier.pkl'):
		print("doesn't find model, now retraining....")
		train_features, test_features, train_labels = load_data()
		rest_cols = train_features.columns.tolist()
		rest_cols.remove('age')
		rest_cols.remove('fnlwgt')
		rest_cols.remove('education-num')
		rest_cols.remove('hours-per-week')
		train_features = hash_variable(train_features, rest_cols)
		X_train, X_test, y_train, y_test = train_test_split(
			train_features, train_labels, test_size=0.2, random_state=50)
	
		model(trainX=X_train, trainY=y_train,testX=X_test, testY=y_test)
	
	remodel = joblib.load('./RandomForestClassifier.pkl')
	pred_result = remodel.predict(X)
	print('prediction result is',pred_result)

	#save result
	if os.path.exists('prediction-result.csv'):
		print("find prediction-result aleady exists, removing...")
		os.remove('prediction-result.csv')
	with open('prediction-result.csv','a') as f:
		for item in pred_result:
			f.write(str(item))
			f.write('\n')

	print('prediction result has been saved successfully')



if __name__ == '__main__':
	train_features, test_features, train_labels = load_data()
	rest_cols = train_features.columns.tolist()
	rest_cols.remove('age')
	rest_cols.remove('fnlwgt')
	rest_cols.remove('education-num')
	rest_cols.remove('hours-per-week')
	#rest_cols.remove('capital-gain')
	#rest_cols.remove('capital-loss')
	train_features = hash_variable(train_features, rest_cols)
	X_train, X_test, y_train, y_test = train_test_split(
		train_features, train_labels, test_size=0.2, random_state=5)
	
	model(trainX=X_train, trainY=y_train,testX=X_test, testY=y_test)
	#test_features = hash_variable(test_features, rest_cols)
	#predictor(test_features)

