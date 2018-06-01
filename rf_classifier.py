import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

train_columns = ['id', 'k1k2', 'locks_signal', 'emergency_signal', 'access_signal', 'THDV_M', 'THDI_M', 'label']
test_columns = ['id', 'k1k2', 'locks_signal', 'emergency_signal', 'access_signal', 'THDV_M', 'THDI_M']
# 读入数据
train = pd.read_csv("/home/zhoujifa/competition/baidu_elecrticity/data_train.csv", names=train_columns)
test = pd.read_csv("/home/zhoujifa/competition/baidu_elecrticity/data_test.csv", names=test_columns)

train = shuffle(train)

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
train_xy, val = train_test_split(train, test_size=0.3, random_state=1)

y = train_xy.label
X = train_xy.drop(['label', 'id', 'THDV_M', 'THDI_M'], axis=1)
val_y = val.label
val_X = val.drop(['label', 'id', 'THDV_M', "THDI_M"], axis=1)

classifier = RandomForestClassifier()
classifier.fit_transform(X, y)
predict_y = classifier.predict(val_X)

accuracy_score = accuracy_score(val_y, predict_y)

y_score = classifier.score(X, y)
# auc_score = roc_auc_score(val_y, y_score)

print(accuracy_score)
test = test.drop(['id', 'THDV_M', "THDI_M"], axis=1)
preds = classifier.predict(test)

np.savetxt('/home/zhoujifa/competition/baidu_elecrticity/rf_submission.csv', np.c_[range(1, len(test) + 1), preds], delimiter=',', header='ImageId,Label', comments='', fmt='%d')

print(classifier.feature_importances_)