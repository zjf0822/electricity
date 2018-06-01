import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_columns = ['id', 'k1k2', 'locks_signal', 'emergency_signal', 'access_signal', 'THDV_M', 'THDI_M', 'label']
# 读入数据
train = pd.read_csv("/home/zhoujifa/competition/baidu_elecrticity/data_train.csv", names=train_columns)

sns.barplot(x='k1k2', y='label', data=train['k1k2'])
plt.show()