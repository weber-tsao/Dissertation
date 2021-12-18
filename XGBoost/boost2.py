# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from Puf_delay import*
from datetime import datetime

### Set start time ###
start_time = datetime.now()

### Load data ###
puf = Puf()
data, data_label = puf.load_data()

### Split train, test data for the model ###
X_train, X_test, y_train, y_test = train_test_split(data, data_label, test_size=.3, random_state=42)
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    n_estimators=3000, 
    learning_rate= 0.5, 
    objective="binary:logistic",
    gpu_id=0,
    tree_method='gpu_hist'
    )
#setattr(xgboostModel, 'verbosity', 2)
xgboostModel.fit(X_train, y_train)
predicted = xgboostModel.predict(X_train)
training_acc = xgboostModel.score(X_train,y_train)
testing_acc = xgboostModel.score(X_test,y_test)
print('Training accuracy: {}%'.format(training_acc*100))
print('Testing accuracy: {}%'.format(testing_acc*100))

### Calculate training time ###
end_time = datetime.now()
print('Runtime: {}'.format(end_time - start_time))







'''
plot_importance(xgboostModel)
print('特徵重要程度: ',xgboostModel.feature_importances_)
'''
'''
# 建立訓練集的 DataFrme
df_train=pd.DataFrame(X_train)
df_train['Class']=y_train
# 建立測試集的 DataFrme
df_test=pd.DataFrame(X_test)
df_test['Class']=y_test

sns.lmplot("PetalLengthCm", "PetalWidthCm", hue='Class', data=df_train, fit_reg=False)

df_train['Predict']=predicted
sns.lmplot("PetalLengthCm", "PetalWidthCm", data=df_train, hue="Predict", fit_reg=False)
plt.show()
'''