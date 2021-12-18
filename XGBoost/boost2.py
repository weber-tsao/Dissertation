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
import warnings
warnings.filterwarnings("ignore")
### Set start time ###
#start_time = datetime.now()

### Load data ###
puf = Puf()
data, data_label = puf.load_data()

### Split train, test data for the model ###
X_train, X_test, y_train, y_test = train_test_split(data, data_label, test_size=.3, random_state=42)
evals_result ={}
eval_s = [(X_train, y_train),(X_test, y_test)]
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

### Set start time ###
start_time = datetime.now()
### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    n_estimators=3000, 
    learning_rate= 0.3, 
    objective="binary:logistic",
    tree_method='gpu_hist',
    evals_result=evals_result,
    min_child_weight=100
    )
#setattr(xgboostModel, 'verbosity', 2)
xgboostModel.fit(X_train, y_train,eval_set=eval_s)
predicted = xgboostModel.predict(X_test)
training_acc = xgboostModel.score(X_train,y_train)
testing_acc = xgboostModel.score(X_test,y_test)
print('Training accuracy: {}%'.format(training_acc*100))
print('Testing accuracy: {}%'.format(testing_acc*100))

### Calculate training time ###
end_time = datetime.now()
print('Runtime: {}'.format(end_time - start_time))

results = xgboostModel.evals_result()
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
# show the legend
plt.legend()
# show the plot
plt.show()