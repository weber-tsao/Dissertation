# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
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
X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.35, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=34)
evals_result ={}
eval_s = [(X_train, y_train),(X_val, y_val)]
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

### Set start time ###
start_time = datetime.now()
### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    n_estimators=2000, 
    learning_rate= 0.3, 
    objective="binary:logistic",
    tree_method='gpu_hist',
    min_child_weight=60,
    max_depth=2,
    use_label_encoder=False,
    eval_metric='logloss'
    #gamma=0.1,
    #subsample=0.8,
    #colsample_bytree=0.8
    )

xgboostModel.fit(X_train, y_train, eval_set=eval_s)
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


### Cross validation 
kfold  = KFold(n_splits=30)
results = cross_val_score(xgboostModel, data, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

'''
param_test1 = {  
    #'max_depth':[2,3,4,5,6] #Best: 2
    #'min_child_weight' :[10,20,30,40,50,60,70,80,90,100,110,120,130,140] #Bset: 60
    #'gamma': [0,0.1,0.5,0.8,1.2,1.5,2.0,3.0,4.0,5.0] #Best: 0.1
    #'subsample': [0,0.1,0.5,0.8,1.2,1.5,2.0,3.0,4.0,5.0] #Best: 0.8
    #'colsample_bytree': [0,0.1,0.5,0.8,1.2,1.5,2.0,3.0,4.0,5.0] #Best: 0.8
}  
gsearch1 = GridSearchCV(estimator=XGBClassifier( learning_rate=0.3, 
                                                 n_estimators=2000, 
                                                 max_depth=2,
                                                 tree_method='gpu_hist',
                                                 evals_result=evals_result,
                                                 min_child_weight=60, 
                                                 objective='binary:logistic', 
                                                 gamma=0.1,
                                                 subsample=0.8,
                                                 colsample_bytree=0.8,
                                                 seed=27),
                                                 param_grid=param_test1,scoring='roc_auc', cv=5, n_jobs=4)  
gsearch1.fit(X_train,y_train)  
evaluate_value = gsearch1.cv_results_
print(evaluate_value)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
'''