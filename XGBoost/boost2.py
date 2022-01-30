# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import auc, plot_roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from Puf_delay import*
from LFSR_simulated import*
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

### Set running start time ###
start_time = datetime.now()

### Load data ###
puf = Puf()
data, data_label = puf.load_data()

### Split train, test data for the model ###
X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.5, random_state=66)
X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=24)
evals_result ={}
eval_s = [(X_train, y_train),(X_val, y_val)]
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_test.shape)

### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    learning_rate=0.05, 
    n_estimators=1000, 
    max_depth=2,
    tree_method='gpu_hist',
    min_child_weight=10, 
    objective='binary:logistic', 
    gamma=0.8,
    subsample=0.8,
    colsample_bytree=0.8,
    #early_stopping_rounds=10,
    seed=27
    )

xgboostModel.fit(X_train, y_train, eval_set=eval_s)
predicted = xgboostModel.predict(X_test)
training_acc = xgboostModel.score(X_train,y_train)
testing_acc = xgboostModel.score(X_test,y_test)
print('Training accuracy: {}%'.format(training_acc*100))
print('Testing accuracy: {}%'.format(testing_acc*100))

### plot loss graph ###
results = xgboostModel.evals_result()
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
# show the legend
plt.legend()
# show the plot
plt.show()

'''# Fit model using each importance as a threshold
thresholds = np.sort(xgboostModel.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(xgboostModel, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	xgboostModel.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	predictions = xgboostModel.predict(select_X_test)
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))'''

selection = SelectFromModel(xgboostModel, threshold=0.001, prefit=True)
data_reduct = selection.transform(data)
print(data_reduct.shape)
### Cross validation ###
kfold = KFold(n_splits=5)
results = cross_val_score(xgboostModel, data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#print(xgboostModel.get_booster().get_score(importance_type="gain"))
#plot_importance(xgboostModel)
#plt.figure(figsize = (30, 30))
#plt.show()

'''plt.bar(range(len(xgboostModel.feature_importances_)), xgboostModel.feature_importances_)
plt.show()'''

'''# Logistic Regression
lr_results = cross_val_score(LogisticRegression(), data, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (lr_results.mean()*100, lr_results.std()*100))

# Decision Tree
dt_results = cross_val_score(DecisionTreeClassifier(), data, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (dt_results.mean()*100, dt_results.std()*100))

# SVM
SVM = svm.SVC(kernel='rbf',C=1,gamma='auto')
svm_results = cross_val_score(SVM, data, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (svm_results.mean()*100, svm_results.std()*100))

# KNeighbors
knn = KNeighborsClassifier(n_neighbors=3)55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555544444444444444444444444444444444444444444444444
knn_results = cross_val_score(knn, data, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (knn_results.mean()*100, knn_results.std()*100))

# Naive Bayes
gnb = GaussianNB()
gnb_results = cross_val_score(gnb, data, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (gnb_results.mean()*100, gnb_results.std()*100))'''

### Cross validation with plotting confidence graph ###
'''tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kfold.split(data, data_label)):
    xgboostModel.fit(data[train], data_label[train])
    viz = plot_roc_curve(xgboostModel, data[test], data_label[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()'''

### Calculate running time ###
end_time = datetime.now()
print('Runtime: {}'.format(end_time - start_time))



'''### Modify training parameter ###
param_test1 = {  
    #'max_depth':[2,3,4,5,6,7,8,9,10] #Best: 2
    #'min_child_weight' :[10,20,30,40,50,60,70,80,90,100,110,120,130,140] #Best: 10
    #'gamma': [0,0.1,0.5,0.8,1.2,1.5,2.0,3.0,4.0,5.0] #Best: 0.8
    #'subsample': [0,0.1,0.5,0.8,1.2,1.5,2.0,3.0,4.0,5.0] #Best: 0.8
    #'colsample_bytree': [0,0.1,0.5,0.8,1.2,1.5,2.0,3.0,4.0,5.0] #Best: 0.8
    #'learning_rate': [0.01,0.05,0.1,0.2,0.3,0.5] #Best: 0.05
    #'n_estimators': [100,500,1000,2000,3000] #Best: 1000 
}  
gsearch1 = GridSearchCV(estimator=XGBClassifier( learning_rate=0.05, 
                                                 n_estimators=1000, 
                                                 max_depth=2,
                                                 tree_method='gpu_hist',
                                                 min_child_weight=10, 
                                                 objective='binary:logistic', 
                                                 gamma=0.8,
                                                 subsample=0.8,
                                                 colsample_bytree=0.8,
                                                 seed=27),
                                                 param_grid=param_test1,scoring='roc_auc', cv=5, n_jobs=4)  
gsearch1.fit(data, data_label)  
evaluate_value = gsearch1.cv_results_
print(evaluate_value)
print(gsearch1.best_params_)
print(gsearch1.best_score_)'''
