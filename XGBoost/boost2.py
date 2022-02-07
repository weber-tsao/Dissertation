# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import auc, plot_roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
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
    label_encoder=False
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

### Feature selction ###
selection = SelectFromModel(xgboostModel, threshold=0.001, prefit=True)
data_reduct = selection.transform(data)

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
lr_results = cross_val_score(LogisticRegression(class_weight='balanced', 
                                                fit_intercept=False,
                                                penalty='l1',
                                                solver='liblinear', 
                                                warm_start=True,
                                                tol=2, 
                                                C=17,
                                                intercept_scaling=16,
                                                max_iter=450), data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (lr_results.mean()*100, lr_results.std()*100))'''

'''# Decision Tree
dt_results = cross_val_score(DecisionTreeClassifier(), data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (dt_results.mean()*100, dt_results.std()*100))'''

'''# SVM
SVM = svm.SVC(kernel='rbf',C=1,gamma='auto')
svm_results = cross_val_score(SVM, data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (svm_results.mean()*100, svm_results.std()*100))'''

'''# KNeighbors
knn = KNeighborsClassifier(n_neighbors=1)
knn_results = cross_val_score(knn, data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (knn_results.mean()*100, knn_results.std()*100))

# Naive Bayes
gnb = BernoulliNB()
gnb_results = cross_val_score(gnb, data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (gnb_results.mean()*100, gnb_results.std()*100))'''

'''### Plot relation graph ###
#0: 95.47, 0.6
#1: 56.91, 0.87
#2: 56.21, 0.79
#3: 57.79, 1.11
#4: 59.94, 0.94
#5: 67.19, 1.24
#7: 75.13, 1.3
#9: 80.71, 1.09
#10: 83.16, 0.34
#15: 89.47, 0.56
#20: 92.01, 0.67

x1 = [0,1,2,3,4,5,7,9,15,20]
y1 = [99.94, 77.97, 78.76, 78.66, 80.71, 79.31, 84.57, 87.19, 92.06, 93.96]
plt.plot(x1, y1, color='red', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=8)

x2 = [0,1,2,3,4,5,7,9,15,20]
y2 = [95.47, 56.91, 56.21, 57.79, 59.94, 67.19, 75.13, 80.71, 89.47, 92.01]
plt.plot(x2, y2, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='green', markersize=8)
 
# setting x and y axis range
plt.ylim(0,100)
plt.xlim(0,25)
plt.xlabel('Base')
plt.ylabel('Accuarcy(%)')
plt.title('LFSR')
plt.legend(['XGBoost', 'SVM'])
plt.show()'''

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

### GridSearch ###
testingModel=XGBClassifier( 
                            n_estimators=1000, 
                            tree_method='gpu_hist',
                            objective='binary:logistic',
                            label_encoder=False)

param_dist = {  
    'max_depth':[1,2,3],
    'min_child_weight' :[9,10,11],
    'gamma': [0.7,0.8,0.9],
    'subsample': [0.7,0.8,0.9],
    'colsample_bytree': [0.7,0.8,0.9],
    'learning_rate': [0.01,0.03,0.06]
    #'n_estimators': [100,500,1000]#100
}  

#grid = RandomizedSearchCV(testingModel,param_dist,cv = 5,scoring = 'roc_auc',n_iter=500,n_jobs = -1,verbose = 2)
grid = GridSearchCV(testingModel, param_dist, scoring='roc_auc', cv=5, n_jobs=4)

grid.fit(data_reduct, data_label)
best_estimator = grid.best_estimator_
print(best_estimator)
print(grid.best_score_)
print(grid.best_params_)