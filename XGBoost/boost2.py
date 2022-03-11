### Import important packages ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold,ShuffleSplit
from sklearn.metrics import auc, plot_roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB,CategoricalNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from arbiter_PUF import*
from XOR_PUF import*
from lightweight_PUF import*
from feedforward_PUF import*
from interpose_PUF import*
from LFSR_simulated import*
from Puf_delay import*
from general_model import*
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

### Set running start time ###
start_time = datetime.now()

### Load data ###
#puf = Puf()
#data, data_label = puf.load_data()
#arbiter_puf = arbiter_PUF()
#data, data_label = arbiter_puf.load_data(68, 6800, 11, 123)
#xor_puf = XOR_PUF()
#data, data_label = xor_puf.load_data(68, 32000, 4, 34)
#lightweight_puf = lightweight_PUF()
#data, data_label = lightweight_puf.load_data(68, 68000, 2, 123)
#feedforward_puf = feedforward_PUF()
#data, data_label = feedforward_puf.load_data(68, 68000, 6, 32, 60, 123)
#interpose_puf = interpose_PUF()
#data, data_label = interpose_puf.load_data(68, 24000, 3, 3, 12)
general_model = general_model()
data, data_label = general_model.load_data(6, 6, 6, 6, 0)

### Split train, test data for the model ###
X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.25, random_state=66)
X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=24)
evals_result ={}
eval_s = [(X_train, y_train),(X_val, y_val)]

### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    learning_rate=0.05, 
    n_estimators=400, 
    max_depth=2,
    tree_method='gpu_hist',
    min_child_weight=10, 
    objective='binary:logistic', 
    gamma=0.8,
    subsample=0.8,
    colsample_bytree=0.8,
    #early_stopping_rounds=100,
    eval_metric='error'
    )

xgboostModel.fit(X_train, y_train, eval_set=eval_s, early_stopping_rounds=100, verbose = 0)
#training_acc = xgboostModel.score(X_train,y_train)
#testing_acc = xgboostModel.score(X_test,y_test)
#print('Training accuracy: {}%'.format(training_acc*100))
#print('Testing accuracy: {}%'.format(testing_acc*100))

'''### plot loss graph ###
results = xgboostModel.evals_result()
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
# show the legend
plt.legend()
# show the plot
plt.show()'''

### Feature selction ###
'''feature_names = [str(x) for x in range(65)]
importance = xgboostModel.feature_importances_
idx_third = importance.argsort()[-3]
threshold = importance[idx_third]
print(threshold)
idx_features = (-importance).argsort()[:3]
name_features = np.array(feature_names)[idx_features]
print(name_features)
print(data[64])'''

selection = SelectFromModel(xgboostModel, threshold=0.00001, prefit=True)
print(xgboostModel.feature_importances_)
data_reduct = selection.transform(data)
data_reduct, data_label = shuffle(data_reduct, data_label)

### Cross validation ###
#ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=3)
ss = StratifiedKFold(n_splits=5)

### Calculate training time ###
end_time = datetime.now()
print('Training time: {}'.format(end_time - start_time))

### Set testing start time ###
test_start_time = datetime.now()
results = cross_val_score(xgboostModel, data_reduct, data_label, cv=ss)
print("cross validation accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

### Check overfitting ###
data_train, data_test, train_label, test_label = train_test_split(data_reduct, data_label, test_size=.25, random_state=66)
xgboostModel.fit(data_train, train_label)
#testingacc = xgboostModel.predict(data_test)
training2 = xgboostModel.score(data_train, train_label)
testing2 = xgboostModel.score(data_test, test_label)
print('For unseen data')
print('Training accuracy: {}%'.format(training2*100))
print('Testing accuracy: {}%'.format(testing2*100))

#cc = f1_score(test_label, testingacc)
#print(cc)

### Calculate testing time ###
test_end_time = datetime.now()
print('Testing time: {}'.format(test_end_time - test_start_time))

'''print(xgboostModel.get_booster().get_score(importance_type="gain"))
plot_importance(xgboostModel)
plt.figure(figsize = (30, 30))
plt.show()'''


'''train_f, test_f, train_label_f, test_label_f = train_test_split(data, data_label, test_size=.25, random_state=66)
random_f = RandomForestClassifier(max_depth=2, random_state=0)
random_f.fit(train_f, train_label_f)

importances = random_f.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_f.estimators_], axis=0)
#print(std)

feature_names = [str(x) for x in range(68)]
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
#forest_importances.plot.bar(yerr=std, ax=ax)
plt.bar(range(len(forest_importances)), forest_importances)
ax.set_title("Feature importances")
ax.set_ylabel("Importances")
fig.tight_layout()
print(importances)'''

'''plt.bar(range(len(xgboostModel.feature_importances_)), xgboostModel.feature_importances_)
plt.show()'''

'''# Logistic Regression
lr_results = cross_val_score(LogisticRegression(C=1, 
                                                class_weight='balanced', 
                                                fit_intercept=False,
                                                intercept_scaling=9,
                                                max_iter=300, 
                                                penalty='l1',
                                                solver='liblinear', 
                                                tol=1, 
                                                warm_start=True), data_reduct, data_label, cv=ss)
print("Accuracy: %.2f%% (%.2f%%)" % (lr_results.mean()*100, lr_results.std()*100))'''

'''# Decision Tree
decisionTree = DecisionTreeClassifier(max_depth=2, 
                                      max_leaf_nodes=2
                                      )
dt_result = cross_val_score(decisionTree, data_reduct, data_label, cv=ss)
print("Accuracy: %.2f%% (%.2f%%)" % (dt_result.mean()*100, dt_result.std()*100))'''

'''dt_results = cross_val_score(DecisionTreeClassifier(), data_reduct, data_label, cv=ss)
print("Accuracy: %.2f%% (%.2f%%)" % (dt_results.mean()*100, dt_results.std()*100))'''

'''# SVM
SVM = svm.SVC(kernel='rbf',
              C=1,
              gamma='auto',
              degree=0,
              coef0=0,
              shrinking=True,
              probability=True,
              tol=0.001,
              cache_size=100,
              class_weight='balanced',
              decision_function_shape='ovo'
              )
svm_results = cross_val_score(SVM, data_reduct, data_label, cv=ss)
print("Accuracy: %.2f%% (%.2f%%)" % (svm_results.mean()*100, svm_results.std()*100))'''

'''# KNeighbors
knn = KNeighborsClassifier(algorithm='ball_tree', 
                           leaf_size=2, 
                           n_neighbors=9)
knn_results = cross_val_score(knn, data_reduct, data_label, cv=ss)
print("Accuracy: %.2f%% (%.2f%%)" % (knn_results.mean()*100, knn_results.std()*100))'''

'''# Naive Bayes
gnb = GaussianNB(var_smoothing=3)
gnb_results = cross_val_score(gnb, data_reduct, data_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (gnb_results.mean()*100, gnb_results.std()*100))'''

'''### Plot relation graph ###

x1 = [0,1,2,3,4,5,6,7,9,15,20]
y1 = [93.03,64.52,67.11,66.14,58.5,64.04,60.26,60.78,64.72,63.58,62.28]
plt.plot(x1, y1, color='red', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=8)

x2 = [0,1,2,3,4,5,6,7,9,15,20]
y2 = [81.19,61.84,62.96,61.42,65.94,61.29,57.36,66.04,63.14,71.7,71.95]
plt.plot(x2, y2, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='green', markersize=8)
 
# setting x and y axis range
plt.ylim(50,100)
plt.xlim(0,20)
plt.xlabel('Base')
plt.ylabel('Accuarcy(%)')
plt.title('LFSR')
plt.legend(['XGBoost', 'DT'])
plt.show()'''

'''x1 = [0,1,2,3,4,6,8,10,15,20]
y1 = [98.85,51.56,52.78,53.53,52.57,51.4,51.04,49.94,50.57,50.94]
plt.plot(x1, y1, color='red', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=8)
 
# setting x and y axis range
plt.ylim(50,100)
plt.xlim(0,20)
plt.xlabel('Base')
plt.ylabel('Accuracy(%)')
plt.show()'''

'''### Cross validation with plotting confidence graph ###
tprs = []
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

'''### GridSearch ###
testingModel=KNeighborsClassifier()

param_dist = {
        'n_neighbors':range(0,10,1), 
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 
        'leaf_size':range(10,50,10), 
        'p':[1,2], 
        'metric':['str', 'callable', 'minkowski'], 
        'metric_params':[dict, None]
        }

grid = RandomizedSearchCV(testingModel,param_dist,cv = 5,scoring = 'roc_auc',n_iter=500,n_jobs = -1,verbose = 2)

grid.fit(data_reduct, data_label)
best_estimator = grid.best_estimator_
print(best_estimator)
print(grid.best_score_)
print(grid.best_params_)

cv_result = pd.DataFrame.from_dict(grid.cv_results_)

cv_result.to_csv()'''