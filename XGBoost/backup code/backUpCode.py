# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:09:11 2022

@author: weber
"""


##boost2.py##


### Split train, test data for the model ###
'''X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.25, random_state=66)
X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=24)
evals_result ={}
eval_s = [(X_train, y_train),(X_val, y_val)]

### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    booster='gbtree', colsample_bytree=1.0,
              eval_metric='error', gamma=0.8,
              learning_rate=0.01, max_depth=5,
              min_child_weight=20, n_estimators=700, subsample=0.8, tree_method='gpu_hist'
    )

xgboostModel.fit(X_train, y_train, eval_set=eval_s, early_stopping_rounds=100, verbose = 0
                 )'''
#training_acc = xgboostModel.score(X_train,y_train)
#testing_acc = xgboostModel.score(X_test,y_test)
#print('Training accuracy: {}%'.format(training_acc*100))
#print('Testing accuracy: {}%'.format(testing_acc*100))

'''selection = SelectFromModel(xgboostModel, threshold=0.01, prefit=True)
print(xgboostModel.feature_importances_)
data_reduct = selection.transform(data)
data_reduct, data_label = shuffle(data_reduct, data_label)
#select_list = selection.get_support()  '''

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

'''### Plot estimator and depth graph
depth_val = [2,3,4]
n_estimators_val = [100,200,300,400,500,600,700,800,900,1000]
for depth in depth_val:
    results = []
    for n_estimators in n_estimators_val:
        xgboostModel = XGBClassifier(
              booster='gbtree', colsample_bytree=1.0,
              eval_metric='error', gamma=0.6,
              learning_rate=0.3, max_depth=depth,
              min_child_weight=20, n_estimators=n_estimators, subsample=0.8, tree_method='gpu_hist'
              )
        score = cross_val_score(xgboostModel, data_reduct, data_label, cv=ss)
        results.append(score.mean()*100)
    plt.plot(n_estimators_val, results, linestyle='solid', marker='o', linewidth = 1.5, markersize=5)

plt.ylim(80,100)
plt.xlim(100,1000)
plt.xlabel('Number of estimators')
plt.ylabel('Accuarcy(%)')
#plt.title('LFSR')
plt.legend([2,3,4,5,6,7,8])
plt.show()'''

### GridSearch ###
'''testingModel=XGBClassifier(tree_method='gpu_hist',
                           objective='binary:logistic', 
                           eval_metric='error'
                           )
'''

#testingModel = XGBClassifier(
#    booster='gbtree', 
#    colsample_bytree=0.8,
#    eval_metric='error', 
#    gamma=0.1,
#    learning_rate=0.01, 
#    #max_depth=5
#    min_child_weight=10, 
#    #n_estimators=200
#    subsample=0.8, 
#    tree_method='gpu_hist'
#    )
#param_dist = {  
#    'max_depth':range(2,6,1),
#    #'min_child_weight' :range(10,30,10),
#    #'gamma': [0.1,0.4,0.6,0.8,1.0],
#    #'subsample': [0.1,0.4,0.6,0.8,1.0],
#    #'colsample_bytree': [0.1,0.4,0.6,0.8,1.0],
#    #'learning_rate': [0.01,0.05,0.1,0.2],
#    'n_estimators': range(100,1000,100)
#}  

#grid =GridSearchCV(testingModel,param_dist,cv = 5,scoring = 'roc_auc',n_jobs = -1,verbose = 2)
#grid =RandomizedSearchCV(testingModel,param_dist,cv = 5,scoring = 'roc_auc',n_iter=50,n_jobs = -1,verbose = 2)
#grid.fit(data, data_label)
#best_estimator = grid.best_estimator_
#print(best_estimator)
#a = grid.best_score_
#print(a)
#c = grid.best_params
#print(c)

#cv_result = pd.DataFrame.from_dict(grid.cv_results_)
#cv_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\tune.csv')


##Parity  vector##
'''data = self.get_parity_vectors(data)
        for d in range(len(data)):
            for j in range(65):
                if data[d][j] == -1:
                    data[d][j] = 0'''