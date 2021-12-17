import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
        
train_df = pd.read_csv('./amazon/train.csv')
test_df = pd.read_csv('./amazon/test.csv')

#print(train_df.head())

X = train_df.drop("ACTION", axis=1)
y = train_df["ACTION"]

cat_features = list(range(0, X.shape[1]))
print(cat_features)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


'''clf = CatBoostClassifier(
    iterations=5, 
    learning_rate=0.1, 
    #loss_function='CrossEntropy'
)


clf.fit(X_train, y_train, 
        cat_features=cat_features, 
        eval_set=(X_val, y_val), 
        verbose=False
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())

clf = CatBoostClassifier(
    iterations=10,
#     verbose=5,
)

clf.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
)
'''

clf = CatBoostClassifier(
    iterations=10,
    custom_loss=['AUC', 'Accuracy']
)

clf.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    plot=True
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())
