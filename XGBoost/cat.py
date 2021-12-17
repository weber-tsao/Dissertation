import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
import catboost
import sklearn
from sklearn.datasets import load_breast_cancer
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from Puf_delay import*


puf = Puf()
data, data_label = puf.load_data()
#print(data)

'''
breast_cancer = load_breast_cancer()
print(type(breast_cancer))
for line in breast_cancer.DESCR.split("\n")[5:31]:
    print(line)

breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns = breast_cancer.feature_names)
breast_cancer_df["TumorType"] = breast_cancer.target

breast_cancer_df.head()
'''

cat_features = list(range(0, data.shape[1]))

best_params = {
            'bagging_temperature': 0.5,
            'depth': 6,
            'iterations': 1000,
            'l2_leaf_reg': 25,
            'learning_rate': 0.5,
            'sampling_frequency': 'PerTreeLevel',
            'leaf_estimation_method': 'Gradient',
            'random_strength': 0.1,
            'boosting_type': 'Ordered',
            'feature_border_type': 'MaxLogSum',
            'l2_leaf_reg': 50,
            'max_ctr_complexity': 2,
            'fold_len_multiplier': 2
    }

X_train, X_test, Y_train, Y_test = train_test_split(data, data_label, train_size=0.5, random_state=5)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=2)

booster = CatBoostClassifier(**best_params,
                               loss_function='Logloss',
                               eval_metric='AUC',
                               nan_mode='Min',
                               thread_count=5,
                               task_type='CPU',
                               verbose=True)

booster.fit(X_train, Y_train,
                              eval_set=(X_val, Y_val),
                              cat_features=cat_features,
                              verbose_eval=50,
                              early_stopping_rounds=500,
                              use_best_model=True,
                              plot=True)


#model_cat.save_model("catmodel")

##Predictions
#cat_predictions = model_cat.predict_proba(test_data)[:, 1]
#cat_predictions_df = pd.DataFrame({'class': cat_predictions})

#booster = CatBoostClassifier(iterations=1000, verbose=100)

#booster.fit(X_train, Y_train, eval_set=(X_test, Y_test));
#booster.set_feature_names(cat_features)

#test_preds = booster.predict(X_test)
#train_preds = booster.predict(X_train)
#print(test_preds)
#print(Y_test)
print("\nTest  Accuracy : %.2f"%booster.score(X_test, Y_test))
print("Train Accuracy : %.2f"%booster.score(X_train, Y_train))
