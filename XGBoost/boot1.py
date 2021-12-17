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

#print("CatBoost Version     : ", catboost.__version__)
#print("Scikit-Learn Version : ", sklearn.__version__)

breast_cancer = load_breast_cancer()

for line in breast_cancer.DESCR.split("\n")[5:31]:
    print(line)

breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns = breast_cancer.feature_names)
breast_cancer_df["TumorType"] = breast_cancer.target

breast_cancer_df.head()


X_train, X_test, Y_train, Y_test = train_test_split(breast_cancer.data, breast_cancer.target, train_size=0.9,
                                                    stratify=breast_cancer.target,
                                                    random_state=123)


booster = CatBoostClassifier(iterations=100, verbose=10)

booster.fit(X_train, Y_train, eval_set=(X_test, Y_test));
booster.set_feature_names(breast_cancer.feature_names)

test_preds = booster.predict(X_test)
train_preds = booster.predict(X_train)

print("\nTest  Accuracy : %.2f"%booster.score(X_test, Y_test))
print("Train Accuracy : %.2f"%booster.score(X_train, Y_train))