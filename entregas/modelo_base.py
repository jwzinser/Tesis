"""
la idea es tomar una base de datos, que la podamos anonimizar a diferentes niveles
despues es hacer un modelo de prediccion respecto a la variable sensible
la matriz seria con 4 opciones ya sea si se generaliza o no la variable dependiente con repecto a la independiente
con sus respectivas combinaciones


"""

"""
posibles bases:
- cancer (todas son numericas)
- ingresos (la dependiente no se puede generalizar porque solo tiene dos niveles, pero las otras tienen incluso varias
variables categóricas)

is it correct to try to predict the sensitive variable?
"""

import pandas as pn
import numpy as np
from sklearn.cross_validation import KFold
import numpy as np
np.random.seed(10)
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

"""
first model no generalization neither in independent nor in dependent (sensitive) variable
"""


for h in range(4):
    data = pn.read_csv("/home/juanzinser/Workspace/Tesis/data/census/census_level_"+str(h) + ".csv")
    le = LabelEncoder()
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    kf = KFold(len(X), n_folds=20)

    # labelencoder
    for col in range(X.shape[1]):
        le = LabelEncoder()
        X[:, col] = le.fit_transform(X[:, col])
    le = LabelEncoder()
    y = le.fit_transform(y)

    n_estimator = 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.5)

    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
        random_state=0)

    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
    # este modelo que es el ganador tiene un total de 84% de correctos respecto al aleatorio que seria 75%.
    print(sum(y_test ==  rf_lm.predict(rf_enc.transform(rf.apply(X_test))))/float(len(y_test)))

    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_grd_lm = grd_lm.predict_proba(
        grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


    # The gradient boosted model by itself
    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)


    # The random forest model by itself
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.5)
    plt.ylim(0.5, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()
