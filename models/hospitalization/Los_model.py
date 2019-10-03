import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from collections import Counter
import time

feat_eng = 'pca'
my_model = 'LogisticRegression'


df_merged = pd.read_csv('src/int_data/feat_eng/merged_data_{}.csv'.format(feat_eng)) # shuffled dataframe
iters = 100

# empty list to store
train_acc_list, test_acc_list, f1_list, precision_list, recall_list, base_acc_list, roc_auc_list, ap_list \
    = [], [], [], [], [], [], [], []

for i in range(iters):
    print('iteration time: ', i)
    df_merged_shu = df_merged.sample(frac=1, random_state=i)
    start = 2
    end = df_merged_shu.shape[1] - 1
    #end = 5
    X = df_merged_shu.iloc[:,start:end].values
    y = df_merged_shu['hours_label_bin'].values

    # splitting the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    if my_model == 'rf':
        clf = RandomForestClassifier(random_state=i, n_jobs=-1)
        param_grid = {
            "max_features": ['auto', 'log2'],
            "min_samples_leaf": [1, 2, 5, 10, 20, 50],
            "n_estimators": [50, 100],
            "class_weight": ['balanced']
        }

    elif my_model == 'linear_svm':
        clf = SVC(kernel='linear', probability=True)
        param_grid = {
            "C": [2**j for j in range(-10, 11)],
            "class_weight": ['balanced']
        }

    elif my_model == 'gradient_boost':
        clf = GradientBoostingClassifier()
        param_grid = {
            "loss" : ['deviance', 'exponential'],
            "learning_rate": [0.1, 0.01, 0.001],
            "max_features": ['auto', 'log2'],
            "n_estimators": [50, 100],
            "min_samples_leaf": [1, 2, 5, 10, 20, 50]
        }

    elif my_model == 'LogisticRegression':
        clf = LogisticRegression()
        param_grid = {
            "C": [2 ** j for j in range(-10, 11)],
            "class_weight": ['balanced']
        }

    elif my_model == 'rbf_svm':
        clf = SVC(kernel='rbf', probability=True)
        param_grid = {
            "C": [2 ** j for j in range(-10, 11)],
            "gamma": [2 ** j for j in range(-10, 11)],
            "class_weight": ['balanced'],
        }


    # run grid search
    grid_search = GridSearchCV(clf, cv=5, param_grid=param_grid, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # population based accuracy
    base_acc = max(np.mean(y_test), 1 - np.mean(y_test))
    base_acc_list.append(base_acc)

    # train accuracy
    y_train_score = grid_search.score(X_train, y_train)
    train_acc_list.append(y_train_score)

    # test accuracy
    y_test_pred = grid_search.predict(X_test)
    y_test_score = grid_search.score(X_test, y_test)
    test_acc_list.append(y_test_score)

    # test prob
    y_test_prob = grid_search.predict_proba(X_test)[:, 1]


    print('Results: ', Counter(y_test_pred), 'Baseline Accuracy: ', base_acc, 'Train Accuracy: ', y_train_score,
          'Test Accuracy: ', y_test_score)
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    f1_list.append(f1_score(y_test, y_test_pred)) #f1-score
    precision_list.append(precision_score(y_test, y_test_pred)) # precision score
    recall_list.append(recall_score(y_test, y_test_pred)) # recall score
    roc_auc_list.append(roc_auc_score(y_test, y_test_prob)) # roc auc
    ap_list.append(average_precision_score(y_test, y_test_prob)) # average precision score


print('Final Results: ',)
print('---------------------------------------------------------------')
print('---------------------------------------------------------------')
print('---------------------------------------------------------------')
print('Mean (Std) Base Accuracy: ', np.mean(base_acc_list), '(', np.std(base_acc_list), ')')
print('Mean (Std) Train Accuracy: ', np.mean(train_acc_list), '(', np.std(train_acc_list), ')')
print('Mean (Std) Test Accuracy: ', np.mean(test_acc_list), '(', np.std(test_acc_list), ')')
print('Mean (Std) Precision Score: ', np.mean(precision_list), '(', np.std(precision_list), ')')
print('Mean (Std) Recall Score: ', np.mean(recall_list), '(', np.std(recall_list), ')')
print('Mean (Std) F1 Score: ', np.mean(f1_list), '(', np.std(f1_list), ')')
print('Mean (Std) ROC AUC Score: ', np.mean(roc_auc_list), '(', np.std(roc_auc_list), ')')
print('Mean (Std) Average Precision Score: ', np.mean(ap_list), '(', np.std(ap_list), ')')
print('---------------------------------------------------------------')
print('---------------------------------------------------------------')
print('---------------------------------------------------------------')