import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

from sklearn.utils import class_weight
from collections import Counter
from functools import reduce
import time


### Precision-Recall Curve
def plot_pr_curve(y_test, y_score, name):

    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.3f}'.format(average_precision))

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
    plt.savefig('Results/Descriptive_figures/Models/PR_curve_' + name + '_v1.pdf')

### ROC Curve
def plot_roc_curve(y_test, y_score, name = ''):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=1, label='ROC curve (AUC = {:0.3f})'.format(roc_auc))


    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.savefig('Results/Descriptive_figures/Models/ROC_curve_' + name + '_v1.pdf')

#report best scores
def report(results, num = 5):
    for i in range(1, num + 1):
        cv_res = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in cv_res:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("**************************")


df_merged_shu = pd.read_csv('Results/Intermediate_output/merged_data_death.csv') #shuffled dataframe
start = 4
#end = 19
end = df_merged_shu.shape[1] - 2

#####################################################################################################
######################################### bin-class label  #########################################
#####################################################################################################

X = df_merged_shu.iloc[:,start:end].values
y = df_merged_shu['death'].values
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###################################################################################################
#################################### RF Model and CV-Grid Search ####################################
###################################################################################################

clf_rf = RandomForestClassifier(random_state=0, n_jobs=-1)

#report best scores
def report(results, num = 5):
    for i in range(1, num + 1):
        cv_res = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in cv_res:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("**************************")

param_grid = {
              "max_features": ['auto','log2'],
              "min_samples_leaf": [1, 2, 5, 10, 20, 50],
              "n_estimators": [50,100],
              "class_weight": ['balanced']
            }

# run grid search
start_time = time.time()
grid_search = GridSearchCV(clf_rf, cv=1, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
       % (time.time() - start_time, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


print(grid_search.score(X_test,y_test))
y_test_pred_rf= grid_search.predict(X_test)
y_test_score_rf = grid_search.score(X_test,y_test)
print('Accuracy for RF: ', y_test_score_rf)
print('Results: ', Counter(y_test_pred_rf))
print('F-1, precision, recall score: ', f1_score(y_test, y_test_pred_rf),
      precision_score(y_test, y_test_pred_rf), recall_score(y_test, y_test_pred_rf))

y_score_rf = grid_search.predict_proba(X_test)[:,-1]
plot_pr_curve(y_test, y_score_rf, 'RF_bin_com_death')
plot_roc_curve(y_test, y_score_rf, 'RF_bin_com_death')


################# Feature Importance from RF #####################
clf_rf_train = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight='balanced', max_features='auto',
                                      random_state=0, n_jobs=-1)
clf_rf_train.fit(X_train, y_train)
importances = clf_rf_train.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_train.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
names = np.array(list(df_merged_shu.iloc[:,start:end]))[indices]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, names[f], importances[indices[f]]))

size = 20
# Plot the feature importances of the forest
fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(size), importances[indices][0:size], color="r", yerr=std[indices][0:size], align="center")
plt.xticks(range(size), names[0:size], rotation=90)
plt.xlim([-1, size])
fig.set_tight_layout(True)
plt.savefig('Results/Descriptive_figures/Models/imp_features.pdf',dpi=900)


### SVM
clf_svm = SVC(kernel='rbf')
param_grid_svm = {
    "C": [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 50, 100, 1000, 10000],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 50, 100, 1000, 10000],
    "class_weight": ['balanced']
}

# run grid search
start = time.time()
grid_search_svm = GridSearchCV(clf_svm, cv=3,param_grid=param_grid_svm, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
print(grid_search_svm.best_score_)
print(grid_search_svm.best_params_)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
       % (time.time() - start, len(grid_search_svm.cv_results_['params'])))
report(grid_search_svm.cv_results_)


print(grid_search_svm.score(X_test,y_test))
y_test_pred_svm = grid_search_svm.predict(X_test)
y_test_score_svm = grid_search_svm.score(X_test,y_test)
print('Accuracy for SVM: ', y_test_score_svm)
print('Results: ', Counter(y_test_pred_svm))
print('F-1, precision, recall score: ', f1_score(y_test, y_test_pred_svm),
      precision_score(y_test, y_test_pred_svm), recall_score(y_test, y_test_pred_svm))

y_score_svm = grid_search_svm.decision_function(X_test)
plot_pr_curve(y_test, y_score_svm, 'SVM_bin_com')
plot_roc_curve(y_test, y_score_svm, 'SVM_bin_com')