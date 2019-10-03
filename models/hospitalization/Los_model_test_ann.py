import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from tensorflow import set_random_seed

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from collections import Counter

method = 'pca'

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
    plt.savefig('src/output/PR_curve_ANN_v1_' + method + '.pdf')

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
    plt.savefig('src/output/ROC_curve_ANN_v1_' + method + '.pdf')

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


df_merged_shu = pd.read_csv('src/int_data/merged_data_' + method + '.csv') #shuffled dataframe
start = 4
#end = 19
end = df_merged_shu.shape[1] - 1
set_random_seed(2)


#####################################################################################################
######################################### bin-class label  #########################################
#####################################################################################################

X = df_merged_shu.iloc[:,start:end].values
y = df_merged_shu['hours_label_bin'].values
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 111)
# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## ANN
def build_clf(optimizer):
    clf = Sequential()
    clf.add(Dense(units = 20, kernel_initializer = 'normal', activation = 'relu', input_dim = X_train.shape[1]))
    clf.add(Dropout(0.2))
    clf.add(Dense(units = 10, kernel_initializer = 'normal',activation = 'relu'))
    clf.add(Dropout(0.2))
    clf.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'sigmoid'))
    clf.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    return clf


clf_nn = KerasClassifier(build_fn = build_clf, epochs=200)
parameters = {'batch_size': [25, 50, 100],
              'optimizer': ['adam', 'rmsprop'],
              'class_weight': ['balanced']}
grid_search_ann = GridSearchCV(estimator = clf_nn,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 3, verbose=0, n_jobs=-2)
grid_search_ann = grid_search_ann.fit(X_train, y_train)
best_parameters = grid_search_ann.best_params_
report(grid_search_ann.cv_results_)

y_test_pred_ann = grid_search_ann.predict(X_test)
y_test_score_ann = grid_search_ann.score(X_test, y_test)
print('Accuracy for ANN: ', y_test_score_ann)
print('Results: ', Counter(y_test_pred_ann[:,0]))
print('F-1, precision, recall score: ', f1_score(y_test, y_test_pred_ann),
      precision_score(y_test, y_test_pred_ann), recall_score(y_test, y_test_pred_ann))

y_score_ann = grid_search_ann.predict_proba(X_test)[:,-1]
plot_pr_curve(y_test, y_score_ann, 'NN_bin_com')
plot_roc_curve(y_test, y_score_ann, 'NN_bin_com')