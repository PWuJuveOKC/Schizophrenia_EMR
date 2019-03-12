import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

feature_size = 10
domain = 'procedure'
code = 'cpt'
input_file = 'procedure_cpt_prior_hosp.csv'


def pca_feat(input_data, seed=0):
    dat = pd.read_csv('src/int_data/' + input_data)
    dat.loc[:, 'idx'] = np.array(dat.groupby('person_id').cumcount())
    dat = dat[['person_id', code, 'idx']].copy()
    dat_pivot = dat.pivot_table(index='person_id', columns=code, aggfunc='count', fill_value = 0)
    X = dat_pivot.values
    pca = PCA(n_components=feature_size, svd_solver='auto', random_state=seed)
    principalComponents = pca.fit_transform(X)
    dat_feat = pd.DataFrame(data = principalComponents,
                               columns = ['PCA_feat_' + str(i) for i in range(1, feature_size + 1)])
    return dat_feat, dat_pivot

my_dat_feat, my_dat_pivot = pca_feat(input_file)

my_dat_feat.index = my_dat_pivot.index
my_dat_feat.to_csv('src/int_data/' + domain + '_' + code + '_pca_feat.csv')

## visualization of PCA
my_dat_feat['person_id'] = my_dat_feat.index
window_size = 10
file = pd.read_csv('src/int_data/hospitalization_window_' + str(window_size) + '.csv')
file = pd.merge(file, my_dat_feat, on = 'person_id', how = 'inner')


# visualization
file['hour_label'] = pd.qcut(file['hospitalization_hours'], 3, labels=[0, 1, 2])
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = file['hour_label'] == target
    ax.scatter(file.loc[indicesToKeep, 'PCA_feat_1'], file.loc[indicesToKeep, 'PCA_feat_2'],
               c = color, s = 50)
plt.ylim(-20, 100)
plt.xlim(-10, 200)
ax.legend(targets)
plt.savefig('src/output/pca_' + domain +'.pdf', dpi=600)

# myfunc =  lambda x: (x > 0) * 1.0
# dat_pivot_binary = dat_pivot.apply(func = myfunc)
# X1 = dat_pivot_binary.values
# from sklearn.neural_network import BernoulliRBM
# rbm = BernoulliRBM(n_components=10, batch_size=32, learning_rate=0.01, n_iter=20, random_state=0, verbose=True)
# rbm.fit(X1)
# weight_matrix = rbm.components_
# dat_feat_rbm = pd.DataFrame(np.dot(X1, weight_matrix.T) + rbm.intercept_hidden_)