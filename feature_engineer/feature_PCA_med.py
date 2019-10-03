import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

feature_size = 10

# medication
domain = 'medication'
input_file = 'medication_prior_hosp.csv'
code = 'drug_concept_id'

# defined schi cohort
schi_dat_v1 = pd.read_csv('src/int_data/visit/schi_cohort_v1.csv')
schi_dat_v2 = pd.read_csv('src/int_data/visit/schi_cohort_v2.csv')
schi_id = set(schi_dat_v1.person_id.values).union(set(set(schi_dat_v2.person_id.values)))

def pca_feat(input_data, seed=0):
    dat0 = pd.read_csv('src/int_data/preprocessed/' + input_data, low_memory=False)
    dat = dat0[dat0.person_id.isin(schi_id)].copy()
    dat.loc[:, 'idx'] = np.array(dat.groupby('person_id').cumcount())
    dat = dat[['person_id', code, 'idx']].copy()
    dat_pivot = dat.pivot_table(index='person_id', columns=code, aggfunc='count', fill_value=0)
    X = dat_pivot.values
    pca = PCA(n_components=feature_size, svd_solver='auto', random_state=seed)
    principalComponents = pca.fit_transform(X)
    dat_feat = pd.DataFrame(data = principalComponents,
                               columns = ['PCA_feat_' + str(i) for i in range(1, feature_size + 1)])
    return dat_feat, dat_pivot

my_dat_feat, my_dat_pivot = pca_feat(input_file)

my_dat_feat.index = my_dat_pivot.index
my_dat_feat.to_csv('src/int_data/feat_eng/{}_pca_feat.csv'.format(domain))

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
# plt.ylim(-40, 5)
# plt.xlim(-10,30)
ax.legend(targets)
plt.savefig('src/output/pca_{}.pdf'.format(domain), dpi=600)