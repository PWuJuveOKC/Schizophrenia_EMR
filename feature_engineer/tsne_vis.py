import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

domain = 'procedure'
dat = pd.read_csv('src/int_data/' + domain + '_topic_model_feat.csv')

## visualization of tSNE
window_size = 10
file = pd.read_csv('src/int_data/hospitalization_window_' + str(window_size) + '.csv')
file = pd.merge(file, dat, on = 'person_id', how = 'inner')
file['hour_label'] = pd.qcut(file['hospitalization_hours'], 3, labels=[0, 1, 2])

labels = file['hour_label'].values
X = file.iloc[:,-11:-1].values

model = TSNE(n_components=2, perplexity=10, metric='mahalanobis', init='pca', random_state=0)
X_ts = model.fit_transform(X)

fig = plt.figure()
xs = X_ts[:, 0]
ys = X_ts[:, 1]

xs1 = xs[labels==0]
xs2 = xs[labels==1]
xs3 = xs[labels==2]
ys1 = ys[labels==0]
ys2 = ys[labels==1]
ys3 = ys[labels==2]

a1=plt.scatter(xs1, ys1, color='blue', marker='*',s=50)
a2=plt.scatter(xs2, ys2, color='red', s=30)
a3=plt.scatter(xs3, ys3, color='green', s=30)

#plt.legend((a1,a2), ('High {}'.format(strat), 'Low {}'.format(strat)),scatterpoints=1, loc='upper left',ncol=3, fontsize=15)
plt.tight_layout()
plt.show()
plt.savefig('src/output/tsne_' + domain + '.pdf',dpi=900)