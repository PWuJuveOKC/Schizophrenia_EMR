import pandas as pd
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# dataPath = r"C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia/"
dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'

# define cohort
cohort = True

domain = 'procedure'
code = 'cpt'
# domain = 'condition'
# code = 'icd9'

# defined schi cohort
# schi_dat_v1 = pd.read_csv('src/int_data/visit/schi_cohort_v1.csv')
# schi_dat_v2 = pd.read_csv('src/int_data/visit/schi_cohort_v2.csv')
schi_dat_v1 = pd.read_csv(dataPath+r'/visit/schi_cohort_v1.csv')
schi_dat_v2 = pd.read_csv(dataPath+r'/visit/schi_cohort_v2.csv')
schi_id = set(schi_dat_v1.person_id.values).union(set(set(schi_dat_v2.person_id.values)))

file0 = pd.read_csv('src/int_data/' + domain + '_' + code + '.csv', low_memory=False)
# file0 = pd.read_csv(dataPath + domain + '_' + code + '.csv', low_memory=False)

if not cohort:
    file = file0.copy()
else:
    file = file0[file0.person_id.isin(schi_id)].copy()

df = file[[code, 'person_id']].groupby(['person_id'])[code].apply(lambda x: ' '.join(x)).reset_index()

# Convert to list
data = df[code].values.tolist()
res = [sent.split(' ') for sent in data]

# Bag of Words
dictionary = gensim.corpora.Dictionary(res)
count = 0

dictionary.filter_extremes(no_below=5, no_above=1.0, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]

# save corpus for networks
# import multiprocessing
import os
pathSave = dataPath+"_".join([domain, code, 'dat'])
os.makedirs(pathSave, exist_ok=True)

replacePattern = {'1':'one', '2':'two', '3':'thr', '4':'fou', '5':'fiv', \
                  '6':'six', '7':'sev', '8':'eig', '9':'nin', '0':'zer', '.':'dot'}
def wordtrans_to(word):
    for i, v in replacePattern.items():
        word = word.replace('%s' % i, v)
    return(word)

inclueded = list(dictionary.token2id.keys())
for r in enumerate(res):
    s = pd.Series((wordtrans_to(k) for k in r[1] if k in inclueded), dtype='str')
    s.to_csv("%s/%s.txt" % (pathSave, df.iat[r[0], 0]), header=False, index=False)

##
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# LDA using BoW
num_topic = 5
seed = 11
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       iterations=500, random_state=seed)

for idx, topic in lda_model.print_topics(-1, 50):
    print('Topic: {} \nWords: {}'.format(idx, topic))

doc_topics = []
for bow in bow_corpus:
    doc_topics.append(dict(lda_model.get_document_topics(bow,  per_word_topics=False)))

topics_emb = pd.DataFrame(doc_topics)
topics_emb.fillna(0, inplace=True)

# for each document
for index, score in sorted(lda_model[bow_corpus[5]], key=lambda tup: -1 * tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 5)))

# visualization
lda_display_bow = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=True)
pyLDAvis.save_html(lda_display_bow, 'src/output/lda_bow_' + domain + '_' + code + '_' + str(cohort) + '.html')

# network generating
from tethne import feature_cooccurrence
#from tethne import gensim_to_theta_featureset
from tethne import gensim_to_phi_featureset

phi = gensim_to_phi_featureset(lda_model)
# theta = gensim_to_theta_featureset(lda_model,bow_corpus,person_id)
graph = feature_cooccurrence(phi,'icd9')
from tethne.networks import topics
graph = topics.terms(graph, threshold=0.015)

import tethne.writers as wr
wr.graph.to_graphml(graph, './mymodel_tc.graphml')

from tethne.readers.plain_text import read
# corpus2 = read(r"/mnt/hgfs/Yuanjia/schi/trac_ 3772_schizophrenia/procedure_cpt_dat")
corpus2 = read(r"/mnt/hgfs/Yuanjia/schi/trac_ 3772_schizophrenia/"+"_".join([domain,code,"dat"]))

from tethne.model.corpus import mallet
model = mallet.LDAModel(corpus2, featureset_name='plain_text')
model.Z=5
model.max_iter=500
model.fit()
model.print_topics(Nwords=15)
from tethne.networks import topics
threshold = 0.007
graph = topics.terms(model, threshold=threshold)
import tethne.writers as wr
wr.graph.to_graphml(graph, r"/mnt/hgfs/Yuanjia/schi/networks/"+"_".join([domain,code,str(threshold)])+".graphml")

# transform id to real id
os.chdir(r"C:/Users/Peter Xu/Desktop/Yuanjia/schi/networks/")
filename = 'procedure_cpt_0.0065' # use the file name that you want to transform
ff = open(filename+".graphml","r+")
graphx = ff.readlines()
ff.close()

replacePattern = {'1':'one', '2':'two', '3':'thr', '4':'fou', '5':'fiv', \
                  '6':'six', '7':'sev', '8':'eig', '9':'nin', '0':'zer', '.':'dot'}
replacePattern = dict(zip(replacePattern.values(), replacePattern.keys()))
replacePattern['w8ht'] = 'weight'
def wordtrans_to(word):
    for i, v in replacePattern.items():
        word = word.replace('%s' % i, v)
    return(word)
g=[wordtrans_to(text[1]) for text in enumerate(graphx)]
gg = open(filename+".graphml","w")
gg.writelines(g)
gg.close()

# transform id to real name
id_name = pd.read_csv(dataPath+"condition_code_to_word.csv", header=None, dtype='str')
id_na = dict(zip(["\""+x+"\"" for x in id_name[0]], \
                 ["\""+x+"\"" for x in id_name[1]]))
ff = open(filename+".graphml","r+")
graphx = ff.readlines()
ff.close()
def wordtrans_to(word):
    for i, v in id_na.items():
        word = word.replace('%s' % i, v)
    return(word)
g=[wordtrans_to(text[1]) for text in enumerate(graphx)]
gg = open(filename+"_word.graphml","w")
gg.writelines(g)
gg.close()