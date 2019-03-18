import pandas as pd
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
domain = 'measurement'
code = 'measurement_concept_id'
# domain = 'drug'
# code = 'drug_concept_id'

file = pd.read_csv(dataPath + prefix + domain + '.csv', delimiter='|')
file1 = file[(file[code]) != 0 & (~file[code].isnull())].copy()
file1[code] = file1[code].apply(lambda x: str(x))
df = file1[[code, 'person_id']].groupby(['person_id'])[code].apply(lambda x: ' '.join(x)).reset_index()

# Convert to list
data = df[code].values.tolist()
res = [sent.split(' ') for sent in data]

# Bag of Words
dictionary = gensim.corpora.Dictionary(res)
count = 0

dictionary.filter_extremes(no_below=5, no_above=0.8, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# LDA using BoW
num_topic = 5
seed = 11
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       iterations=200, random_state=seed)

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
lda_display_bow = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display_bow, 'src/output/lda_bow_' + domain + '.html')