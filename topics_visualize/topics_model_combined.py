import pandas as pd
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'

# define cohort
cohort = True

file_condition = pd.read_csv('src/int_data/condition_icd9.csv', low_memory=False)
file_procedure = pd.read_csv('src/int_data/procedure_cpt.csv', low_memory=False)
file_medication = pd.read_csv(dataPath + prefix + 'drug.csv', delimiter='|', low_memory=False)

file_condition1 = file_condition[['person_id', 'condition_concept_id', 'condition_start_date', 'icd9']].copy()
file_condition1.columns = ['person_id', 'concept_id', 'start_date', 'id_code']

file_procedure1 = file_procedure[['person_id', 'procedure_concept_id', 'procedure_date', 'cpt']].copy()
file_procedure1.columns = ['person_id', 'concept_id', 'start_date', 'id_code']

file_medication0 = file_medication[(file_medication['drug_concept_id']) != 0 &
                                   (~file_medication['drug_concept_id'].isnull())].copy()
file_medication1 = file_medication0[['person_id', 'drug_concept_id', 'drug_exposure_start_date',
                                     'drug_concept_id']].copy()
file_medication1.columns = [['person_id', 'concept_id', 'start_date', 'id_code']]
file_medication1['id_code'] = file_medication1['id_code'].apply(lambda x: str(x))

dat_frames = [file_condition1, file_procedure1, file_medication1]
file_combine0 = pd.concat(dat_frames)
file_combine0.sort_values(by=['person_id', 'start_date'], inplace=True)

# defined schi cohort
schi_dat = pd.read_csv('src/int_data/visit/schi_cohort.csv')
schi_id = set(schi_dat.person_id.values)

if not cohort:
    file_combine = file_combine0.copy()
else:
    file_combine = file_combine0[file_combine0.person_id.isin(schi_id)].copy()

df = file_combine[['id_code', 'person_id']].groupby(['person_id'])['id_code'].apply(lambda x: ' '.join(x)).reset_index()

# Convert to list
data = df['id_code'].values.tolist()
res = [sent.split(' ') for sent in data]

# Bag of Words
dictionary = gensim.corpora.Dictionary(res)
count = 0

dictionary.filter_extremes(no_below=5, no_above=1.0, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# LDA using BoW
num_topic = 6
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
pyLDAvis.save_html(lda_display_bow, 'src/output/lda_bow_cond_proc_med_' + str(num_topic) + '_' + str(cohort) + '.html')
