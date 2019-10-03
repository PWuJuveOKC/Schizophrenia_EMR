import pandas as pd
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


num_topics = 5


# lab
# domain = 'lab'
# input_code = 'measurement_concept_id'
# input_file = 'lab_prior_hosp.csv'

# medication
domain = 'medication'
input_code = 'drug_concept_id'
input_file = 'medication_prior_hosp.csv'

# defined schi cohort
schi_dat_v1 = pd.read_csv('src/int_data/visit/schi_cohort_v1.csv')
schi_dat_v2 = pd.read_csv('src/int_data/visit/schi_cohort_v2.csv')
schi_id = set(schi_dat_v1.person_id.values).union(set(set(schi_dat_v2.person_id.values)))

def clean_data(code, file_name):
    dat0 = pd.read_csv('src/int_data/preprocessed/' + file_name)
    dat = dat0[dat0.person_id.isin(schi_id)].copy()
    dat[code] = dat[code].apply(lambda x: str(x)).copy()
    df = dat[[code, 'person_id']].groupby(['person_id'])[code].apply(lambda x: ' '.join(x)).reset_index()
    data = df[code].values.tolist()
    out = [sent.split(' ') for sent in data]
    return out, df


def lda_model(input_list, num_topic=num_topics, seed=111):
    # Bag of Words in Dataset
    dictionary = gensim.corpora.Dictionary(input_list)
    dictionary.filter_extremes(no_below=5, no_above=1, keep_n=10000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in res]
    model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       minimum_probability=1e-4, iterations=500, random_state=seed)
    return model, bow_corpus, dictionary

res, my_dat = clean_data(input_code, input_file)
my_lda_model, my_bow_corpus, my_dictionary = lda_model(res)

for idx, topic in my_lda_model.print_topics(-1, 50):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# visualization
lda_display_bow = pyLDAvis.gensim.prepare(my_lda_model, my_bow_corpus, my_dictionary, sort_topics=True)
pyLDAvis.save_html(lda_display_bow, 'src/output/lda_bow_' + domain + '_prior_hosp.html')

# create dataframe for condition topics
doc_topics = []
for bow in my_bow_corpus:
    doc_topics.append(dict(my_lda_model.get_document_topics(bow, per_word_topics=False)))
topics_emb = pd.DataFrame(doc_topics)
topics_emb.fillna(0, inplace=True)
topics_emb.columns = [domain + '_' + 'topic_' + str(i) for i in range(1, num_topics + 1)]
topics_emb['person_id'] = my_dat.person_id
topics_emb.to_csv('src/int_data/feat_eng/' + domain + '_topic_model_feat_' + str(num_topics) + '.csv', index=None)
