import pandas as pd
import gensim
from gensim import corpora, models
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


num_topics = 10
# condition
# domain = 'condition'
# input_code = 'icd9'
# input_file = 'condition_icd9_prior_hosp.csv'

# procedure
domain = 'procedure'
input_code = 'cpt'
input_file = 'procedure_cpt_prior_hosp.csv'


def clean_data(code, file_name):
    dat = pd.read_csv('src/int_data/' + file_name)
    df = dat[[code, 'person_id']].groupby(['person_id'])[code].apply(lambda x: ' '.join(x)).reset_index()
    data = df[code].values.tolist()
    out = [sent.split(' ') for sent in data]
    return out, df


def lda_model(input_list, num_topic=10, seed=111):
    # Bag of Words in Dataset
    dictionary = gensim.corpora.Dictionary(input_list)
    dictionary.filter_extremes(no_below=5, no_above=0.8, keep_n=10000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in res]
    model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       minimum_probability=1e-4, iterations=200, random_state=seed)
    return model, bow_corpus, dictionary

res, my_dat = clean_data(input_code, input_file)
my_lda_model, my_bow_corpus, _ = lda_model(res)

for idx, topic in my_lda_model.print_topics(-1, 50):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# create dataframe for condition topics
doc_topics = []
for bow in my_bow_corpus:
    doc_topics.append(dict(my_lda_model.get_document_topics(bow, per_word_topics=False)))
topics_emb = pd.DataFrame(doc_topics)
topics_emb.fillna(0, inplace=True)
topics_emb.columns = [domain + '_' + 'topic_' + str(i) for i in range(1, num_topics + 1)]
topics_emb['person_id'] = my_dat.person_id
topics_emb.to_csv('src/int_data' + domain + '_topic_model_feat.csv', index=None)
