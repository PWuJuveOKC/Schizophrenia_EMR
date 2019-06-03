import pandas as pd
import warnings
import numpy as np
import os
os.chdir("C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia")
warnings.filterwarnings("ignore", category=DeprecationWarning)
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# Load Data
# file1 = pd.read_csv('trac_ 3772_schizophrenia_measurement.csv',delimiter='|',nrows=300,
                    # dtype={'visit_occurrence_id':object},parse_dates=True)
file = pd.read_csv('trac_ 3772_schizophrenia_measurement.csv',delimiter='|',
                    dtype={'visit_occurrence_id':object},parse_dates=True)

# personId = file.person_id.unique()

parse_dates = ['valid_start_date', 'valid_end_date']
concept = pd.read_csv('trac_ 3772_schizophrenia_concept.csv', delimiter='|', date_parser=parse_dates, low_memory=False)

concept_sub = concept[['concept_id', 'concept_name']]

# lab_concept1 = pd.merge(file1, concept_sub, left_on='measurement_concept_id',right_on='concept_id', how='left')
lab_concept = pd.merge(file, concept_sub, left_on='measurement_concept_id',right_on='concept_id', how='left')
# lab_concept1 = lab_concept1[lab_concept1.measurement_concept_id != 0]
lab_concept = lab_concept[lab_concept.measurement_concept_id != 0]


to_take = ['measurement_id','value_as_number','unit_concept_id','unit_source_value']
def to_cover(org, new):
    tt = pd.merge(org[['measurement_id','value_as_number','unit_concept_id','unit_source_value']], new, on='measurement_id', how='left')
    org.loc[~tt.value_as_number_y.isnull().values,'value_as_number']=tt.loc[~tt.value_as_number_y.isnull(),'value_as_number_y'].tolist()
    org.loc[~tt.value_as_number_y.isnull().values,'unit_concept_id']=tt.loc[~tt.value_as_number_y.isnull(),'unit_concept_id_y'].tolist()
    org.loc[~tt.value_as_number_y.isnull().values,'unit_source_value']=tt.loc[~tt.value_as_number_y.isnull(),'unit_source_value_y'].tolist()
    return org
# Body temperature
tmp=lab_concept.query('concept_name=="Body temperature"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8653.0')
concept_sub.query('concept_id==9289.0')
tmp['unit_source_value'] = 'degree Celsius'
tmp['unit_concept_id'] = 8653.0
tmp.loc[(tmp.value_as_number<122) & (tmp.value_as_number>86),'value_as_number'] = \
    (tmp.loc[(tmp.value_as_number<122) & (tmp.value_as_number>86),'value_as_number']-32)*5/9
wrong_id = tmp.loc[~((tmp.value_as_number<50) & (tmp.value_as_number>30)),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# mean         36.791336
# min          30.333333
# max          45.000000
val = tmp['value_as_number'].copy()
val[(val >= 41.5) | (val <= 35)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'body_tmp_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
tp1 = tmp.copy()
w_id = wrong_id.copy()

# Respiratory rate
tmp=lab_concept.query('concept_name=="Respiratory rate"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
wrong_id = tmp.loc[(tmp.value_as_number.isnull()) | (tmp.value_as_number <=0) | (tmp.value_as_number >= 150),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# mean         18.990322
# min           1.000000
# max         149.000000
val = tmp['value_as_number'].copy()
val[(val >= 20) | (val <= 8)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 18) | (tmp.value_as_number <= 12))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'R_rate_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# BP systolic
tmp=lab_concept.query('concept_name=="BP systolic"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8876')
tmp['unit_source_value'] = 'millimeter mercury column'
wrong_id = tmp.loc[(tmp.value_as_number.isnull()) | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# mean        124.590914
# min           1.000000
# max        1110.000000
val = tmp['value_as_number'].copy()
val[(val >= 180) | (val <= 90)] = 'panic'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'BP_systolic_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# BP diastolic
tmp=lab_concept.query('concept_name=="BP diastolic"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8876')
tmp['unit_source_value'] = 'millimeter mercury column'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    812204.000000
# mean         70.814018
# min           1.000000
# max         996.000000
val = tmp['value_as_number'].copy()
val[(val >= 120) | (val <= 60)] = 'panic'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'BP_diastolic_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Heart rate
tmp=lab_concept.query('concept_name=="Heart rate"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8483.0')
tmp['unit_source_value'] = 'counts per minute'
wrong_id = tmp.loc[~((tmp.value_as_number<500) & (tmp.value_as_number>1)),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    766593.000000
# mean         84.302014
# min           2.000000
# max         477.000000
val = tmp['value_as_number'].copy()
val[(val >= 100) | (val <= 60)] = 'panic'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'heart_rate_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
tp1 = tmp.copy()
w_id = wrong_id.copy()


# Glucose [Mass/volume] in Blood | Glucose lab
tmp=lab_concept.query('concept_name=="Glucose [Mass/volume] in Blood" | concept_name=="Glucose lab"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    675815.000000
# mean        153.136808
# min           1.000000
# max        4008.000000
val = tmp['value_as_number'].copy()
val[(val >= 500) | (val <= 40)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 99) | (tmp.value_as_number <= 65))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'glucose_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Hemoglobin
tmp=lab_concept.query('concept_name=="Hemoglobin"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8713.0')
tmp['unit_source_value'] = 'g/dL'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    345368.000000
# mean         12.007982
# min           0.500000
# max          24.700000
person = pd.read_csv('trac_ 3772_schizophrenia_person.csv',delimiter='|',parse_dates=True)
tmp = pd.merge(tmp, file[['measurement_id','person_id']])
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(val >= 21.4) | (val <= 7.0)] = 'panic'
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 15.9) | (tmp.value_as_number <= 11.1))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 17.7) | (tmp.value_as_number <= 13))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Hemoglobin_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)


# Platelet count
tmp=lab_concept.query('concept_name=="Platelet count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9444.0')
concept_sub.query('concept_id==8554.0')
tmp['unit_concept_id'] = 9444.0
tmp['unit_source_value'] = 'x10(9)/L'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    329717.000000
# mean        246.771880
# min           0.000000
# max        1457.000000
val = tmp['value_as_number'].copy()
val[(val <= 10) | (val >= 1000)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 450) | (tmp.value_as_number <= 140))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'platelet_count_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)


# eGFR with normals for black | eGFR with normals for non-black
tmp=lab_concept.query('concept_name=="eGFR with normals for black" | concept_name=="eGFR with normals for non-black"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mL/min/1.73m2'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    224918.000000
# mean         53.135930
# min           1.000000
# max         143.000000
val = tmp['value_as_number'].copy()
val[(val <= 15)] = 'panic'
val[val.apply(np.isreal) & tmp.value_as_number <= 60] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'eGFR_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Body height Measured
# tmp=lab_concept.query('concept_name=="Body height Measured"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==8582.0')
# concept_sub.query('concept_id==9327.0')
# tmp['unit_source_value'] = 'centimeter'
# tmp['unit_concept_id'] = 8582.0
# tmp.loc[(tmp.value_as_number<80) & (tmp.value_as_number>25),'value_as_number'] = \
#     tmp.loc[(tmp.value_as_number<80) & (tmp.value_as_number>25),'value_as_number']*2.54
# wrong_id = tmp.loc[~((tmp.value_as_number<203) & (tmp.value_as_number>63.5)),'measurement_id']
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# # Body weight
# tmp=lab_concept.query('concept_name=="Body weight"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==9529.0')
# concept_sub.query('concept_id==8739.0')
# tmp.loc[tmp.unit_concept_id==8739.0 ,'value_as_number'] = \
#     tmp.loc[tmp.unit_concept_id==8739.0 ,'value_as_number']/2.205
# tmp.loc[tmp.unit_source_value=='pound' ,'value_as_number'] = \
#     tmp.loc[tmp.unit_source_value=='pound' ,'value_as_number']/2.205
# tmp.loc[tmp.unit_source_value=='lb' ,'value_as_number'] = \
#     tmp.loc[tmp.unit_source_value=='lb' ,'value_as_number']/2.205
# wrong_id = tmp.loc[(tmp.value_as_number.isnull())|((tmp.unit_concept_id.isnull()) & tmp.unit_source_value.isnull()),'measurement_id']
# tmp['unit_source_value'] = 'kilogram'
# tmp['unit_concept_id'] = 9529.0
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Body mass index
tmp=lab_concept.query('concept_name=="Body mass index"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9531.0')
wrong_id = tmp.loc[(tmp.value_as_number>90) | tmp.value_as_number.isnull() | (tmp.value_as_number<=1) ,'measurement_id']
tmp['unit_source_value'] = 'kilogram per square meter'
tmp['unit_concept_id'] = 9531.0
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    78488.000000
# mean        29.981638
# min          1.100000
# max         86.849998
val = tmp['value_as_number'].copy()
val[(val >= 30)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 25) | (tmp.value_as_number <= 18.5))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'bmi_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    49476.000000
# mean       190.175075
# min         18.000000
# max       3368.000000
val = tmp['value_as_number'].copy()
val[(val >= 240)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 200) | (tmp.value_as_number <= 125))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'cho_mass_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol in HDL [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol in HDL [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    39728.000000
# mean        47.528494
# min          0.000000
# max        323.000000
person = pd.read_csv('trac_ 3772_schizophrenia_person.csv',delimiter='|',parse_dates=True)
tmp = pd.merge(tmp, file[['measurement_id','person_id']])
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(val >= 95) | (val <= 40)] = 'panic'
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 93) | (tmp.value_as_number <= 50))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 73) | (tmp.value_as_number <= 40))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'cho_hdl_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Cholesterol in LDL [Mass/volume] in Serum or Plasma
# tmp=lab_concept.query('concept_name=="Cholesterol in LDL [Mass/volume] in Serum or Plasma"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==8840.0')
# wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(w_id)]
lab_concept = to_cover(lab_concept, tp1)

tp1 = pd.merge(tp1, file[['measurement_id','person_id']])
tp1.to_csv('Cleaned_lab.csv', index=None)

# model
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
df = tp1[['val', 'person_id']].groupby(['person_id'])['val'].apply(lambda x: ' '.join(x)).reset_index()
data = df['val'].values.tolist()
res = [sent.split(' ') for sent in data]
dictionary = gensim.corpora.Dictionary(res)
count = 0
dictionary.filter_extremes(no_below=5, no_above=1.0, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]
num_topic = 5
seed = 11
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       iterations=500, random_state=seed)
# visualization
lda_display_bow = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=True)
pyLDAvis.save_html(lda_display_bow, 'lab_topic_model.html')
