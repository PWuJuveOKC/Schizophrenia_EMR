import pandas as pd
import gensim
import re
from gensim import corpora, models
import pyLDAvis.gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
dataName = 'procedure'

file = pd.read_csv(dataPath + prefix + dataName + '.csv', delimiter='|', low_memory=False)

# remove_set = set(['M998', '738.19', 'M99.8', 'Z00.00', 'Z02.79', 'V68.09',
#                   'V76.12', 'Z12.31', 'V58.82', 'Z46.82', 'I9:799.9', '799.9', 'R69',
#                   'V42.1', 'Z94.1', 'V70.0', 'V58.83', 'Z51.81', 'I10:R69', 'I9:738.19'])

#file = file[~file.condition_source_value.isin(remove_set)]
# replace_file = pd.read_csv('src/int_data/icd9_icd10_pw.csv')
# replace_dict = dict(zip(replace_file.ICD10, replace_file.ICD9))

file['cpt'] = file['procedure_source_value']
file['cpt'] = file.cpt.apply(lambda x: x.split(':')[-1])
#file['icd9'] = file['icd9'].apply(lambda x: replace_dict[x] if x in replace_dict else x)

# condition file with cleaned ICD9 code
file.to_csv('src/int_data/procedure_cpt.csv', index=None)

# Prior to hospitalization
window_size = 10
Hosp = pd.read_csv('src/int_data/hospitalization_window_' + str(window_size) + '.csv')
file_hosp = pd.merge(file, Hosp, on='person_id', how='left')
dat = file_hosp[file_hosp.hospitalization_hours.notnull()].copy()
dat['visit_start_date'] = pd.to_datetime(dat['visit_start_date'])
dat['procedure_date'] = pd.to_datetime(dat['procedure_date'])
dat_prior_hosp = dat[dat.procedure_date < dat.visit_start_date].copy()
dat_prior_sub = dat_prior_hosp[['procedure_occurrence_id', 'person_id', 'procedure_concept_id', 'cpt', 'procedure_date', 'visit_start_date', 'hospitalization_hours']].copy()

# condition file prior to hospitalization
dat_prior_sub.to_csv('src/int_data/procedure_cpt_prior_hosp.csv', index=None)