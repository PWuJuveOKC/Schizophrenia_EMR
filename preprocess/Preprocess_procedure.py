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

remove_set = set(['NC111', 'NS111'])

file = file[~file.procedure_source_value.isin(remove_set)].copy()
replace_dict = dict()

# additional transforms
replace_dict['GZHZZZZ'] = '94.44'
replace_dict['Z51.12'] = 'V58.12'
replace_dict['Z00.00'] = 'V70.0'

file['cpt'] = file['procedure_source_value']
file['cpt'] = file.cpt.apply(lambda x: x.split(':')[-1])
file['cpt'] = file['cpt'].apply(lambda x: replace_dict[x] if x in replace_dict else x)

# procedure file with cleaned CPT code
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

# procedure file prior to hospitalization
dat_prior_sub.to_csv('src/int_data/procedure_cpt_prior_hosp.csv', index=None)
