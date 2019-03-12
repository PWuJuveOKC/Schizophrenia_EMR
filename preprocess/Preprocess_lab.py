import pandas as pd
import numpy as np
import re
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
from gensim.test.utils import datapath
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

## Load Data
dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
dataName = 'measurement'
file = pd.read_csv(dataPath + prefix + dataName +'.csv',delimiter='|')

## Some check
print(file.shape) #(21734609, 18)
print(file.dtypes)

personId = file.person_id.unique() #16900
## check missing portion
file.isnull().sum()

parse_dates = ['valid_start_date', 'valid_end_date' ]
concept = pd.read_csv(dataPath + prefix + 'concept' +'.csv', delimiter='|', date_parser=parse_dates, low_memory=False)

concept_sub = concept[['concept_id', 'concept_name']]
concept_sub.columns = ['measurement_concept_id', 'concept_name']

lab_concept = pd.merge(file, concept_sub, on='measurement_concept_id', how='left')
lab_concept = lab_concept[lab_concept.measurement_concept_id != 0]

window_size = 10
Hosp = pd.read_csv('src/int_data/hospitalization_window_' + str(window_size) + '.csv')

file_hosp = pd.merge(lab_concept, Hosp, on='person_id', how='left')
dat = file_hosp[file_hosp.hospitalization_hours.notnull()]

dat['measurement_date'] = pd.to_datetime(dat['measurement_date'])
dat['visit_start_date'] = pd.to_datetime(dat['visit_start_date'])

dat_prior_hosp = dat[dat.measurement_date < dat.visit_start_date] #5857 unique ids
dat_prior_hosp.to_csv('src/int_data/lab_prior_hosp.csv', index=None)