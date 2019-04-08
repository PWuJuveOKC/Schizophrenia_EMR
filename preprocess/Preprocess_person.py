import pandas as pd
import numpy as np



dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
dataName = 'person'

parse_dates = ['birth_date_time']
file = pd.read_csv(dataPath + prefix + dataName +'.csv',delimiter='|',date_parser=parse_dates,low_memory=False)

file_demo = file[['person_id', 'gender_concept_id', 'year_of_birth', 'race_concept_id', 'ethnicity_concept_id']].copy()

concept = pd.read_csv(dataPath + prefix + 'concept' + '.csv', delimiter='|', date_parser=parse_dates, low_memory=False)
concept = concept[['concept_id', 'concept_name']].copy()

# check race & ethnicity & gender
print(file_demo.race_concept_id.value_counts())
print(file_demo.ethnicity_concept_id.value_counts())
print(file_demo.gender_concept_id.value_counts())

race_id = set(file_demo.race_concept_id.unique())
ethnicity_id = set(file_demo.ethnicity_concept_id.unique())
gender_id = set(file_demo.gender_concept_id.unique())
race_concept = concept[concept.concept_id.isin(race_id)].copy()
ethnicity_concept = concept[concept.concept_id.isin(ethnicity_id)].copy()
gender_concept = concept[concept.concept_id.isin(gender_id)].copy()

# check age and correct
print(file_demo.year_of_birth.describe())
file_demo['birth_year'] = np.where(file_demo.year_of_birth <= 1860,
                                   file_demo['year_of_birth'] + 100, file_demo['year_of_birth'])

# -1: missing, 1: white, 0: others
file_demo['race'] = np.where(file_demo.race_concept_id.isin({44814653, 0, 8552}), -1,
                             np.where(file_demo.race_concept_id == 8527, 1, 0))

# - 1: missing, 1: hispanic, 0: non-hispanic
file_demo['ethnicity'] = np.where(file_demo.ethnicity_concept_id == 0, -1,
                                  np.where(file_demo.ethnicity_concept_id == 38003563, 1, 0))

# - 1: missing, 1: male, 0: female
file_demo['gender'] = np.where(file_demo.gender_concept_id == 8551, -1,
                                  np.where(file_demo.gender_concept_id == 8507, 1, 0))

file_demo1 = file_demo[['person_id', 'gender', 'race', 'ethnicity', 'birth_year']]
file_demo1.to_csv('src/int_data/preprocessed/demographics.csv', index=None)

