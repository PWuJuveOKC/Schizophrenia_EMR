# v3: based on condition_type_concept_id

# v1: definition based on start_date

import pandas as pd

visit_in = pd.read_csv('src/int_data/visit/visit_in.csv')
schi_code = pd.read_csv('src/int_data/visit/schi_code.csv')

condition = pd.read_csv('src/int_data/preprocessed/condition_icd9.csv', low_memory=False)
condition_sub = condition[['person_id', 'condition_start_date', 'visit_occurrence_id', 'icd9',
                           'condition_type_concept_id']].copy()

visit_in_sub = visit_in[['person_id', 'visit_start_date', 'visit_occurrence_id']].copy()
visit_in_sub.columns = ['person_id', 'start_date', 'visit_occurrence_id']

condition_sub.columns = ['person_id', 'start_date', 'visit_occurrence_id', 'icd9', 'condition_type_concept_id']

# Step 2: restrict to schi diagnosis based on condition type concept id
condition_schi = condition_sub[condition_sub.condition_type_concept_id.isin([42894222, 44786627])].copy()

# Step 3: merge files respectively: inpatient with schi diagnosis
match_var1 = 'visit_occurrence_id'
schi_in = pd.merge(visit_in_sub, condition_schi, on=['person_id', match_var1], how='inner')


# 13242
uid_schi_in = set(schi_in.person_id.unique())

# dump this file to int_data
schi_in.to_csv('src/int_data/visit/preprocessed/schi_cohort_inpatient.csv', index=None)

