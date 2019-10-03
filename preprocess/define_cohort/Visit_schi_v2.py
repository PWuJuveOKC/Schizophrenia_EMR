# v2: definition based on visit_occurrence_id

import pandas as pd

visit_in = pd.read_csv('src/int_data/visit/visit_in.csv')
visit_out = pd.read_csv('src/int_data/visit/visit_out.csv')
schi_code = pd.read_csv('src/int_data/visit/schi_code.csv')

condition = pd.read_csv('src/int_data/condition_icd9.csv', low_memory=False)
condition_sub = condition[['person_id', 'condition_start_date', 'icd9', 'visit_occurrence_id']].copy()

visit_in_sub = visit_in[['person_id', 'visit_start_date', 'visit_occurrence_id']].copy()
visit_in_sub.columns = ['person_id', 'start_date', 'occurrence_id']
visit_out_sub = visit_out[['person_id', 'visit_start_date', 'visit_occurrence_id']].copy()
visit_out_sub.columns = ['person_id', 'start_date', 'occurrence_id']

condition_sub.columns = ['person_id', 'condition_start_date', 'icd9', 'occurrence_id']

# Step 2: restrict to schi diagnosis
schi_code_set = set([str(x) for x in schi_code.ICD9_Schi.values])
condition_schi = condition_sub[condition_sub.icd9.isin(schi_code_set)].copy()

# Step 3: merge files respectively: inpatient with schi diagnosis / outpatient with schi_diagnosis (at least twice)
# specify inpatient diagnosis = 1 for var "in_out_diag"
match_var = 'occurrence_id'
schi_in = pd.merge(visit_in_sub, condition_schi, on=['person_id', match_var], how='inner')
schi_in['in_out_diag'] = 1
schi_out0 = pd.merge(visit_out_sub, condition_schi, on=['person_id', match_var], how='inner')
schi_out = schi_out0[schi_out0.duplicated('person_id') | schi_out0.duplicated('person_id', keep='last')].copy()
schi_out['in_out_diag'] = 0

# 4596
uid_schi_in = set(schi_in.person_id.unique())
# 1705
uid_schi_out = set(schi_out.person_id.unique())
# 5280
uid_schi = uid_schi_in.union(uid_schi_out)

# Step 4: concatenate schi_in and schi_out
schi = pd.concat([schi_in, schi_out])
schi_sort = schi.sort_values(by = ['person_id', match_var])

# dump this file to int_data
schi_sort.to_csv('src/int_data/visit/schi_cohort_v2.csv', index=None)
