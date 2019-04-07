# v1: definition based on start_date
import pandas as pd
dataPath = r'src/int_data/'
# dataPath = r"C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia/"
visit_in = pd.read_csv(dataPath+r'/visit/visit_in.csv')
visit_out = pd.read_csv(dataPath+r'/visit/visit_out.csv')
schi_code = pd.read_csv(dataPath+r'/visit/schi_code.csv')

# condition = pd.read_csv('src/int_data/condition_icd9.csv', low_memory=False)
condition = pd.read_csv(dataPath+r'/condition_icd9.csv', low_memory=False)
condition_sub = condition[['person_id', 'condition_start_date', 'icd9']].copy()

visit_in_sub = visit_in[['person_id', 'visit_start_date']].copy()
visit_in_sub.columns = ['person_id', 'start_date']
visit_out_sub = visit_out[['person_id', 'visit_start_date', 'visit_occurrence_id']].copy()
visit_out_sub.columns = ['person_id', 'start_date', 'occurrence_id']

condition_sub.columns = ['person_id', 'start_date', 'icd9']

# Step 2: restrict to schi diagnosis
schi_code_set = set([str(x) for x in schi_code.ICD9_Schi.values])
condition_schi = condition_sub[condition_sub.icd9.isin(schi_code_set)].copy()

# Step 3: merge files respectively: inpatient with schi diagnosis / outpatient with schi_diagnosis (at least twice)
# specify inpatient diagnosis = 1 for var "in_out_diag"
match_var = 'start_date'
schi_in = pd.merge(visit_in_sub, condition_schi, on=['person_id', match_var], how='inner')
schi_in['in_out_diag'] = 1
schi_out0 = pd.merge(visit_out_sub, condition_schi, on=['person_id', match_var], how='inner')
schi_out = schi_out0[schi_out0.duplicated('person_id') | schi_out0.duplicated('person_id', keep='last')].copy()
schi_out['in_out_diag'] = 0

# 2752
uid_schi_in = set(schi_in.person_id.unique())
# 2371
uid_schi_out = set(schi_out.person_id.unique())
# 3882
uid_schi = uid_schi_in.union(uid_schi_out)

# Step 4: concatenate schi_in and schi_out
schi = pd.concat([schi_in, schi_out])
schi_sort = schi.sort_values(by = ['person_id', match_var])

# dump this file to int_data
schi_sort.to_csv('src/int_data/visit/schi_cohort_v1.csv', index=None)
# schi_sort.to_csv(dataPath+'/visit/schi_cohort_v1.csv', index=None)

# check missing occurence id and start date in condition file
mis_occur = condition[condition.visit_occurrence_id.isnull()].copy()
mis_occur_id = mis_occur.person_id.unique()

mis_date = condition[condition.condition_start_date.isnull()].copy()
