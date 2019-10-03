import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
dataName = 'condition'

file = pd.read_csv(dataPath + prefix + dataName + '.csv', delimiter='|', low_memory=False)

xls = pd.ExcelFile('Data/Suicide.xlsx')
df1 = pd.read_excel(xls, 'Suicide')
df2 = pd.read_excel(xls, 'Suicidal')
df3 = pd.read_excel(xls, 'Self-Harm')
df4 = pd.read_excel(xls, 'Self-Injury')

# suicide concept code
sui_code = set(df1.Id).union(set(df2.Id)).union(set(df3.Id)).union(set(df4.Id))

hosp = pd.read_csv('src/int_data/hospitalization_window_10.csv', low_memory=False)

# suicide code in condition
file_condition = pd.read_csv('src/int_data/preprocessed/condition_icd9.csv', low_memory=False)
hosp_condition = pd.merge(file_condition, hosp, on='person_id', how='left')
dat = hosp_condition[hosp_condition.hospitalization_hours.notnull()].copy()
dat['visit_start_date'] = pd.to_datetime(dat['visit_start_date'])
dat['condition_start_date'] = pd.to_datetime(dat['condition_start_date'])
# post first visit
dat_post_hosp = dat[dat.condition_start_date >= dat.visit_start_date].copy()
dat_post_hosp = dat_post_hosp[['person_id', 'condition_concept_id', 'condition_start_date',
                               'visit_start_date', 'icd9']].copy()
dat_sui_con = dat_post_hosp[dat_post_hosp.condition_concept_id.isin(sui_code)].copy()

dat_suicide = dat_sui_con[['person_id', 'icd9']].groupby('person_id').aggregate('count')
dat_suicide['person_id'] = dat_suicide.index
dat_suicide.to_csv('src/int_data/suicide_count.csv', index=None)
