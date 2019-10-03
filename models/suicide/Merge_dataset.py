import pandas as pd
from functools import reduce

desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

feat_eng = 'pca'
hosp = pd.read_csv('src/int_data/hospitalization_window_10.csv', low_memory=False)

# demographics and restrict to schi cohort
dat_demo0 = pd.read_csv('src/int_data/preprocessed/demographics.csv')
schi_dat_v1 = pd.read_csv('src/int_data/visit/schi_cohort_v1.csv')
schi_dat_v2 = pd.read_csv('src/int_data/visit/schi_cohort_v2.csv')
schi_id = set(schi_dat_v1.person_id.values).union(set(set(schi_dat_v2.person_id.values)))
dat_demo = dat_demo0[dat_demo0.person_id.isin(schi_id)].copy()

dat_hosp = pd.merge(hosp, dat_demo, how='inner', on='person_id')
dat_hosp['visit_year'] =  pd.DatetimeIndex(dat_hosp['visit_start_date']).year
dat_hosp['age'] = dat_hosp['visit_year'] - dat_hosp['birth_year']
dat_hosp1 = dat_hosp[(dat_hosp.age >= 30) & (dat_hosp.age <= 100)].copy()
dat_base = dat_hosp1[['person_id', 'hospitalization_hours', 'age', 'ethnicity', 'race', 'gender']].copy()

dat_drug = pd.read_csv('src/int_data/feat_eng/medication_{}_feat.csv'.format(feat_eng))
dat_condition = pd.read_csv('src/int_data/feat_eng/condition_{}_feat.csv'.format(feat_eng))
dat_procedure = pd.read_csv('src/int_data/feat_eng/procedure_{}_feat.csv'.format(feat_eng))
dat_measure = pd.read_csv('src/int_data/feat_eng/lab_{}_feat.csv'.format(feat_eng))

# exclude dat_drug at this moment due to missing
topics_frames = [dat_base, dat_condition, dat_procedure, dat_measure]
topics_merged = reduce(lambda left, right: pd.merge(left, right, on=['person_id'], how='inner'), topics_frames)

del topics_merged['hospitalization_hours']
dat_suicide = pd.read_csv('src/int_data/suicide_count.csv')

topics_merged = pd.merge(topics_merged, dat_suicide, on='person_id', how='left')
topics_merged.fillna(0, inplace=True)
topics_merged['suicide'] = topics_merged['icd9'].map(lambda x: 1 if x >= 1 else 0)
del topics_merged['icd9']

df_shuffled = topics_merged.sample(frac=1, random_state=0)
df_shuffled.to_csv('src/int_data/feat_eng/merged_data_{}_suicide.csv'.format(feat_eng), index=None)