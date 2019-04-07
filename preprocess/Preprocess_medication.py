import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


dataPath = "./Data/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'
domain = 'drug'

file = pd.read_csv(dataPath + prefix + domain + '.csv', delimiter='|', low_memory=False)

# medication to concept name mapping
med_concept_id_uni = set(file[domain + '_concept_id'].values)

parse_dates = ['valid_start_date', 'valid_end_date']
concept = pd.read_csv(dataPath + prefix + 'concept' + '.csv', delimiter='|', date_parser=parse_dates, low_memory=False)
concept_sub = concept[concept['concept_id'].isin(med_concept_id_uni)].copy()
concept_sub = concept_sub[['concept_id', 'concept_name']]
# concept_sub.to_csv('src/int_data/medication_id_to_name.csv', index=None)

concept_id_freq = pd.DataFrame(file.drug_concept_id.value_counts())
concept_id_freq.columns = [['frequency']]
concept_id_freq['concept_id'] = concept_id_freq.index
concept_id_name = pd.merge(concept_id_freq, concept_sub, on='concept_id', how='inner')
concept_id_name.to_csv('src/int_data/medication_id_to_name.csv', index=None)

# Prior to hospitalization
window_size = 10
Hosp = pd.read_csv('src/int_data/hospitalization_window_' + str(window_size) + '.csv')
file_hosp = pd.merge(file, Hosp, on='person_id', how='left')
dat = file_hosp[file_hosp.hospitalization_hours.notnull()].copy()
dat['visit_start_date'] = pd.to_datetime(dat['visit_start_date'])
dat['drug_exposure_start_date'] = pd.to_datetime(dat['drug_exposure_start_date'])
dat_prior_hosp = dat[dat.drug_exposure_start_date < dat.visit_start_date].copy()
dat_prior_sub = dat_prior_hosp[['drug_exposure_id', 'person_id', 'drug_concept_id', 'drug_exposure_start_date', 'visit_start_date', 'hospitalization_hours']].copy()
dat_prior_sub = dat_prior_sub[dat_prior_sub.drug_concept_id != 0].copy()

# procedure file prior to hospitalization
dat_prior_sub.to_csv('src/int_data/preprocessed/medication_prior_hosp.csv', index=None)