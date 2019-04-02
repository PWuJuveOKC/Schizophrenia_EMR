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
concept_sub = concept_sub[['concept_id', 'concept_name']].copy()
concept_sub.columns = [['drug_concept_id', 'drug_name']]

file = file[file.drug_concept_id != 0]
med_concept = pd.merge(file, concept_sub, on='drug_concept_id', how='left')

# first generation meds
drug_first_list = ['chlorpromazine', 'fluphenazine', 'haloperidol', 'loxapine', 'molindone', 'perphenazine',
                   'pimozide', 'prochlorperazine', 'thioridazine', 'trifluoperazine', 'triflupromazine']
for drug_first in drug_first_list:
    drug_dat = med_concept[med_concept.drug_name.apply(lambda x: drug_first in x.lower())].copy()
    drug_dat_id = drug_dat.person_id.unique()
    print('drug_name: ', drug_first)
    print('observations #: ', drug_dat.shape[0]),
    print('unique id #: ', len(drug_dat_id)),
    print('----------------------------------')

# second generation meds
drug_second_list = ['aripiprazole', 'clozapine', 'iloperidone', 'olanzapine', 'paliperidone', 'quetiapine',
                       'risperidone', 'ziprasidone']
for drug_second in drug_second_list:
    drug_dat = med_concept[med_concept.drug_name.apply(lambda x: drug_second in x.lower())].copy()
    drug_dat_id = drug_dat.person_id.unique()
    print('drug_name: ', drug_second)
    print('observations #: ', drug_dat.shape[0]),
    print('unique id #: ', len(drug_dat_id)),
    print('----------------------------------')