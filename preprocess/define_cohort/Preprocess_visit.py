import pandas as pd

dataPath = "./Data/trac_ 3772_schizophrenia/"
# dataPath = r"C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia/"
prefix = 'trac_ 3772_schizophrenia_'

parse_dates = ['visit_start_date', 'visit_start_datetime', 'visit_end_date', 'visit_end_datetime']
file = pd.read_csv(dataPath + prefix + 'visit.csv', delimiter='|', date_parser=parse_dates, low_memory=False)
concept = pd.read_csv(dataPath + prefix + 'concept.csv',delimiter='|',date_parser=parse_dates,low_memory=False)

# (1183223, 12)
print(file.shape)
print(file.dtypes)

# 17499
personId = file.person_id.unique()
# check missing portion
file.isnull().sum()

out_set = {8756, 8809, 8870, 9202, 9203, 4150254, 4213245, 4214465, 4214691, 4249897, 45765807, 45765824,
           45765829, 45765841, 45765869, 45765880, 45765994, 45773134}

# Step 1: separate in or out patient files
# 72421 obs
file_in = file[(file.visit_concept_id.isin([8717, 9201])) | (file.visit_type_concept_id == 9201)].copy()

# 1085466 obs
file_out = file[(file.visit_concept_id.isin(out_set)) |
                (file.visit_type_concept_id.isin([9202, 9203, 42898160]))].copy()

# dump to temp file
# file_in.to_csv('src/int_data/visit/visit_in.csv', index=None)
# file_out.to_csv('src/int_data/visit/visit_out.csv', index=None)
file_in.to_csv(dataPath+r'/visit/visit_in.csv', index=None)
file_out.to_csv(dataPath+r'/visit/visit_out.csv', index=None)