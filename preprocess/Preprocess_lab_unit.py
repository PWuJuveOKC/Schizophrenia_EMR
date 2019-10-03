import pandas as pd
import warnings
import numpy as np
import os
os.chdir("C:/Users/Peter Xu/Desktop/Yuanjia/schi/trac_ 3772_schizophrenia")
warnings.filterwarnings("ignore", category=DeprecationWarning)
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# Load Data
# file1 = pd.read_csv('trac_ 3772_schizophrenia_measurement.csv',delimiter='|',nrows=300,
                    # dtype={'visit_occurrence_id':object},parse_dates=True)
file = pd.read_csv('trac_ 3772_schizophrenia_measurement.csv',delimiter='|',
                    dtype={'visit_occurrence_id':object},parse_dates=True)


parse_dates = ['valid_start_date', 'valid_end_date']
concept = pd.read_csv('trac_ 3772_schizophrenia_concept.csv', delimiter='|', date_parser=parse_dates, low_memory=False)

concept_sub = concept[['concept_id', 'concept_name']]

# lab_concept1 = pd.merge(file1, concept_sub, left_on='measurement_concept_id',right_on='concept_id', how='left')
lab_concept = pd.merge(file, concept_sub, left_on='measurement_concept_id',right_on='concept_id', how='left')
# lab_concept1 = lab_concept1[lab_concept1.measurement_concept_id != 0]
lab_concept = lab_concept[lab_concept.measurement_concept_id != 0]


to_take = ['measurement_id','value_as_number','unit_concept_id','unit_source_value']
person = pd.read_csv('trac_ 3772_schizophrenia_person.csv',delimiter='|',parse_dates=True)
def to_cover(org, new):
    tt = pd.merge(org[['measurement_id','value_as_number','unit_concept_id','unit_source_value']], new, on='measurement_id', how='left')
    org.loc[~tt.value_as_number_y.isnull().values,'value_as_number']=tt.loc[~tt.value_as_number_y.isnull(),'value_as_number_y'].tolist()
    org.loc[~tt.value_as_number_y.isnull().values,'unit_concept_id']=tt.loc[~tt.value_as_number_y.isnull(),'unit_concept_id_y'].tolist()
    org.loc[~tt.value_as_number_y.isnull().values,'unit_source_value']=tt.loc[~tt.value_as_number_y.isnull(),'unit_source_value_y'].tolist()
    return org
# Body temperature
tmp=lab_concept.query('concept_name=="Body temperature"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8653.0')
concept_sub.query('concept_id==9289.0')
tmp['unit_source_value'] = 'degree Celsius'
tmp['unit_concept_id'] = 8653.0
tmp.loc[(tmp.value_as_number<122) & (tmp.value_as_number>86),'value_as_number'] = \
    (tmp.loc[(tmp.value_as_number<122) & (tmp.value_as_number>86),'value_as_number']-32)*5/9
wrong_id = tmp.loc[~((tmp.value_as_number<50) & (tmp.value_as_number>30)),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# mean         36.791336
# min          30.333333
# max          45.000000
val = tmp['value_as_number'].copy()
val[(val >= 41.5) | (val <= 35)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'body_tmp_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
tp1 = tmp.copy()
w_id = wrong_id.copy()

# Respiratory rate
tmp=lab_concept.query('concept_name=="Respiratory rate"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
wrong_id = tmp.loc[(tmp.value_as_number.isnull()) | (tmp.value_as_number <=0) | (tmp.value_as_number >= 150),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# mean         18.990322
# min           1.000000
# max         149.000000
val = tmp['value_as_number'].copy()
val[(val >= 20) | (val <= 8)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 18) | (tmp.value_as_number <= 12))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'R_rate_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# BP systolic
tmp=lab_concept.query('concept_name=="BP systolic"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8876')
tmp['unit_source_value'] = 'millimeter mercury column'
wrong_id = tmp.loc[(tmp.value_as_number.isnull()) | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# mean        124.590914
# min           1.000000
# max        1110.000000
val = tmp['value_as_number'].copy()
val[(val >= 180) | (val <= 90)] = 'panic'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'BP_systolic_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# BP diastolic
tmp=lab_concept.query('concept_name=="BP diastolic"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8876')
tmp['unit_source_value'] = 'millimeter mercury column'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    812204.000000
# mean         70.814018
# min           1.000000
# max         996.000000
val = tmp['value_as_number'].copy()
val[(val >= 120) | (val <= 60)] = 'panic'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'BP_diastolic_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Heart rate
tmp=lab_concept.query('concept_name=="Heart rate"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8483.0')
tmp['unit_source_value'] = 'counts per minute'
wrong_id = tmp.loc[~((tmp.value_as_number<500) & (tmp.value_as_number>1)),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    766593.000000
# mean         84.302014
# min           2.000000
# max         477.000000
val = tmp['value_as_number'].copy()
val[(val >= 100) | (val <= 60)] = 'panic'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'heart_rate_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)


# Glucose [Mass/volume] in Blood | Glucose lab
tmp=lab_concept.query('concept_name=="Glucose [Mass/volume] in Blood" | concept_name=="Glucose lab"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    675815.000000
# mean        153.136808
# min           1.000000
# max        4008.000000
val = tmp['value_as_number'].copy()
val[(val >= 500) | (val <= 40)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 99) | (tmp.value_as_number <= 65))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'glucose_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Hematocrit
tmp=lab_concept.query('concept_name=="Hematocrit"')[to_take + ['person_id']]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8554.0')
tmp['unit_source_value'] = '%'
tmp['unit_concept_id'] = 8554.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    347087.000000
# mean         36.236387
# min           2.400000
# max          75.000000
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(val >= 65.0) | (val <= 21.0)] = 'panic'
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 46.6) | (tmp.value_as_number <= 34.0))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 51.0) | (tmp.value_as_number <= 37.5))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Hematocrit_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)


# Hemoglobin
tmp=lab_concept.query('concept_name=="Hemoglobin"')[to_take + ['person_id']]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8713.0')
tmp['unit_source_value'] = 'g/dL'
tmp['unit_concept_id'] = 8713.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    345368.000000
# mean         12.007982
# min           0.500000
# max          24.700000
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(val >= 21.4) | (val <= 7.0)] = 'panic'
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 15.9) | (tmp.value_as_number <= 11.1))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 17.7) | (tmp.value_as_number <= 13))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Hemoglobin_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)


# Platelet count
tmp=lab_concept.query('concept_name=="Platelet count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9444.0')
concept_sub.query('concept_id==8554.0')
tmp['unit_concept_id'] = 9444.0
tmp['unit_source_value'] = 'x10(9)/L'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    329717.000000
# mean        246.771880
# min           0.000000
# max        1457.000000
val = tmp['value_as_number'].copy()
val[(val <= 10) | (val >= 1000)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 450) | (tmp.value_as_number <= 140))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'platelet_count_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

 # Erythrocyte mean corpuscular hemoglobin [Entitic mass] by Automated count
tmp=lab_concept.query('concept_name=="Erythrocyte mean corpuscular hemoglobin [Entitic mass] by Automated count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8564.0')
tmp['unit_source_value'] = 'picogram'
tmp['unit_concept_id'] = 8564.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    330262.000000
# mean         29.209371
# min          13.200000
# max          60.500000
val = tmp['value_as_number'].copy()
val[(val >= 33) | (val <= 27)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Erythrocyte_Entitic_count_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Leukocytes [#/volume] in Blood by Automated count
tmp=lab_concept.query('concept_name=="Leukocytes [#/volume] in Blood by Automated count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9444.0')
tmp['unit_source_value'] = 'x10(9)/L'
tmp['unit_concept_id'] = 9444.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    330524.000000
# mean          8.437866
# min           0.030000
# max         435.900000
val = tmp['value_as_number'].copy()
val[(val >= 49.9) | (val <= 1.1)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 10.8) | (tmp.value_as_number <= 3.4))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Leukocytes_Blood_count_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)


# Erythrocytes [#/volume] in Blood by Automated count
tmp=lab_concept.query('concept_name=="Erythrocytes [#/volume] in Blood by Automated count"')[to_take + ['person_id']]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8734.0')
tmp['unit_source_value'] = 'x10(9)/L'
tmp['unit_concept_id'] = 8734.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    330603.000000
# mean          4.127480
# min           0.290000
# max           8.540000
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 5.13) | (tmp.value_as_number <= 3.92))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 5.65) | (tmp.value_as_number <= 4.35))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Erythrocytes_Blood_count_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)\


# Erythrocyte mean corpuscular hemoglobin concentration [Mass/volume] by Automated count
tmp=lab_concept.query('concept_name=="Erythrocyte mean corpuscular hemoglobin concentration [Mass/volume] by Automated count"')[to_take + ['person_id']]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8713.0')
tmp['unit_source_value'] = 'g/dL'
tmp['unit_concept_id'] = 8713.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    330347.000000
# mean         33.049497
# min          17.800000
# max          55.800000
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & (tmp.value_as_number <= 12.0)] = 'panic'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & (tmp.value_as_number <= 13.5)] = 'panic'
val[(val.apply(np.isreal)) & ((tmp.value_as_number >= 34) | (tmp.value_as_number <= 30))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Erythrocyte_concentration_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Erythrocyte mean corpuscular volume [Entitic volume] by Automated count
tmp=lab_concept.query('concept_name=="Erythrocyte mean corpuscular volume [Entitic volume] by Automated count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8583.0')
tmp['unit_source_value'] = 'fL'
tmp['unit_concept_id'] = 8583.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    330454.000000
# mean         88.330236
# min          50.100000
# max         144.700000
val = tmp['value_as_number'].copy()
val[((tmp.value_as_number >= 95) | (tmp.value_as_number <= 80))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Erythrocyte_Entitic_count_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Erythrocyte distribution width [Ratio] by Automated count
tmp=lab_concept.query('concept_name=="Erythrocyte distribution width [Ratio] by Automated count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8554.0')
tmp['unit_source_value'] = '%'
tmp['unit_concept_id'] = 8554.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    328507.000000
# mean         14.802117
# min           8.700000
# max          53.600000
val = tmp['value_as_number'].copy()
val[(val >= 14.5) | (val <= 11.5)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Erythrocyte_Ratio_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Potassium serum/plasma
tmp=lab_concept.query('concept_name=="Potassium serum/plasma" | concept_name=="Potassium [Moles/volume] in Blood"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8753.0')
tmp['unit_source_value'] = 'mM/L'
tmp['unit_concept_id'] = 8753.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    335752.000000
# mean          4.245378
# min           1.100000
# max         114.000000
val = tmp['value_as_number'].copy()
val[(val >= 6) | (val <= 3)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 5.2) | (tmp.value_as_number <= 3.5))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Potassium_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Platelet mean volume [Entitic volume] in Blood
# tmp=lab_concept.query('concept_name=="Platelet mean volume [Entitic volume] in Blood"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==8583.0')
# tmp['unit_source_value'] = 'fL'
# tmp['unit_concept_id'] = 8583.0
# wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
# tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
# tmp.value_as_number.describe()
# # count    317530.000000
# # mean         10.519633
# # min           4.400000
# # max          23.700000
# val = tmp['value_as_number'].copy()
# val[(val >= 41.5) | (val <= 35)] = 'panic'
# val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
# val[val.apply(np.isreal)] = 'normal'
# tmp['val'] = 'Platelet_Entitic_' + val
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Creatinine serum/plasma
tmp=lab_concept.query('concept_name=="Creatinine serum/plasma"')[to_take + ['person_id']]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    319722.000000
# mean          1.285983
# min           0.100000
# max          36.630000
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 90) | (tmp.value_as_number <= 45))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 110) | (tmp.value_as_number <= 60))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Creatinine_plasma_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Sodium serum/plasma | Sodium [Moles/volume] in Blood
tmp=lab_concept.query('concept_name=="Sodium serum/plasma" | concept_name=="Sodium [Moles/volume] in Blood"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8753.0')
tmp['unit_source_value'] = 'mM/L'
tmp['unit_concept_id'] = 8753.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    335901.000000
# mean        138.036832
# min           7.000000
# max         200.000000
val = tmp['value_as_number'].copy()
val[(val >= 160) | (val <= 125)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 144) | (tmp.value_as_number <= 134))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Sodium_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Urea nitrogen serum/plasma
tmp=lab_concept.query('concept_name=="Urea nitrogen serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    319560.000000
# mean         19.658383
# min           1.000000
# max         293.000000
val = tmp['value_as_number'].copy()
val[(val >= 20) | (val <= 7)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Urea_nitrogen_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Chloride serum/plasma
tmp=lab_concept.query('concept_name=="Chloride serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8753.0')
tmp['unit_source_value'] = 'mM/L'
tmp['unit_concept_id'] = 8753.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    320149.000000
# mean        102.761555
# min          29.000000
# max         150.000000
val = tmp['value_as_number'].copy()
val[(val >= 120) | (val <= 80)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 107) | (tmp.value_as_number <= 98))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Chloride_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Calcium serum/plasma serum/plasma
tmp=lab_concept.query('concept_name=="Calcium serum/plasma serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    303585.000000
# mean          8.940735
# min           0.100000
# max          60.200000
val = tmp['value_as_number'].copy()
val[(val >= 13) | (val <= 6.5)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 10.2) | (tmp.value_as_number <= 8.7))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Calcium_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# eGFR with normals for black | eGFR with normals for non-black
tmp=lab_concept.query('concept_name=="eGFR with normals for black" | concept_name=="eGFR with normals for non-black"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mL/min/1.73m2'
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    224918.000000
# mean         53.135930
# min           1.000000
# max         143.000000
val = tmp['value_as_number'].copy()
val[(val <= 15)] = 'panic'
val[val.apply(np.isreal) & tmp.value_as_number <= 60] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'eGFR_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Nucleated erythrocytes [#/volume] in Blood
# tmp=lab_concept.query('concept_name=="Nucleated erythrocytes [#/volume] in Blood"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==9444.0')
# tmp['unit_source_value'] = 'x10(9)/L'
# tmp['unit_concept_id'] = 9444.0
# wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
# tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
# tmp.value_as_number.describe()
# # count    12856.000000
# # mean         0.312359
# # min          0.020000
# # max        400.000000
# val = tmp['value_as_number'].copy()
# val[(val >= 41.5) | (val <= 35)] = 'panic'
# val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
# val[val.apply(np.isreal)] = 'normal'
# tmp['val'] = 'Nucleated_Blood_' + val
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# # Nucleated erythrocytes/100 leukocytes [Ratio] in Blood
# tmp=lab_concept.query('concept_name=="Nucleated erythrocytes/100 leukocytes [Ratio] in Blood"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==8653.0')
# concept_sub.query('concept_id==9289.0')
# tmp['unit_source_value'] = 'degree Celsius'
# tmp['unit_concept_id'] = 8653.0
# wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
# tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
# tmp.value_as_number.describe()
# # mean         36.791336
# # min          30.333333
# # max          45.000000
# val = tmp['value_as_number'].copy()
# val[(val >= 41.5) | (val <= 35)] = 'panic'
# val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
# val[val.apply(np.isreal)] = 'normal'
# tmp['val'] = 'Nucleated_erythrocytes_Ratio_' + val
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Aspartate aminotransferase serum/plasma
tmp=lab_concept.query('concept_name=="Aspartate aminotransferase serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_name=="U/L"')
concept_sub.query('concept_id==4118000')
tmp['unit_source_value'] = 'U/L'
tmp['unit_concept_id'] = 4118000
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    207937.000000
# mean         44.195978
# min           0.000000
# max       20000.000000
val = tmp['value_as_number'].copy()
val[(val >= 48)] = 'panic'
val[val.apply(np.isreal) & (tmp.value_as_number >= 40)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Aspartate_aminotransferase_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Dry body weight Measured
# tmp=lab_concept.query('concept_name=="Dry body weight Measured"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==9529')
# tmp['unit_source_value'] = 'kg'
# tmp['unit_concept_id'] = 9529.0
# wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
# tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
# tmp.value_as_number.describe()
# # count    189707.000000
# # mean         79.297025
# # min           0.170000
# # max         980.000000
# val = tmp['value_as_number'].copy()
# val[(val >= 1)] = 'abnormal'
# val[val.apply(np.isreal)] = 'normal'
# tmp['val'] = 'Dry_body_' + val
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Alanine aminotransferase serum/plasma
tmp=lab_concept.query('concept_name=="Alanine aminotransferase serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'U/L'
tmp['unit_concept_id'] = 4118000
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    187777.000000
# mean         38.647811
# min           0.000000
# max        9734.000000
val = tmp['value_as_number'].copy()
val[(val >= 55)] = 'panic'
val[val.apply(np.isreal) & (tmp.value_as_number >= 48)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Alanine_aminotransferase_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Albumin serum/plasma
tmp=lab_concept.query('concept_name=="Albumin serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8713.0')
concept_sub.query('concept_id==8840.0')
tmp.loc[(tmp.unit_concept_id == 8840),'value_as_number'] = \
    (tmp.loc[(tmp.unit_concept_id == 8840),'value_as_number'])/1000
tmp['unit_source_value'] = 'g/dL'
tmp['unit_concept_id'] = 8713.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    186997.000000
# mean          3.945976
# min           0.200000
# max           8.700000
val = tmp['value_as_number'].copy()
val[(val >= 5) | (val <= 3)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Albumin_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Total Bilirubin serum/plasma
tmp=lab_concept.query('concept_name=="Total Bilirubin serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    185808.000000
# mean          0.785986
# min           0.000000
# max          56.100000
val = tmp['value_as_number'].copy()
val[(val >= 15) | (val <= 35)] = 'panic'
val[val.apply(np.isreal) & (tmp.value_as_number >= 1)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Bilirubin_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Alkaline phosphatase serum/plasma
tmp=lab_concept.query('concept_name=="Alkaline phosphatase serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'U/L'
tmp['unit_concept_id'] = 4118000
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    184387.000000
# mean         95.145748
# min           2.000000
# max        5738.000000
val = tmp['value_as_number'].copy()
val[(val >= 115) | (val <= 30)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Alkaline_phosphatase_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Direct bilirubin serum/plasma
tmp=lab_concept.query('concept_name=="Direct bilirubin serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    183065.000000
# mean          0.259282
# min           0.000000
# max          39.800000
val = tmp['value_as_number'].copy()
val[(val >= 0.2) | (val <= 35)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Direct_bilirubin_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Protein serum/plasma
tmp=lab_concept.query('concept_name=="Protein serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'g/dL'
tmp['unit_concept_id'] = 8713.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    177797.000000
# mean          6.977875
# min           0.200000
# max          20.000000
val = tmp['value_as_number'].copy()
val[(val >= 8.3) | (val <= 6)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Protein_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Body height Measured
# tmp=lab_concept.query('concept_name=="Body height Measured"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==8582.0')
# concept_sub.query('concept_id==9327.0')
# tmp['unit_source_value'] = 'centimeter'
# tmp['unit_concept_id'] = 8582.0
# tmp.loc[(tmp.value_as_number<80) & (tmp.value_as_number>25),'value_as_number'] = \
#     tmp.loc[(tmp.value_as_number<80) & (tmp.value_as_number>25),'value_as_number']*2.54
# wrong_id = tmp.loc[~((tmp.value_as_number<203) & (tmp.value_as_number>63.5)),'measurement_id']
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Bicarbonate [Moles/volume] in Serum
tmp=lab_concept.query('concept_name=="Bicarbonate [Moles/volume] in Serum"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mM/L'
tmp['unit_concept_id'] = 8753.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    162828.000000
# mean         24.167582
# min           1.000000
# max          62.000000
val = tmp['value_as_number'].copy()
val[(val >= 40) | (val <= 10)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 29) | (tmp.value_as_number <= 22))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Bicarbonate_Serum_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Lymphocytes %
tmp=lab_concept.query('concept_name=="Lymphocytes %"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = '%'
tmp['unit_concept_id'] = 8554.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    161412.000000
# mean         26.946529
# min           0.600000
# max         100.000000
val = tmp['value_as_number'].copy()
val[(val >= 44) | (val <= 18)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Lymphocytes_%_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Neutrophils %
# tmp=lab_concept.query('concept_name=="Neutrophils %"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# tmp['unit_source_value'] = '%'
# tmp['unit_concept_id'] = 8554.0
# wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
# tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
# tmp.value_as_number.describe()
# # count    161438.000000
# # mean         61.215180
# # min           1.000000
# # max         100.000000
# val = tmp['value_as_number'].copy()
# val[(val >= 41.5) | (val <= 35)] = 'panic'
# val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
# val[val.apply(np.isreal)] = 'normal'
# tmp['val'] = 'Neutrophils_%_' + val
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Carbon dioxide serum/plasma
tmp=lab_concept.query('concept_name=="Carbon dioxide serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mM/L'
tmp['unit_concept_id'] = 8753.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    156489.000000
# mean         25.099279
# min           2.000000
# max         212.000000
val = tmp['value_as_number'].copy()
val[(val >= 40) | (val <= 10)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 29) | (tmp.value_as_number <= 23))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Carbon_dioxide_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Basophils/100 leukocytes in Blood by Automated count
tmp=lab_concept.query('concept_name=="Basophils/100 leukocytes in Blood by Automated count"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = '%'
tmp['unit_concept_id'] = 8554.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    145154.000000
# mean          0.395523
# min           0.000000
# max          35.000000
val = tmp['value_as_number'].copy()
val[(val >= 2)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Basophils/100_leukocytes_Blood_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Magnesium [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Magnesium [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    133606.000000
# mean          1.925238
# min           0.100000
# max          27.000000
val = tmp['value_as_number'].copy()
val[(val >= 9) | (val <= 1)] = 'panic'
val[val.apply(np.isreal) & (tmp.value_as_number >= 3.5)] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Magnesium_Plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Monocytes/100 leukocytes in Blood by Automated count | Monocytes [#/volume] in Blood
# tmp=lab_concept.query('concept_name=="Monocytes/100 leukocytes in Blood by Automated count" | concept_name=="Monocytes [#/volume] in Blood"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# tmp['unit_source_value'] = '%'
# tmp['unit_concept_id'] = 8554.0
# wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
# tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
# tmp.value_as_number.describe()
# # count    149317.000000
# # mean          8.236583
# # min           0.200000
# # max         100.000000
# val = tmp['value_as_number'].copy()
# val[(val >= 41.5) | (val <= 35)] = 'panic'
# val[val.apply(np.isreal) & ((tmp.value_as_number >= 39.4) | (tmp.value_as_number <= 36.1))] = 'abnormal'
# val[val.apply(np.isreal)] = 'normal'
# tmp['val'] = 'body_tmp_' + val
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Anion gap serum/plasma
tmp=lab_concept.query('concept_name=="Anion gap serum/plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
wrong_id = tmp.loc[tmp.value_as_number.isnull() | (tmp.value_as_number<=0),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    120314.000000
# mean         10.892890
# min           1.000000
# max         110.000000
val = tmp['value_as_number'].copy()
# count    120314.000000
# mean         10.892890
# min           1.000000
# max         110.000000
val[(val >= 30) ] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 10) | (tmp.value_as_number <= 3))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'Anion_gap_plasma_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)



# # Body weight
# tmp=lab_concept.query('concept_name=="Body weight"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==9529.0')
# concept_sub.query('concept_id==8739.0')
# tmp.loc[tmp.unit_concept_id==8739.0 ,'value_as_number'] = \
#     tmp.loc[tmp.unit_concept_id==8739.0 ,'value_as_number']/2.205
# tmp.loc[tmp.unit_source_value=='pound' ,'value_as_number'] = \
#     tmp.loc[tmp.unit_source_value=='pound' ,'value_as_number']/2.205
# tmp.loc[tmp.unit_source_value=='lb' ,'value_as_number'] = \
#     tmp.loc[tmp.unit_source_value=='lb' ,'value_as_number']/2.205
# wrong_id = tmp.loc[(tmp.value_as_number.isnull())|((tmp.unit_concept_id.isnull()) & tmp.unit_source_value.isnull()),'measurement_id']
# tmp['unit_source_value'] = 'kilogram'
# tmp['unit_concept_id'] = 9529.0
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

# Body mass index
tmp=lab_concept.query('concept_name=="Body mass index"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==9531.0')
wrong_id = tmp.loc[(tmp.value_as_number>90) | tmp.value_as_number.isnull() | (tmp.value_as_number<=1) ,'measurement_id']
tmp['unit_source_value'] = 'kilogram per square meter'
tmp['unit_concept_id'] = 9531.0
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    78488.000000
# mean        29.981638
# min          1.100000
# max         86.849998
val = tmp['value_as_number'].copy()
val[(val >= 30)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 25) | (tmp.value_as_number <= 18.5))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'bmi_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol [Mass/volume] in Serum or Plasma"')[to_take]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    49476.000000
# mean       190.175075
# min         18.000000
# max       3368.000000
val = tmp['value_as_number'].copy()
val[(val >= 240)] = 'panic'
val[val.apply(np.isreal) & ((tmp.value_as_number >= 200) | (tmp.value_as_number <= 125))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'cho_mass_' + val
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# Cholesterol in HDL [Mass/volume] in Serum or Plasma
tmp=lab_concept.query('concept_name=="Cholesterol in HDL [Mass/volume] in Serum or Plasma"')[to_take + ['person_id']]
tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
concept_sub.query('concept_id==8840.0')
tmp['unit_source_value'] = 'mg/dL'
tmp['unit_concept_id'] = 8840.0
wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
tmp = tmp.loc[~tmp.measurement_id.isin(wrong_id)]
tmp.value_as_number.describe()
# count    39728.000000
# mean        47.528494
# min          0.000000
# max        323.000000
tmp = pd.merge(tmp, person[['person_id','gender_source_value']], how = 'left')
val = tmp['value_as_number'].copy()
val[(val >= 95) | (val <= 40)] = 'panic'
val[(tmp.gender_source_value =='F') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 93) | (tmp.value_as_number <= 50))] = 'abnormal'
val[(tmp.gender_source_value =='M') & (val.apply(np.isreal)) & ((tmp.value_as_number >= 73) | (tmp.value_as_number <= 40))] = 'abnormal'
val[val.apply(np.isreal)] = 'normal'
tmp['val'] = 'cho_hdl_' + val
tmp = tmp.drop(columns=['person_id', 'gender_source_value'])
# lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# lab_concept = to_cover(lab_concept, tmp)
w_id = w_id.append(wrong_id)
tp1 = tp1.append(tmp)

# # Cholesterol in LDL [Mass/volume] in Serum or Plasma
# tmp=lab_concept.query('concept_name=="Cholesterol in LDL [Mass/volume] in Serum or Plasma"')[to_take]
# tmp.drop_duplicates(subset=['unit_concept_id','unit_source_value'])
# concept_sub.query('concept_id==8840.0')
# wrong_id = tmp.loc[tmp.value_as_number.isnull(),'measurement_id']
# # lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(wrong_id)]
# # lab_concept = to_cover(lab_concept, tmp)
# w_id = w_id.append(wrong_id)
# tp1 = tp1.append(tmp)

lab_concept = lab_concept.loc[~lab_concept.measurement_id.isin(w_id)]
lab_concept = to_cover(lab_concept, tp1)

###
tp1 = pd.merge(tp1, file[['measurement_id','person_id','measurement_date']])
tp1.to_csv('Cleaned_lab.csv', index=None)

tp1 = pd.read_csv('Cleaned_lab.csv',low_memory=False)
# read the visit data
parse_dates = ['visit_start_date', 'visit_start_datetime', 'visit_end_date', 'visit_end_datetime']
visit = pd.read_csv('trac_ 3772_schizophrenia_visit.csv', delimiter='|', date_parser=parse_dates, low_memory=False)
# keep only inpatient records (from Natalie, April 11, 2019 at 5:13 PM)
visit = visit.query('(visit_concept_id == 9201) | (visit_concept_id == 8717) | (visit_type_concept_id == 9201)')
# sort by date
visit = visit.sort_values(by=['visit_start_date'])
# keep first inpatient time for each person
visit = visit.drop_duplicates(['person_id'])
# add first inpatient time to lab record. (tp1 is the cleaned lab record)
tp1 = pd.merge(tp1, visit[['person_id','visit_start_date']], how='inner')
# screen
tp1 = tp1.query('measurement_date <= visit_start_date')
# further reduce to the defined cohort. (schi_cohort.csv is sent by Peng, 2019/3/29 (11:37))
cohort1 = pd.read_csv('./visit/schi_cohort_v1.csv')
cohort2 = pd.read_csv('./visit/schi_cohort_v2.csv')
cohort = set(cohort1.person_id).union(set(cohort2.person_id))
tp1 = tp1[tp1.person_id.isin(cohort)]


# model
import gensim
from gensim import corpora, models
import pyLDAvis.gensim
df = tp1[['val', 'person_id']].groupby(['person_id'])['val'].apply(lambda x: ' '.join(x)).reset_index()
data = df['val'].values.tolist()
res = [sent.split(' ') for sent in data]
dictionary = gensim.corpora.Dictionary(res)
count = 0
dictionary.filter_extremes(no_below=5, no_above=1.0, keep_n=10000)
bow_corpus = [dictionary.doc2bow(doc) for doc in res]
num_topic = 3
seed = 11
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topic, id2word=dictionary, passes=20, workers=10,
                                       iterations=500, random_state=seed)
# visualization
lda_display_bow = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics=True)
pyLDAvis.save_html(lda_display_bow, 'lab_topic_model_t3.html')


# create dataframe for condition topics
doc_topics = []
for bow in bow_corpus:
    doc_topics.append(dict(lda_model.get_document_topics(bow, per_word_topics=False)))
topics_emb = pd.DataFrame(doc_topics)
topics_emb.fillna(0, inplace=True)
topics_emb.columns = ['lab_prior_' + str(i) for i in range(1, num_topic + 1)]
topics_emb['person_id'] = df.person_id
topics_emb.to_csv('lab_prior_hospitalization_model_feat.csv', index=None)
